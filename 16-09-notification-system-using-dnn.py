# Production-grade Deep Neural Network for CTR Prediction
# This implementation includes best practices for scalability, monitoring, and deployment

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from abc import ABC, abstractmethod
import json
import os
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CTRConfig:
    """Configuration class for CTR model parameters"""
    # Model architecture
    embedding_dim: int = 64
    hidden_units: List[int] = None
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = 'relu'
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 1024
    epochs: int = 10
    validation_split: float = 0.2
    
    # Regularization
    l1_reg: float = 1e-5
    l2_reg: float = 1e-5
    
    # Feature engineering
    numerical_features: List[str] = None
    categorical_features: List[str] = None
    max_vocab_size: int = 10000
    
    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [512, 256, 128]
        if self.numerical_features is None:
            self.numerical_features = []
        if self.categorical_features is None:
            self.categorical_features = []

class FeatureProcessor:
    """Handles feature preprocessing for CTR prediction"""
    
    def __init__(self, config: CTRConfig):
        self.config = config
        self.feature_columns = []
        self.feature_specs = {}
        
    def create_feature_columns(self, train_df: pd.DataFrame) -> List[tf.feature_column.FeatureColumn]:
        """Create TensorFlow feature columns"""
        feature_columns = []
        
        # Numerical features
        for feature in self.config.numerical_features:
            if feature in train_df.columns:
                # Normalize numerical features
                normalizer = tf.feature_column.numeric_column(
                    feature, 
                    normalizer_fn=lambda x: tf.nn.l2_normalize(x, axis=1)
                )
                feature_columns.append(normalizer)
                logger.info(f"Added numerical feature: {feature}")
        
        # Categorical features
        for feature in self.config.categorical_features:
            if feature in train_df.columns:
                vocab_size = min(train_df[feature].nunique(), self.config.max_vocab_size)
                categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                    feature, hash_bucket_size=vocab_size
                )
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column, dimension=self.config.embedding_dim
                )
                feature_columns.append(embedding_column)
                logger.info(f"Added categorical feature: {feature} with vocab size: {vocab_size}")
        
        self.feature_columns = feature_columns
        return feature_columns
    
    def create_feature_spec(self) -> Dict[str, tf.io.FixedLenFeature]:
        """Create feature specification for parsing"""
        feature_spec = {}
        
        for feature in self.config.numerical_features:
            feature_spec[feature] = tf.io.FixedLenFeature([], tf.float32)
            
        for feature in self.config.categorical_features:
            feature_spec[feature] = tf.io.FixedLenFeature([], tf.string)
            
        self.feature_specs = feature_spec
        return feature_spec

class CTRModel(tf.keras.Model):
    """Production-grade Deep Neural Network for CTR prediction"""
    
    def __init__(self, config: CTRConfig, feature_columns: List[tf.feature_column.FeatureColumn]):
        super(CTRModel, self).__init__()
        self.config = config
        self.feature_columns = feature_columns
        
        # Feature layer
        self.feature_layer = tf.keras.utils.DenseFeatures(feature_columns)
        
        # Deep neural network layers
        self.dnn_layers = []
        for i, units in enumerate(config.hidden_units):
            # Dense layer
            dense_layer = tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=config.l1_reg, l2=config.l2_reg
                ),
                name=f'dense_{i}'
            )
            self.dnn_layers.append(dense_layer)
            
            # Batch normalization
            if config.use_batch_norm:
                bn_layer = tf.keras.layers.BatchNormalization(name=f'bn_{i}')
                self.dnn_layers.append(bn_layer)
            
            # Activation
            activation_layer = tf.keras.layers.Activation(
                config.activation, name=f'activation_{i}'
            )
            self.dnn_layers.append(activation_layer)
            
            # Dropout
            dropout_layer = tf.keras.layers.Dropout(
                config.dropout_rate, name=f'dropout_{i}'
            )
            self.dnn_layers.append(dropout_layer)
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            1, 
            activation='sigmoid',
            name='ctr_prediction'
        )
        
        # Compile metrics
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    
    def call(self, inputs, training=None):
        """Forward pass"""
        # Feature transformation
        x = self.feature_layer(inputs)
        
        # Deep neural network
        for layer in self.dnn_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Output prediction
        output = self.output_layer(x)
        return output

class CTRTrainer:
    """Training pipeline for CTR model"""
    
    def __init__(self, config: CTRConfig):
        self.config = config
        self.model = None
        self.feature_processor = FeatureProcessor(config)
        self.training_history = None
        
    def prepare_dataset(self, df: pd.DataFrame, is_training: bool = True) -> tf.data.Dataset:
        """Prepare TensorFlow dataset from pandas DataFrame"""
        # Separate features and labels
        if 'label' in df.columns:
            labels = df['label'].values
            features_df = df.drop('label', axis=1)
        else:
            labels = None
            features_df = df
        
        # Create feature dictionary
        feature_dict = {}
        for col in features_df.columns:
            if col in self.config.numerical_features:
                feature_dict[col] = features_df[col].values.astype(np.float32)
            elif col in self.config.categorical_features:
                feature_dict[col] = features_df[col].astype(str).values
        
        # Create dataset
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((feature_dict, labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(feature_dict)
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(self.config.batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        else:
            dataset = dataset.batch(self.config.batch_size)
            
        return dataset
    
    def build_model(self, train_df: pd.DataFrame) -> CTRModel:
        """Build and compile the CTR model"""
        # Create feature columns
        feature_columns = self.feature_processor.create_feature_columns(train_df)
        
        # Build model
        self.model = CTRModel(self.config, feature_columns)
        
        # Build the model by calling it once
        sample_input = {}
        for feature in self.config.numerical_features:
            if feature in train_df.columns:
                sample_input[feature] = tf.constant([[0.0]])
        for feature in self.config.categorical_features:
            if feature in train_df.columns:
                sample_input[feature] = tf.constant([['sample']])
        
        if sample_input:
            _ = self.model(sample_input)
            logger.info(f"Model built successfully with {self.model.count_params()} parameters")
        
        return self.model
    
    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the CTR model"""
        logger.info("Starting model training...")
        
        # Build model
        self.build_model(train_df)
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df, is_training=True)
        
        validation_data = None
        if val_df is not None:
            validation_data = self.prepare_dataset(val_df, is_training=False)
        elif self.config.validation_split > 0:
            # Use built-in validation split
            validation_data = None
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc' if validation_data else 'auc',
                patience=3,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_ctr_model.h5',
                monitor='val_auc' if validation_data else 'auc',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train model
        self.training_history = self.model.fit(
            train_dataset,
            epochs=self.config.epochs,
            validation_data=validation_data,
            validation_split=self.config.validation_split if validation_data is None else 0,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed successfully")
        return self.training_history.history
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the model on test data"""
        logger.info("Evaluating model...")
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Prepare test dataset
        test_dataset = self.prepare_dataset(test_df, is_training=False)
        
        # Model evaluation
        results = self.model.evaluate(test_dataset, verbose=1)
        
        # Create results dictionary
        eval_results = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            eval_results[metric_name] = results[i]
        
        # Additional metrics using sklearn
        predictions = self.predict(test_df.drop('label', axis=1))
        true_labels = test_df['label'].values
        
        eval_results['sklearn_auc'] = roc_auc_score(true_labels, predictions)
        eval_results['sklearn_logloss'] = log_loss(true_labels, predictions)
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Prepare dataset
        dataset = self.prepare_dataset(df, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(dataset)
        return predictions.flatten()
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Save model
        self.model.save(path)
        
        # Save configuration
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'CTRTrainer':
        """Load a trained model"""
        # Load configuration
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = CTRConfig(**config_dict)
        trainer = cls(config)
        
        # Load model
        trainer.model = tf.keras.models.load_model(path)
        
        logger.info(f"Model loaded from {path}")
        return trainer

# Production serving utilities
class CTRServingModel:
    """Production serving wrapper for CTR model"""
    
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        
        # Load configuration
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            self.config_dict = json.load(f)
    
    def predict_single(self, features: Dict[str, Any]) -> float:
        """Predict CTR for a single example"""
        # Convert to batch format
        batch_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                batch_features[key] = tf.constant([[value]], dtype=tf.float32)
            else:
                batch_features[key] = tf.constant([[str(value)]])
        
        prediction = self.model(batch_features)
        return float(prediction.numpy()[0, 0])
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Predict CTR for a batch of examples"""
        predictions = []
        for features in features_list:
            pred = self.predict_single(features)
            predictions.append(pred)
        return predictions

# Example usage and demonstration
def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    data = {
        # Numerical features
        'user_age': np.random.randint(18, 70, n_samples),
        'ad_price': np.random.exponential(10, n_samples),
        'user_activity_score': np.random.beta(2, 5, n_samples),
        
        # Categorical features
        'user_gender': np.random.choice(['M', 'F'], n_samples),
        'ad_category': np.random.choice(['electronics', 'clothing', 'sports', 'books'], n_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
        
        # Label (CTR)
        'label': np.random.binomial(1, 0.1, n_samples)  # 10% CTR
    }
    
    return pd.DataFrame(data)

# Example training script
def main():
    """Main training function"""
    print("Creating production-grade CTR prediction system...")
    
    # Create sample data
    df = create_sample_data(10000)
    
    # Split data
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"Data split: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")
    
    # Configure model
    config = CTRConfig(
        embedding_dim=32,
        hidden_units=[256, 128, 64],
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=512,
        epochs=10,
        numerical_features=['user_age', 'ad_price', 'user_activity_score'],
        categorical_features=['user_gender', 'ad_category', 'device_type']
    )
    
    # Initialize trainer
    trainer = CTRTrainer(config)
    
    # Train model
    history = trainer.train(train_df, val_df)
    
    # Evaluate model
    results = trainer.evaluate(test_df)
    print(f"Test Results: AUC = {results['sklearn_auc']:.4f}, LogLoss = {results['sklearn_logloss']:.4f}")
    
    # Save model
    trainer.save_model('production_ctr_model')
    
    # Demonstrate serving
    serving_model = CTRServingModel('production_ctr_model')
    
    # Single prediction example
    sample_features = {
        'user_age': 25,
        'ad_price': 15.5,
        'user_activity_score': 0.8,
        'user_gender': 'M',
        'ad_category': 'electronics',
        'device_type': 'mobile'
    }
    
    ctr_prediction = serving_model.predict_single(sample_features)
    print(f"CTR Prediction for sample user: {ctr_prediction:.4f}")

if __name__ == "__main__":
    main()
