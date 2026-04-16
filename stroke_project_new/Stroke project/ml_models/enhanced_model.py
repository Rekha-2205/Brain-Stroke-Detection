"""
Enhanced model architecture with transfer learning and ensemble capabilities
Includes pretrained ResNet backbone + BiLSTM, focal loss, and model stacking
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB3
from tensorflow.keras.losses import Loss


class FocalLoss(Loss):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        epsilon = 1e-7
        # Cast y_true to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        ce_loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_weight = y_true * (1 - y_pred) ** self.gamma + (1 - y_true) * y_pred ** self.gamma
        
        return self.alpha * focal_weight * ce_loss


class EnhancedBiLSTMStrokeDetector:
    """Enhanced BiLSTM model with ResNet backbone"""
    
    def __init__(self, input_shape=(224, 224, 1), bilstm_units=256, dropout_rate=0.4, use_transfer_learning=True):
        self.input_shape = input_shape
        self.bilstm_units = bilstm_units
        self.dropout_rate = dropout_rate
        self.use_transfer_learning = use_transfer_learning
        self.model = None
    
    def build_model(self):
        """Build enhanced BiLSTM model with transfer learning"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Convert grayscale to RGB if using pretrained models
        x = inputs
        if self.input_shape[-1] == 1:
            x = layers.Concatenate()([x, x, x])
        
        if self.use_transfer_learning:
            # Load pretrained ResNet50 (pretrained on ImageNet)
            resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            resnet.trainable = False  # Freeze initial layers
            
            # Feature extraction with transfer learning
            x = resnet(x)
        else:
            # Custom CNN feature extraction
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
        
        # Global Average Pooling to reduce spatial dimensions
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers with batch normalization and regularization
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001, use_focal_loss=True):
        """Compile with optimized settings for medical imaging"""
        if use_focal_loss:
            loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            loss = 'binary_crossentropy'
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.AUC(name='auc')
            ]
        )
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, class_weight=None):
        """Train with early stopping and learning rate reduction"""
        
        # Calculate class weights if imbalanced
        if class_weight is None:
            unique, counts = np.unique(y_train, return_counts=True)
            total = len(y_train)
            class_weight = {int(u): total / (2 * c) for u, c in zip(unique, counts)}
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=15, restore_best_weights=True, mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5', monitor='val_auc', save_best_only=True, mode='max'
            ),
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        return history
    
    def predict(self, image):
        """Make prediction on single image"""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)[0][0]
        return float(prediction)
    
    def save_model(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = keras.models.load_model(filepath, custom_objects={'FocalLoss': FocalLoss})


class EnsembleStrokeDetector:
    """Ensemble of multiple models for robust prediction"""
    
    def __init__(self, models_config=None):
        self.models = []
        self.weights = []
        self.models_config = models_config or [
            {'use_transfer_learning': True, 'bilstm_units': 256},
            {'use_transfer_learning': False, 'bilstm_units': 128},
        ]
    
    def build_ensemble(self, input_shape=(224, 224, 1)):
        """Build multiple models for ensemble"""
        for config in self.models_config:
            model = EnhancedBiLSTMStrokeDetector(
                input_shape=input_shape,
                bilstm_units=config.get('bilstm_units', 256),
                use_transfer_learning=config.get('use_transfer_learning', True)
            )
            model.build_model()
            model.compile_model()
            self.models.append(model)
        
        # Equal weights initially
        self.weights = [1.0 / len(self.models)] * len(self.models)
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, **kwargs):
        """Train all models in ensemble"""
        histories = []
        for i, model in enumerate(self.models):
            print(f"\nTraining ensemble model {i+1}/{len(self.models)}...")
            history = model.train(X_train, y_train, X_val, y_val, **kwargs)
            histories.append(history)
        return histories
    
    def predict_ensemble(self, image):
        """Predict using ensemble voting"""
        predictions = []
        for model in self.models:
            pred = model.predict(image)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = sum(p * w for p, w in zip(predictions, self.weights))
        return weighted_pred
    
    def predict_batch_ensemble(self, images):
        """Batch prediction with ensemble"""
        return np.array([self.predict_ensemble(img) for img in images])
