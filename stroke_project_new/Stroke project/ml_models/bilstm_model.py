import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, Bidirectional, LSTM,
    Dense, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class BiLSTMStrokeDetector:
    def __init__(self, input_shape, bilstm_units=128, dropout_rate=0.3):
        self.input_shape = input_shape
        self.bilstm_units = bilstm_units
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # -------- CNN Feature Extractor --------
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        # -------- Reshape for BiLSTM --------
        x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

        # -------- BiLSTM --------
        x = Bidirectional(LSTM(self.bilstm_units, return_sequences=False))(x)
        x = Dropout(self.dropout_rate)(x)

        # -------- Output --------
        outputs = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs, outputs)
        return self.model

    def compile_model(self, learning_rate=0.0001):
        self.model.compile(
            optimizer=Adam(learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, batch_size, epochs, class_weight=None):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=1
        )

    def save_model(self, path):
        self.model.save(path)
