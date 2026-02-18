import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

class PredictiveModel:
    def __init__(self):
        self.model = None
        
    def build_model(self, input_shape):
        """
        Builds a neural network model for price prediction.
        Args:
            input_shape (tuple): Shape of the input data.
        Returns:
            Model: Compiled Keras model.
        """
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_model(self, X_train, y_train, batch_size=32, epochs=10):
        """
        Trains the neural network model.
        Args:
            X_train (numpy array): Training data features.
            y_train (numpy array): Training data labels.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs.
        """
        self.model = self.build_model(X_train.shape[1:])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5', save_best_only=True
        )
        
        self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[checkpoint],
            validation_split=0.2
        )

    def predict(self, X):
        """
        Makes predictions using the trained model.
        Args:
            X (numpy array): Input data for prediction.
        Returns:
            numpy array: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        return self.model.predict(X)