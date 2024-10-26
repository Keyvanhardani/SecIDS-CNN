import tensorflow as tf
import pandas as pd

class SecIDSModel:
    def __init__(self, model_path="SecIDS-CNN.h5"):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, data):
        # Preprocess data if needed (assume data is a Pandas DataFrame)
        processed_data = self.preprocess_data(data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        # Convert predictions to readable format if needed
        return ["Attack" if pred > 0.5 else "Benign" for pred in predictions]

    def preprocess_data(self, data):
        # Placeholder for preprocessing logic, adjust according to your needs
        # For example, you may need to scale or reshape data
        return data.values
