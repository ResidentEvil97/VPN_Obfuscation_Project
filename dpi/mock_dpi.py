import numpy as np
import joblib
import os

class MockDPI:
    def __init__(self, model_path = "dpi/models/random_forest_dpi.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ML model not found at {model_path}. Train and save a model first.")
        self.model = joblib.load(model_path)

    def features_to_vector(self, features):
        # Map your environment's features to the model input order
        # You must provide 'duration', 'flowPktsPerSecond', 'flowBytesPerSecond'
        return np.array([
            features.get('duration', 0.0),
            features.get('flowPktsPerSecond', 0.0),
            features.get('flowBytesPerSecond', 0.0)
        ]).reshape(1, -1)

    def detect(self, features):
        vector = self.features_to_vector(features)
        pred = self.model.predict(vector)
        return bool(pred[0])  # True if VPN, False if Non-VPN