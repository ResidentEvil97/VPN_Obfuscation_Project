"""
MockDPI: A simulated Deep Packet Inspection (DPI) system
=========================================================

This class provides a simulated DPI system that can be used to detect
VPN traffic in your environment. It uses a pre-trained scikit-learn
random forest model to classify traffic as either VPN or Non-VPN.

The model is trained on the CICFlowMeter dataset, which can be found
at https://www.unb.ca/cic/datasets/ids.html. The dataset contains
labeled traffic captures of various protocols, including VPN.

The class provides a simple interface for detecting VPN traffic. You
can use the `detect` method to provide a dictionary of features and
get a boolean result indicating whether the traffic is VPN or Non-VPN.

The features required for detection are:

* `duration`: The duration of the flow in seconds
* `flowPktsPerSecond`: The number of packets per second in the flow
* `flowBytesPerSecond`: The number of bytes per second in the flow

You can use the `features_to_vector` method to convert your features
into the format required by the model.

The class is designed to be easy to use and flexible. You can use it
as-is or modify it to fit your specific needs.
"""

import numpy as np
import joblib
import os


class MockDPI:
    """
    A simulated Deep Packet Inspection (DPI) system.

    This class provides a simulated DPI system that can be used to detect
    VPN traffic in your environment. It uses a pre-trained scikit-learn
    random forest model to classify traffic as either VPN or Non-VPN.
    """

    def __init__(self, model_path="dpi/models/random_forest_dpi.pkl"):
        """
        Initialize the MockDPI instance.

        Args:
            model_path (str): The path to the pre-trained model. Defaults to
                "dpi/models/random_forest_dpi.pkl".
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ML model not found at {model_path}. Train and save a model first."
            )
        self.model = joblib.load(model_path)

    def features_to_vector(self, features):
        """
        Convert a dictionary of features into a vector for the model.

        Args:
            features (dict): A dictionary containing the features to convert.

        Returns:
            numpy.ndarray: A vector containing the features in the order
                required by the model.
        """
        # Map your environment's features to the model input order
        # You must provide 'duration', 'flowPktsPerSecond', 'flowBytesPerSecond'
        return np.array([
            features.get('duration', 0.0),
            features.get('flowPktsPerSecond', 0.0),
            features.get('flowBytesPerSecond', 0.0)
        ]).reshape(1, -1)

    def detect(self, features):
        """
        Detect whether the traffic is VPN or Non-VPN.

        Args:
            features (dict): A dictionary containing the features to use for
                detection.

        Returns:
            bool: True if the traffic is VPN, False if it is Non-VPN.
        """
        vector = self.features_to_vector(features)
        pred = self.model.predict(vector)
        return bool(pred[0])  # True if VPN, False if Non-VPN