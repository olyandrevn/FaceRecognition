import cv2
import torch
import numpy as np


class FaceRecognizer:
    def __init__(self, config):
        self.config = config
        
        model_path = config.get("model_path")
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load a pretrained face recognition model.
        
        Args:
            model_path (str): Path to the model file.
        
        Returns:
            Loaded model.
        """
        pass

    def __call__(self, img: np.ndarray) -> int:
        """
        Recognize face in an image and return their ID.
        
        Args:
            img (np.ndarray): Input image in BGR format.
        
        Returns:
            int: An ID for the recognized face.
        """
        pass
