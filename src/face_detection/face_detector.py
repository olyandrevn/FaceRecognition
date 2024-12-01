import cv2
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class Bbox:
    """
    Represents a bounding box for a detected face with top-left and bottom-right corners.
    """
    x1: int
    y1: int
    x2: int
    y2: int


class FaceDetector:
    def __init__(self, config):
        self.config = config

        model_path = config.get("model_path")
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load a pretrained face detection model.
        
        Args:
            model_path (str): Path to the model file.
        
        Returns:
            Loaded model.
        """
        pass

    def __call__(self, img: np.ndarray) -> list[Bbox]:
        """
        Detect faces in an image and return their bounding boxes.
        
        Args:
            img (np.ndarray): Input image in BGR format.
        
        Returns:
            list[Bbox]: A list of bounding boxes for detected faces.
        """
        pass
