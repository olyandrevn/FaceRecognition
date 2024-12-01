import cv2
import numpy as np
from typing import Optional
from src.face_detection.face_detector import Bbox

class FaceExtractor:
    def __init__(self, config):
        pass

    def extract(self, image: np.ndarray, bbox: Bbox) -> np.ndarray:
        """
        Extracts the face region from the image using the bounding box.
        
        Args:
            image (np.ndarray): Input image in BGR format.
            bbox (Bbox): Bounding box specifying the face region.
        
        Returns:
            np.ndarray: Cropped face region.
        """
        pass
