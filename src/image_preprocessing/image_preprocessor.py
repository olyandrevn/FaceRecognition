import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, config):
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for model inference.
       
        Args:
            image (np.ndarray): Input image in BGR format.
        
        Returns:
            np.ndarray: Preprocessed image.
        """
        pass