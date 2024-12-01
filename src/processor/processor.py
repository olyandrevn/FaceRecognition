import threading
import queue
import cv2
from datetime import datetime
from src.image_preprocessing.image_preprocessor import ImagePreprocessor
from src.face_extraction.face_extractor import FaceExtractor
from src.face_detection.face_detector import FaceDetector
from src.face_recognition.face_recognizer import FaceRecognizer
from src.database.database import Database

class Processor:
    """
    Processor pipeline:
    1. Preprocess images.
    2. Detect faces.
    3. Extract faces.
    4. Recognize faces and output IDs with metadata.
    """

    def __init__(self, config, database):
        self.preprocessor = ImagePreprocessor(config.img_preprocessor)
        self.extractor = FaceExtractor(config.face_extractor)
        self.detector = FaceDetector(config.face_detection)
        self.recognizer = FaceRecognizer(config.face_recognition)
        self.database = database

        self.input_queue = queue.Queue()  # Queue for metadata and image paths
        self.image_queue = queue.Queue()  # Queue for preprocessed images with metadata
        self.face_queue = queue.Queue()  # Queue for cropped faces with metadata
        self.output_queue = queue.Queue()  # Queue for recognized IDs with metadata

        self._start_workers()

    def _start_workers(self):
        """
        Start worker threads for each processing stage.
        """
        threading.Thread(target=self._preprocess_worker, daemon=True).start()
        threading.Thread(target=self._detect_faces_worker, daemon=True).start()
        threading.Thread(target=self._recognize_faces_worker, daemon=True).start()
        threading.Thread(target=self._write_to_db_worker, daemon=True).start()

    def enqueue_image(self, metadata):
        """
        Add a raw image path with metadata to the input queue.

        Args:
            metadata (dict): Metadata dictionary, must include `image_path`.
        """
        if "image_path" not in metadata or not isinstance(metadata["image_path"], str):
            print(f"Invalid metadata or missing image_path: {metadata}")
            return
        self.input_queue.put(metadata)
        print(f"Enqueued image with metadata: {metadata}")

    def _preprocess_worker(self):
        """
        Worker to preprocess images and enqueue them for face detection.
        """
        while True:
            metadata = self.input_queue.get()
            image_path = metadata["image_path"]
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image at {image_path}")
                continue

            # Preprocess the image
            preprocessed_image = self.preprocessor(image)
            metadata["preprocessed_image"] = preprocessed_image
            self.image_queue.put(metadata)

    def _detect_faces_worker(self):
        """
        Worker to detect faces in preprocessed images and enqueue face crops with metadata.
        """
        while True:
            metadata = self.image_queue.get()
            preprocessed_image = metadata["preprocessed_image"]
            bbox_detections = self.detector(preprocessed_image)
            print(f"Detected {len(bbox_detections)} faces.")

            for bbox in bbox_detections:
                face_image = self.extractor(preprocessed_image, bbox)
                face_metadata = metadata.copy()
                face_metadata["face_image"] = face_image
                face_metadata["bbox"] = bbox
                self.face_queue.put(face_metadata)

    def _recognize_faces_worker(self):
        """
        Worker to recognize faces and enqueue results with metadata.
        """
        while True:
            metadata = self.face_queue.get()
            face_image = metadata["face_image"]
            customer_id = self.recognizer(face_image)
            print(f"Recognized Face ID: {customer_id}")

            # Add face ID to metadata
            metadata["customer_id"] = customer_id
            self.output_queue.put(metadata)

    def _write_to_db_worker(self):
        """
        Worker to write recognized face results to the database.
        """
        while True:
            metadata = self.output_queue.get()
            self.database.insert_record(metadata)
            print(f"Inserted Face ID: {metadata['customer_id']} into the database.")


if __name__ == "__main__":
    config = Config()

    db = Database(config.database)
    columns = ["customer_id", "timestamp", "store_id"]
    db.create_table("store_001_customers", columns)

    processor = Processor(config)

    # Enqueue images with metadata
    processor.enqueue_image({"image_path": "path/to/image1.jpg", "timestamp": "2024-11-26T10:05:00", "store_id": "store_001"})
    processor.enqueue_image({"image_path": "path/to/image2.jpg", "timestamp": "2024-11-26T10:07:00", "store_id": "store_001"})
