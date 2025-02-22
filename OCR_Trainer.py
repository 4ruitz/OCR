import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import keras
import numpy as np
import cv2
import requests
from PIL import Image
from typing import List, Optional
from huggingface_hub import hf_hub_download
import tensorflow as tf
import pickle

class ImageTokenizer:
    def __init__(self):
        self.unique_pixels = set()
        self.pixel_to_token = {}
        self.token_to_pixel = {}

    def fit(self, images):
        for image in images:
            self.unique_pixels.update(np.unique(image))
        self.pixel_to_token = {pixel: i for i, pixel in enumerate(sorted(self.unique_pixels))}
        self.token_to_pixel = {i: pixel for pixel, i in self.pixel_to_token.items()}

    def tokenize(self, images):
        return np.vectorize(self.pixel_to_token.get)(images)

    def detokenize(self, tokens):
        return np.vectorize(self.token_to_pixel.get)(tokens)

class MNISTPredictor:
    def __init__(self, model_name):
        # Download the model and tokenizer files
        model_path = hf_hub_download(repo_id=model_name, filename="mnist_model.keras")
        tokenizer_path = hf_hub_download(repo_id=model_name, filename="mnist_tokenizer.pkl")

        # Load the model and tokenizer
        self.model = keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)

    def extract_features(self, image: Image.Image) -> List[np.ndarray]:
        """Extract features from the image for multiple digits."""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_images = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) > 50:  # Adjust this threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                roi = thresh[y:y+h, x:x+w]
                resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                digit_images.append(resized.reshape((28, 28, 1)).astype('float32') / 255)

        return digit_images

    def predict(self, image: Image.Image) -> Optional[List[int]]:
        """Predict digits in the image."""
        try:
            digit_images = self.extract_features(image)
            tokenized_images = [self.tokenizer.tokenize(img) for img in digit_images]
            predictions = self.model.predict(np.array(tokenized_images), verbose=0)
            return np.argmax(predictions, axis=1).tolist()
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

def save_predictions_to_file(predictions: List[int], output_path: str) -> None:
    """Save predictions to a text file."""
    try:
        with open(output_path, 'w') as f:
            f.write(f"Predicted digits are: {', '.join(map(str, predictions))}\n")
    except Exception as e:
        print(f"Error saving predictions to file: {e}")

class DirectoryWatcher(FileSystemEventHandler):
    def __init__(self, predictor, output_path):
        self.predictor = predictor
        self.output_path = output_path

    def on_created(self, event):
        if event.is_directory:
            return
        print(f"New image detected: {event.src_path}")
        try:
            image = Image.open(event.src_path)
            digits = self.predictor.predict(image)
            if digits is not None:
                print(f"Predicted digits: {digits}")
                save_predictions_to_file(digits, self.output_path)
                print(f"Predictions saved to {self.output_path}")
            else:
                print("Failed to predict digits.")
        except Exception as e:
            print(f"Error processing image {event.src_path}: {e}")

def main(watch_directory: str, model_name: str, output_path: str) -> None:
    try:
        predictor = MNISTPredictor(model_name)

        # Set up the directory watcher
        event_handler = DirectoryWatcher(predictor, output_path)
        observer = Observer()
        observer.schedule(event_handler, watch_directory, recursive=False)

        # Start watching the directory
        print(f"Watching directory: {watch_directory} for new images...")
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    watch_directory = "C:\dev\OCR\watch_images"  # Directory to watch for new images
    model_name = "0xnu/mnist-ocr"  # Model name from Hugging Face Hub
    output_path = "predictions.txt"  # File to save predictions

    main(watch_directory, model_name, output_path)
