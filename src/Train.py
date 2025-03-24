import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from Model import CNN
import os
from PIL import Image
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import queue
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class Trainer:
    def __init__(self, model_path="untuned_model.pth", lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.model = CNN().to(self.device)
        self.model_path = model_path
        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.custom_images = []
        self.custom_labels = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_mnist(self, epochs=10, batch_size=64):
        logging.info("Training on MNIST dataset...")

        train_loader = DataLoader(
            datasets.MNIST("data", train=True, download=True, transform=self.transform),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = DataLoader(
            datasets.MNIST("data", train=False, transform=self.transform),
            batch_size=1000,
        )

        best_accuracy = 0

        for epoch in range(epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    logging.info(
                        f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}]	Loss: {loss.item():.6f}"
                    )

            accuracy = self.test(test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model()

        logging.info(f"Best accuracy: {best_accuracy:.2f}%")

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                correct += output.argmax(dim=1).eq(target).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        logging.info(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )
        return accuracy

    def save_model(self):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.model_path,
        )
        logging.info(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logging.info("Model loaded successfully")
            return True
        logging.info("No saved model found. Will train from scratch.")
        return False

    def wait_for_file_ready(self, file_path, timeout=5, check_interval=0.1):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with open(file_path, "rb") as f:
                    f.read()
                    return True
            except (IOError, OSError):
                time.sleep(check_interval)
        return False

    def is_valid_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def process_new_image(self, image_path):
        image_path = os.path.abspath(image_path)

        if not self.wait_for_file_ready(image_path):
            logging.info(f"Timeout waiting for file to be ready: {image_path}")
            return

        if not self.is_valid_image(image_path):
            logging.info(f"Invalid or corrupted image file: {image_path}")
            return

        image = Image.open(image_path).convert("L")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = output.argmax(dim=1).item()

        print(f"\nPrediction for {os.path.basename(image_path)}: {pred}")
        label = input(
            "Enter the correct digit (0-9) or press Enter to confirm: "
        ).strip()

        if label == "":
            logging.info("Prediction confirmed")
            return

        try:
            label = int(label)
            if 0 <= label <= 9:
                self.custom_images.append(image)
                self.custom_labels.append(label)
                logging.info(f"Image added with label {label}")
                if len(self.custom_images) >= 10:
                    self.fine_tune()
            else:
                logging.info("Invalid label. Must be between 0 and 9")
        except ValueError:
            logging.error("Invalid input. Must be a number between 0 and 9")

    def fine_tune(self, epochs=3):
        if len(self.custom_images) == 0:
            return

        logging.info("\nFine-tuning model on custom images...")
        custom_dataset = CustomDataset(
            self.custom_images, self.custom_labels, transform=self.transform
        )
        custom_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for data, target in custom_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

            logging.info(f"Fine-tuning Epoch: {epoch+1}/{epochs}")

        self.save_model()
        logging.info("Fine-tuning completed")


class ImageHandler(FileSystemEventHandler):
    def __init__(self, trainer):
        self.trainer = trainer
        self.processing_queue = queue.Queue()
        threading.Thread(target=self._process_queue, daemon=True).start()

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            logging.info(f"New image detected: {event.src_path}")
            self.processing_queue.put(event.src_path)

    def _process_queue(self):
        while True:
            try:
                file_path = self.processing_queue.get()
                time.sleep(0.5)
                self.trainer.process_new_image(file_path)
            except Exception as e:
                logging.error(f"Error in processing thread: {str(e)}")
            finally:
                self.processing_queue.task_done()


def main():
    watch_dir = Path("FTP_Folder").resolve()
    watch_dir.mkdir(exist_ok=True)
    logging.info(f"Watch directory: {watch_dir}")

    trainer = Trainer()
    if not trainer.load_model():
        trainer.train_mnist()

    event_handler = ImageHandler(trainer)
    observer = Observer()
    observer.schedule(event_handler, path=str(watch_dir), recursive=False)
    observer.start()

    logging.info(
        f"\nWatching directory '{watch_dir}' for new images...\nSupported formats: .png, .jpg, .jpeg"
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.error("\nStopping directory watch...")
    observer.join()


if __name__ == "__main__":
    main()
