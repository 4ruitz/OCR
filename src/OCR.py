import time
import queue
import threading
import logging
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from torchvision import transforms
from PIL import Image
import torch
from snap7.client import Client
from snap7.util import set_int, set_bool, get_bool
from Model import CNN


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PLCConnection:
    def __init__(self, plc_ip, db_number, start_offset):
        self.plc_ip = plc_ip
        self.db_number = db_number
        self.start_offset = start_offset
        self.plc = Client()

    def connect(self):
        try:
            self.plc.connect(self.plc_ip, 0, 1)
            logging.info(f"Connected to PLC at {self.plc_ip}.")
        except Exception as e:
            logging.error(f"Error connecting to PLC: {str(e)}")
            raise

    def disconnect(self):
        try:
            self.plc.disconnect()
            logging.info("PLC connection closed.")
        except Exception as e:
            logging.error(f"Error disconnecting from PLC: {str(e)}")

    def send_data(self, int_value):
        try:
            bool_data = bytearray(1)
            data = bytearray(2)
            set_int(data, 0, int_value)
            self.plc.db_write(self.db_number, self.start_offset, data)
            logging.info(
                f"Integer {int_value} written to DB{self.db_number} at offset {self.start_offset}."
            )
            set_bool(bool_data, 0, 0, True)
            self.plc.db_write(self.db_number, self.start_offset + 2, bool_data)
            logging.info(f"Boolean bit set to True.")

        except Exception as e:
            logging.error(f"PLC communication error: {str(e)}")

class OCR:
    def __init__(self, model_path="model.pth", plc_connection=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)["model_state_dict"]
        )
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.plc_connection = plc_connection

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert("L")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(image_tensor)
                pred = output.argmax(dim=1).item()

            logging.info(f"Prediction for {image_path}: {pred}")
            self.send_PLC(pred)

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")

    def send_PLC(self, int_value):
        if self.plc_connection:
            self.plc_connection.send_data(int_value)





class ImageHandler(FileSystemEventHandler):
    def __init__(self, ocr):
        self.ocr = ocr
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
                time.sleep(0.5)  # Small delay to simulate image processing
                self.ocr.predict(file_path)
            except Exception as e:
                logging.error(f"Error in processing thread: {str(e)}")
            finally:
                self.processing_queue.task_done()


def main():
    watch_dir = "FTP_Folder"
    logging.info(f"Watching directory: {watch_dir}")
    plc_ip = "192.168.0.1"

    plc_connection = PLCConnection(plc_ip=plc_ip, db_number=1, start_offset=0)
    plc_connection.connect()

    ocr = OCR(model_path="model.pth", plc_connection=plc_connection)
    event_handler = ImageHandler(ocr)
    observer = Observer()
    observer.schedule(event_handler, path=watch_dir, recursive=False)
    observer.start()

    logging.info("\nWatching for new images...\nSupported formats: .png, .jpg, .jpeg")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Stopping directory watch...")
        plc_connection.disconnect()
    finally:
        observer.join()


if __name__ == "__main__":
    main()
