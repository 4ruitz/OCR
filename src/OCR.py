import time
import queue
import threading
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from torchvision import transforms
from PIL import Image
import torch
from Model import CNN

class BasePLCClient:
    def connect(self):
        raise NotImplementedError("Connect method must be implemented.")

    def disconnect(self):
        raise NotImplementedError("Disconnect method must be implemented.")

    def write_int(self, value):
        raise NotImplementedError("write_int method must be implemented.")

    def write_bool(self, value, bit_index=0):
        raise NotImplementedError("write_bool method must be implemented.")

    def read_int(self):
        raise NotImplementedError("read_int method must be implemented.")

    def read_bool(self, bit_index=0):
        raise NotImplementedError("read_bool method must be implemented.")

class Snap7PLCClient(BasePLCClient):
    from snap7 import Client
    from snap7.util import set_int, set_bool, get_int, get_bool

    def __init__(self, ip, rack=0, slot=1, db_number=1, start_offset=0):
        self.client = self.Client()
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.db_number = db_number
        self.start_offset = start_offset

    def connect(self):
        try:
            self.client.connect(self.ip, self.rack, self.slot)
            print("PLC Connected.")
        except Exception as e:
            print(f"PLC connection error: {str(e)}")

    def disconnect(self):
        if self.client.get_connected():
            self.client.disconnect()
            print("PLC connection closed.")

    def write_int(self, value):
        try:
            int_data = bytearray(2)
            self.set_int(int_data, 0, value)
            self.client.db_write(self.db_number, self.start_offset, int_data)
            print(f"Integer {value} written to DB{self.db_number} at offset {self.start_offset}.")
        except Exception as e:
            print(f"PLC write error (INT): {str(e)}")

    def write_bool(self, value, bit_index=0):
        try:
            bool_data = bytearray(1)
            self.set_bool(bool_data, 0, bit_index, value)
            self.client.db_write(self.db_number, self.start_offset + 2, bool_data)
            print(f"Boolean bit {bit_index} set to {value}.")
        except Exception as e:
            print(f"PLC write error (BOOL): {str(e)}")

    def read_int(self):
        try:
            data = self.client.db_read(self.db_number, self.start_offset, 2)
            return self.get_int(data, 0)
        except Exception as e:
            print(f"PLC read error (INT): {str(e)}")

    def read_bool(self, bit_index=0):
        try:
            data = self.client.db_read(self.db_number, self.start_offset + 2, 1)
            return self.get_bool(data, 0, bit_index)
        except Exception as e:
            print(f"PLC read error (BOOL): {str(e)}")

class OCR:
    def __init__(self, model_path="model.pth", plc_client=None):
        self.device = torch.device("cpu")
        self.model = CNN().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)["model_state_dict"]
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.plc_client = plc_client

    def predict(self, image_path):
        image = Image.open(image_path).convert("L")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = output.argmax(dim=1).item()

        print(f"Prediction for {image_path}: {prediction}")

        if self.plc_client:
            self.send_PLC(prediction)

    def send_PLC(self, int_value):
        try:
            self.plc_client.write_int(int_value)
            time.sleep(0.1)
            self.plc_client.write_bool(True)
            time.sleep(0.1)
            self.plc_client.write_bool(False)
        except Exception as e:
            print(f"Error sending data to PLC: {str(e)}")

class ImageHandler(FileSystemEventHandler):
    def __init__(self, ocr):
        self.ocr = ocr
        self.processing_queue = queue.Queue()
        threading.Thread(target=self._process_queue, daemon=True).start()

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"New image detected: {event.src_path}")
            self.processing_queue.put(event.src_path)

    def _process_queue(self):
        while True:
            try:
                file_path = self.processing_queue.get()
                time.sleep(0.5)
                self.ocr.predict(file_path)
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")
            finally:
                self.processing_queue.task_done()

def main():
    plc_client = Snap7PLCClient(ip="192.168.0.1", db_number=1, start_offset=0)
    plc_client.connect()

    ocr = OCR(model_path="model.pth", plc_client=plc_client)
    event_handler = ImageHandler(ocr)
    observer = Observer()
    observer.schedule(event_handler, path="FTP_Folder", recursive=False)
    observer.start()
    print("\nWatching for new images...\nSupported formats: .png, .jpg, .jpeg")

    try:
        while True:
            time.sleep(1)
            print(f"Read INT: {plc_client.read_int()}, Read BOOL: {plc_client.read_bool()}")
    except KeyboardInterrupt:
        observer.stop()
        plc_client.disconnect()
        print("\nStopping directory watch...")
    observer.join()

if __name__ == "__main__":
    main()
