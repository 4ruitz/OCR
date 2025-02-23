import time
import queue
import threading
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from torchvision import transforms
from PIL import Image
import torch
import snap7
from snap7.util import set_int
from Model import CNN 

class OCR:
    def __init__(self, model_path="model.pth", plc_ip='192.168.0.1', db_number=1, start_offset=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.plc_ip = plc_ip
        self.db_number = db_number
        self.start_offset = start_offset

    def predict(self, image_path):
        image = Image.open(image_path).convert("L")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            pred = output.argmax(dim=1).item()

        print(f"Prediction for {image_path}: {pred}")

        self.send_PLC(pred)

    def send_PLC(self, int_value):
        plc = snap7.client.Client()
        plc.connect(self.plc_ip, 0, 1)  

        data = bytearray(2)
        set_int(data, 0, int_value)
        
        plc.db_write(self.db_number, self.start_offset, data)
        
        plc.disconnect()
        print(f"Integer {int_value} written to DB{self.db_number} at offset {self.start_offset}.")


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
    watch_dir = "FTP_Folder"
    print(f"Watching directory: {watch_dir}")


    ocr = OCR(model_path="model.pth", plc_ip='192.168.0.1', db_number=1, start_offset=0)
    event_handler = ImageHandler(ocr)
    observer = Observer()
    observer.schedule(event_handler, path=watch_dir, recursive=False)
    observer.start()

    print("\nWatching for new images...\nSupported formats: .png, .jpg, .jpeg")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping directory watch...")
    observer.join()


if __name__ == "__main__":
    main()
