import time
import os
import queue
from queue import Queue
import threading
import torch
from PIL import Image
from torchvision import transforms
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from snap7.client import Client
from snap7.util import set_int, set_bool, get_int, get_bool
from Model import CNN

plc = None
model = None
transform = None
image_queue = Queue()
stop_processing = False

def connect_plc():
    global plc
    plc = Client()
    try:
        plc.connect("192.168.0.1", 0, 1)
        print("PLC connected")
        return True
    except Exception as e:
        print(f"PLC connection error: {e}")
        return False

def load_model():
    global model, transform
    model = CNN()
    model.load_state_dict(torch.load("model.pth", map_location="cpu")["model_state_dict"])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    print("Model loaded")

def process_image(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return
            
        time.sleep(0.5)
        
        image = Image.open(image_path).convert("L")
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()
        
        print(f"Prediction for {image_path}: {prediction}")
        
        if plc and plc.get_connected():
            send_to_plc(prediction)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def send_to_plc(value):
    try:
        data = bytearray(2)
        set_int(data, 0, value)
        plc.db_write(1, 0, data)
        
        bool_data = bytearray(1)
        set_bool(bool_data, 0, 0, True)
        plc.db_write(1, 2, bool_data)
        time.sleep(0.5)
        
        set_bool(bool_data, 0, 0, False)
        plc.db_write(1, 2, bool_data)
        
        print(f"Sent value {value} to PLC")
    except Exception as e:
        print(f"Error sending to PLC: {e}")

def read_plc_values():
    try:
        if plc and plc.get_connected():
            int_data = plc.db_read(1, 0, 2)
            int_val = get_int(int_data, 0)
            
            bool_data = plc.db_read(1, 2, 1)
            bool_val = get_bool(bool_data, 0, 0)
            
            return int_val, bool_val
        return None, None
    except Exception as e:
        print(f"Error reading from PLC: {e}")
        return None, None

def queuing():
    global stop_processing
    print("Image processing worker started")
    while not stop_processing:
        try:
            try:
                image_path = image_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            process_image(image_path)
            
            image_queue.task_done()
            
        except Exception as e:
            print(f"Error in processing thread: {e}")
    
    print("Image processing worker stopped")

class ImageWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"New image detected: {event.src_path}")
            image_queue.put(event.src_path)

def main():
    global stop_processing
    
    os.makedirs("FTP_Folder", exist_ok=True)
    
    load_model()
    
    connect_plc()
    
    worker_thread = threading.Thread(target=queuing, daemon=True)
    worker_thread.start()
    
    observer = Observer()
    observer.schedule(ImageWatcher(), "FTP_Folder", recursive=False)
    observer.start()
    print("Watching for new images in FTP_Folder")
    
    try:
        while True:
            time.sleep(1)
            int_val, bool_val = read_plc_values()
            if int_val is not None:
                print(f"PLC values - INT: {int_val}, BOOL: {bool_val}")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_processing = True
        
        worker_thread.join(timeout=2.0)
        
        observer.stop()
        
        if plc and plc.get_connected():
            plc.disconnect()
            
        observer.join()
        print("Program stopped")

if __name__ == "__main__":
    main()