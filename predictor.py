import os
import time
import torch
import torchvision.transforms as transforms
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image

# Define the model (same architecture as before)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define a function to load the trained model
def load_model(model_path='mnist_cnn.pth'):
    model = SimpleCNN()  # Reuse the model architecture
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define the transformation for input images
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                               transforms.Resize((28, 28)),  # Resize to 28x28 pixels
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])

# Function to process an image and predict digits
def process_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Handler for new files in the directory
class ImageHandler(FileSystemEventHandler):
    def __init__(self, model):
        self.model = model

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.png', '.jpg', '.jpeg')):  # Process image files
            print(f"New image found: {event.src_path}")
            prediction = process_image(event.src_path, self.model)
            print(f"Predicted digit: {prediction}")

# Watch the directory for new images
def watch_directory(directory="/watch_images"):
    model = load_model()
    event_handler = ImageHandler(model)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    watch_directory("/watch_images")
