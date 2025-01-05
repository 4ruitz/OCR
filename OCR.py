import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import snap7test
from PIL import Image
from torchvision import transforms
import struct

# Define the OCR neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize the model
model = Net()
model.load_state_dict(torch.load("model.pth"))  # Load pre-trained model weights
model.eval()

# Preprocessing transformations for the input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to process image and run OCR
def run_ocr(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Get the OCR result (index of the highest logit)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Function to send result to Siemens PLC via Snap7
def send_to_plc(value):
    plc = snap7test.client.Client()
    plc.connect('192.168.0.2', 0, 1)  # Replace with your PLC's IP and rack/slot
    
    # Data Block (DB) and offset
    db_number = 1  # Replace with the correct DB number
    offset = 0  # Replace with the correct offset within the DB

    # Convert the integer value to bytes (using struct.pack)
    data = struct.pack('>i', value)  # Pack the integer as a big-endian 4-byte integer

    # Write the data to the PLC
    plc.db_write(db_number, offset, data)

    plc.disconnect()

# Monitor the /images directory for new files
def monitor_directory(path):
    seen_files = set()

    while True:
        # List all files in the directory
        files = set(os.listdir(path))
        
        # Find new files by comparing with the previously seen files
        new_files = files - seen_files
        
        if new_files:
            for new_file in new_files:
                image_path = os.path.join(path, new_file)
                print(f"New image detected: {image_path}")
                
                result = run_ocr(image_path)  # Run OCR on the image
                send_to_plc(result)  # Send OCR result to PLC
                print(f"OCR Result: {result}")
            
            # Update the seen files set
            seen_files.update(new_files)

        # Sleep for a short time before checking again
        time.sleep(1)

# Start monitoring the /images directory
monitor_directory('/home/pi/images')
