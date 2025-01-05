import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from opcua import Client
from django.http import JsonResponse

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Load model and set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()  # Create an instance of the model
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))  # Load the saved model weights
model.to(device)
model.eval()

# Transformation for images
transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to process the image and pass it to the model
def process_image(image_path):
    image = Image.open(image_path).convert('L')  # Open image as grayscale (MNIST expects grayscale)
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)  # Get the predicted label (number)
        return pred.item()

# Function to send the result to Kepware OPC UA
def send_to_kepware(detected_number):
    kepware_url = "opc.tcp://127.0.0.1:49320"  # Replace with your Kepware IP and port
    client = Client(kepware_url)
    try:
        client.connect()
        plc_tag_node = client.get_node("ns=2;s=Channel1.Device1.Tag1")  # Replace with your actual Node ID
        plc_tag_node.set_value(detected_number)  # Send detected number to PLC
        print(f"Detected number {detected_number} sent to PLC")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.disconnect()

# Django view to handle the image processing and sending to PLC
def ocr_from_ftp_and_send(request):
    ftp_directory = "C:\cogneximages"  # Path to FTP directory with images
    image_files = [f for f in os.listdir(ftp_directory) if f.endswith(".png") or f.endswith(".jpg")]

    results = []
    
    for image_file in image_files:
        image_path = os.path.join(ftp_directory, image_file)
        
        # Step 1: Process the image and get OCR result (detected number)
        detected_number = process_image(image_path)
        results.append({
            'image': image_file,
            'detected_number': detected_number
        })
        
        # Step 2: Send detected number to Kepware OPC UA
        send_to_kepware(detected_number)

    return JsonResponse({'status': 'success', 'results': results})