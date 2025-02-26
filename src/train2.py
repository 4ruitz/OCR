import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define the neural network class (same as before)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Custom Dataset class for new images
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for filename in os.listdir(image_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                self.images.append(filename)
                self.labels.append(-1)  # Placeholder for manual labeling
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_path).convert("L")  # Open image in grayscale
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Function to fine-tune the model on new images
def fine_tune(model, optimizer, criterion, train_loader, num_epochs=1):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Fine-tuning Epoch {epoch+1}/{num_epochs} complete, Loss: {loss.item():.4f}")

# Watchdog event handler
class ImageEventHandler(FileSystemEventHandler):
    def __init__(self, model, optimizer, criterion, transform, train_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.transform = transform
        self.train_loader = train_loader

    def on_created(self, event):
        if event.is_directory:
            return
        filename = event.src_path
        if filename.endswith(".png") or filename.endswith(".jpg"):
            print(f"New image detected: {filename}")
            label = int(input("Please enter the label for this image (0-9): "))
            
            # Add the new image with label to the training dataset
            train_loader.dataset.images.append(os.path.basename(filename))
            train_loader.dataset.labels.append(label)

            # Fine-tune the model on the new image
            fine_tune(self.model, self.optimizer, self.criterion, train_loader)
            torch.save(self.model.state_dict(), 'fine_tuned_model.pth')
            print("Model fine-tuned and saved.")

# Setup and training loop
def train():
    # Set up dataset transformation and model
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create a custom dataset for the new images
    image_dir = "./FTP_Folder"  # Directory to watch for new images
    train_dataset = CustomImageDataset(image_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Set up the file system observer
    event_handler = ImageEventHandler(model, optimizer, criterion, transform, train_loader)
    observer = Observer()
    observer.schedule(event_handler, image_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    train()
