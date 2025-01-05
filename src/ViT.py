import torch
from torchvision import transforms
import torch.optim as optim
from torchvision import datasets, transforms, models
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.data import DataLoader
from train_model2 import device

# Preprocessing for ViT
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
def preprocess_data(images, labels):
    inputs = feature_extractor(images, return_tensors="pt", padding=True)
    inputs['labels'] = labels
    return inputs

# Load Dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Pretrained ViT
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10)
model.to(device)

# Training and Evaluation Functions
optimizer = optim.Adam(model.parameters(), lr=5e-5)

def train_vit(model, train_loader, epochs=3):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            inputs = preprocess_data(images, labels.to(device))
            outputs = model(**inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def evaluate_vit(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            inputs = preprocess_data(images, labels.to(device))
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

# Train and Evaluate
train_vit(model, train_loader, epochs=3)
evaluate_vit(model, test_loader)
