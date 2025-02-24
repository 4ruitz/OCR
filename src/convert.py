import torch

# Load your trained model
from Model import CNN  # Assuming your CNN model is defined in cnn.py

model = CNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Convert model to TorchScript format
scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pt")

print("TorchScript model saved as scripted_model.pt")
