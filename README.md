# PyTorch Handwritten Numbers OCR

A simple project to recognize handwritten numbers (0-9) using a convolutional neural network (CNN) built with PyTorch. The model is trained on the MNIST dataset and can be used to predict handwritten digits from images.
- (This project was a learning tool and should not be used in production or as a teaching resource as it is not validated outside of my own testing conditions and I am still very new to the AI space.)

## Features

- **Pretrained Model:** Trained on the MNIST dataset for quick deployment.
- **Custom Input:** Accepts user-provided images for inference via live webcam feed.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/4ruitz/OCR.git

2. Install required dependancies
   ```bash
   pip install -r requirements.txt
   
3. Ensure PyTorch is installed. Refer to the official [PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions tailored to your hardware.

## Usage

1. Train the model (optional):
   ```bash:
   python train_model.py
Default training configuration uses the MNIST dataset.

2. Run inference: 
   ```bash:
   python main.py
3. Adjust slider for contrast (click within the window to enable the ocr) 
![image](https://github.com/user-attachments/assets/68319b7e-dd5e-4123-bb59-f7100da12857)

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/4ruitz/OCR/blob/main/LICENSE) file for details.
