import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

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


def predict(model, img):
    img = cv2.resize(img, (28, 28)) 
    img = np.expand_dims(img, axis=0) 
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float32)
    img = img / 255.0

    with torch.no_grad():
        output = model(img)
    index = output.argmax(dim=1, keepdim=True).item()
    return str(index)

startInference = False

def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

threshold = 100

def on_threshold(x):
    global threshold
    threshold = x

def start_cv(model):
    global threshold
    cap = cv2.VideoCapture(0)
    frame = cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', 150, 255, on_threshold)
    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    while True:
        ret, frame = cap.read()

        if startInference:
            frameCount += 1
            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)

            resizedFrame = thr[240-75:240+75, 320-75:320+75]
            background[240-75:240+75, 320-75:320+75] = resizedFrame

            iconImg = cv2.resize(resizedFrame, (28, 28))

            res = predict(model, iconImg)

            if frameCount == 5:
                background[0:480, 0:80] = 0
                frameCount = 0

            cv2.putText(background, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(background, (320-80, 240-80), (320+80, 240+80), (255, 255, 255), thickness=3)

            cv2.imshow('background', background)
        else:
            cv2.imshow('background', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    model = Net()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()

    print("starting cv...")

    start_cv(model)

if __name__ == '__main__':
    main()
