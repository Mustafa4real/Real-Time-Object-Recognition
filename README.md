# Real-Time-Object-Recognition
In this project I developed a Python-based application that uses a pre-trained deep learning model integrated with OpenCV to recognize and label objects from a live webcam feed in real-time.

## Model
`SSD-MobileNet-V2-FPNlite` which can deffrentiate between 80 different classes, was used as the pretrained model for the project.

## Features
- Real-time object detection with bounding boxes and class labels
- Uses a lightweight SSD MobileNet V2 model for fast inference
- Supports all 90 classes from the COCO dataset
- Runs directly from webcam (no need for dataset or training)

## How to run
1. Download requirements using the following command:
`pip install -r requirements.txt`
2. run `main.py`

## Detected classes example

<img width="907" height="743" alt="Screenshot (725)" src="https://github.com/user-attachments/assets/e6d0fcca-90c8-41ee-9ac7-118fd2bc4bec" />

