## pip install --upgrade pip
## pip install ultralytics opencv-contrib-python numpy scipy torch torchvision torchaudio ffmpeg-python dtaidistance yt-dlp

import cv2
import torch
from ultralytics import YOLO
import ffmpeg

print("✅ OpenCV:", cv2.__version__)
print("✅ PyTorch:", torch.__version__)
print("✅ CUDA Available:", torch.cuda.is_available())

# Load YOLO model to check installation
model = YOLO("yolov8n.pt")
print("✅ YOLO Model Loaded Successfully!")
