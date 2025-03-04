import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")

# Load a test image
image_path = "Test_media/test_photos/image.png"  # Replace with an actual frame
frame = cv2.imread(image_path)

# Run YOLO detection
results = model(frame)

# Extract first detection result
result = results[0]  # YOLO returns a list, so we take the first item

# Plot the detections directly
result.show()  # This should now work correctly
