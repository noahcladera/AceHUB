import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
video_path = "Test_media/test_videos/video_1.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize trackers dictionary
trackers = {}  # e.g., {"person": tracker_object, "racket": tracker_object}
tracking_ids = {"person": False, "racket": False}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection every frame (or every few frames)
    results = model(frame, conf=0.2)  # Lower threshold if necessary

    # Process detections for both person (class 0) and tennis racket (class 39)
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            class_id = int(cls.item())
            if class_id == 0 and not tracking_ids["person"]:
                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2 - x1, y2 - y1)
                # Initialize tracker for person
                trackers["person"] = cv2.TrackerCSRT_create()
                trackers["person"].init(frame, bbox)
                tracking_ids["person"] = True
            elif class_id == 39 and not tracking_ids["racket"]:
                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2 - x1, y2 - y1)
                # Initialize tracker for racket
                trackers["racket"] = cv2.TrackerCSRT_create()
                trackers["racket"].init(frame, bbox)
                tracking_ids["racket"] = True

    # Update trackers if initialized
    for label in trackers.keys():
        success, bbox = trackers[label].update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0) if label == "racket" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            tracking_ids[label] = False  # If tracking fails, allow re-detection

    # Show the frame
    cv2.imshow("Multi-Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
