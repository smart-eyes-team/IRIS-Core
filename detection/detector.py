from ultralytics import YOLO


class ObjectDetector:
    def __init__(self):
        # Load mô hình YOLOv8 Nano
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        # Nhận diện vật thể trong frame
        results = self.model(frame)
        return results