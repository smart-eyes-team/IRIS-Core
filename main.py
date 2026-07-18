import cv2
from detection.detector import ObjectDetector
from tracking.tracker import ObjectTracker
from recognition.distance import DistanceEstimator
from recognition.face_recognition import FaceRecognizer

# Khởi tạo các thành phần

detector = ObjectDetector()
tracker = ObjectTracker(max_tracks=10)
distance_estimator = DistanceEstimator()
face_recognizer = FaceRecognizer()

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Nhận diện vật thể
    results = detector.detect(frame)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = result.names.get(cls_id, "object")

            detections.append([x1, y1, x2, y2, conf, cls_id, class_name])

    # Theo dõi đối tượng
    tracked_objects = tracker.update(frame, detections)

    # Nhận diện khuôn mặt
    recognized_faces = face_recognizer.recognize(frame)

    # Vẽ kết quả lên ảnh
    for det in detections:
        x1, y1, x2, y2, conf, _, class_name = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        distance_info = distance_estimator.estimate((x1, y1, x2, y2), frame.shape, class_name)
        cv2.putText(frame, f"{distance_info['distance_m']}m", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    for track in tracked_objects:
        x1, y1, x2, y2 = map(int, track["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f"ID {track['id']} {track['class_name']}", (x1, max(0, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    for face in recognized_faces:
        x1, y1, x2, y2 = face["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{face['name']} ({face['confidence']})", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("SMART-EYES", frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()