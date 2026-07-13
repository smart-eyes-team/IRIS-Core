import cv2
from detection.detector import ObjectDetector

# Khởi tạo detector
detector = ObjectDetector()

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Nhận diện vật thể
    results = detector.detect(frame)

    # Vẽ kết quả lên ảnh
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("SMART-EYES", annotated_frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()