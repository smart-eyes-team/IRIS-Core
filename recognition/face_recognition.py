import os
import glob

import cv2


class FaceRecognizer:
    def __init__(self, known_faces_dir=None, confidence_threshold=0.35):
        self.known_faces_dir = known_faces_dir or os.path.join(os.path.dirname(__file__), "known_faces")
        self.confidence_threshold = confidence_threshold
        self.face_cascade = None
        self._load_face_cascade()
        self.known_faces = self._load_known_faces()

    def _load_face_cascade(self):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml") # type: ignore
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = None

    def _load_known_faces(self):
        if not os.path.isdir(self.known_faces_dir):
            return []

        known_faces = []
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

        for pattern in image_patterns:
            for image_path in glob.glob(os.path.join(self.known_faces_dir, pattern)):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = cv2.resize(image, (100, 100))
                label = os.path.splitext(os.path.basename(image_path))[0]
                known_faces.append((label, image))

        return known_faces

    def _match_score(self, face_gray, known_face_gray):
        if face_gray.size == 0 or known_face_gray.size == 0:
            return 0.0

        result = cv2.matchTemplate(face_gray, known_face_gray, cv2.TM_CCOEFF_NORMED)
        if result.size == 0:
            return 0.0
        return float(result[0][0])

    def recognize(self, frame):
        if self.face_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )

        results = []
        for x, y, w, h in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (100, 100))

            best_name = "Unknown"
            best_score = 0.0

            for label, known_face in self.known_faces:
                score = self._match_score(face_roi, known_face)
                if score > best_score:
                    best_score = score
                    best_name = label

            if best_score < self.confidence_threshold:
                best_name = "Unknown"

            results.append(
                {
                    "bbox": (x, y, x + w, y + h),
                    "name": best_name,
                    "confidence": round(best_score, 2),
                }
            )

        return results
