import os
import cv2


class FaceRecognizer:
    def __init__(self, known_faces_dir=None, confidence_threshold=0.45):

        self.known_faces_dir = known_faces_dir or os.path.join(
            os.path.dirname(__file__),
            "known_faces",
        )

        self.confidence_threshold = confidence_threshold

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        self.known_faces = self._load_known_faces()

        print(f"[FaceRecognizer] Loaded {len(self.known_faces)} images.")

    def _load_known_faces(self):

        known_faces = []

        if not os.path.exists(self.known_faces_dir):
            print("known_faces folder not found.")
            return known_faces

        # Mỗi thư mục là tên một người
        for person_name in os.listdir(self.known_faces_dir):

            person_dir = os.path.join(
                self.known_faces_dir,
                person_name,
            )

            if not os.path.isdir(person_dir):
                continue

            for filename in os.listdir(person_dir):

                if not filename.lower().endswith(".jpg"):
                    continue

                image_path = os.path.join(
                    person_dir,
                    filename,
                )

                image = cv2.imread(
                    image_path,
                    cv2.IMREAD_GRAYSCALE,
                )

                if image is None:
                    continue

                image = cv2.resize(
                    image,
                    (100, 100),
                )

                known_faces.append(
                    (
                        person_name,
                        image,
                    )
                )

        return known_faces

    def _match_score(self, face1, face2):

        result = cv2.matchTemplate(
            face1,
            face2,
            cv2.TM_CCOEFF_NORMED,
        )

        return float(result[0][0])

    def recognize(self, frame):

        if self.face_cascade.empty():
            return []

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        results = []

        for (x, y, w, h) in faces:

            face_roi = gray[y:y+h, x:x+w]

            face_roi = cv2.resize(
                face_roi,
                (100, 100),
            )

            best_name = "Unknown"
            best_score = -1

            for name, known_face in self.known_faces:

                score = self._match_score(
                    face_roi,
                    known_face,
                )

                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score < self.confidence_threshold:
                best_name = "Unknown"

            results.append(
                {
                    "bbox": (
                        x,
                        y,
                        x + w,
                        y + h,
                    ),
                    "name": best_name,
                    "confidence": round(best_score, 2),
                }
            )

        return results