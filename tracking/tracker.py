import math
import cv2
import numpy as np

class ObjectTracker:
    def __init__(self, max_tracks=50, max_age=270, tracker_type="csrt"):
        """
        max_tracks: Số lượng vật thể tối đa lưu trong bộ nhớ.
        max_age: Số lượng frame tối đa cho phép track "tạm mất dấu" trước khi xóa hoàn toàn.
        """
        self.max_tracks = max_tracks
        self.max_age = max_age
        self.tracker_type = tracker_type
        self.trackers = {}
        self.next_id = 1

    def _create_tracker(self):
        try:
            if self.tracker_type.lower() == "mosse":
                if hasattr(cv2, 'legacy'):
                    return cv2.legacy.TrackerMOSSE_create() # type: ignore
                return cv2.TrackerMOSSE.create() # type: ignore
            else:
                if hasattr(cv2, 'legacy'):
                    return cv2.legacy.TrackerCSRT_create() # type: ignore
                return cv2.TrackerCSRT.create() # type: ignore
        except AttributeError:
            # Fallback nếu không có module contrib tracking
            return None

    def _create_kalman_filter(self):
        kalman = cv2.KalmanFilter(8, 4, 0)
        kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.25
        kalman.errorCovPost = np.eye(8, dtype=np.float32)
        return kalman

    def _iou(self, box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _extract_appearance_feature(self, frame, bbox):
        if frame is None:
            return np.zeros(64, dtype=np.float32)

        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(frame.shape[1] - 1, x1))
        y1 = max(0, min(frame.shape[0] - 1, y1))
        x2 = max(0, min(frame.shape[1], x2))
        y2 = max(0, min(frame.shape[0], y2))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(64, dtype=np.float32)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32) / 255.0
        feature = gray.reshape(-1)
        norm = np.linalg.norm(feature)
        if norm < 1e-6:
            return np.zeros(64, dtype=np.float32)
        return feature / norm

    def _cosine_distance(self, feature_a, feature_b):
        denom = np.linalg.norm(feature_a) * np.linalg.norm(feature_b)
        if denom < 1e-6:
            return 1.0
        return 1.0 - float(np.dot(feature_a, feature_b) / denom)

    def _mahalanobis_distance(self, predicted_bbox, detection_bbox):
        predicted_state = np.array(
            [predicted_bbox[0], predicted_bbox[1], max(1e-5, predicted_bbox[2] - predicted_bbox[0]), max(1e-5, predicted_bbox[3] - predicted_bbox[1])],
            dtype=np.float32,
        )
        detection_state = np.array(
            [detection_bbox[0], detection_bbox[1], max(1e-5, detection_bbox[2] - detection_bbox[0]), max(1e-5, detection_bbox[3] - detection_bbox[1])],
            dtype=np.float32,
        )
        diff = detection_state - predicted_state
        covariance = np.diag([max(1.0, abs(diff[0]) + 1.0), max(1.0, abs(diff[1]) + 1.0), max(1.0, abs(diff[2]) + 1.0), max(1.0, abs(diff[3]) + 1.0)])
        try:
            inv_cov = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(covariance)
        mahal = diff @ inv_cov @ diff
        return float(np.sqrt(max(0.0, mahal)))

    def _predict_bbox(self, track_info):
        kalman = track_info.get("kalman")
        if kalman is None:
            return track_info["bbox"]

        prediction = kalman.predict()
        x = float(prediction[0][0])
        y = float(prediction[1][0])
        w = max(0.0, float(prediction[2][0]))
        h = max(0.0, float(prediction[3][0]))
        return [x, y, x + w, y + h]

    def _update_from_detection(self, track_info, detection_bbox, frame):
        x1, y1, x2, y2 = detection_bbox
        measurement = np.array([x1, y1, max(1e-5, x2 - x1), max(1e-5, y2 - y1)], dtype=np.float32).reshape(4, 1)
        track_info["kalman"].correct(measurement)
        track_info["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
        track_info["feature"] = self._extract_appearance_feature(frame, detection_bbox)
        track_info["age"] = 0
        track_info["active"] = True

    def _match_detections(self, detections, active_tracks, frame):
        if not detections or not active_tracks:
            return [], list(range(len(detections))), list(range(len(active_tracks)))

        track_ids = list(active_tracks.keys())
        costs = []
        
        for det_idx, det in enumerate(detections):
            det_bbox = det[:4]
            det_feature = self._extract_appearance_feature(frame, det_bbox)
            row = []
            for track_id in track_ids:
                track_info = active_tracks[track_id]
                predicted_bbox = self._predict_bbox(track_info)
                
                motion_cost = self._mahalanobis_distance(predicted_bbox, det_bbox)
                appearance_cost = self._cosine_distance(track_info.get("feature", np.zeros(64, dtype=np.float32)), det_feature)
                iou_score = self._iou(predicted_bbox, det_bbox)
                
                # Kết hợp chi phí: Nếu IoU cao -> Chi phí thấp, ưu tiên tuyệt đối IoU trước
                iou_cost = 1.0 - iou_score
                combined_cost = 0.4 * motion_cost + 0.3 * appearance_cost + 0.3 * iou_cost
                row.append(combined_cost)
            costs.append(row)

        matched = []
        used_track_indices = set()
        used_detection_indices = set()
        candidates = []
        
        for det_idx, row in enumerate(costs):
            for track_idx, cost in enumerate(row):
                candidates.append((cost, det_idx, track_idx))

        # Thuật toán so khớp tham lam (Greedy Matching) tránh chồng chéo ID
        candidates.sort(key=lambda item: item[0])
        for cost, det_idx, track_idx in candidates:
            if det_idx in used_detection_indices or track_idx in used_track_indices:
                continue
            
            track_id = track_ids[track_idx]
            predicted_bbox = self._predict_bbox(active_tracks[track_id])
            det_bbox = detections[det_idx][:4]
            iou_score = self._iou(predicted_bbox, det_bbox)
            center_dist = math.dist(self._center(predicted_bbox), self._center(det_bbox))
            
            # Điều kiện khớp: Có IoU hình học HOẶC khoảng cách tâm đủ gần, tránh ngưỡng cố định 0.8
            if cost < 1.2 or iou_score > 0.15 or center_dist < 100:
                matched.append((track_idx, det_idx))
                used_track_indices.add(track_idx)
                used_detection_indices.add(det_idx)

        unmatched_detections = [idx for idx in range(len(detections)) if idx not in used_detection_indices]
        unmatched_tracks = [idx for idx in range(len(track_ids)) if idx not in used_track_indices]
        return matched, unmatched_detections, unmatched_tracks

    def update(self, frame, detections):
        """
        detections: list dạng [[x1, y1, x2, y2, conf, class_id, class_name], ...]
        returns: danh sách dict chứa thông tin tracking còn hoạt động
        """
        # Lọc sạch dictionary: Xóa hẳn các track quá tuổi để giải phóng bộ nhớ RAM
        for track_id in list(self.trackers.keys()):
            if self.trackers[track_id]["age"] > self.max_age:
                del self.trackers[track_id]

        active_tracks = {track_id: info for track_id, info in self.trackers.items() if info.get("active", True)}

        # Nếu không có detection nào ở frame này, tăng tuổi của tất cả track hiện tại lên 1
        if not detections:
            for track_id, track_info in active_tracks.items():
                track_info["age"] += 1
                predicted_bbox = self._predict_bbox(track_info)
                track_info["bbox"] = predicted_bbox
                if track_info["age"] > self.max_age:
                    track_info["active"] = False
            
            return [
                {"id": tid, "bbox": tinfo["bbox"], "class_name": tinfo["class_name"]}
                for tid, tinfo in self.trackers.items() if tinfo.get("active", True)
            ]

        # Khớp đối tượng
        matched, unmatched_detections, unmatched_tracks = self._match_detections(detections, active_tracks, frame)
        matched_track_ids = []

        # 1. Cập nhật các đối tượng khớp thành công
        for track_idx, det_idx in matched:
            track_id = list(active_tracks.keys())[track_idx]
            matched_track_ids.append(track_id)
            det = detections[det_idx]
            track_info = self.trackers[track_id]
            self._update_from_detection(track_info, det[:4], frame)
            if len(det) > 6:
                track_info["class_name"] = det[6]

        # 2. Khởi tạo Track mới cho các detection không trùng khớp
        for det_idx in unmatched_detections:
            if len(self.trackers) >= self.max_tracks:
                continue
            det = detections[det_idx]
            x1, y1, x2, y2 = det[:4]
            
            # Bộ lọc chống nhiễu: Loại bỏ nhầm lẫn các thiết bị nhỏ ví dụ như điện thoại thành người
            c_name = det[6] if len(det) > 6 else "object"
            if c_name.lower() in ["person", "human"]:
                # Kích thước hộp quá nhỏ (ví dụ bé hơn 50 pixel) thì không tạo track người mới
                if (x2 - x1) < 40 or (y2 - y1) < 60:
                    continue

            kalman = self._create_kalman_filter()
            kalman.statePost = np.array([x1, y1, max(1e-5, x2 - x1), max(1e-5, y2 - y1), 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
            
            self.trackers[self.next_id] = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class_name": c_name,
                "age": 0,
                "active": True,
                "kalman": kalman,
                "feature": self._extract_appearance_feature(frame, det[:4]),
            }
            self.next_id += 1

        # 3. Xử lý các track cũ bị mất dấu ở frame này (Unmatched Tracks)
        track_ids = list(active_tracks.keys())
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            track_info = self.trackers[track_id]
            track_info["age"] += 1
            
            # Sử dụng dự đoán của Kalman Filter để giữ nguyên vị trí ước tính
            predicted_bbox = self._predict_bbox(track_info)
            track_info["bbox"] = predicted_bbox
            
            if track_info["age"] > self.max_age:
                track_info["active"] = False

        # Chỉ trả về các đối tượng active thực sự
        return [
            {
                "id": track_id,
                "bbox": track_info["bbox"],
                "class_name": track_info["class_name"],
            }
            for track_id, track_info in self.trackers.items()
            if track_info.get("active", True) and track_info["age"] <= 3 # Chỉ hiển thị UI nếu mất dấu không quá 3 frame
        ]