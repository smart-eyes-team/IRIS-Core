class DistanceEstimator:
    def __init__(self):
        self.reference_sizes = {
            "person": 1.0,
            "car": 1.5,
            "motorbike": 0.8,
            "bicycle": 0.8,
            "chair": 0.6,
            "table": 0.8,
            "door": 0.7,
            "stairs": 0.9,
            "truck": 1.8,
            "bus": 2.0,
        }

    def estimate(self, bbox, frame_shape, class_name=None):
        x1, y1, x2, y2 = bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        box_area = width * height
        frame_area = frame_shape[1] * frame_shape[0]

        if frame_area <= 0:
            ratio = 0.0
        else:
            ratio = box_area / frame_area

        base_size = self.reference_sizes.get((class_name or "object").lower(), 0.8)
        distance_m = round(max(0.3, min(8.0, base_size / max(ratio, 0.01) ** 0.5)), 1)

        if distance_m < 1.5:
            message = "rất gần"
        elif distance_m < 3.0:
            message = "gần"
        elif distance_m < 5.0:
            message = "ở khoảng trung bình"
        else:
            message = "xa"

        return {
            "distance_m": distance_m,
            "message": message,
        }
