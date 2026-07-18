import unittest
import numpy as np

from tracking.tracker import ObjectTracker


class TrackerTests(unittest.TestCase):
    def test_tracker_keeps_id_across_consecutive_frames(self):
        tracker = ObjectTracker(max_tracks=5, max_age=3)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first_detections = [[10, 10, 45, 80, 0.98, 0, "person"]]
        first_result = tracker.update(frame, first_detections)

        self.assertEqual(len(first_result), 1)
        first_id = first_result[0]["id"]

        second_detections = [[12, 12, 47, 82, 0.97, 0, "person"]]
        second_result = tracker.update(frame, second_detections)

        self.assertEqual(len(second_result), 1)
        self.assertEqual(second_result[0]["id"], first_id)


if __name__ == "__main__":
    unittest.main()
