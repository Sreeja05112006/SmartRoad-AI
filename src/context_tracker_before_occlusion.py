
from collections import Counter, deque

import numpy as np
from scipy.optimize import linear_sum_assignment


def center(bbox):
    """Return the center point of a bounding box."""
    return np.array(
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        dtype=float,
    )


def iou(box_a, box_b):
    """Calculate intersection over union of two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection
    return intersection / (union + 1e-9)


class ContextAwareTracker:
    """
    Experimental tracker combining motion, bounding-box overlap,
    and class-history consistency.
    """

    def __init__(
        self,
        max_age=30,
        max_distance=180,
        max_cost=1.0,
        history_size=7,
    ):
        self.tracks = {}
        self.next_id = 1

        self.max_age = max_age
        self.max_distance = max_distance
        self.max_cost = max_cost
        self.history_size = history_size

    def update(self, detections):
        """Associate current detections with existing tracks."""
        track_ids = list(self.tracks.keys())

        assigned = {}
        matched_tracks = set()
        matched_detections = set()

        # Match existing tracks with new detections.
        if track_ids and detections:
            costs = np.full(
                (len(track_ids), len(detections)),
                1e6,
                dtype=float,
            )

            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]

                # Predict one frame ahead from the last observed center.
                predicted_center = (
                    track["center"] + track["velocity"] * (track["age"] + 1)
                )

                for j, detection in enumerate(detections):
                    detection_center = center(detection["bbox"])

                    distance = np.linalg.norm(
                        predicted_center - detection_center
                    )

                    if distance > self.max_distance:
                        continue

                    overlap = iou(
                        track["bbox"], detection["bbox"]
                    )

                    # Use the majority class from recent observations.
                    stable_class_id = Counter(
                        track["class_history"]
                    ).most_common(1)[0][0]

                    class_mismatch = (
                        stable_class_id != detection["class_id"]
                    )

                    # Lower cost means a better match.
                    costs[i, j] = (
                        0.50 * distance / self.max_distance
                        + 0.35 * (1.0 - overlap)
                        + 0.15 * float(class_mismatch)
                    )

            rows, cols = linear_sum_assignment(costs)

            for row, col in zip(rows, cols):
                if costs[row, col] <= self.max_cost:
                    track_id = track_ids[row]

                    assigned[col] = track_id
                    matched_tracks.add(track_id)
                    matched_detections.add(col)

        # Age unmatched tracks and remove expired tracks.
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]["age"] += 1

                if self.tracks[track_id]["age"] > self.max_age:
                    del self.tracks[track_id]

        output = []

        for index, detection in enumerate(detections):
            detection_center = center(detection["bbox"])
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            bbox = np.asarray(detection["bbox"], dtype=float)

            if index in assigned:
                track_id = assigned[index]
                track = self.tracks[track_id]

                # Estimate velocity using observed center displacement.
                displacement = detection_center - track["center"]

                track["velocity"] = (
                    0.65 * track["velocity"]
                    + 0.35 * displacement
                )

                track["center"] = detection_center
                track["bbox"] = bbox
                track["age"] = 0

                track["class_history"].append(class_id)
                track["name_history"].append(class_name)

            else:
                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = {
                    "center": detection_center,
                    "bbox": bbox,
                    "velocity": np.zeros(2, dtype=float),
                    "age": 0,
                    "class_history": deque(
                        [class_id], maxlen=self.history_size
                    ),
                    "name_history": deque(
                        [class_name], maxlen=self.history_size
                    ),
                }

            track = self.tracks[track_id]

            # Derive the displayed class from recent observations.
            stable_class_id = Counter(
                track["class_history"]
            ).most_common(1)[0][0]

            stable_class = Counter(
                track["name_history"]
            ).most_common(1)[0][0]

            output.append({
                **detection,
                "track_id": track_id,
                "stable_class": stable_class,
            })

        return output
