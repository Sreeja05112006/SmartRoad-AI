from collections import Counter, deque
import numpy as np
from scipy.optimize import linear_sum_assignment


def center(bbox):
    return np.array(
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        dtype=float,
    )


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / (union + 1e-9)


def direction_name(vector, min_speed=2.0):
    dx, dy = float(vector[0]), float(vector[1])

    if np.hypot(dx, dy) < min_speed:
        return "Stationary"

    if abs(dx) >= abs(dy):
        return "Right" if dx > 0 else "Left"

    return "Down" if dy > 0 else "Up"


class ContextAwareTracker:
    """
    Custom tracker using:
    1. Motion prediction
    2. IoU overlap
    3. Class-history stability
    4. Road-zone consistency
    5. Direction consistency
    """

    def __init__(
        self,
        max_age=20,
        max_distance=140,
        max_cost=1.0,
        history_size=8,
        zone_penalty=0.25,
        direction_penalty=0.25,
    ):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.max_distance = max_distance
        self.max_cost = max_cost
        self.history_size = history_size
        self.zone_penalty = zone_penalty
        self.direction_penalty = direction_penalty

    def update(self, detections):
        track_ids = list(self.tracks.keys())
        assigned = {}
        matched_tracks = set()

        if track_ids and detections:
            costs = np.full(
                (len(track_ids), len(detections)),
                1e6,
                dtype=float,
            )

            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                predicted_center = track["center"] + track["velocity"]

                stable_class_id = Counter(
                    track["class_history"]
                ).most_common(1)[0][0]

                stable_zone = Counter(
                    track["zone_history"]
                ).most_common(1)[0][0]

                old_direction = direction_name(track["velocity"])

                for j, detection in enumerate(detections):
                    detection_center = center(detection["bbox"])

                    distance = np.linalg.norm(
                        predicted_center - detection_center
                    )

                    if distance > self.max_distance:
                        continue

                    overlap = iou(track["bbox"], detection["bbox"])

                    class_mismatch = (
                        stable_class_id != detection["class_id"]
                    )

                    candidate_motion = detection_center - track["center"]
                    new_direction = direction_name(candidate_motion)

                    direction_mismatch = (
                        old_direction != "Stationary"
                        and new_direction != "Stationary"
                        and old_direction != new_direction
                    )

                    detection_zone = detection.get("zone", "Unassigned")

                    zone_mismatch = (
                        stable_zone != "Unassigned"
                        and detection_zone != "Unassigned"
                        and stable_zone != detection_zone
                    )

                    costs[i, j] = (
                        0.40 * (distance / self.max_distance)
                        + 0.25 * (1.0 - overlap)
                        + 0.15 * float(class_mismatch)
                        + self.zone_penalty * float(zone_mismatch)
                        + self.direction_penalty * float(direction_mismatch)
                    )

            rows, cols = linear_sum_assignment(costs)

            for row, col in zip(rows, cols):
                if costs[row, col] <= self.max_cost:
                    track_id = track_ids[row]
                    assigned[col] = track_id
                    matched_tracks.add(track_id)

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]["age"] += 1

                if self.tracks[track_id]["age"] > self.max_age:
                    del self.tracks[track_id]

        output = []

        for index, detection in enumerate(detections):
            detection_center = center(detection["bbox"])
            bbox = np.asarray(detection["bbox"], dtype=float)
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            zone = detection.get("zone", "Unassigned")

            if index in assigned:
                track_id = assigned[index]
                track = self.tracks[track_id]

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
                track["zone_history"].append(zone)

            else:
                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = {
                    "center": detection_center,
                    "bbox": bbox,
                    "velocity": np.zeros(2, dtype=float),
                    "age": 0,
                    "class_history": deque([class_id], maxlen=self.history_size),
                    "name_history": deque([class_name], maxlen=self.history_size),
                    "zone_history": deque([zone], maxlen=self.history_size),
                }

            track = self.tracks[track_id]

            stable_class = Counter(
                track["name_history"]
            ).most_common(1)[0][0]

            stable_zone = Counter(
                track["zone_history"]
            ).most_common(1)[0][0]

            output.append(
                {
                    **detection,
                    "track_id": track_id,
                    "stable_class": stable_class,
                    "stable_zone": stable_zone,
                    "direction": direction_name(track["velocity"]),
                }
            )

        return output