from collections import Counter, deque
import numpy as np
from scipy.optimize import linear_sum_assignment

def center(b):
    return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2], dtype=float)

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (aa + ab - inter + 1e-9)

class ContextAwareTracker:
    """Educational tracker: predicted motion + IoU + short class history."""
    def __init__(self, max_age=15, max_distance=120):
        self.tracks, self.next_id = {}, 1
        self.max_age, self.max_distance = max_age, max_distance

    def update(self, detections):
        tids = list(self.tracks)
        assigned, matched_t, matched_d = {}, set(), set()
        if tids and detections:
            costs = np.full((len(tids), len(detections)), 1e6)
            for i, tid in enumerate(tids):
                tr = self.tracks[tid]
                pred = tr["center"] + tr["velocity"] * (tr["age"] + 1)
                for j, d in enumerate(detections):
                    c = center(d["bbox"])
                    dist = np.linalg.norm(pred-c)
                    if dist <= self.max_distance:
                        mismatch = tr["class_id"] != d["class_id"]
                        costs[i, j] = .65*dist/self.max_distance + .45*(1-iou(tr["bbox"], d["bbox"])) + (.35 if mismatch else 0)
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] < 1.35:
                    tid = tids[i]
                    assigned[j] = tid
                    matched_t.add(tid); matched_d.add(j)
        for tid in list(self.tracks):
            if tid not in matched_t:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        out = []
        for j, d in enumerate(detections):
            if j in assigned:
                tid = assigned[j]; tr = self.tracks[tid]
                c = center(d["bbox"])
                delta = c - tr["center"]
                tr["velocity"] = .65*tr["velocity"] + .35*delta
                tr["center"], tr["bbox"], tr["age"] = c, np.asarray(d["bbox"], float), 0
                tr["history"].append(d["class_name"])
                stable = Counter(tr["history"]).most_common(1)[0][0]
            else:
                tid = self.next_id; self.next_id += 1
                c = center(d["bbox"])
                self.tracks[tid] = {"center": c, "bbox": np.asarray(d["bbox"], float),
                    "velocity": np.zeros(2), "age": 0, "class_id": d["class_id"],
                    "history": deque([d["class_name"]], maxlen=7)}
                stable = d["class_name"]
            self.tracks[tid]["class_id"] = d["class_id"]
            out.append({**d, "track_id": tid, "stable_class": stable})
        return out
