import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.context_tracker import ContextAwareTracker


ROAD_CLASSES = {0, 1, 2, 3, 5, 6, 7}


def get_zone(point, zones):
    if not zones:
        return "Unassigned"

    px, py = float(point[0]), float(point[1])

    for zone in zones:
        name = zone["name"]
        polygon = np.array(zone["points"], dtype=np.int32)

        inside = cv2.pointPolygonTest(
            polygon,
            (px, py),
            False,
        )

        if inside >= 0:
            return name

    return "Unassigned"


def draw(frame, detections, zones):
    image = frame.copy()
    height, width = image.shape[:2]

    zone_colors = [
        (255, 150, 60),
        (180, 80, 255),
        (60, 180, 255),
        (255, 100, 180),
    ]

    for index, zone in enumerate(zones or []):
        polygon = np.array(zone["points"], dtype=np.int32)
        color = zone_colors[index % len(zone_colors)]

        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon], color)
        image = cv2.addWeighted(image, 0.88, overlay, 0.12, 0)

        cv2.polylines(image, [polygon], True, color, 2)

        x, y = polygon[0]
        cv2.putText(
            image,
            zone["name"],
            (int(x), int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        track_id = int(d.get("track_id", -1))
        class_name = d.get(
            "stable_class",
            d.get("class_name", "object"),
        )
        zone = d.get("stable_zone", d.get("zone", "Unassigned"))
        direction = d.get("direction", "Unknown")
        confidence = float(d.get("confidence", 0.0))

        color = (40, 190, 90)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw the main object bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        # Large ID text
        id_label = f"ID:{track_id}"
        id_font_scale = 1.3
        id_thickness = 4

        (id_w, id_h), id_baseline = cv2.getTextSize(
            id_label,
            font,
            id_font_scale,
            id_thickness,
        )

        id_x = x1
        id_y = y1 - 10

        # If the ID would go outside the top of the frame,
        # place it below the bounding box instead.
        if id_y - id_h < 0:
            id_y = y2 + id_h + 12

        cv2.rectangle(
            image,
            (id_x, id_y - id_h - 8),
            (id_x + id_w + 12, id_y + id_baseline + 4),
            color,
            -1,
        )

        cv2.putText(
            image,
            id_label,
            (id_x + 6, id_y),
            font,
            id_font_scale,
            (255, 255, 255),
            id_thickness,
            cv2.LINE_AA,
        )

        # Smaller object details text
        detail_label = (
            f"{class_name} | {direction} | {zone} | {confidence:.2f}"
        )
        detail_font_scale = 0.55
        detail_thickness = 2

        (detail_w, detail_h), detail_baseline = cv2.getTextSize(
            detail_label,
            font,
            detail_font_scale,
            detail_thickness,
        )

        detail_y = id_y + detail_h + 18

        if detail_y > height - 5:
            detail_y = id_y - detail_h - 18

        cv2.rectangle(
            image,
            (id_x, detail_y - detail_h - 6),
            (id_x + detail_w + 10, detail_y + detail_baseline + 4),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            image,
            detail_label,
            (id_x + 5, detail_y),
            font,
            detail_font_scale,
            (255, 255, 255),
            detail_thickness,
            cv2.LINE_AA,
        )

    return image


def process_video(
    input_path,
    output_path,
    csv_path,
    mode,
    conf,
    imgsz,
    max_frames,
    zones=None,
    progress=None,
):
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    limit = min(total, max_frames) if max_frames > 0 else total

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    custom_tracker = ContextAwareTracker()

    records = []
    all_track_ids = set()
    unique_classes = {}
    observations_by_class = Counter()
    observations_by_zone = Counter()
    direction_flow = Counter()

    processed = 0
    start = time.perf_counter()

    try:
        while cap.isOpened():
            ok, frame = cap.read()

            if not ok:
                break

            if max_frames > 0 and processed >= max_frames:
                break

            detections = []

            if mode == "ByteTrack baseline":
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=list(ROAD_CLASSES),
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                )[0]

                if result.boxes is not None:
                    for b in result.boxes:
                        cid = int(b.cls[0].item())
                        tid = int(b.id[0].item()) if b.id is not None else -1
                        bbox = b.xyxy[0].cpu().numpy().tolist()
                        object_center = (
                            (bbox[0] + bbox[2]) / 2,
                            (bbox[1] + bbox[3]) / 2,
                        )

                        detections.append(
                            {
                                "bbox": bbox,
                                "class_id": cid,
                                "class_name": model.names.get(cid, str(cid)),
                                "stable_class": model.names.get(cid, str(cid)),
                                "confidence": float(b.conf[0].item()),
                                "track_id": tid,
                                "zone": get_zone(object_center, zones),
                                "stable_zone": get_zone(object_center, zones),
                                "direction": "Baseline",
                            }
                        )

            else:
                result = model.predict(
                    frame,
                    classes=list(ROAD_CLASSES),
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                )[0]

                raw = []

                if result.boxes is not None:
                    for b in result.boxes:
                        cid = int(b.cls[0].item())
                        bbox = b.xyxy[0].cpu().numpy().tolist()
                        object_center = (
                            (bbox[0] + bbox[2]) / 2,
                            (bbox[1] + bbox[3]) / 2,
                        )

                        raw.append(
                            {
                                "bbox": bbox,
                                "class_id": cid,
                                "class_name": model.names.get(cid, str(cid)),
                                "confidence": float(b.conf[0].item()),
                                "zone": get_zone(object_center, zones),
                            }
                        )

                detections = custom_tracker.update(raw)

            annotated = draw(frame, detections, zones or [])
            writer.write(annotated)

            for d in detections:
                tid = int(d.get("track_id", -1))
                class_name = d.get(
                    "stable_class",
                    d.get("class_name", "object"),
                )
                zone = d.get(
                    "stable_zone",
                    d.get("zone", "Unassigned"),
                )
                direction = d.get("direction", "Unknown")

                x1, y1, x2, y2 = map(float, d["bbox"])

                if tid >= 0:
                    all_track_ids.add(tid)
                    unique_classes[tid] = class_name

                observations_by_class[class_name] += 1
                observations_by_zone[zone] += 1

                if direction not in {"Baseline", "Stationary", "Unknown"}:
                    direction_flow[direction] += 1

                records.append(
                    {
                        "frame": processed,
                        "time_seconds": round(processed / fps, 2),
                        "track_id": tid,
                        "class": class_name,
                        "zone": zone,
                        "direction": direction,
                        "confidence": round(
                            float(d.get("confidence", 0.0)),
                            4,
                        ),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "center_x": round((x1 + x2) / 2, 2),
                        "center_y": round((y1 + y2) / 2, 2),
                    }
                )

            processed += 1

            if progress and (processed == 1 or processed % 5 == 0):
                progress(
                    processed,
                    limit or processed,
                    f"Processed {processed} frame(s)",
                )

    finally:
        cap.release()
        writer.release()

    elapsed = max(time.perf_counter() - start, 1e-9)

    unique_by_class = Counter(unique_classes.values())

    columns = [
        "frame",
        "time_seconds",
        "track_id",
        "class",
        "zone",
        "direction",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
    ]

    pd.DataFrame(records, columns=columns).to_csv(csv_path, index=False)

    return {
        "frames_processed": processed,
        "unique_track_ids": len(all_track_ids),
        "unique_by_class": dict(unique_by_class),
        "detection_records": len(records),
        "fps": processed / elapsed,
        "observations_by_class": dict(observations_by_class),
        "observations_by_zone": dict(observations_by_zone),
        "direction_flow": dict(direction_flow),
        "records": records,
    }