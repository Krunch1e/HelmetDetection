from ultralytics import YOLO
from config import (
    MODEL_PATH,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    HEAD_REGION_RATIO,
    VOTE_WINDOW,
)
from utils.geometry import inside_region, aspect_ratio_valid


class HelmetViolationDetector:
    def __init__(self):
        self.model = YOLO(str(MODEL_PATH))
        self.history = {}

    def process_frame(self, frame):
        results = self.model.track(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
        )

        rider_boxes = []
        helmet_boxes = []
        nohelmet_boxes = []

        for r in results:
            if r.boxes.id is None:
                continue

            for box, cls, track in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                label = self.model.names[int(cls)]

                if label == "rider":
                    rider_boxes.append((int(track), x1, y1, x2, y2))
                elif label == "helmet":
                    helmet_boxes.append((x1, y1, x2, y2))
                elif label == "no-helmet":
                    nohelmet_boxes.append((x1, y1, x2, y2))

        detections = []

        for track_id, x1, y1, x2, y2 in rider_boxes:
            head_y2 = int(y1 + (y2 - y1) * HEAD_REGION_RATIO)
            head_region = (x1, y1, x2, head_y2)

            helmet = any(
                aspect_ratio_valid(h) and inside_region(h, head_region)
                for h in helmet_boxes
            )
            nohelmet = any(inside_region(n, head_region) for n in nohelmet_boxes)

            if track_id not in self.history:
                self.history[track_id] = []

            if helmet:
                self.history[track_id].append("helmet")
            elif nohelmet:
                self.history[track_id].append("nohelmet")

            if len(self.history[track_id]) > VOTE_WINDOW:
                self.history[track_id].pop(0)

            votes = self.history[track_id]
            violation = votes.count("nohelmet") > votes.count("helmet")

            detections.append(
                {
                    "track_id": track_id,
                    "box": (x1, y1, x2, y2),
                    "head_region": head_region,
                    "violation": violation,
                }
            )

        return detections, helmet_boxes, nohelmet_boxes
