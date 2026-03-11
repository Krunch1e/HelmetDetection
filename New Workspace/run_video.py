import cv2
import os
from config import SCREEN_WIDTH, SCREEN_HEIGHT, VIOLATION_DIR
from detector.helmet_violation_detector import HelmetViolationDetector

VIDEO_PATH = "test_video.mp4"

os.makedirs(VIOLATION_DIR, exist_ok=True)

detector = HelmetViolationDetector()
cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0


def scale_frame(frame):
    h, w = frame.shape[:2]
    scale = min(SCREEN_WIDTH / w, SCREEN_HEIGHT / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    detections, helmet_boxes, nohelmet_boxes = detector.process_frame(frame)

    for det in detections:
        track_id = det["track_id"]
        x1, y1, x2, y2 = det["box"]
        hx1, hy1, hx2, hy2 = det["head_region"]
        violation = det["violation"]

        color = (0, 0, 255) if violation else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)

        text = f"ID {track_id}"
        if violation:
            text += " NO HELMET"
            if frame_id % 20 == 0:
                cv2.imwrite(
                    f"{VIOLATION_DIR}/violation_{track_id}_{frame_id}.jpg", frame
                )

        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for x1, y1, x2, y2 in helmet_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame, "helmet", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    for x1, y1, x2, y2 in nohelmet_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            "no helmet",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    frame = scale_frame(frame)
    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
