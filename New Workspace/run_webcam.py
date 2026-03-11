import cv2
from detector.helmet_violation_detector import HelmetViolationDetector

detector = HelmetViolationDetector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam error")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections, helmet_boxes, nohelmet_boxes = detector.process_frame(frame)

    for det in detections:
        track_id = det["track_id"]
        x1, y1, x2, y2 = det["box"]
        hx1, hy1, hx2, hy2 = det["head_region"]
        violation = det["violation"]

        color = (0, 0, 255) if violation else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)

        text = f"ID {track_id} - {'NO HELMET' if violation else 'HELMET'}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not detections:
        for x1, y1, x2, y2 in helmet_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                "helmet",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
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

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
