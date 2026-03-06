from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "runs/detect/runs/helmet_yolo12m_refine/weights/best.pt"

counted_ids = set()

id_frame_count = {}

MIN_FRAMES = 15


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening webcam")
        return

    prev_time = 0
    violation_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_ids = set()

        results = model.track(frame, conf=0.7, iou=0.6, persist=True, verbose=False)

        for result in results:
            boxes = result.boxes

            if boxes.id is None:
                continue

            current_ids.update([int(i) for i in boxes.id])

            for box, track_id in zip(boxes, boxes.id):
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(track_id)

                label = model.names[cls]

                if label == "helmet":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                    id_frame_count[track_id] = id_frame_count.get(track_id, 0) + 1

                    if (
                        id_frame_count[track_id] >= MIN_FRAMES
                        and track_id not in counted_ids
                    ):
                        violation_count += 1
                        counted_ids.add(track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"{label} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        for tracked_id in list(id_frame_count.keys()):
            if tracked_id not in current_ids:
                id_frame_count.pop(tracked_id)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Violations: {violation_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Helmet Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
