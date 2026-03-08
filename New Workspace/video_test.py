from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "train/weights/best.pt"
VIDEO_PATH = "test_video.mp4"

HEAD_REGION_RATIO = 0.4
CONF_THRESHOLD = 0.35

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

VOTE_WINDOW = 10

VIOLATION_DIR = "violations"
os.makedirs(VIOLATION_DIR, exist_ok=True)

history = {}


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def inside_region(box, region):
    cx, cy = box_center(box)
    x1, y1, x2, y2 = region
    return x1 < cx < x2 and y1 < cy < y2


def aspect_ratio_valid(box):

    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    if h == 0:
        return False

    ratio = w / h

    return 0.6 < ratio < 1.6


def scale_frame(frame):

    h, w = frame.shape[:2]

    scale = min(SCREEN_WIDTH / w, SCREEN_HEIGHT / h)

    nw = int(w * scale)
    nh = int(h * scale)

    return cv2.resize(frame, (nw, nh))


def main():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        results = model.track(
            frame,
            conf=CONF_THRESHOLD,
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

                label = model.names[int(cls)]

                if label == "rider":
                    rider_boxes.append((int(track), x1, y1, x2, y2))

                elif label == "helmet":
                    helmet_boxes.append((x1, y1, x2, y2))

                elif label == "no-helmet":
                    nohelmet_boxes.append((x1, y1, x2, y2))

        for track_id, x1, y1, x2, y2 in rider_boxes:
            head_y2 = int(y1 + (y2 - y1) * HEAD_REGION_RATIO)

            head_region = (x1, y1, x2, head_y2)

            helmet = False
            nohelmet = False

            for h in helmet_boxes:
                if not aspect_ratio_valid(h):
                    continue

                if inside_region(h, head_region):
                    helmet = True

            for n in nohelmet_boxes:
                if inside_region(n, head_region):
                    nohelmet = True

            if track_id not in history:
                history[track_id] = []

            if helmet:
                history[track_id].append("helmet")

            elif nohelmet:
                history[track_id].append("nohelmet")

            if len(history[track_id]) > VOTE_WINDOW:
                history[track_id].pop(0)

            votes = history[track_id]

            helmet_votes = votes.count("helmet")
            nohelmet_votes = votes.count("nohelmet")

            violation = nohelmet_votes > helmet_votes

            color = (0, 255, 0)

            if violation:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, head_y2), (255, 0, 0), 1)

            text = f"ID {track_id}"

            if violation:
                text += " NO HELMET"

                if frame_id % 20 == 0:
                    cv2.imwrite(
                        f"{VIOLATION_DIR}/violation_{track_id}_{frame_id}.jpg", frame
                    )

            cv2.putText(
                frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

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

        frame = scale_frame(frame)

        cv2.imshow("Helmet Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
