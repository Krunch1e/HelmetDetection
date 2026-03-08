from ultralytics import YOLO
import cv2

MODEL_PATH = "train/weights/best.pt"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def main():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture("test_video.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.35, verbose=False)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "rider":
                    color = (255, 200, 0)
                    thickness = 1

                elif label == "helmet":
                    color = (0, 255, 0)
                    thickness = 3

                else:  # no-helmet
                    color = (0, 0, 255)
                    thickness = 3

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        cv2.imshow("Helmet Detection", display_frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
