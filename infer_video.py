from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "runs/detect/runs/helmet_yolo12m_refine/weights/best.pt"
INPUT_FOLDER = "Test Videos"
OUTPUT_FOLDER = "outputs/videos"


def process_video(input_path, output_path, model):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.35)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()


def main():
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    video_files = [
        f
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    if not video_files:
        print("No videos found.")
        return

    for video_name in video_files:
        input_path = os.path.join(INPUT_FOLDER, video_name)

        base_name = os.path.splitext(video_name)[0]
        output_name = f"{base_name}_processed.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        print(f"Processing {video_name}...")

        process_video(input_path, output_path, model)

        print(f"Saved → {output_path}")

    print("All videos processed.")


if __name__ == "__main__":
    main()
