import cv2
import os

BASE_VIDEO_PATH = r"D:\MajorProject\HelmetDetection\dashcam_videos"
BASE_OUTPUT_PATH = r"D:\MajorProject\HelmetDetection\dataset_new"


def extract_from_folder(video_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(video_folder):
        if file.lower().endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(video_folder, file)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"❌ Could not open {file}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps)  # 1 frame per second
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"\n🎥 Processing: {file}")
            print(f"Total Frames: {total_frames} | FPS: {fps}")

            video_name = os.path.splitext(file)[0]
            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    filename = f"{video_name}_frame_{saved_count}.jpg"
                    save_path = os.path.join(output_folder, filename)
                    cv2.imwrite(save_path, frame)
                    saved_count += 1

                if frame_count % 300 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")

                frame_count += 1

            cap.release()
            print(f"✅ Saved {saved_count} frames from {file}")

    print(f"\n🎯 Finished processing folder: {video_folder}")


if __name__ == "__main__":
    extract_from_folder(
        os.path.join(BASE_VIDEO_PATH, "night"),
        os.path.join(BASE_OUTPUT_PATH, "night_raw"),
    )

    extract_from_folder(
        os.path.join(BASE_VIDEO_PATH, "day"), os.path.join(BASE_OUTPUT_PATH, "day_raw")
    )

    print("\n🚀 ALL EXTRACTION COMPLETE")
