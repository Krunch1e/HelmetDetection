from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "runs/detect/runs/helmet_yolo12m_refine/weights/best.pt"
INPUT_FOLDER = "Test Images"
OUTPUT_FOLDER = "outputs/images"


def process_image(input_path, output_path, model):
    image = cv2.imread(input_path)

    if image is None:
        print(f"Failed to load {input_path}")
        return

    # Optional: Slight brightness correction (helps night a bit)
    image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)

    results = model(image, conf=0.35, verbose=False)
    annotated = results[0].plot()

    cv2.imwrite(output_path, annotated)


def main():
    model = YOLO(MODEL_PATH)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("No images found in Test Images folder.")
        return

    for image_name in image_files:
        input_path = os.path.join(INPUT_FOLDER, image_name)

        base_name = os.path.splitext(image_name)[0]
        output_name = f"{base_name}_processed.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        print(f"Processing {image_name}...")

        process_image(input_path, output_path, model)

        print(f"Saved → {output_path}")

    print("All images processed successfully.")


if __name__ == "__main__":
    main()
