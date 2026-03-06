from ultralytics import YOLO
import os


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(script_dir, "data", "data.yaml")

    model = YOLO("yolo26m.pt")

    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=768,
        batch=4,
        device=0,
        workers=4,
        cache=False,
        mosaic=1.0,
        close_mosaic=20,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        cos_lr=True,
        save=True,
        save_period=10,
        project=script_dir,
    )


if __name__ == "__main__":
    main()
