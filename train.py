from ultralytics import YOLO


def main():

    model = YOLO("runs/detect/runs/helmet_yolo12m_final/weights/last.pt")

    model.train(
        data="data/data.yaml",
        epochs=40,  # 20–30 more epochs
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        amp=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        lr0=0.003,
        lrf=0.01,
        cos_lr=True,
        patience=10,
        save_period=5,
        project="runs",
        name="helmet_yolo12m_refine",  # NEW NAME
        resume=False,  # IMPORTANT
    )


if __name__ == "__main__":
    main()
