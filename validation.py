from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/runs/helmet_yolo12m_refine/weights/best.pt")
    metrics = model.val(data="data/data.yaml", workers=0)  # important
    print(metrics)


if __name__ == "__main__":
    main()
