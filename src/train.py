from ultralytics import YOLO

def main():
    print("Initializing YOLO11s Model...")
    model = YOLO("yolo11s.pt")

    print("Starting Training Pipeline...")
    model.train(
        data="../data/ppe_data.yaml", # Points to the data folder
        epochs=50,
        imgsz=640,
        batch=16,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        fliplr=0.5, mosaic=1.0, erasing=0.4,
        project="../models",
        name="YOLO11_PPE_Augmented",
        save=True
    )

if __name__ == "__main__":
    main()
