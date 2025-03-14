from ultralytics import YOLO

if __name__ == "__main__":
    pth_path = r"C:\Users\Lenovo\Desktop\ultralytics-main\yolov8\runs\detect\train\weights\best.pt"

    test_path = r"C:\Users\Lenovo\Desktop\ultralytics-main\yolov8\ceshi"
    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model

    # Predict with the model
    results = model(test_path, save=True, conf=0.5)  # predict on an image