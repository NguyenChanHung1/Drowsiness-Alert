from ultralytics import YOLO

model = YOLO("ml/weights/yolo11n.pt")

model.export(format="tflite")