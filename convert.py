from ultralytics import YOLO

model = YOLO("models/yolov8s.pt")
model.export(format = "onnx")