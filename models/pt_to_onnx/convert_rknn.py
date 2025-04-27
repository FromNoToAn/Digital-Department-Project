from ultralytics import YOLO
model = YOLO("yolov8.pt")
path = model.export(format="onnx")  # Internal method written by airockchip, don't be fooled by the format name
