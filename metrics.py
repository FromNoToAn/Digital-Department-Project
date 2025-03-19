import argparse
from ultralytics import YOLO

# Создаем парсер для аргументов командной строки
parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
parser.add_argument("model_path", type=str, help="Path to the YOLO .pt model file")
parser.add_argument("yaml_path", type=str, help="Path to the yaml file")

# Парсим аргументы
args = parser.parse_args()

# Загружаем модель с указанного пути
model = YOLO(args.model_path)
metrics = model.val(
    data=args.yaml_path,  # Path to your dataset YAML
    split='test',            # Use test set
    plots=True,              # Generate confusion matrix
    save_json=True,          # Optional: Save metrics to JSON
    conf=0.5,                # Confidence threshold
    iou=0.5                 # IoU threshold
)