import argparse
from ultralytics import YOLO

# Создаем парсер для аргументов командной строки
parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
parser.add_argument("model_path", type=str, help="Path to the YOLO .pt model file")

# Парсим аргументы
args = parser.parse_args()

# Загружаем модель с указанного пути
model = YOLO(args.model_path)

# Экспортируем модель в формат ONNX
model.export(format="onnx")

print(f"Model {args.model_path} exported to ONNX format.")
