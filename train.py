import argparse
from ultralytics import YOLO
import torch

# Создаем парсер для аргументов командной строки
parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
parser.add_argument("-m","--model_path", type=str, help="Path to the YOLO .pt model file")
parser.add_argument("yaml_path", type=str, help="Path to the yaml file")
parser.add_argument("-e","--epochs", type=int, default= 40, help="Number of epochs")
parser.add_argument("-b","--batchsize", type=int, default = 32, help="Number of batchsize")
parser.add_argument("-d","--device", type=str, choices=["cpu","0"], default = "gpu", help="Number of batchsize")


# Парсим аргументы
args = parser.parse_args()
if not(torch.cuda.is_available()) and args.device == "gpu":
    print("cuda is unavailable, choose cpu")
    quit()

# Загружаем модель с указанного пути
if args.model_path:
     model = YOLO(args.model_path)
else:
    model = YOLO(args.yaml_path)
model.train(data=args.yaml_path,epochs=args.epochs,device = args.device, batch =args.batchsize)