from detector import Detector


class BaseDetector(Detector):
    pass


if __name__ == "base_detector":
    detector = BaseDetector("./models/yolov8s_576x1024.onnx")
