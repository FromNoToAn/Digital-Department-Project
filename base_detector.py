import cv2
import numpy as np
import uvicorn

from detector import Detector

class BaseDetector(Detector):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.app.add_api_route("/process_video", self.process_video, methods=["POST"])
        
    async def process_video(self, video_url: str, task_id: int):
        """
        Запускает обработку видео через API.
        """
        self.logger.info(f"Received request to process video: {video_url}")
        self._perform_inference_async(video_url, task_id)
        return {"message": "Video processing started", "task_id": task_id}


if __name__ == "__main__":
    # Пример использования BaseDetector с путем к модели
    # detector = BaseDetector("./models/yolov8s_576x1024.onnx")
    detector = BaseDetector("./models/best.onnx")
    uvicorn.run(detector.app, host="127.0.0.1", port=8000)
