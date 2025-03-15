import cv2
import numpy as np
import uvicorn
import os
import shutil
import json

from fastapi import UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException

from detector import Detector

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
VIDEOS_DIR = "videos"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

PLACEHOLDER_IMAGE = f"{VIDEOS_DIR}/no_img.jpg"

class BaseDetector(Detector):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        
        self.app.add_api_route("/process_video", self.process_video, methods=["POST"])
        self.app.add_api_route("/upload", self.upload_video, methods=["POST"])
        self.app.add_api_route("/task/stream/{task_id}", self.stream_task, methods=["POST"])
        self.app.add_api_route("/task/data/{task_id}", self.task_data, methods=["POST"])
        self.app.add_api_route("/task/results/{task_id}", self.task_results, methods=["POST"])
        
        self.app.add_api_route("/task/status/{task_id}", self.get_task_status, methods=["GET"])
        
        self.app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")

    def get_free_task_id(self) -> int:
        """
        Возвращает минимальный незанятый task_id.
        """
        task_id = 1
        while task_id in self.task_params or os.path.exists(os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")):
            task_id += 1
        return task_id

    async def process_video(self, video_url: str, task_id: int):
        """
        Запускает обработку видео через API.
        """
        self.logger.info(f"Received request to process video: {video_url}")
        self._perform_inference_async(video_url, task_id)
        return {"message": "Video processing started", "task_id": task_id}

    async def upload_video(self, video: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
        """
        Получает загруженный файл, сохраняет его и передает на обработку.
        """
        video_path = os.path.join(UPLOAD_DIR, video.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        task_id = self.get_free_task_id()
        
        # Добавляем обработку в фон

        # Добавляем обработку в фон
        background_tasks.add_task(self._perform_inference_async, video_path, task_id)

        return JSONResponse(content={
            "message": "Video uploaded and processing started",
            "task_id": task_id,
            "video_url": PLACEHOLDER_IMAGE
        })
        
    async def get_task_status(self, task_id: int):
        """
        Возвращает текущий статус задачи по ее task_id.
        """
        status_file_path = os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")
        
        if os.path.exists(status_file_path):
            with open(status_file_path, "r") as status_file:
                status = json.load(status_file)
            return JSONResponse(content=status)
        else:
            return JSONResponse(content={"task_id": task_id, "success": False, "message": "Task not ready yet"})
        
    

    async def stream_task(self, task_id: int, stream_data: dict):
        """
        Обрабатывает поток данных для заданного task_id.
        """
        self.logger.info(f"Received stream data for task_id {task_id}: {stream_data}")
        return {"message": "Stream data processed", "task_id": task_id, "stream_info": stream_data}

    async def task_data(self, task_id: int, task_data: dict):
        """
        Обрабатывает данные задачи для заданного task_id.
        """
        self.logger.info(f"Received task data for task_id {task_id}: {task_data}")
        return {"message": "Task data processed", "task_id": task_id, "data": task_data}

    async def task_results(self, task_id: int, results: dict):
        """
        Принимает результаты выполнения задачи для заданного task_id и сохраняет их.
        """
        self.logger.info(f"Received results for task_id : {task_id}")
        
        # Создаем путь для сохранения результатов
        result_file_path = os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")
        
        # Сохраняем результаты в файл
        with open(result_file_path, "w") as result_file:
            json.dump(results, result_file)
        
        # Добавляем информацию о сохранении
        return {"message": "Task results saved", "task_id": task_id, "results_path": result_file_path}

if __name__ == "__main__":
    # Пример использования BaseDetector с путем к модели
    detector = BaseDetector("./models/best.onnx")
    uvicorn.run(detector.app, host="127.0.0.1", port=8000)
