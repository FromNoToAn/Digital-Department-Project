import os
import sys
import cv2
import numpy as np
import uvicorn
import shutil
import json
import aiohttp

from fastapi import UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException

from detector import Detector

from config import general_cfg

RESULTS_DIR = "results"
VIDEOS_DIR = "videos"

os.makedirs(VIDEOS_DIR, exist_ok=True)

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
        
        self.app.add_api_route("/upload_video", self.upload_video, methods=["POST"])
        
        self.app.add_api_route("/task/status/{task_id}", self.get_status_task, methods=["GET"])
        self.app.add_api_route("/task/status/{task_id}", self.delete_status_task, methods=["DELETE"])
        
        self.app.add_api_route("/task/results/{task_id}", self.results_task, methods=["POST"])
        
        self.app.add_api_route("/task/stream/{task_id}", self.get_stream_task, methods=["GET"])
        self.app.add_api_route("/task/data/{task_id}", self.task_data, methods=["POST"])
        
        self.app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")


    def get_free_task_id(self) -> int:
        """
        Возвращает минимальный незанятый task_id.
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        task_id = 1
        while task_id in self.task_params or os.path.exists(os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")):
            task_id += 1
        return task_id


    async def upload_video(self, video: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
        """
        Принимает видео и отправляет его на эндпоинт обработки без сохранения на диск.
        """
        task_id = self.get_free_task_id()
        
        properties = {
        "isRealtime": "true",
        "cornerUp": 0,
        "cornerLeft": 0,
        "cornerBottom": 1080,
        "cornerRight": 1920
        }
        
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field("file", video.file, filename=video.filename, content_type=video.content_type)
            form_data.add_field("properties", json.dumps(properties), content_type="application/json")
            
            response = await session.post(f"http://{general_cfg['manager_host']}:{general_cfg['manager_port']}/api/inference/{task_id}", data=form_data)

            if response.status != 200:
                return JSONResponse(content={"message": "Error sending video", "status": response.status}, status_code=500)

        return JSONResponse(content={
            "message": "Video uploaded and sent for processing",
            "task_id": task_id
        })


    async def get_status_task(self, task_id: int):
        """
        Возвращает текущий статус задачи, запрашивая его через GET-запрос.
        Всегда запрашивает актуальную информацию, не используя хранилище.
        """
        # Выполняем GET-запрос к API, чтобы получить актуальный статус
        async with aiohttp.ClientSession() as session:
            try:
                url = f"http://{general_cfg['manager_host']}:{general_cfg['manager_port']}/api/inference/{task_id}"
                async with session.get(url) as response:
                    if response.status == 200:
                        status = await response.json()
                        return JSONResponse(content=status)
                    else:
                        # Если GET-запрос не вернул статус, проверяем файл с результатами
                        status_file_path = os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")
                        
                        if os.path.exists(status_file_path):
                            with open(status_file_path, "r") as status_file:
                                status = json.load(status_file)
                            return JSONResponse(content=status)
                        else:
                            # Если и файл не найден, возвращаем ошибку
                            return JSONResponse(content={
                                "task_id": task_id,
                                "success": False,
                                "message": f"Task {task_id} not found or error occurred (Status code: {response.status})"
                            }, status_code=404)

            except Exception as e:
                return JSONResponse(content={
                    "task_id": task_id,
                    "success": False,
                    "message": f"Error occurred: {str(e)}"
                }, status_code=500)
                
    async def delete_status_task(self, task_id: int):
        """
        Инициирует удаление самой задачи через API.
        """
        # Удалим саму задачу, отправив запрос на /api/inference/{task_id} с методом DELETE
        async with aiohttp.ClientSession() as session:
            try:
                url = f"http://{general_cfg['manager_host']}:{general_cfg['manager_port']}/api/inference/{task_id}"
                async with session.delete(url) as response:
                    if response.status == 200:
                        return JSONResponse(content={
                            "task_id": task_id,
                            "message": "Task status and task deleted successfully"
                        })
                    else:
                        # Если удаление задачи через API не удалось, возвращаем сообщение об ошибке
                        return JSONResponse(content={
                            "task_id": task_id,
                            "message": f"Task deletion failed with status code {response.status}"
                        }, status_code=response.status)

            except Exception as e:
                # В случае ошибки при отправке запроса на удаление задачи через API
                return JSONResponse(content={
                    "task_id": task_id,
                    "message": f"Error occurred while trying to delete task: {str(e)}"
                }, status_code=500)
        
    # async def status_task(self, task_id: int):
    #     """
    #     Возвращает текущий статус задачи по ее task_id.
    #     """
        
    #     status_file_path = os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")
        
    #     now_progress = 0.00
    #     if self.task_params:
    #         now_progress = self.task_params[task_id].progress
        
    #     if os.path.exists(status_file_path):
    #         with open(status_file_path, "r") as status_file:
    #             status = json.load(status_file)
    #         return JSONResponse(content=status)
    #     else:
    #         return JSONResponse(content={"task_id": task_id, "success": False, "progress": now_progress, "message": "Task not ready yet"})
        
    

    async def get_stream_task(self, task_id: int):
        url = f"http://{general_cfg['manager_host']}:{general_cfg['manager_port']}/task/stream/{task_id}"
        
        # Используем aiohttp для асинхронного получения потока
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    frame = await response.read()  # Получаем кадр
                    return frame
                else:
                    raise HTTPException(status_code=response.status, detail="Failed to get task stream")

    async def task_data(self, task_id: int, task_data: dict):
        """
        Обрабатывает данные задачи для заданного task_id.
        """
        self.logger.info(f"Received task data for task_id {task_id}: {task_data}")
        return {"message": "Task data processed", "task_id": task_id, "data": task_data}

    async def results_task(self, task_id: int, results: dict):
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
    
    detector = BaseDetector("./models/best.onnx")
    uvicorn.run(detector.app, host=general_cfg["manager_host"], port=general_cfg["manager_port"])
