import os
import glob
import sys
import cv2
import numpy as np
import uvicorn
import shutil
import json
import aiohttp

from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException

from detector import Detector

from config import general_cfg

RESULTS_DIR = "results"
VIDEOS_DIR = "videos"


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
        
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        
        for file in glob.glob(os.path.join(RESULTS_DIR, "task_*_data.json")):
            os.remove(file)

        # Удаляем все PNG-файлы в videos/
        for file in glob.glob(os.path.join(VIDEOS_DIR, "*.jpg")):
            os.remove(file)
        
        self.app.add_api_route("/upload_video", self.upload_video, methods=["POST"])
        
        self.app.add_api_route("/task/status/{task_id}", self.get_status_task, methods=["GET"])
        self.app.add_api_route("/task/status/{task_id}", self.delete_status_task, methods=["DELETE"])
        
        self.app.add_api_route("/task/results/{task_id}", self.results_task, methods=["POST"])
        
        self.app.add_api_route("/task/stream/{task_id}", self.stream_task, methods=["POST"])
        self.app.add_api_route("/task/stream/{task_id}", self.get_stream_task, methods=["GET"])
        
        self.app.add_api_route("/task/data/{task_id}", self.data_task, methods=["POST"])
        self.app.add_api_route("/task/data/{task_id}", self.get_data_task, methods=["GET"])
        
        self.app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")


    def get_free_task_id(self) -> int:
        """
        Возвращает минимальный незанятый task_id.
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Удаляем файлы task_X_data.json в results/
        
        task_id = 1
        while task_id in self.task_params or os.path.exists(os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")):
            task_id += 1
        return task_id


    async def upload_video(self, video: UploadFile = File(...), is_realtime: bool = Form(False)):
        """
        Принимает видео и отправляет его на эндпоинт обработки без сохранения на диск.
        """
        task_id = self.get_free_task_id()
        
        properties = {
        "isRealtime": str(is_realtime),
        # "cornerUp": 0,
        # "cornerLeft": 0,
        # "cornerBottom": 1080,
        # "cornerRight": 1920
        }
        
        self.logger.info(f"is_realtime: {is_realtime}")
        
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            video_bytes = await video.read()
            
            form_data.add_field("file", video_bytes, filename=video.filename, content_type=video.content_type)
            form_data.add_field("properties", json.dumps(properties), content_type="text/plain")
            
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
                        status["mp4"] = False
                        return JSONResponse(content=status)
                    else:
                        # Если GET-запрос не вернул статус, проверяем файл с результатами
                        status_file_path = os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")
                        
                        if os.path.exists(status_file_path):
                            with open(status_file_path, "r") as status_file:
                                status = json.load(status_file)
                            status["mp4"] = True
                            return JSONResponse(content=status)
                        else:
                            # Если и файл не найден, возвращаем ошибку
                            return JSONResponse(content={
                                "task_id": task_id,
                                "success": False,
                                "mp4": False,
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


    async def stream_task(self, task_id: int, file: UploadFile = File(...)):
        file_path = f"{VIDEOS_DIR}/{task_id}.jpg"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"Image received for task {task_id}", "file_path": file_path}

    async def get_stream_task(self, task_id: int):
        
        img_path = f"{VIDEOS_DIR}/{task_id}.jpg"
        video_path = os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")
        
        if os.path.exists(video_path):
            return JSONResponse(content={"stream": False, "message": "Stream has ended"})
        elif not os.path.exists(img_path):
            return JSONResponse(content={"stream": True, "message": "Stream file not found"})
        else:
            file_url = f"http://{general_cfg['manager_host']}:{general_cfg['manager_port']}/{img_path}"
            return JSONResponse(content={"stream": True, "message": "Stream file found", "file_url": file_url})
        
    async def data_task(self, task_id: int, task_data: dict):
        """
        Обрабатывает данные задачи для заданного task_id.
        """
        file_path = os.path.join(RESULTS_DIR, f"task_{task_id}_data.json")
    
        # Сохраняем данные задачи в файл JSON
        with open(file_path, "w") as json_file:
            json.dump(task_data, json_file)
                
        # self.logger.info(f"Received task data for task_id {task_id}: {task_data}")
        return {"message": "Task data processed", "task_id": task_id, "data": task_data}
    
    async def get_data_task(self, task_id: int):
        """
        Получает данные задачи для заданного task_id из JSON-файла и возвращает их как JSON-ответ.
        """
        # Определяем путь для файла данных задачи
        file_path = os.path.join(RESULTS_DIR, f"task_{task_id}_data.json")
        
        # Проверяем, существует ли файл данных
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        # Читаем данные из файла JSON
        with open(file_path, "r") as json_file:
            task_data = json.load(json_file)
        
        # Возвращаем данные как JSON-ответ
        return JSONResponse(content=task_data)
    
    
    async def results_task(self, task_id: int, results: dict):
        """
        Принимает результаты выполнения задачи для заданного task_id и сохраняет их.
        """
        # self.logger.info(f"Received results for task_id : {task_id}")
        
        # Создаем путь для сохранения результатов
        result_file_path = os.path.join(RESULTS_DIR, f"task_{task_id}_results.json")
        
        # Сохраняем результаты в файл
        with open(result_file_path, "w") as result_file:
            json.dump(results, result_file)
            
        img_path = f"{VIDEOS_DIR}/{task_id}.jpg"
        data_path = os.path.join(RESULTS_DIR, f"task_{task_id}_data.json")
        
        if os.path.exists(img_path):
            os.remove(img_path)
            
        if os.path.exists(data_path):
            os.remove(data_path)
        
        # Добавляем информацию о сохранении
        return {"message": "Task results saved", "task_id": task_id, "results_path": result_file_path}

if __name__ == "__main__":
    
    detector = BaseDetector("./models/best.onnx")
    uvicorn.run(detector.app, host=general_cfg["manager_host"], port=general_cfg["manager_port"])
