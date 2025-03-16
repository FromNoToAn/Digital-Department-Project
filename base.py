"""
This module is the core for the Base Detector.
It contains an API for communicating with the video analytics manager.
It also has a method for getting frames and time from a video stream,
processing them and saving the results. It also has methods for other detectors
that should be redefined for processing by a specific detector.
"""
import os
import queue
import re
import sys
import time
import json
from collections import defaultdict
import traceback
from abc import ABC, abstractmethod
from logging import Logger
from subprocess import Popen, TimeoutExpired
import threading
from threading import Thread

import cv2
import ffmpeg
import numpy as np
import requests
from fastapi import FastAPI

from api import create_app
from api.request_data import RequestPostData
from config import general_cfg
from logger import create_logger
from schemas.inference_parameters import FFprobeParameters, InferenceCycleParameters
from SFSORT import SFSORT
from utils.dataclasses import StatusTask, TaskParameters
from utils.validate import validate_unix_timestamp


class Base(ABC):
    """
    Core of the Base Detector.
    """
    class YOLODataLogger:
        def __init__(self, task_id):
            self.active_objects = defaultdict(dict)
            self.lock = threading.Lock()
            self.start_time = time.time()
            self.task_id = task_id
            
            # Запускаем периодическое обновление JSON
            self.timer = threading.Timer(5.0, self.update_json)
            self.timer.daemon = True
            self.timer.start()

        def _calculate_duration(self, obj_id):
            """Вычисляет время пребывания объекта в кадре."""
            return round(time.time() - self.active_objects[obj_id]['first_seen'], 2)

        def process_detections(self, detections):
            """
            Обновляет информацию о текущих объектах в кадре.
            Args:
                detections (list): Список обнаружений в формате 
                                [{"class": "", "bbox": [x,y,w,h]}, ...]
            """
            current_time = time.time()
            current_ids = set()
            
            with self.lock:
                # Генерируем уникальные ID на основе хеша параметров
                for det in detections:
                    obj_id = hash((det['class'], tuple(det['bbox'])))
                    
                    if obj_id not in self.active_objects:
                        self.active_objects[obj_id] = {
                            'class': det['class'],
                            'bbox': det['bbox'],
                            'first_seen': current_time,
                            'last_seen': current_time
                        }
                    else:
                        self.active_objects[obj_id]['last_seen'] = current_time
                    current_ids.add(obj_id)
                
                # Удаляем объекты, которые больше не в кадре
                expired_ids = [obj_id for obj_id in self.active_objects 
                            if obj_id not in current_ids]
                for obj_id in expired_ids:
                    del self.active_objects[obj_id]

        def generate_data(self):
            """Формирует данные для сохранения в JSON."""
            data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "objects": []
            }
            
            with self.lock:
                for obj_id, obj in self.active_objects.items():
                    data["objects"].append({
                        "class": obj['class'],
                        "bbox": obj['bbox'],
                        "duration": self._calculate_duration(obj_id)
                    })
            
            return data

        def update_json(self):
            """Обновляет JSON-файл с данными и перезапускает таймер."""
            try:
                data = self.generate_data()
                with open('./json_logs/yolo_detections.json', "w") as f:
                    json.dump(data, f, indent=4)
            finally:
                # Перезапускаем таймер
                self.timer = threading.Timer(5.0, self.update_json)
                self.timer.start()

        def stop(self):
            """Останавливает периодическое обновление."""
            self.timer.cancel()

    def __init__(self):
        self.logger: Logger = create_logger(self.__class__.__name__)
        """A logger for displaying various information."""
        self.task_params: dict[int, TaskParameters] = {}
        """It stores all the parameters for each running task."""
        self.trackers: dict[int, SFSORT] = {}
        """It stores all the object trackers in the video for each running task."""
        self.timestamps: dict[int, queue.Queue] = {}
        """It stores all timestamps of the last processed frame for each running task."""
        self.app: FastAPI = create_app(
            self.task_params, self._perform_inference_async, self.logger
        )
        """The application object for communication with the video analytics manager."""
        self.data_loggers = {}

    @abstractmethod
    def pre_process(
        self, input_img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """
        Adjusts the dimensions of the original image to those required
        for processing without changing the aspect ratio.

        Parameters
        ----------
        input_img : ndarray
            The original image.

        Returns
        -------
        img : ndarray
            The resized image.
        ratio : float
            The aspect ratio of the image to be processed relative
            to the original one.
        dwdh : tuple[float, float]
            It contains paddings in width and height, respectively.
        """
        raise NotImplementedError("Subclasses must implement pre_process")

    @abstractmethod
    def inference(self, img: np.ndarray) -> np.ndarray:
        """
        Processes the image with a neural network.

        Parameters
        ----------
        img : ndarray
            The image to be processed.

        Returns
        -------
        output : ndarray
            The results of image processing by a neural network.
        """
        raise NotImplementedError("Subclasses must implement inference")

    @abstractmethod
    def post_process(
        self, outputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """
        Processes the results from the neural network and presents them as:
        bbox coordinates, class, accuracy - for each found object in the frame.

        Parameters
        ----------
        outputs : ndarray
            The results obtained from neural network processing.

        Returns
        -------
        boxes : np.ndarray | None
            An array of bounding box coordinates, where each box is represented
            as [x_min, y_min, x_max, y_max]. None if no objects are detected.
        classes : np.ndarray | None
            An array of class indices corresponding to the detected objects.
            None if no objects are detected.
        scores : np.ndarray | None
            An array of confidence scores (or probabilities) associated with
            each detected object. None if no objects are detected.
        """
        raise NotImplementedError("Subclasses must implement post_process")

    @staticmethod
    def draw_ROI(
        img: np.ndarray,
        corners: list[int],
    ) -> np.ndarray:
        """
        Highlights the region of interest by coordinates

        Parameters
        ----------
        img : ndarray
            The original image.
        cornerUp : int
            The coordinate of the upper-left corner in height
        cornerLeft : int
            The coordinate of the upper-left corner in width
        cornerBottom : int
            The coordinate of the lower-right corner in height
        cornerRight : int
            The coordinate of the lower-right corner in width

        Returns
        -------
        inf_img : ndarray
            An image with the selected region of interest
        """
        inf_img = img.copy()
        mask = np.zeros(inf_img.shape[:2], np.uint8)

        cornerUp = corners[0]
        cornerLeft = corners[1]
        cornerBottom = corners[2]
        cornerRight = corners[3]

        mask[cornerUp:cornerBottom, cornerLeft:cornerRight] = 255
        inf_img = cv2.bitwise_and(inf_img, inf_img, mask=mask)
        return inf_img

    @staticmethod
    def draw_results(img: np.ndarray, dets: list) -> np.ndarray:
        """
        Draws all found objects, their classes, confidence, and
        track ID on the source frame.

        Parameters
        ----------
        img : ndarray
            The original image.
        dets : list of list
            A list of all the properties of each object in the frame
            (coordinates of the bounding box, class, accuracy).

        Returns
        -------
        inf_img : ndarray
            An image with all the selected objects and their properties.
        """
        inf_img = img.copy()

        for det in dets:
            cv2.rectangle(
                img=inf_img,
                pt1=(det[0], det[1]),
                pt2=(det[2], det[3]),
                color=(235, 215, 50),
                thickness=3,
            )
            cv2.putText(
                img=inf_img,
                text=f"{det[4]}, {det[5]}, {det[6]:.2f}",
                # text=f"({det[4]}){det[5]} - {round(det[6], 2)}",
                org=(int(det[0]) + 13, int(det[1]) - 13),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1.5,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        return inf_img

    def _get_timestamp(self, process: Popen, task_id: str) -> None:
        """
        A loop that takes and records timestamps from the ffmpeg process
        for each received frame.

        Parameters
        ----------
        process : :obj:`Popen`
            An ffmpeg process that reads frames from a video stream and text data.
        task_id : str
            The ID of the video processing task.
        """
        sei = False
        zero_is_twice = False

        while self.task_params[task_id].inference_status == StatusTask.RUNNING:
            while self.task_params[task_id].frame_processed <= 0 and not sei:
                err_line = process.stderr.readline()  # type: ignore
                if not err_line:
                    break
                msg: str = err_line.decode().strip()

                match = re.search(r"SEI.*ts: (\d+)", msg)
                if match:
                    timestamp = int(match.group(1))
                    sei = validate_unix_timestamp(timestamp)
                    if sei:
                        self.timestamps[task_id].put(timestamp)
                        break

                match = re.search(r"pts_time:\s*([-+]?\d*\.\d+|\d+)", msg)
                if match:
                    if not int(float(match.group(1)) * 1000):
                        if zero_is_twice:
                            continue
                        zero_is_twice = True
                    timestamp = int(float(match.group(1)) * 1000)
                    self.timestamps[task_id].put(timestamp)

            err_line = process.stderr.readline()  # type: ignore
            if not err_line:
                break
            msg: str = err_line.decode().strip()

            if sei:
                match = re.search(r"SEI.*ts: (\d+)", msg)
                if match:
                    timestamp = int(match.group(1))
            else:
                match = re.search(r"pts_time:\s*([-+]?\d*\.\d+|\d+)", msg)
                if match:
                    timestamp = int(float(match.group(1)) * 1000)
            if timestamp not in self.timestamps[task_id].queue:
                self.timestamps[task_id].put(timestamp)
                

    def run(self, img: np.ndarray, task_id: int) -> list:
        """
        Runs the object detection and tracking pipeline on the given image.

        This method processes an input image, performs inference to detect objects,
        applies post-processing to refine detections, and uses a tracker to maintain
        object identities across frames.

        Parameters
        ----------
        img : np.ndarray
            The input image for object detection, represented as a NumPy array.
        task_id : int
            The identifier for the current task, used to manage trackers for
            multi-task scenarios.

        Returns
        -------
        dets : list
            A list of detected and tracked objects. Each element in the list is a
            sub-list containing the following information:
            - x0, y0, x1, y1 (int): Bounding box coordinates for the detected object.
            - class_id (int): The class index of the detected object.
            - tracker_id (int): The ID assigned to the object by the tracker.
            - score (float): The confidence score of the detection.
            Returns an empty list if no objects are detected.
        """
        dets = []
        pre_img, ratio, dwdh = self.pre_process(img)

        outputs = self.inference(pre_img)
        if outputs is not None:

            boxes, classes, scores = self.post_process(outputs, dwdh, ratio)
            if boxes is not None:

                match general_cfg["tracker"]:
                    case "sfsort":
                        tracks = self.trackers[task_id].update(boxes, scores, classes)
                        if len(tracks):
                            for track in tracks:
                                x0, y0, x1, y1 = map(int, track[0])
                                dets.append(
                                    [
                                        x0,
                                        y0,
                                        x1,
                                        y1,
                                        int(track[1]),  # Track ID
                                        int(track[2]),  # Class ID
                                        float(track[3]),# Confidence
                                    ]
                                )
                    case "botsort":
                        tracks = self.trackers[task_id].update(
                            np.hstack((boxes,
                                       scores.reshape(-1,1),
                                       classes.reshape(-1,1))),
                            img
                            )
                        if len(tracks):
                            for track in tracks:
                                x0, y0, x1, y1 = map(int, track[0:4])
                                dets.append(
                                    [
                                        x0,
                                        y0,
                                        x1,
                                        y1,
                                        int(track[4]),  # Track ID
                                        int(track[6]),  # Class ID
                                        float(track[5]),# Confidence
                                    ]
                                )
        return dets

    @staticmethod
    def _ffprobe_read(video_url: str) -> FFprobeParameters:
        """
        Extracts video metadata using FFmpeg and returns relevant parameters.

        Parameters
        ----------
        video_url : str
            The URL or file path of the video to probe.

        Returns
        -------
        FFprobeParameters
            An object containing video properties such as width, height, frame rate,
            frame interval, and duration.
        """
        print(video_url)
        try:
            probe = ffmpeg.probe(video_url)
            # print(probe)
        except ffmpeg.Error as e:
            print("ffprobe stderr output:")
            print(e.stderr.decode())  # Выводим stderr
            raise RuntimeError(e.stderr.decode())
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        fps = video_info["r_frame_rate"].split("/")
        fps = int(fps[0]) / int(fps[1])
        duration = (
            float(video_info["duration"])
            if "duration" in video_info
            else general_cfg["working_time_sec"]
        )

        params = FFprobeParameters(
            width=int(video_info["width"]),
            height=int(video_info["height"]),
            fps=fps,
            frame_interval=fps / general_cfg["framerate"],
            duration=duration,
        )

        return params

    @staticmethod
    def _update_tracker(tracker: SFSORT, ffprobe_params: FFprobeParameters) -> None:
        """
        Updates tracker configuration based on FFmpeg probe parameters.

        Parameters
        ----------
        tracker : SFSORT
            The tracker instance to update.
        ffprobe_params : FFprobeParameters
            The video parameters extracted from FFmpeg probe.

        Returns
        -------
        None
        """
        match general_cfg["tracker"]:
            case "sfsort":
                general_cfg["tracker_args_sfsort"].update(
                    {
                        "marginal_timeout": (7 * ffprobe_params.fps // 10),
                        "central_timeout": ffprobe_params.fps,
                        "horizontal_margin": ffprobe_params.width // 10,
                        "vertical_margin": ffprobe_params.height // 10,
                        "frame_width": ffprobe_params.width,
                        "frame_height": ffprobe_params.height,
                    }
                )
                general_cfg.upload()
                # tracker.update_args(general_cfg["tracker_args_sfsort"])

    @staticmethod
    def _create_ffmpeg_processes(
        video_url: str,
        ffprobe_params: FFprobeParameters,
        create_write_process: bool = True,
        task_id: int = 0,
    ) -> tuple[Popen, Popen | None]:
        """
        Creates FFmpeg processes for reading and optionally writing video data.

        Parameters
        ----------
        video_url : str
            The URL or file path of the video to process.
        ffprobe_params : FFprobeParameters
            The video parameters extracted from FFmpeg probe.
        create_write_process : bool, optional
            Flag to indicate whether a write process should be created (default is True).
        task_id : int, optional
            The task identifier used to name the output video file (default is 0).

        Returns
        -------
        tuple[Popen, Popen | None]
            A tuple containing the read process and the write process (if created).
        """
        read_process = (
            ffmpeg.input(
                video_url,
                t=ffprobe_params.duration,
                r=ffprobe_params.fps,
            )
            .filter("showinfo")
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                loglevel="trace",
                r=ffprobe_params.fps,
            )
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        write_process = None
        if create_write_process:
            # Prepare for save video results
            video_dir = os.path.join(os.path.abspath(os.path.curdir), "videos")
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            video_file = os.path.join(video_dir, f"{task_id}.mp4")

            write_process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="bgr24",
                    s="{}x{}".format(ffprobe_params.width, ffprobe_params.height),
                    r=min(general_cfg["framerate"], ffprobe_params.fps),
                )
                .output(
                    video_file,
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    r=min(general_cfg["framerate"], ffprobe_params.fps),
                    loglevel="quiet",
                )
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

        return read_process, write_process

    @staticmethod
    def _get_frame(read_process, ffprobe_params: FFprobeParameters):
        """
        Retrieves a video frame from the FFmpeg read process.

        Parameters
        ----------
        read_process : Popen
            The FFmpeg read process from which to retrieve the frame.
        ffprobe_params : FFprobeParameters
            The video parameters extracted from FFmpeg probe.

        Returns
        -------
        np.ndarray | None
            A NumPy array representing the video frame, or None if no frame is available.
        """
        # Get frame from stdout
        in_bytes = read_process.stdout.read(
            ffprobe_params.width * ffprobe_params.height * 3
        )
        if not in_bytes or sys.getsizeof(in_bytes) < (
            ffprobe_params.width * ffprobe_params.height * 3
        ):
            return None
        # Convert frame from bytes to numpy array
        frame = (
            np.frombuffer(in_bytes, np.uint8)
            .reshape([ffprobe_params.height, ffprobe_params.width, 3])
            .copy()
        )

        return frame

    def _inference_cycle(self, video_url: str, task_id: int, properties: dict) -> dict:
        """
        The main processing cycle of the video stream. It takes frames
        using the ffmpeg reading process, runs them through
        preprocess-inference-postprocess and saves the results,
        as well as records video with annotated frames
        using the ffmpeg recording process.

        Parameters
        ----------
        video_url : str
            A link to the RTSP video stream to be processed.
        task_id : int
            The ID of the video processing task.
        properties: dict
            Additional parameters for processing

        Returns
        -------
        results : dict
            An excerpt with all the saved results
            of an excerpt from an RTSP video stream.
        """
        params = InferenceCycleParameters(
            is_realtime=properties.get("isRealtime", False),
            ffprobe_params=self._ffprobe_read(video_url),
        )

        self._update_tracker(
            tracker=self.trackers[task_id], ffprobe_params=params.ffprobe_params
        )

        read_process, write_process = self._create_ffmpeg_processes(
            video_url=video_url,
            ffprobe_params=params.ffprobe_params,
            create_write_process=True,
            task_id=task_id,
        )

        session = RequestPostData(
            url_frame=f"http://{self.task_params[task_id].host_ip}:{general_cfg['manager_port']}/task/stream/{task_id}",
            url_data=f"http://{self.task_params[task_id].host_ip}:{general_cfg['manager_port']}/task/data/{task_id}",
            logger=self.logger,
        )
        # Start session
        if params.is_realtime:
            session.start()

        timestamp_thread = Thread(
            target=self._get_timestamp, args=(read_process, task_id)
        )
        timestamp_thread.start()

        fps_start_time = time.time()
        fps_counter = 0

        self.logger.debug("frame_id\tframe_timestamp\tprogress")
        while self.task_params[task_id].inference_status == StatusTask.RUNNING:
            frame = self._get_frame(read_process, params.ffprobe_params)
            if frame is None:
                self.logger.warning("End of frames or broken frame!")
                self.task_params[task_id].inference_status = StatusTask.ERROR
                break
            if "corners" in properties and len(properties["corners"]):
                frame = self.draw_ROI(img=frame, corners=properties["corners"])
            # Get frame's timestamp
            current_timestamp = self.timestamps[task_id].get()

            # Setting the frame rate.
            if self.task_params[task_id].frame_processed < params.current_frame:
                self.task_params[task_id].frame_processed += 1
                continue
            self.task_params[task_id].frame_processed += 1
            params.current_frame += params.ffprobe_params.frame_interval

            # Get results
            result = self.run(frame, task_id)
            converted_dets = []
            for det in result:
                bbox = det[0:3]
                track_id = det[4]
                class_id = det[5]
                scores = det[6] if len(det) > 7 else 'unknown'
                converted_dets.append({
                    'track': track_id,
                    'score': scores,
                    'class': class_id,
                    'bbox': bbox
                })
            #Передаем данные в логгер
            if task_id in self.data_loggers:
                self.data_loggers[task_id].process_detections(converted_dets)
            inf_img = self.draw_results(frame, result)
             # Расчет и отображение FPS
            fps_counter += 1
            fps_end_time = time.time()
            fps = fps_counter / (fps_end_time - fps_start_time)
            cv2.putText(
                inf_img,
                f"FPS: {round(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            if fps_end_time - fps_start_time >= 1.0:
                fps_counter = 0
                fps_start_time = time.time()

            if write_process is not None:
                write_process.stdin.write(inf_img.astype(np.uint8).tobytes())
            params.results[f"{current_timestamp}"] = result

            # Sending data to the sending queue
            if params.is_realtime:
                data_frame = {
                    "fps": params.ffprobe_params.fps,
                    "duration": params.ffprobe_params.duration,
                    "timestamp": current_timestamp,
                    "result": result,
                }
                session.put(cv2.cvtColor(inf_img, cv2.COLOR_BGR2RGB), data_frame)

            self.task_params[task_id].progress = (
                self.task_params[task_id].frame_processed
            ) / (params.total_frame)
            self.logger.debug(
                "%s\t\t%s\t%s",
                self.task_params[task_id].frame_processed,
                current_timestamp,
                self.task_params[task_id].progress,
            )
            self.task_params[task_id].ts_last_processed = current_timestamp

        self.task_params[task_id].progress = max(self.task_params[task_id].progress, 1)

        try:
            read_process.stdout.close()
            read_process.stderr.close()
            read_process.wait(timeout=5)
        except TimeoutExpired:
            read_process.kill()

        
        if write_process is not None:
            try:
                write_process.stdin.close()
                write_process.wait(timeout=5)
            except TimeoutExpired:
                write_process.kill()

        timestamp_thread.join()

        session.stop()
        if session.is_alive():
            session.join()
        results = {"results": params.results}

        return results

    def _perform_inference_async(
        self, video_url: str, task_id: int, properties: dict = {}
    ) -> None:
        """
        Starts the main processing cycle, receives the results from it,
        sends them to the video analytics manager and cleans up the data
        about the completed task.

        Parameters
        ----------
        video_url : str
            A link to the RTSP video stream to be processed.
        task_id : int
            The ID of the video processing task.
        properties: dict
            Additional parameters for processing
        """
        match general_cfg["tracker"]:
            case "sfsort":
                self.trackers[task_id] = SFSORT.SFSORT(general_cfg["tracker_args_sfsort"])
            case "botsort":
                from boxmot import BotSort
                self.trackers[task_id] = BotSort(
                    frame_rate=general_cfg["framerate"],
                    **general_cfg["tracker_args_botsort"])
        
        self.timestamps[task_id] = queue.Queue()

        self.data_loggers[task_id] = self.YOLODataLogger(task_id)
        
        # print("lol")
        lol = StatusTask.RUNNING
        # print(task_id)
        # print("lol-")
        self.task_params[task_id] = TaskParameters(host_ip="127.0.0.1")
        self.task_params[task_id].inference_status = StatusTask.RUNNING
        success = True
        # inferencing
        try:
            results = self._inference_cycle(video_url, task_id, properties)
            if self.task_params[task_id].inference_status == StatusTask.RUNNING:
                self.task_params[task_id].inference_status = StatusTask.COMPLETED
        except Exception as e:
            self.logger.error("%s:\n%s", e, traceback.format_exc())
            self.task_params[task_id].inference_status = StatusTask.ERROR
            results = {}
            success = False
        finally:
            self.data_loggers.pop(task_id, None)

        response_content = {
            "task_id": task_id,
            "state": self.task_params[task_id].inference_status,
            "success": success,
            "framesProcessed": self.task_params[task_id].frame_processed,
            "progress": self.task_params[task_id].progress,
            "tsLastFrame": self.task_params[task_id].ts_last_processed,
            "results": results,
        }
        print(response_content["success"])

        try:
            response = requests.post(
                f"http://{self.task_params[task_id].host_ip}:{general_cfg['manager_port']}/task/results/{task_id}",
                json=response_content,
                timeout=30,
            )
        except requests.exceptions.ReadTimeout:
            self.logger.error(f"Timeout for task {task_id}, skipping response.")
            response = None

        if response is not None:
            try:
                self.logger.info("Response from manager after complete task: %s", response.json())
            except Exception as e:
                self.logger.error("Error while logging response: %s", str(e))

        self.task_params.pop(task_id)
        self.trackers.pop(task_id)
        while not self.timestamps[task_id].empty():
            self.timestamps[task_id].get()
        del self.timestamps[task_id]
