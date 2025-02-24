import os
import shutil
from logging import Logger
from typing import Any, Callable, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from utils.dataclasses import StatusTask, TaskParameters


def create_router(
    task_params: dict[int, TaskParameters], func: Callable[..., Any], logger: Logger
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/inference/{task_id}")
    async def initiate_inference(
        task_id: int,
        background_tasks: BackgroundTasks,
        request: Request,
        file: Optional[UploadFile] = File(None),
    ) -> JSONResponse:
        """
        Creates a task based on a request from the video analytics manager
        and runs it in the background.

        Parameters
        ----------
        task_id : int
            The ID of the video processing task.
        background_tasks : :obj:`BackgroundTasks`
            A set of background tasks that will process the video
            after sending a response to the client.
        request : :obj:`Request`
            A request from the video analytics manager that contains data
            for video processing (a link to the video stream
            and various parameters to save to class parameters).
        file : Optional[UploadFile], optional
            An optional file upload containing the video to be processed.
            If provided, the file is saved and used for processing. Defaults to None.

        Returns
        -------
        response : :obj:`JSONResponse`
            The response is in the form of JSON, which notifies
            the video analytics manager about the successful/unsuccessful
            start of the processing task.
        """

        def get_properties(data: dict) -> dict[str]:
            """
            Extracts and processes properties from the provided data.

            Parameters
            ----------
            data : dict
                A dictionary containing properties data from the request.

            Returns
            -------
            properties : dict[str, Any]
                A dictionary containing processed properties, including:
                - `isRealtime` (bool): Indicates if the task is running in real-time.
                - `corners` (list[int]): A list of integers representing corner coordinates.
            """
            properties = {
                "isRealtime": data.get("isRealtime", False),
                "corners": [],
            }
            if isinstance(properties["isRealtime"], str):
                properties["isRealtime"] = (
                    True if properties["isRealtime"].lower() == "true" else False
                )
            check_corners = ["cornerUp", "cornerLeft", "cornerBottom", "cornerRight"]
            # Check corner
            for check in check_corners:
                try:
                    properties["corners"].append(int(data[check]))
                except (KeyError, ValueError):
                    break
            return properties

        if file:
            # Input and processing properties
            data = await request.form()
            properties = get_properties(data=data)

            # Prepare for save video results
            video_dir = os.path.join(os.path.abspath(os.path.curdir), "uploaded_videos")
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            video_url = os.path.join(video_dir, f"{file.filename}")
            with open(video_url, "wb+") as buffer:
                shutil.copyfileobj(file.file, buffer)
        else:
            # Input and processing properties
            data: dict = await request.json()
            properties = get_properties(data=data.get("properties", {}))
            video_url = data["cameraUrls"][0]["video"]
        if task_id in task_params:
            if not task_params[task_id].inference_status >= 2:
                logger.warning(
                    '%s:%s - "%s %s" Task %s already exists.',
                    request.client.host,  # type: ignore
                    request.client.port,  # type: ignore
                    request.method,
                    request.url,
                    task_id,
                )

                return JSONResponse(
                    status_code=405,
                    content={
                        "task_id": task_id,
                        "success": False,
                        "state": task_params[task_id].inference_status,
                    },
                )

        task_params[task_id] = TaskParameters(host_ip=request.client.host)  # type: ignore

        background_tasks.add_task(func, video_url, task_id, properties)

        logger.info(
            '%s:%s - "%s %s" Task %s created.',
            request.client.host,  # type: ignore
            request.client.port,  # type: ignore
            request.method,
            request.url,
            task_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "task_id": task_id,
                "state": task_params[task_id].inference_status,
                "success": True,
                "framesProcessed": task_params[task_id].frame_processed,
                "progress": task_params[task_id].progress,
                "tsLastFrame": task_params[task_id].ts_last_processed,
            },
        )

    @router.get("/api/inference/{task_id}")
    async def get_inference_status(task_id: int, request: Request) -> JSONResponse:
        """
        Collects and sends information about the current status of the task.

        Parameters
        ----------
        task_id : int
            The ID of the video processing task.
        request : :obj:`Request`
            A request from the video analytics manager
            containing data for sending information back.

        Returns
        -------
        response : :obj:`JSONResponse`
            The response is in the form of json, which transmits
            the current status of the task (processing status,
            ID of the last processed frame, video processing progress,
            timestamp of the last processed frame).
            If the task with the transmitted ID does not exist,
            it returns a message stating that there is no such task.
        """
        if task_id not in task_params:
            logger.warning(
                '%s:%s - "%s %s" Task %s not found.',
                request.client.host,  # type: ignore
                request.client.port,  # type: ignore
                request.method,
                request.url,
                task_id,
            )

            raise HTTPException(status_code=404, detail="Task not found")

        logger.info(
            '%s:%s - "%s %s" Info about task %s sended.',
            request.client.host,  # type: ignore
            request.client.port,  # type: ignore
            request.method,
            request.url,
            task_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "task_id": task_id,
                "state": task_params[task_id].inference_status,
                "success": True,
                "framesProcessed": task_params[task_id].frame_processed,
                "progress": task_params[task_id].progress,
                "tsLastFrame": task_params[task_id].ts_last_processed,
            },
        )

    @router.delete("/api/inference/{task_id}")
    async def stop_inference(task_id: int, request: Request) -> JSONResponse:
        """
        Sets an event to stop the specified task.

        Parameters
        ----------
        task_id : int
            The ID of the video processing task.
        request : :obj:`Request`
            A request from the video analytics manager
            containing data for stopping the task.

        Returns
        -------
        response : :obj:`JSONResponse`
            The response is in the form of json, which transmits
            the current status of the task with the status of the task stop
            (processing status, ID of the last processed frame,
            video processing progress, timestamp of the last processed frame).
            If the task with the passed ID does not exist,
            it returns a message stating that such a task does not exist.
        """
        if task_id not in task_params:
            logger.warning(
                '%s:%s - "%s %s" Task %s not found.',
                request.client.host,  # type: ignore
                request.client.port,  # type: ignore
                request.method,
                request.url,
                task_id,
            )

            raise HTTPException(status_code=404, detail="Task not found")

        task_params[task_id].inference_status = StatusTask.STOPPED

        logger.info(
            '%s:%s - "%s %s" Task %s stopped.',
            request.client.host,  # type: ignore
            request.client.port,  # type: ignore
            request.method,
            request.url,
            task_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "task_id": task_id,
                "state": task_params[task_id].inference_status,
                "success": True,
                "framesProcessed": task_params[task_id].frame_processed,
                "progress": task_params[task_id].progress,
                "tsLastFrame": task_params[task_id].ts_last_processed,
            },
        )

    @router.get("/api/inference/video/{task_id}")
    async def get_video_result(task_id: int) -> FileResponse:
        """
        Sends a file with the recorded annotated video.

        Parameters
        ----------
        task_id : int
            The ID of the video processed task.

        Returns
        -------
        response : :obj:`FileResponse`
            The response is in the form of a video file with the results
            of processing.
        """
        video_dir = os.path.join(os.path.abspath(os.path.curdir), "videos")
        video_name = f"{task_id}.mp4"
        video_file = os.path.join(video_dir, video_name)
        return FileResponse(video_file)

    return router
