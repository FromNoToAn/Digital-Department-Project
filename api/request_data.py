import asyncio
from io import BytesIO
from queue import Queue
from threading import Event, Thread
from logging import Logger
from typing import Any, Callable, Iterable, Mapping

from PIL import Image
import numpy as np
import aiohttp

from config import general_cfg


class BackgroundThread(Thread):
    """
    A base class for creating background threads with controlled startup, handling,
    and shutdown operations.

    This class extends `Thread` and provides methods for starting, stopping, and
    performing tasks in a loop. Subclasses should implement the `startup`,
    `handle`, and `shutdown` methods.
    """


    def __init__(
        self,
        group: None = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        args: Iterable[Any] = ...,
        kwargs: Mapping[str, Any] | None = None,
        *,
        daemon: bool | None = None
    ) -> None:
        """
        Initializes the BackgroundThread.

        Parameters
        ----------
        group : None, optional
            Reserved for future extension, always set to None.
        target : Callable[..., object], optional
            The callable object to invoke in the thread.
        name : str, optional
            The thread name.
        args : Iterable[Any], optional
            The arguments to pass to the target callable.
        kwargs : Mapping[str, Any], optional
            A dictionary of keyword arguments to pass to the target callable.
        daemon : bool, optional
            If True, the thread runs as a daemon.
        """
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self._stop_event = Event()

    def stop(self) -> None:
        """
        Signals the thread to stop by setting the stop event.
        """
        self._stop_event.set()

    @property
    def _stopped(self) -> bool:
        """
        Checks if the thread has been signaled to stop.

        Returns
        -------
        bool
            True if the thread has been signaled to stop, False otherwise.
        """
        return self._stop_event.is_set()

    def startup(self) -> None:
        """
        Initializes resources or performs preliminary setup for the thread.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def handle(self) -> None:
        """
        Contains the main logic to execute repeatedly in the thread.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def shutdown(self) -> None:
        """
        Cleans up resources or performs final operations after the thread stops.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def run(self) -> None:
        """
        Starts the thread's workflow, running `startup`, then repeatedly calling
        `handle` until stopped, and finally calling `shutdown`.
        """
        self.startup()

        while not self._stopped:
            self.handle()

        self.shutdown()

class RequestPostData(BackgroundThread):
    """
    A specialized background thread class for periodically sending image data and
    related metadata to specified endpoints via HTTP POST requests.
    """
    def __init__(
        self,
        group: None = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        args: Iterable[Any] = ...,  # type: ignore
        kwargs: Mapping[str, Any] | None = None,
        *,
        daemon: bool | None = None,
        logger: Logger,
        url_frame:str,
        url_data:str
    ) -> None:
        """
        Initializes the RequestPostData thread with specified endpoints and a logger.

        Parameters
        ----------
        group : None, optional
            Reserved for future extension, always set to None.
        target : Callable[..., object], optional
            The callable object to invoke in the thread.
        name : str, optional
            The thread name.
        args : Iterable[Any], optional
            The arguments to pass to the target callable.
        kwargs : Mapping[str, Any], optional
            A dictionary of keyword arguments to pass to the target callable.
        daemon : bool, optional
            If True, the thread runs as a daemon.
        logger : Logger
            Logger for logging thread operations.
        url_frame : str
            The endpoint for sending image data.
        url_data : str
            The endpoint for sending metadata.
        """
        super().__init__(
            group, target, name, args, kwargs, daemon=daemon
        )

        self.url_frame = url_frame
        self.url_data = url_data
        self.logger = logger
        self.queue = Queue()
        self.is_run = True

    def handle(self) -> None:
        """
        Executes the thread's main task, asynchronously sending data and images.
        """
        asyncio.run(self.__run_post())

    async def __run_post(self):
        """
        Sends image data and related metadata asynchronously to predefined endpoints.

        This method retrieves items from the queue, processes the image data,
        and sends two separate POST requests: one for the image and one for the metadata.

        Raises
        ------
        Exception
            Stops the thread if an error occurs during the POST request.
        """
        async with aiohttp.ClientSession() as session:
            while not self.queue.empty():
                inf_img, data = self.queue.get()
                if (self._stopped):
                    continue
                # Convert an image from BGR to RGB
                frame = Image.fromarray(inf_img)
                buffer = BytesIO()
                frame.save(buffer, format="JPEG", quality=general_cfg["quality_post_image"])
                try:
                    await session.post(
                        url = self.url_frame,
                        data = buffer.getvalue(),
                        timeout=2
                    )
                    await session.post(
                        url = self.url_data,
                        json = data,
                        timeout=2
                    )
                except Exception:
                    self.stop() # ?
            self._stop_event.wait(1)

    def startup(self) -> None:
        """
        Logs the startup of the RequestPostData thread.
        """
        self.logger.info("RequestData started")

    def shutdown(self) -> None:
        """
        Logs the shutdown of the RequestPostData thread.
        """
        self.logger.info("RequestData stopped")

    def put(self, frame:np.ndarray, data: dict):
        """
        Adds an image and its associated metadata to the queue for processing.

        Parameters
        ----------
        frame : np.ndarray
            The processed image in NumPy array format.
        data : dict
            Metadata or additional information associated with the frame.
        """
        self.queue.put((frame, data))