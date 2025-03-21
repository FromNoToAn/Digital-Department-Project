from dataclasses import dataclass
from enum import IntEnum


class StatusTask(IntEnum):
    """Represents the various statuses a task can have during its lifecycle."""
    INIT = 0
    """The task is ready to be completed"""
    RUNNING = 1
    """The task is running"""
    STOPPED = 2
    """The task was interrupted"""
    COMPLETED = 3
    """The task did completed correctrly"""
    ERROR = 4
    """The task did not finish correctly"""
    UNKNOWN = 5
    """reserved"""

@dataclass
class TaskParameters:
    """A class for all parameters of the processing task."""

    host_ip: str
    """The address of the device from which the request came, for the running task."""
    inference_status: StatusTask = StatusTask.INIT
    """The status of the running task."""
    frame_processed: int = 0
    """The ID of processed frames for the running task."""
    progress: float = 0
    """The video processing progress of the running task."""
    ts_last_processed: float = 0
    """The timestamp of the last processed frame for the running task."""
