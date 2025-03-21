from pydantic import BaseModel, model_validator


class FFprobeParameters(BaseModel):
    """
    Represents video metadata parameters extracted using FFprobe.

    Attributes
    ----------
    width : int
        The width of the video in pixels.
    height : int
        The height of the video in pixels.
    fps : float
        The frames per second (FPS) of the video.
    frame_interval : float
        The interval between frames, calculated as REAL_FPS / DESIRED_FPS.
    duration : float
        The total duration of the video in seconds.
    """
    width: int
    height: int
    fps: float
    frame_interval: float
    duration: float


class InferenceCycleParameters(BaseModel):
    """
    Represents the parameters for an inference cycle, including video metadata,
    real-time settings, and frame counts.

    Attributes
    ----------
    results : dict[int, list]
        A dictionary in which the keys are the timestamps of the frame,
        and the values are lists of the results of processing this frame by the detector.
        Defaults to an empty dictionary.
    is_realtime : bool
        Specifies if the inference cycle operates in real-time mode. Defaults to `False`.
    current_frame : int
        The index of the current frame being processed. Defaults to 0.
    ffprobe_params : FFprobeParameters
        An instance of `FFprobeParameters` containing video metadata.
    total_frame : float
        The total number of frames to process. Defaults to infinity.

    Methods
    -------
    compute_total_frame_and_duration()
        Adjusts `total_frame` and `duration` based on whether the cycle is in real-time mode.
    """
    results: dict[int, list] = {}
    is_realtime: bool = False
    current_frame: int = 0
    ffprobe_params: FFprobeParameters
    total_frame: float = float("inf")

    @model_validator(mode="after")
    def compute_total_frame_and_duration(self):
        if self.is_realtime:
            self.total_frame = float("inf")
            self.ffprobe_params.duration = 0
        else:
            duration = self.ffprobe_params.duration
            fps = self.ffprobe_params.fps
            self.total_frame = duration * fps if duration != 0 else float("inf")
        return self
