"""
This module is a basic detector. It is the basis for other detectors
using neural networks. By itself, it is able to run the basic YOLOv8 model
and process a segment from the provided video stream, returning the results
to the video analytics manager in the form of a json file and an annotated video.
"""

import threading
import cv2
import numpy as np
import onnxruntime as ort

from base import Base


class Detector(Base):
    """
    A base detector designed to be inherited by other detectors
    that use neural networks.

    Parameters
    ----------
    model_path : str
        Filename or serialized ONNX or ORT format model in a byte string.
    """

    CONF_TH = 0.1
    """The confidence threshold for the results of the onnx model."""
    IOU_TH = 0.9
    """The threshold of intersection over union for the results of the onnx model."""

    def __init__(self, model_path: str):
        super().__init__()
        self.session = ort.InferenceSession(
            model_path, 
            providers=["CPUExecutionProvider"] # запуск на CPU
            # providers=["CUDAExecutionProvider"] # запуск на GPU
        )
        """An onnx session to launch the model."""

        self.input_name: str = self.session.get_inputs()[0].name
        """The name of the input metadata."""
        self.output_name: list[str] = [self.session.get_outputs()[0].name]
        """The name of the output metadata."""
        self.lock = threading.Lock()

    @staticmethod
    def letterbox(
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleup=True,
        stride=32,
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """
        Resizes and pads an image to fit a new shape while maintaining aspect ratio.

        Parameters
        ----------
        im : np.ndarray
            The input image to be resized and padded.
        new_shape : tuple[int, int], optional
            The desired output shape (height, width). Default is (640, 640).
        color : tuple[int, int, int], optional
            The color for padding. Default is (114, 114, 114).
        auto : bool, optional
            If True, adjusts padding to be a multiple of stride. Default is True.
        scaleup : bool, optional
            If True, allows scaling up the image. If False, only scales down. Default is True.
        stride : int, optional
            The stride for padding adjustment. Default is 32.

        Returns
        -------
        tuple[np.ndarray, float, tuple[float, float]]
            A tuple containing the resized and padded image,
            the scaling ratio, and the padding values.
        """

        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, r, (dw, dh)

    def pre_process(
        self, input_img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        img = input_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, ratio, dwdh = self.letterbox(img, auto=False)
        img = np.expand_dims(img, axis=0).astype("float32") / 255.0
        img = np.transpose(img, [0, 3, 1, 2])

        return img, ratio, dwdh

    def inference(self, img: np.ndarray) -> np.ndarray:
        with self.lock:
            return self.session.run(self.output_name, {self.input_name: img})[0]

    def post_process(
        self, output: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        outputs = np.transpose(np.squeeze(output))
        classes_scores = outputs[:, 4:]
        max_scores = np.amax(classes_scores, axis=1)
        conf_indices = np.where(max_scores >= self.CONF_TH)[0]

        boxes, classes, scores = [], [], []

        for i in conf_indices:
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            x0 = x - w / 2
            x1 = x + w / 2
            y0 = y - h / 2
            y1 = y + h / 2

            boxes.append([x0, y0, x1, y1])
            classes.append(class_id)
            scores.append(max_score)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.CONF_TH, self.IOU_TH)

        if isinstance(indices, tuple):
            return None, None, None

        boxes = np.array(boxes)[indices]
        classes = np.array(classes)[indices]
        scores = np.array(scores)[indices]

        return boxes, classes, scores
