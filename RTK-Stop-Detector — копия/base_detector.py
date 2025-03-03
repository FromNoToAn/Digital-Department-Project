import numpy as np
from SFSORT_tracker.SFSORT import SFSORT
from detector import Detector


class BaseDetector(Detector):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.tracker_arguments = {
                "dynamic_tuning": True,
                "cth": 0.7,
                "high_th": 0.7,
                "high_th_m": 0.1,
                "match_th_first": 0.6,
                "match_th_first_m": 0.05,
                "match_th_second": 0.4,
                "low_th": 0.2,
                "new_track_th": 0.5,
                "new_track_th_m": 0.1,
                "marginal_timeout": 14,
                "central_timeout": 20,
                "horizontal_margin": 128,
                "vertical_margin": 72,
                "frame_width": 1280,
                "frame_height": 720
            }
        self.tracker = SFSORT(self.tracker_arguments)
        
        
    def run(self, img: np.ndarray) -> list:
        dets = []
        tracks = []
        pre_img, ratio, dwdh = self.pre_process(img)
        outputs = self.inference(pre_img)
        
        if outputs is not None:
            boxes, classes, scores = self.post_process(outputs)

            if boxes is not None:
                boxes -= np.array(dwdh * 2)
                boxes /= ratio
                boxes = boxes.round().astype(np.int32)

                tracks = self.tracker.update(boxes, scores, classes)

                if len(tracks):
                    for track in tracks:
                        x0, y0, x1, y1 = map(int, track[0])
                        dets.append(
                            [
                                x0,
                                y0,
                                x1,
                                y1,
                                int(track[1]),
                                int(track[2]),
                                float(track[3]),
                            ]
                        )

        return tracks



