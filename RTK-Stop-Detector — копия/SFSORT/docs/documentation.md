Quickstart
----------
There are two files available for a quickstart with this tracker:

* through ONNX model in form of YOLOv7 [here].

[here]: https://github.com/Applied-Deep-Learning-Lab/SFSORT/blob/main/yolov7_onnxruntime_sfsort.py

* for any model in Ultralytics framework [here].

[here]: https://github.com/Applied-Deep-Learning-Lab/SFSORT/blob/main/ultralytics_sfsort.py

Usage
-----
Simply import the module in your Python code:

    from SFSORT import SFSORT

Initialize the tracker:

    # Organize tracker arguments into standard format
    tracker_arguments = {"dynamic_tuning": True, "cth": 0.7,
                        "high_th": 0.7, "high_th_m": 0.1,
                        "match_th_first": 0.6, "match_th_first_m": 0.05,
                        "match_th_second": 0.4, "low_th": 0.2,
                        "new_track_th": 0.5, "new_track_th_m": 0.1,
                        "marginal_timeout": (7 * fps // 10),
                        "central_timeout": fps,
                        "horizontal_margin": width // 10,
                        "vertical_margin": height // 10,
                        "frame_width": width,
                        "frame_height": height,}
    # Instantiate a tracker
    tracker = SFSORT(tracker_arguments)

The parameters for the tracker are as follows:

* `cth`                 Threshold that determines the number of set members whose detection score exceeds this value
* `high_th`             Minimum score for high-score detections
* **`low_th`**          Minimum score for intermediate-score detections
* `match_th_first`      Maximum allowable cost in the first association module
* **`match_th_second`** Maximum allowable cost in the second association module
* `new_track_th`        Minimum score for detections identified as new tracks
* `horizontal_margin`   Margin to determine the horizontal boundaries of central areas
* `vertical_margin`     Margin to determine the vertical boundaries of central areas
* `central_timeout`     Time-out for tracks lost at central areas
* `marginal_timeout`    Time-out for tracks lost at marginal areas

Parameters in bold are the most important and should be modified based on the detector being used.
For example, YOLOv10 might give low scores to small objects while Unet doesn't and thus `low_th`
should be set lower for the former.
Parameters with `_m` suffix are margins that are being used with `dynamic_tuning` key. Basically if
your detector is being used in an environment with a lot of occlusions/disocclusions/overlap it might
be better to leave this parameter turned on. It should dynamically change threshold values when the
detector is confused in crowded environments or rapidly moving objects. In other cases you can turn it off.

After that simply pass three arrays, containing bounding boxes, confidence and classes values.

    # Update the tracker with the latest detections
    tracks = tracker.update(
        prediction_results.xyxy,
        prediction_results.conf,
        prediction_results.cls)

As a response you will get an array with track IDs and your input values. Be aware that all values are in `np.float32`.

    # Extract tracking data from the tracker
    bbox_list      = tracks[:, 0]
    track_id_list  = tracks[:, 1]
    cls_id_list    = tracks[:, 2]
    scores_list    = tracks[:, 3]
