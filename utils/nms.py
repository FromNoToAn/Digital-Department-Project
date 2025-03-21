import numpy as np


def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4).
                               Each box is represented as [x1, y1, x2, y2].
        scores (numpy.ndarray): Array of scores with shape (N,).
        iou_threshold (float): Intersection-over-Union (IoU) threshold for suppression.

    Returns:
        numpy.ndarray: Indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # Compute areas of all boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by scores in descending order
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        # Select the box with the highest score and add it to the keep list
        i = order[0]
        keep.append(i)

        # Compute IoU of the selected box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection

        iou = intersection / union

        # Keep boxes with IoU less than the threshold
        remaining_indices = np.where(iou <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    return np.array(keep, dtype=int)