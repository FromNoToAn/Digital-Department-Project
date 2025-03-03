import random
import time
import cv2
import numpy as np
import onnxruntime as ort
from SFSORT import SFSORT

# Model loading
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('cartype_v2.onnx', providers=providers)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    """
    Resize and pad an image to a new shape while meeting stride-multiple constraints.
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
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

# All classes
names = ['bike', 'bus', 'car', 'construction equipment', 'emergency', 'motorbike', 'personal mobility', 'quad bike', 'truck']

# Load the video file
cap = cv2.VideoCapture('test.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the MP4 codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Organize tracker arguments into standard format
tracker_arguments = {"dynamic_tuning": True, "cth": 0.5,
                     "high_th": 0.7, "high_th_m": 0.1,
                     "match_th_first": 0.5, "match_th_first_m": 0.05,
                     "match_th_second": 0.1, "low_th": 0.2,
                     "new_track_th": 0.3, "new_track_th_m": 0.1,
                     "marginal_timeout": (7 * fps // 10),
                     "central_timeout": fps,
                     "horizontal_margin": width // 10,
                     "vertical_margin": height // 10,
                     "frame_width": 640,
                     "frame_height": 640}
# Instantiate a tracker
tracker = SFSORT(tracker_arguments)
# Define a color list for track visualization
colors = {}

# Process each frame of the video
while cap.isOpened():
   ret, frame = cap.read()
   if not ret:
         break

   start_time = time.time()

   # Preprocessing steps
   img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   image = img.copy()
   image, ratio, dwdh = letterbox(image, auto=False)
   image = image.transpose((2, 0, 1))
   image = np.expand_dims(image, 0)
   image = np.ascontiguousarray(image)
   im = image.astype(np.float32)
   im /= 255
   im.shape
   # Model layers names required by onnxruntime
   outname = [i.name for i in session.get_outputs()]
   inname = [i.name for i in session.get_inputs()]
   inp = {inname[0]:im}

   # ONNX inference
   outputs = session.run(outname, inp)[0]    # batch_id,x0,y0,x1,y1,cls_id,score

   start_tracker_time = time.time()
   # Update the tracker with the latest detections
   tracks = tracker.update(outputs[:, 1:5], outputs[:, 6], outputs[:, 5])
   end_tracker_time = time.time() - start_tracker_time

   # Skip additional analysis if the tracker is not currently tracking anyone
   if len(tracks) == 0:
      out.write(frame)
      continue
   
   # Extract tracking data from the tracker
   bbox_list      = tracks[:, 0]
   track_id_list  = tracks[:, 1]
   cls_id_list    = tracks[:, 2]
   scores_list    = tracks[:, 3]

   # Visualize tracks
   start_postprocess_time = time.time()
   for _, (track_id, bbox, cls_id, score) in enumerate(
       zip(track_id_list, bbox_list, cls_id_list, scores_list)):

      # Define a new color for newly detected tracks
      if track_id not in colors:
         colors[track_id] = (random.randrange(255),
                             random.randrange(255),
                             random.randrange(255))

      color = colors[track_id]

      # Extract the bounding box coordinates
      x0, y0, x1, y1 = bbox
      # Scale the box back to original frame size
      box = np.array([x0,y0,x1,y1])
      box -= np.array(dwdh*2)
      box /= ratio
      box = box.round().astype(np.int32).tolist()

      # Assign names to detected classes
      name = names[int(cls_id)]
      name += ' '+str(score)

      # Draw the bounding boxes on the frame
      annotated_frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
      cv2.putText(annotated_frame, name+' '+str(track_id),
                  (box[0], box[1]-5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2) 

   # Measure and visualize timers
   end_postprocess_time = time.time() - start_postprocess_time
   elapsed_time = time.time() - start_time
   fps = 1 / elapsed_time
   cv2.putText(annotated_frame, f'{fps:.1f} FPS (overall)',
      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,)
   cv2.putText(annotated_frame, f'{end_tracker_time*1000:.2f} ms (SFSORT)',
      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,)
   cv2.putText(annotated_frame, f'{end_postprocess_time*1000:.2f} ms (post-process)',
      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,)

   # If key is pressed, close the window
   key = cv2.waitKey(1)
   if key == 27: # ESC
      break
   
   # cv2.imshow("test", annotated_frame)

   # Write the frame to the output video file
   out.write(annotated_frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
