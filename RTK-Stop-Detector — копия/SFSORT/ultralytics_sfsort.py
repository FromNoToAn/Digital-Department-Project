import random
import time
import cv2
from ultralytics import YOLO
from SFSORT import SFSORT

# Model loading
session = YOLO('yolov8m.pt', task='detect')
# All classes
names = session.names

# Load the video file
cap = cv2.VideoCapture('Sample.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the MP4 codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

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
# Define a color list for track visualization
colors = {}

# Process each frame of the video
while cap.isOpened():
   ret, frame = cap.read()
   if not ret:
         break

   start_time = time.time()

   # Detect people in the frame
   prediction = session.predict(frame, imgsz=640, conf=0.1, iou=0.45,
                                half=False, max_det=100, verbose=False)
   # Exclude additional information from the predictions
   prediction_results = prediction[0].boxes.cpu().numpy()

   start_tracker_time = time.time()
   # Update the tracker with the latest detections
   tracks = tracker.update(
       prediction_results.xyxy,
       prediction_results.conf,
       prediction_results.cls)
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
      x0, y0, x1, y1 = map(int, bbox)
      # Assign names to detected classes
      name = names[cls_id]
      name += ' '+str(score)

      # Draw the bounding boxes on the frame
      annotated_frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
      cv2.putText(annotated_frame, name+' '+str(track_id),
                  (x0, y0-5),
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
