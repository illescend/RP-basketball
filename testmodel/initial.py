import os
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0" # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100" # Try to prioritize FFMPEG
import cv2
from ultralytics import YOLO

# 1. Load Models
# For ball detection: yolov8n.pt already includes 'sports ball' (class 32)
# You can specify classes=[32] to only look for sports balls.
try:
    ball_model = YOLO('Yolo-Weights/model2.pt')
    print("Ball detection custom model loaded successfully.")
    print(f"Classes: {ball_model.names}")
except Exception as e:
    print(f"Error loading ball_model: {e}")
    exit()

# For pose estimation
try:
    pose_model = YOLO('yolov8n-pose.pt') # Standard pose model
    print("Pose estimation model (yolov8n-pose.pt) loaded successfully.")

    RIGHT_WRIST_KP_INDEX = 10
    LEFT_WRIST_KP_INDEX = 9
    # Choose which wrist you are primarily interested in, e.g., the shooting hand
    TARGET_WRIST_KP_INDEX = RIGHT_WRIST_KP_INDEX  # Example: focusing on right wrist
except Exception as e:
    print(f"Error loading pose_model: {e}")
    exit()


# 2. Video Input
# video_path = "curry_segment_slowed2.mp4" # Make sure this path is correct
availableBackends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]
print(availableBackends)

video_path = "footage/freethrow_3v2.mp4"  # Make sure this path is correct
cap = cv2.VideoCapture(video_path,cv2.CAP_MSMF)
if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit()

# 3. Video Output (get properties from input video)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# Ensure fps is a reasonable value, e.g., not 0
if fps == 0:
    print("Warning: Video FPS is 0, defaulting to 25 FPS for output.")
    fps = 30

output_path = "outputs/freethrow_1_detect.avi"
# Using 'mp4v' codec for .mp4 files, common and widely supported
availableBackends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_path}")
    cap.release()
    exit()

print(f"Processing video: {video_path}")
print(f"Outputting to: {output_path}")
# 4. Process Frames
frame_count = 0
max_frames = float('inf') # Set to a number (e.g., 100) to process only a few frames for testing

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 == 0: # Print progress every 10 frames
        print(f"Processing frame {frame_count}...")

    # --- Object Detection (Ball) ---
    # verbose=False to reduce console spam from YOLO
    # We are interested in class 32 ('sports ball') for yolov8n.pt
    # conf=0.3 means confidence threshold of 30%
    ball_results_list = ball_model(frame, conf=0.6, verbose=False,classes=[0,1,3,4])

    # .plot() method draws the detections on a copy of the frame and returns it.
    # ball_results_list is a list of Results objects (usually one if one frame is passed)
    if ball_results_list and len(ball_results_list) > 0:
        annotated_frame_with_balls = ball_results_list[0].plot() # This returns a new frame with ball detections
    else:
        annotated_frame_with_balls = frame.copy() # No detections, use original

    # --- Pose Estimation ---
    # Now run pose estimation on the frame that *already has ball annotations*
    # (This is a simplification; for advanced coordinate extraction, you'd run both on the original frame
    # and combine coordinates, but for display, this is fine)
    pose_results_list = pose_model(annotated_frame_with_balls, verbose=False)

    wrist_x, wrist_y, wrist_conf = -1, -1, -1.0  # Initialize
    if pose_results_list and len(pose_results_list) > 0:
        res_pose = pose_results_list[0]  # Results for the first (and only) frame processed

        # Draw all pose keypoints and skeletons using YOLO's plot method
        # but we'll draw our target wrist separately for emphasis.
        # To avoid double drawing, you can pass `kpt_line=False` or similar to res_pose.plot()
        # if you only want boxes, or draw keypoints manually.
        # For simplicity, let's let it draw all, then we add our specific info.
        display_frame = res_pose.plot(img=annotated_frame_with_balls, labels=True, conf=True,
                                      boxes=True)  # Plot pose on display_frame

        # Extract keypoints for the first detected person (person_idx = 0)
        if res_pose.keypoints and res_pose.keypoints.xy is not None and len(res_pose.keypoints.xy) > 0:
            # keypoints.xy is a tensor of shape [num_persons, num_keypoints, 2] (x,y coordinates)
            # keypoints.conf is a tensor of shape [num_persons, num_keypoints] (confidence scores)

            first_person_kps_xy = res_pose.keypoints.xy[0].cpu().numpy()  # For the first person
            first_person_kps_conf = None
            if res_pose.keypoints.conf is not None:  # Confidence might be None if no kpts found
                first_person_kps_conf = res_pose.keypoints.conf[0].cpu().numpy()

            # Check if the target wrist keypoint index is valid
            if len(first_person_kps_xy) > TARGET_WRIST_KP_INDEX:
                wrist_x, wrist_y = first_person_kps_xy[TARGET_WRIST_KP_INDEX].astype(int)

                if first_person_kps_conf is not None and len(first_person_kps_conf) > TARGET_WRIST_KP_INDEX:
                    wrist_conf = first_person_kps_conf[TARGET_WRIST_KP_INDEX]

                # Draw a circle and text for the target wrist if detected
                if wrist_x > 0 and wrist_y > 0:  # Basic check if keypoint is valid (not 0,0)
                    cv2.circle(display_frame, (wrist_x, wrist_y), 8, (255, 0, 255), -1)  # Magenta circle
                    wrist_text = f"Wrist ({wrist_conf:.2f})"
                    cv2.putText(display_frame, wrist_text, (wrist_x + 10, wrist_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # --- Display and Save ---
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Combined Detections", display_frame)
    out.write(display_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user.")
        break

# 5. Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished processing. Output saved to {output_path}")