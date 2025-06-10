import math
import os

import numpy as np

os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0" # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100" # Try to prioritize FFMPEG
import cv2
from ultralytics import YOLO
import utils

# 1. Load Models
# For ball detection: yolov8n.pt already includes 'sports ball' (class 32)
# You can specify classes=[32] to only look for sports balls.

# --- Constants from utils (or define them here if not importing) ---
# Make sure these match the values in your utils.py or define them explicitly
SHOOTING_WRIST_KP_INDEX = utils.SHOOTING_WRIST_KP_INDEX
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
WRIST_BALL_PROXIMITY_MARGIN = utils.WRIST_BALL_PROXIMITY_MARGIN
CONFIRMATION_FRAMES_FOR_RELEASE = utils.CONFIRMATION_FRAMES_FOR_RELEASE

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
except Exception as e:
    print(f"Error loading pose_model: {e}")
    exit()


# 2. Video Input
# video_path = "curry_segment_slowed2.mp4" # Make sure this path is correct
availableBackends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]
print(availableBackends)

video_path = "footage/freethrow_arc3.mp4"  # Make sure this path is correct
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
    fps = 240

output_path = "outputs/freethrow3_speed_arc_5frames.avi"
# Using 'mp4v' codec for .mp4 files, common and widely supported
availableBackends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_path}")
    cap.release()
    exit()

print(f"Processing video: {video_path}")
print(f"Outputting to: {output_path}")


#### SETUP
TARGET_BALL_CLASS_NAME = "ball"
id_of_ball = 0

# --- Release Detection State Variables ---
release_detected_in_shot = False
release_frame_info = None # To store {"frame_no": X, "ball_bbox": Y, "wrist_pos": Z}
potential_release_buffer = [] # Stores (frame_no, ball_bbox, wrist_pos) for frames where wrist is OUTSIDE ball
trajectory_points = [] # <--- New: List to store ball center points post-release (pixels)

# Variables for a scaled trajectory
origin_x_pxl, origin_y_pxl = 0,0
ball_radius_pxl = 1.0
trajectory_points_scaled = []

FRAMES_FOR_NAIVE_ESTIMATION = 5 # Number of frames *after* release to consider for displacement
BASKETBALL_REAL_RADIUS_M = 0.12 # Meters

estimated_speed_mps = 0.0
estimated_angle_deg = 0.0

# 4. Process Frames
frame_count = 0
max_frames = float('inf') # Set to a number (e.g., 100) to process only a few frames for testing

while cap.isOpened() and frame_count < max_frames:
    ret, frame_orig = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 30 == 0: print(f"Processing frame {frame_count}...")

    current_frame_annotated = frame_orig.copy()

    # --- Object Detection (Ball) ---
    ball_results_list = ball_model(frame_orig, conf=0.6, verbose=False, classes=[id_of_ball])
    ball_bbox = utils.get_ball_bbox(ball_results_list, id_of_ball)

    # --- Pose Estimation ---
    pose_results_list = pose_model(frame_orig, conf=0.35, verbose=False)

    # --- Identify Likely Shooter's Wrist ---
    shooter_wrist_pos, shooter_person_idx = None, -1
    if ball_bbox is not None:
        shooter_wrist_pos, shooter_person_idx = utils.get_likely_shooter_wrist_and_person_idx(
            pose_results_list,
            ball_bbox,
            SHOOTING_WRIST_KP_INDEX,
            MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
        )

    # --- Release Detection Logic ---
    if not release_detected_in_shot and ball_bbox is not None and shooter_wrist_pos is not None:
        # ... (Your existing release detection logic using potential_release_buffer) ...
        wrist_in_ball_bbox = utils.is_point_inside_bbox(shooter_wrist_pos, ball_bbox, margin=WRIST_BALL_PROXIMITY_MARGIN)
        if not wrist_in_ball_bbox:
            dist_to_bbox_edge = utils.calculate_distance_point_to_bbox_edge(shooter_wrist_pos, ball_bbox)
            potential_release_buffer.append({
                "frame_no": frame_count, "ball_bbox": ball_bbox,
                "wrist_pos": shooter_wrist_pos, "dist_to_edge": dist_to_bbox_edge
            })
            if len(potential_release_buffer) > CONFIRMATION_FRAMES_FOR_RELEASE:
                 potential_release_buffer.pop(0)
            if len(potential_release_buffer) == CONFIRMATION_FRAMES_FOR_RELEASE:
                release_frame_info = potential_release_buffer[0]
                release_detected_in_shot = True

                trajectory_points = [] # <--- Reset trajectory on new release detection
                # Add the ball position from the release frame itself as the start
                trajectory_points_scaled = []

                rel_ball_bbox = release_frame_info['ball_bbox']
                origin_x_pxl = (rel_ball_bbox[0] + rel_ball_bbox[2]) // 2
                origin_y_pxl = (rel_ball_bbox[1] + rel_ball_bbox[3]) // 2

                print(f"RELEASE DETECTED at frame: {release_frame_info['frame_no']}")
                #... (rest of print statements)

                # Calculate apparent radius at release for scaling
                # Using average of width/2 and height/2 for radius
                release_ball_width_px = rel_ball_bbox[2] - rel_ball_bbox[0]
                release_ball_height_px = rel_ball_bbox[3] - rel_ball_bbox[1]
                ball_radius_pxl = (release_ball_width_px + release_ball_height_px) / 4.0  # Average radius
                if ball_radius_pxl < 1: ball_radius_pxl = 1.0  # Avoid division by zero

                trajectory_points.append((origin_x_pxl, origin_y_pxl))
                trajectory_points_scaled.append({'x': 0.0, 'y': 0.0, 'frame': release_frame_info['frame_no']})

                print(f"RELEASE DETECTED at frame: {release_frame_info['frame_no']}")
                print(f"  Release Ball Center (px): ({origin_x_pxl}, {origin_y_pxl})")
                print(f"  Release Ball Apparent Radius (px): {ball_radius_pxl:.2f}")
        else:
            potential_release_buffer = [] # Clear buffer if wrist is back inside

    # --- Store Trajectory Points AFTER Release ---
    if release_detected_in_shot and ball_bbox is not None:
        # Calculate ball center for the *current* frame's detection
        current_ball_center_x = (ball_bbox[0] + ball_bbox[2]) // 2
        current_ball_center_y = (ball_bbox[1] + ball_bbox[3]) // 2
        current_point = (current_ball_center_x, current_ball_center_y)

        # Add point if it's different from the last one (optional, avoids static points)
        if not trajectory_points or current_point != trajectory_points[-1]:
            trajectory_points.append(current_point)
            scaled_x = (current_ball_center_x - origin_x_pxl) / ball_radius_pxl
            scaled_y = (origin_y_pxl - current_ball_center_y) / ball_radius_pxl

            #trajectory_points_scaled.append((scaled_x, scaled_y))
            trajectory_points_scaled.append({'x': scaled_x, 'y': scaled_y, 'frame': frame_count})

            # Naive Speed/Angle Estimation (once enough points are collected)
            # We need 1 (release) + FRAMES_FOR_NAIVE_ESTIMATION points
            if len(trajectory_points_scaled) == 1 + FRAMES_FOR_NAIVE_ESTIMATION:
                point_start = trajectory_points_scaled[0]  # This is (0,0) at release_frame_info['frame_no']
                point_end = trajectory_points_scaled[
                    FRAMES_FOR_NAIVE_ESTIMATION]  # The point FRAMES_FOR_NAIVE_ESTIMATION after release

                print(f"Start: {point_start}, End: {point_end}")
                delta_scaled_x = point_end['x'] - point_start['x']  # point_start['x'] is 0.0
                delta_scaled_y = point_end['y'] - point_start['y']  # point_start['y'] is 0.0

                real_delta_x_m = delta_scaled_x * BASKETBALL_REAL_RADIUS_M
                real_delta_y_m = delta_scaled_y * BASKETBALL_REAL_RADIUS_M

                delta_frames = point_end['frame'] - point_start['frame']
                print(f"Delta frames: delta_frames")
                #isn't this the same as frames for naive
                if fps > 0 and delta_frames > 0:
                    delta_t_s = delta_frames / fps

                    avg_v_x_mps = real_delta_x_m / delta_t_s
                    avg_v_y_mps = real_delta_y_m / delta_t_s

                    estimated_speed_mps = math.sqrt(avg_v_x_mps ** 2 + avg_v_y_mps ** 2)
                    # atan2(y,x). Y is vertical component, X is horizontal.
                    estimated_angle_rad = math.atan2(avg_v_y_mps, avg_v_x_mps)
                    estimated_angle_deg = 180 - math.degrees(estimated_angle_rad)

                    print(f"  NAIVE ESTIMATION (first {FRAMES_FOR_NAIVE_ESTIMATION} frames post-release):")
                    print(f"    Delta_t: {delta_t_s:.3f}s")
                    print(f"    Real Delta X: {real_delta_x_m:.3f}m, Real Delta Y: {real_delta_y_m:.3f}m")
                    print(f"    Avg Vx: {avg_v_x_mps:.2f} m/s, Avg Vy: {avg_v_y_mps:.2f} m/s")
                    print(f"    Speed: {estimated_speed_mps:.2f} m/s, Angle: {estimated_angle_deg:.2f} degrees")


    # --- Annotations for Visualization ---
    # Draw ball bbox (only the target class)
    if ball_results_list and len(ball_results_list) > 0:
        # Plotting only the ball class from model2.pt
        current_frame_annotated = ball_results_list[0].plot(img=current_frame_annotated, boxes=True, labels=True) # Removed classes filter here if get_ball_bbox already ensures correct class

    # Draw pose (keypoints only, no person box from pose_model)
    if pose_results_list and len(pose_results_list) > 0:
        current_frame_annotated = pose_results_list[0].plot(img=current_frame_annotated, boxes=False, labels=False)

    # Highlight the identified shooter's wrist (if found)
    if shooter_wrist_pos is not None:
        cv2.circle(current_frame_annotated, tuple(shooter_wrist_pos), 7, (0, 255, 255), -1) # Yellow

    # --- Draw Trajectory ---
    if release_detected_in_shot and len(trajectory_points) >= 2:
        # Draw lines connecting consecutive points in the trajectory
        points_np = np.array(trajectory_points, dtype=np.int32)
        # cv2.polylines is efficient for drawing connected lines
        cv2.polylines(current_frame_annotated, [points_np], isClosed=False, color=(0, 255, 0), thickness=2)
        # Optional: Draw circles at each tracked point
        # for point in trajectory_points:
        #     cv2.circle(current_frame_annotated, point, 3, (0, 255, 0), -1)


    # Indicate release (draw release marker AFTER trajectory so it's on top)
    if release_detected_in_shot and release_frame_info:
        cv2.putText(current_frame_annotated, f"Release Frame: {release_frame_info['frame_no']}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Draw a marker on the ball's position *at release* (first point of trajectory)
        if trajectory_points: # Make sure list is not empty
             cv2.circle(current_frame_annotated, trajectory_points[0], 15, (0,0,255), 2) # Red circle at release point
             # Display scaled trajectory info (optional)
             # Display scaled trajectory info (optional)
        if estimated_speed_mps > 0:
            cv2.putText(current_frame_annotated, f"Speed: {estimated_speed_mps:.2f} m/s", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(current_frame_annotated, f"Angle: {estimated_angle_deg:.1f} deg", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    # --- Display and Save ---
    cv2.imshow("Release Detection & Scaled Trajectory", current_frame_annotated) # Renamed window
    out.write(current_frame_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user.")
        break

# 5. Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished processing. Output saved to {output_path}")