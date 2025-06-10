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

SAVE_RELEASE_FRAMES = True # Set to False to disable
RELEASE_FRAME_OUTPUT_DIR = "outputs/release_frames"
if SAVE_RELEASE_FRAMES and not os.path.exists(RELEASE_FRAME_OUTPUT_DIR):
    os.makedirs(RELEASE_FRAME_OUTPUT_DIR)
# --- END NEW ---

# --- NEW: Camera Intrinsic Parameters (from Gyroflow) ---
# K matrix for GoPro HERO12 Black Linear 1080p
K = np.array([
    [1134.4909447935638, 0.0, 968.5293267526777],
    [0.0, 1133.2194561973472, 553.0426928367185],
    [0.0, 0.0, 1.0]
])

# --- Configuration for Detection and Tracking ---
PRE_RELEASE_BALL_CONF = 0.6  # Confidence for ball detection before release
POST_RELEASE_BALL_CONF = 0.25 # Lowered confidence for ball tracking after release
MAX_PIXEL_DISPLACEMENT_POST_RELEASE = 75 # Max pixels ball can move for post-release accept

# Distortion Coefficients (D) for GoPro HERO12 Black Linear 1080p (fisheye model params)
D = np.array([
    0.43466848378748985,  # k1
    -0.6373157421587791, # k2
    2.297946857510941,    # k3
    -2.0968503612499836   # k4
])


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

video_path = "footage/freethrow_1.mp4"  # Make sure this path is correct
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

output_path = "outputs/freethrow_check.avi"
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

# --- Choose ONE processing mode for a given run ---
# True: Detect on original frame, then undistort key points (ball center, wrist)
UNDISTORT_POINTS_AFTER_DETECTION = True
# True: Undistort the entire frame first, then run detections on it
DETECT_ON_FULLY_UNDISTORTED_FRAME = True # Set one of these to True, the other to False, or both False for original behavior
# ---

# Precompute maps if detecting on fully undistorted frame
map1, map2 = None, None
if DETECT_ON_FULLY_UNDISTORTED_FRAME:
    print("Precomputing undistortion maps for full frame...")
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (frame_width, frame_height), cv2.CV_16SC2)
    print("Maps computed.")

frame_count = 0
max_frames = float('inf')
shot_count = 0 # To differentiate release frame images if multiple shots occur

while cap.isOpened() and frame_count < max_frames:
    ret, frame_orig = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 10 == 0:  # Changed print frequency to match other log
        print(
            f"Frame {frame_count}, Shot: {shot_count}, Release: {release_detected_in_shot}, Traj Scaled Pts: {len(trajectory_points_scaled)}")

    frame_to_detect_on = frame_orig
    current_frame_annotated = frame_orig.copy()  # Start with original for annotation

    if DETECT_ON_FULLY_UNDISTORTED_FRAME and map1 is not None:
        frame_to_detect_on = cv2.remap(frame_orig, map1, map2, interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)
        current_frame_annotated = frame_to_detect_on.copy()  # Annotate on the (fully) undistorted frame

    #Dynamic Confidence for Ball Detection
    current_ball_conf_threshold = PRE_RELEASE_BALL_CONF
    if release_detected_in_shot:
        current_ball_conf_threshold = POST_RELEASE_BALL_CONF

    ball_results_list = ball_model(frame_to_detect_on, conf=current_ball_conf_threshold, verbose=False,
                                   classes=[id_of_ball])
    ball_bbox_from_detection = utils.get_ball_bbox(ball_results_list, id_of_ball)

    print(f"  Frame {frame_count}: V_A - Initial ball_bbox_from_detection: {ball_bbox_from_detection} (conf_thresh_used={current_ball_conf_threshold:.2f})")

    ball_center_for_trajectory = None
    ball_bbox_for_logic = ball_bbox_from_detection  # This will be used for IOU checks

    if ball_bbox_from_detection is not None:
        center_x_detected = (ball_bbox_from_detection[0] + ball_bbox_from_detection[2]) / 2.0
        center_y_detected = (ball_bbox_from_detection[1] + ball_bbox_from_detection[3]) / 2.0

        temp_ball_center_processed = None  # Undistorted version of detected center
        if UNDISTORT_POINTS_AFTER_DETECTION and not DETECT_ON_FULLY_UNDISTORTED_FRAME:
            distorted_points = np.array([[[center_x_detected, center_y_detected]]], dtype=np.float32)
            undistorted_points_px = cv2.fisheye.undistortPoints(distorted_points, K, D, P=K)
            if undistorted_points_px is not None and len(undistorted_points_px) > 0:
                ball_center_for_trajectory = (
                int(round(undistorted_points_px[0][0][0])), int(round(undistorted_points_px[0][0][1])))
        else:  # Either detecting on already undistorted frame, or not undistorting points
            ball_center_for_trajectory = (int(round(center_x_detected)), int(round(center_y_detected)))

        print(f'Proximity condition check: {release_detected_in_shot and temp_ball_center_processed is not None and len(trajectory_points_pixels) > 0}')
        # --- MODIFIED: Proximity Validation for Post-Release Detections ---
        if release_detected_in_shot and temp_ball_center_processed is not None and len(
                trajectory_points_pixels) > 0:
            last_known_ball_center = trajectory_points_pixels[-1]
            dist_sq = (temp_ball_center_processed[0] - last_known_ball_center[0]) ** 2 + \
                      (temp_ball_center_processed[1] - last_known_ball_center[1]) ** 2
            print(f"    Frame {frame_count}: V_A Post-release proximity check: dist_sq={dist_sq:.1f} vs threshold_sq={MAX_PIXEL_DISPLACEMENT_POST_RELEASE**2:.1f}")
            if dist_sq <= MAX_PIXEL_DISPLACEMENT_POST_RELEASE ** 2:
                ball_center_for_trajectory = temp_ball_center_processed  # Accept it
            else:
                print(f"    Frame {frame_count}: V_A REJECTED by proximity.")
                # print(f"  Frame {frame_count}: Post-release ball detection too far ({math.sqrt(dist_sq):.1f}px > {MAX_PIXEL_DISPLACEMENT_POST_RELEASE}px), rejected.")
                ball_center_for_trajectory = None  # Reject it
                ball_bbox_for_logic = None  # If center is rejected, bbox associated with it is also invalid for logic
        elif temp_ball_center_processed is not None:  # Before release, or first point after release
            ball_center_for_trajectory = temp_ball_center_processed
        # --- END MODIFIED ---
    else:  # No ball detected at current_ball_conf_threshold
        ball_bbox_for_logic = None
        print(f"  Frame {frame_count}: V_A No ball detected at conf={current_ball_conf_threshold:.2f}, so ball_center_for_trajectory is None.")

    pose_results_list = pose_model(frame_to_detect_on, conf=0.35, verbose=False)
    shooter_wrist_pos_for_logic = None  # This will be the point used for release logic

    if ball_bbox_for_logic is not None:
        shooter_wrist_pos_detected, shooter_person_idx = utils.get_likely_shooter_wrist_and_person_idx(
            pose_results_list, ball_bbox_for_logic, SHOOTING_WRIST_KP_INDEX, MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
        )
        if shooter_wrist_pos_detected is not None:
            if UNDISTORT_POINTS_AFTER_DETECTION and not DETECT_ON_FULLY_UNDISTORTED_FRAME:
                wrist_dist_pts = np.array([[[shooter_wrist_pos_detected[0], shooter_wrist_pos_detected[1]]]],
                                          dtype=np.float32)
                wrist_undist_px = cv2.fisheye.undistortPoints(wrist_dist_pts, K, D, P=K)
                if wrist_undist_px is not None:
                    shooter_wrist_pos_for_logic = (
                    int(round(wrist_undist_px[0][0][0])), int(round(wrist_undist_px[0][0][1])))
            else:  # Detecting on already undistorted frame or not undistorting points
                shooter_wrist_pos_for_logic = (int(shooter_wrist_pos_detected[0]), int(shooter_wrist_pos_detected[1]))

    if not release_detected_in_shot and ball_center_for_trajectory is not None and shooter_wrist_pos_for_logic is not None and ball_bbox_for_logic is not None:
        wrist_in_ball_bbox = utils.is_point_inside_bbox(shooter_wrist_pos_for_logic, ball_bbox_for_logic,
                                                        margin=WRIST_BALL_PROXIMITY_MARGIN)
        if not wrist_in_ball_bbox:
            dist_to_bbox_edge = utils.calculate_distance_point_to_bbox_edge(shooter_wrist_pos_for_logic,
                                                                            ball_bbox_for_logic)
            potential_release_buffer.append({
                "frame_no": frame_count,
                "ball_center_processed": ball_center_for_trajectory,  # This is the key point for trajectory
                "ball_bbox_detection": ball_bbox_for_logic,  # Bbox from detection (used for radius)
                "wrist_pos_processed": shooter_wrist_pos_for_logic,
                "dist_to_edge": dist_to_bbox_edge,
                "original_frame_at_this_moment": frame_orig.copy()  # Store the original frame when this event is added
            })
            if len(potential_release_buffer) > CONFIRMATION_FRAMES_FOR_RELEASE:
                potential_release_buffer.pop(0)
            if len(potential_release_buffer) == CONFIRMATION_FRAMES_FOR_RELEASE:
                release_frame_info_dict = potential_release_buffer[0]
                release_detected_in_shot = True
                shot_count += 1  # Increment shot counter
                estimated_speed_mps = 0.0
                estimated_angle_deg = 0.0

                trajectory_points_pixels = []
                trajectory_points_scaled = []

                origin_x_pxl, origin_y_pxl = release_frame_info_dict['ball_center_processed']

                rel_ball_bbox_det = release_frame_info_dict['ball_bbox_detection']
                release_ball_width_px = rel_ball_bbox_det[2] - rel_ball_bbox_det[0]
                release_ball_height_px = rel_ball_bbox_det[3] - rel_ball_bbox_det[1]
                ball_radius_pxl = (release_ball_width_px + release_ball_height_px) / 4.0
                if ball_radius_pxl < 1: ball_radius_pxl = 1.0

                trajectory_points_pixels.append((origin_x_pxl, origin_y_pxl))
                trajectory_points_scaled.append({'x': 0.0, 'y': 0.0, 'frame': release_frame_info_dict['frame_no']})

                print(f"RELEASE DETECTED (Shot {shot_count}) at frame: {release_frame_info_dict['frame_no']}")
                print(f"  Release Ball Center (processed px): ({origin_x_pxl}, {origin_y_pxl})")
                print(f"  Release Ball Apparent Radius (from detected bbox, px): {ball_radius_pxl:.2f}")

                # --- NEW: Save the frame at release ---
                if SAVE_RELEASE_FRAMES:
                    # Use the original frame that was stored when this event was added to buffer
                    frame_to_save = release_frame_info_dict["original_frame_at_this_moment"]
                    # Annotate this specific frame for saving
                    annotated_release_frame_for_save = frame_to_save.copy()
                    cv2.circle(annotated_release_frame_for_save, (origin_x_pxl, origin_y_pxl), 15, (0, 0, 255),
                               2)  # Release point
                    if shooter_wrist_pos_for_logic:  # Draw wrist used for logic if available
                        cv2.circle(annotated_release_frame_for_save, shooter_wrist_pos_for_logic, 7, (0, 255, 255),
                                   -1)  # Wrist

                    # Add text indicating processing mode
                    mode_text = "DistortedPoints"
                    if DETECT_ON_FULLY_UNDISTORTED_FRAME:
                        mode_text = "FullFrameUndist"
                    elif UNDISTORT_POINTS_AFTER_DETECTION:
                        mode_text = "UndistortPoints"

                    save_path = os.path.join(RELEASE_FRAME_OUTPUT_DIR,
                                             f"release_shot{shot_count}_frame{release_frame_info_dict['frame_no']}_{mode_text}.png")
                    cv2.imwrite(save_path, annotated_release_frame_for_save)
                    print(f"  Saved release frame to: {save_path}")
                # --- END NEW ---
        else:
            potential_release_buffer = []
            # If ball is back in hand, reset release state for potential new shot
            release_detected_in_shot = False
            # trajectory_points_pixels = [] # Optional: clear trajectory if shot is "cancelled"
            # trajectory_points_scaled = []

    #Post release detection
    if release_detected_in_shot and ball_center_for_trajectory is not None:
        current_point_processed_px = ball_center_for_trajectory

        if not trajectory_points_pixels or current_point_processed_px != trajectory_points_pixels[-1]:
            trajectory_points_pixels.append(current_point_processed_px)
            scaled_x = (current_point_processed_px[0] - origin_x_pxl) / ball_radius_pxl
            scaled_y = (origin_y_pxl - current_point_processed_px[1]) / ball_radius_pxl
            trajectory_points_scaled.append({'x': scaled_x, 'y': scaled_y, 'frame': frame_count})

            if estimated_speed_mps == 0.0 and len(trajectory_points_scaled) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
                # ... (naive estimation logic as before) ...
                point_start_index = 0
                point_end_index = FRAMES_FOR_NAIVE_ESTIMATION
                if point_end_index < len(trajectory_points_scaled):
                    point_start = trajectory_points_scaled[point_start_index]
                    point_end = trajectory_points_scaled[point_end_index]
                    delta_scaled_x = point_end['x'] - point_start['x']
                    delta_scaled_y = point_end['y'] - point_start['y']
                    real_delta_x_m = delta_scaled_x * BASKETBALL_REAL_RADIUS_M
                    real_delta_y_m = delta_scaled_y * BASKETBALL_REAL_RADIUS_M
                    delta_frames = point_end['frame'] - point_start['frame']
                    if fps > 0 and delta_frames > 0:
                        delta_t_s = delta_frames / fps
                        avg_v_x_mps = real_delta_x_m / delta_t_s
                        avg_v_y_mps = real_delta_y_m / delta_t_s
                        estimated_speed_mps = math.sqrt(avg_v_x_mps ** 2 + avg_v_y_mps ** 2)
                        estimated_angle_rad = math.atan2(avg_v_y_mps, avg_v_x_mps)
                        estimated_angle_deg = math.degrees(estimated_angle_rad)
                        print(
                            f"  NAIVE ESTIMATION (Shot {shot_count}, points at index {point_start_index} and {point_end_index}):")
                        # ... (rest of print statements) ...

    # --- Annotations for Visualization (on current_frame_annotated) ---
    # Draw the raw detected bbox (ball_bbox_from_detection)
    if ball_bbox_from_detection is not None:
        left, top, right, bottom = [int(c) for c in ball_bbox_from_detection]
        # Color based on whether this raw detection was ultimately used
        is_bbox_used_for_traj = (ball_center_for_trajectory is not None and
                                 abs((left + right) / 2 - ball_center_for_trajectory[
                                     0] if UNDISTORT_POINTS_AFTER_DETECTION and not DETECT_ON_FULLY_UNDISTORTED_FRAME else (
                                                                                                                                       left + right) / 2 -
                                                                                                                           ball_center_for_trajectory[
                                                                                                                               0]) < 5 and  # Heuristic
                                 abs((top + bottom) / 2 - ball_center_for_trajectory[
                                     1] if UNDISTORT_POINTS_AFTER_DETECTION and not DETECT_ON_FULLY_UNDISTORTED_FRAME else (
                                                                                                                                       top + bottom) / 2 -
                                                                                                                           ball_center_for_trajectory[
                                                                                                                               1]) < 5)
        color = (
        0, 255, 255) if is_bbox_used_for_traj and current_ball_conf_threshold == POST_RELEASE_BALL_CONF else (
        255, 0, 0)
        cv2.rectangle(current_frame_annotated, (left, top), (right, bottom), color, 1)

    if ball_center_for_trajectory is not None:  # The final, validated, undistorted center
        cv2.circle(current_frame_annotated, ball_center_for_trajectory, 5, (0, 0, 255), -1)  # Red

    if pose_results_list and len(pose_results_list) > 0:
        current_frame_annotated = pose_results_list[0].plot(img=current_frame_annotated, boxes=False, labels=False)
    if shooter_wrist_pos_for_logic is not None:
        cv2.circle(current_frame_annotated, tuple(shooter_wrist_pos_for_logic), 7, (0, 255, 255), -1)

    if release_detected_in_shot and len(trajectory_points_pixels) >= 2:
        points_np = np.array(trajectory_points_pixels, dtype=np.int32)
        cv2.polylines(current_frame_annotated, [points_np], isClosed=False, color=(0, 255, 0), thickness=2)

    if release_detected_in_shot:  # Display text if release has occurred in this shot segment
        if release_frame_info_dict:
            cv2.putText(current_frame_annotated, f"Shot {shot_count} Release: Fr{release_frame_info_dict['frame_no']}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if trajectory_points_pixels:
            cv2.circle(current_frame_annotated, trajectory_points_pixels[0], 15, (0, 0, 255), 2)
        if estimated_speed_mps > 0:
            cv2.putText(current_frame_annotated, f"Speed: {estimated_speed_mps:.2f} m/s", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(current_frame_annotated, f"Angle: {estimated_angle_deg:.1f} deg", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    cv2.imshow("Basketball Analysis", current_frame_annotated)
    out.write(current_frame_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished processing. Output saved to {output_path}")