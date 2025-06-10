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
RELEASE_FRAME_OUTPUT_DIR = "outputs/release_frames_ccs"
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

# Distortion Coefficients (D) for GoPro HERO12 Black Linear 1080p (fisheye model params)
D = np.array([
    0.43466848378748985,  # k1
    -0.6373157421587791, # k2
    2.297946857510941,    # k3
    -2.0968503612499836   # k4
])

fx = K[0,0]
fy = K[1,1]
cx = K[0,2]
cy = K[1,2]


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

video_path = "footage/freethrow_arc2.mp4"  # Make sure this path is correct
cap = cv2.VideoCapture(video_path,cv2.CAP_MSMF)
if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit()

# 3. Video Output (get properties from input video)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
# Ensure fps is a reasonable value, e.g., not 0
if video_fps == 0:
    print("Warning: Video FPS is 0, defaulting to 25 FPS for output.")
    video_fps = 240

output_path = "outputs/freethrow2_speed_ccs.avi"
# Using 'mp4v' codec for .mp4 files, common and widely supported
availableBackends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), video_fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_path}")
    cap.release()
    exit()

print(f"Processing video: {video_path}")
print(f"Outputting to: {output_path}")


#### SETUP
TARGET_BALL_CLASS_NAME = "ball"
id_of_ball = 0

# --- State Variables ---
release_detected_in_shot = False
release_frame_info_dict = None
potential_release_buffer = []
trajectory_points_pixels_undistorted = [] # Stores (u_undistorted, v_undistorted)

# --- CCS Specific Variables ---
trajectory_points_ccs = [] # Stores {'frame': frame_no, 'Pc': [Xc, Yc, Zc]}
Zc_at_release = None # Estimated depth of ball from camera at release

FRAMES_FOR_NAIVE_ESTIMATION = 5 # Number of frames *after* release to consider for displacement
BASKETBALL_REAL_DIAMETER_M = 0.24

estimated_speed_mps = 0.0
estimated_angle_deg_XY_ccs = 0.0 # Angle in camera's XY plane
estimated_angle_deg_elev_ccs = 0.0 # Elevation angle from camera's XZ plane

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

while cap.isOpened():
    ret, frame_orig = cap.read()
    if not ret: break
    frame_count += 1
    if frame_count % 30 == 0: print(f"Processing frame {frame_count}...")

    frame_to_detect_on = frame_orig
    current_frame_annotated = frame_orig.copy()
    if DETECT_ON_FULLY_UNDISTORTED_FRAME and map1 is not None:
        frame_to_detect_on = cv2.remap(frame_orig, map1, map2, interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)
        current_frame_annotated = frame_to_detect_on.copy()

    ball_results_list = ball_model(frame_to_detect_on, conf=0.6, verbose=False, classes=[id_of_ball])
    ball_bbox_from_detection = utils.get_ball_bbox(ball_results_list, id_of_ball)

    ball_center_undistorted_px = None  # This will be (u,v)
    if ball_bbox_from_detection is not None:
        center_x_detected = (ball_bbox_from_detection[0] + ball_bbox_from_detection[2]) / 2.0
        center_y_detected = (ball_bbox_from_detection[1] + ball_bbox_from_detection[3]) / 2.0
        if UNDISTORT_POINTS_AFTER_DETECTION and not DETECT_ON_FULLY_UNDISTORTED_FRAME:
            dist_pts = np.array([[[center_x_detected, center_y_detected]]], dtype=np.float32)
            undist_pts_px = cv2.fisheye.undistortPoints(dist_pts, K, D, P=K)
            if undist_pts_px is not None:
                ball_center_undistorted_px = (int(round(undist_pts_px[0][0][0])), int(round(undist_pts_px[0][0][1])))
        else:
            ball_center_undistorted_px = (int(round(center_x_detected)), int(round(center_y_detected)))

    pose_results_list = pose_model(frame_to_detect_on, conf=0.35, verbose=False)
    shooter_wrist_pos_processed = None  # This will be (u,v)
    if ball_bbox_from_detection is not None:  # Use detected bbox for proximity
        shooter_wrist_pos_detected, _ = utils.get_likely_shooter_wrist_and_person_idx(
            pose_results_list, ball_bbox_from_detection, SHOOTING_WRIST_KP_INDEX,
            MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
        )
        if shooter_wrist_pos_detected is not None:
            if UNDISTORT_POINTS_AFTER_DETECTION and not DETECT_ON_FULLY_UNDISTORTED_FRAME:
                wrist_dist_pts = np.array([[[shooter_wrist_pos_detected[0], shooter_wrist_pos_detected[1]]]],
                                          dtype=np.float32)
                wrist_undist_px = cv2.fisheye.undistortPoints(wrist_dist_pts, K, D, P=K)
                if wrist_undist_px is not None:
                    shooter_wrist_pos_processed = (
                    int(round(wrist_undist_px[0][0][0])), int(round(wrist_undist_px[0][0][1])))
            else:
                shooter_wrist_pos_processed = (int(shooter_wrist_pos_detected[0]), int(shooter_wrist_pos_detected[1]))

    if not release_detected_in_shot and ball_center_undistorted_px is not None and shooter_wrist_pos_processed is not None and ball_bbox_from_detection is not None:
        wrist_in_ball_bbox = utils.is_point_inside_bbox(shooter_wrist_pos_processed, ball_bbox_from_detection,
                                                        margin=WRIST_BALL_PROXIMITY_MARGIN)
        if not wrist_in_ball_bbox:
            dist_edge = utils.calculate_distance_point_to_bbox_edge(shooter_wrist_pos_processed,
                                                                    ball_bbox_from_detection)
            potential_release_buffer.append({
                "frame_no": frame_count,
                "ball_center_undistorted_px": ball_center_undistorted_px,
                "ball_bbox_detection": ball_bbox_from_detection,
                "wrist_pos_processed": shooter_wrist_pos_processed,
                "original_frame_at_this_moment": frame_orig.copy()
            })
            if len(potential_release_buffer) > CONFIRMATION_FRAMES_FOR_RELEASE:
                potential_release_buffer.pop(0)
            if len(potential_release_buffer) == CONFIRMATION_FRAMES_FOR_RELEASE:
                release_frame_info_dict = potential_release_buffer[0]
                release_detected_in_shot = True
                shot_count += 1
                estimated_speed_mps = 0.0
                estimated_angle_deg_XY_ccs = 0.0
                estimated_angle_deg_elev_ccs = 0.0

                trajectory_points_pixels_undistorted = []
                trajectory_points_ccs = []
                Zc_at_release = None  # Reset

                u_release, v_release = release_frame_info_dict['ball_center_undistorted_px']
                trajectory_points_pixels_undistorted.append((u_release, v_release))

                # Estimate Zc_at_release
                rel_ball_bbox_det = release_frame_info_dict['ball_bbox_detection']
                apparent_width_px = rel_ball_bbox_det[2] - rel_ball_bbox_det[0]
                apparent_height_px = rel_ball_bbox_det[3] - rel_ball_bbox_det[1]
                apparent_diameter_px_release = (apparent_width_px + apparent_height_px) / 2.0

                if apparent_diameter_px_release > 0:
                    # Using average focal length (fx+fy)/2 for simplicity, or pick one e.g. fx
                    avg_focal_length_px = (fx + fy) / 2.0
                    Zc_at_release = (BASKETBALL_REAL_DIAMETER_M * avg_focal_length_px) / apparent_diameter_px_release
                else:
                    Zc_at_release = 1.0  # Fallback, should not happen if bbox is valid

                # Calculate initial CCS point
                xn_release = (u_release - cx) / fx
                yn_release = (v_release - cy) / fy
                Xc_release = xn_release * Zc_at_release
                Yc_release = yn_release * Zc_at_release
                Pc_release = np.array([Xc_release, Yc_release, Zc_at_release])
                trajectory_points_ccs.append({'frame': release_frame_info_dict['frame_no'], 'Pc': Pc_release})

                print(f"RELEASE DETECTED (Shot {shot_count}) at frame: {release_frame_info_dict['frame_no']}")
                print(f"  Release Ball Center (undistorted px): ({u_release}, {v_release})")
                print(f"  Est. Zc at Release (m): {Zc_at_release:.3f}")
                print(f"  Release Ball Coords (CCS, m): {Pc_release.round(3)}")

                if SAVE_RELEASE_FRAMES:
                    # ... (save frame logic as before, using appropriate mode text) ...
                    frame_to_save = release_frame_info_dict["original_frame_at_this_moment"]
                    annotated_release_frame_for_save = frame_to_save.copy()
                    cv2.circle(annotated_release_frame_for_save, (u_release, v_release), 15, (0, 0, 255), 2)
                    if shooter_wrist_pos_processed:
                        cv2.circle(annotated_release_frame_for_save, shooter_wrist_pos_processed, 7, (0, 255, 255), -1)
                    mode_text = "CCS_UndistortPoints"  # Example
                    if DETECT_ON_FULLY_UNDISTORTED_FRAME: mode_text = "CCS_FullFrameUndist"
                    save_path = os.path.join(RELEASE_FRAME_OUTPUT_DIR,
                                             f"release_shot{shot_count}_frame{release_frame_info_dict['frame_no']}_{mode_text}.png")
                    cv2.imwrite(save_path, annotated_release_frame_for_save)
                    print(f"  Saved release frame to: {save_path}")
        else:
            potential_release_buffer = []
            release_detected_in_shot = False  # Allow re-detection if hand regains ball

    if release_detected_in_shot and ball_center_undistorted_px is not None and Zc_at_release is not None:
        current_u, current_v = ball_center_undistorted_px

        # Add to pixel trajectory if different
        if not trajectory_points_pixels_undistorted or (current_u, current_v) != trajectory_points_pixels_undistorted[
            -1]:
            trajectory_points_pixels_undistorted.append((current_u, current_v))

            # Convert current point to CCS, assuming Zc = Zc_at_release
            xn_current = (current_u - cx) / fx
            yn_current = (current_v - cy) / fy
            Xc_current = xn_current * Zc_at_release  # Assume constant depth Zc
            Yc_current = yn_current * Zc_at_release  # Assume constant depth Zc
            Pc_current = np.array([Xc_current, Yc_current, Zc_at_release])  # Zc_current is assumed = Zc_at_release

            # Add to CCS trajectory if different from last CCS point (optional, if Pc is stored)
            # This check is more complex for numpy arrays, simplified to just add for now
            trajectory_points_ccs.append({'frame': frame_count, 'Pc': Pc_current})

            # Naive Speed/Angle Estimation using CCS points
            if estimated_speed_mps == 0.0 and len(trajectory_points_ccs) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
                point_start_ccs_info = trajectory_points_ccs[0]
                point_end_ccs_info = trajectory_points_ccs[FRAMES_FOR_NAIVE_ESTIMATION]

                Pc_start = point_start_ccs_info['Pc']
                Pc_end = point_end_ccs_info['Pc']

                delta_Pc = Pc_end - Pc_start  # Vector subtraction [delta_Xc, delta_Yc, delta_Zc]
                # delta_Zc will be small if Zc_at_release is used for all points;
                # if Zc was re-estimated per frame, delta_Zc would be more meaningful

                delta_frames = point_end_ccs_info['frame'] - point_start_ccs_info['frame']
                if video_fps > 0 and delta_frames > 0:
                    delta_t_s = delta_frames / video_fps

                    Vc_avg = delta_Pc / delta_t_s  # Average velocity vector in CCS [Vcx, Vcy, Vcz]

                    estimated_speed_mps = np.linalg.norm(Vc_avg)  # Magnitude of the 3D velocity vector

                    # Angle in camera's XY plane (view from top/bottom of camera)
                    # Yc typically points down in image, Xc right.
                    # For atan2(y,x), if Yc points down, a positive angle means "down and right" or "up and right" from Xc axis
                    # Let's use -Vc_avg[1] to make "upwards" in camera Y result in positive angle.
                    estimated_angle_rad_XY_ccs = math.atan2(-Vc_avg[1], Vc_avg[0])
                    estimated_angle_deg_XY_ccs = math.degrees(estimated_angle_rad_XY_ccs)

                    # Elevation angle from camera's XZ plane (how much "up/down" in camera view from its forward direction)
                    # Vc_avg[0] is Xc (sideways), Vc_avg[1] is Yc (up/down in image), Vc_avg[2] is Zc (depth)
                    # We want angle relative to the plane formed by Xc and Zc (camera's "horizontal plane")
                    magnitude_in_XZ_plane = math.sqrt(Vc_avg[0] ** 2 + Vc_avg[2] ** 2)
                    estimated_angle_rad_elev_ccs = math.atan2(-Vc_avg[1], magnitude_in_XZ_plane)  # -Yc for up positive
                    estimated_angle_deg_elev_ccs = math.degrees(estimated_angle_rad_elev_ccs)

                    print(
                        f"  NAIVE CCS ESTIMATION (Shot {shot_count}, points at index 0 and {FRAMES_FOR_NAIVE_ESTIMATION}):")
                    print(f"    Delta_t: {delta_t_s:.3f}s")
                    print(f"    Delta_Pc (m): {delta_Pc.round(3)}")
                    print(f"    Avg Vc (m/s): {Vc_avg.round(2)}")
                    print(f"    Speed (CCS): {estimated_speed_mps:.2f} m/s")
                    print(
                        f"    Angle XY (CCS): {estimated_angle_deg_XY_ccs:.2f} deg (relative to camera X-axis in XY plane)")
                    print(
                        f"    Angle Elev (CCS): {estimated_angle_deg_elev_ccs:.2f} deg (elevation from camera XZ-plane)")

    # --- Annotations ---
    if ball_bbox_from_detection is not None:
        left, top, right, bottom = [int(c) for c in ball_bbox_from_detection]
        cv2.rectangle(current_frame_annotated, (left, top), (right, bottom), (255, 0, 0), 1)
    if ball_center_undistorted_px is not None:
        cv2.circle(current_frame_annotated, ball_center_undistorted_px, 5, (0, 0, 255), -1)

    if pose_results_list and len(pose_results_list) > 0:
        current_frame_annotated = pose_results_list[0].plot(img=current_frame_annotated, boxes=False, labels=False)
    if shooter_wrist_pos_processed is not None:
        cv2.circle(current_frame_annotated, shooter_wrist_pos_processed, 7, (0, 255, 255), -1)

    if release_detected_in_shot and len(trajectory_points_pixels_undistorted) >= 2:
        points_np = np.array(trajectory_points_pixels_undistorted, dtype=np.int32)
        cv2.polylines(current_frame_annotated, [points_np], isClosed=False, color=(0, 255, 0), thickness=2)

    if release_detected_in_shot:
        if release_frame_info_dict:
            cv2.putText(current_frame_annotated, f"S{shot_count} Rls:Fr{release_frame_info_dict['frame_no']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if trajectory_points_pixels_undistorted:
            cv2.circle(current_frame_annotated, trajectory_points_pixels_undistorted[0], 15, (0, 0, 255), 2)
        if estimated_speed_mps > 0:
            cv2.putText(current_frame_annotated, f"Speed(CCS): {estimated_speed_mps:.2f}m/s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            # Displaying elevation angle as it's more comparable to typical launch angle
            cv2.putText(current_frame_annotated, f"Elev(CCS): {estimated_angle_deg_elev_ccs:.1f}deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    cv2.imshow("CCS Estimation", current_frame_annotated)
    out.write(current_frame_annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished processing. Output saved to {output_path}")