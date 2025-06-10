import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np

os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0" # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100" # Try to prioritize FFMPEG

import cv2
from ultralytics import YOLO
import utils  # Your helper functions
from scipy.optimize import curve_fit  # For model fitting

# --- Camera Intrinsic Parameters (from Gyroflow) ---
K = np.array([
    [1134.4909447935638, 0.0, 968.5293267526777],  # fx, 0, cx
    [0.0, 1133.2194561973472, 553.0426928367185],  # 0, fy, cy
    [0.0, 0.0, 1.0]
])
D = np.array([  # Fisheye distortion coefficients
    0.43466848378748985, -0.6373157421587791, 2.297946857510941, -2.0968503612499836
])
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]
# ---

# --- Constants ---
SHOOTING_WRIST_KP_INDEX = utils.SHOOTING_WRIST_KP_INDEX
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
WRIST_BALL_PROXIMITY_MARGIN = utils.WRIST_BALL_PROXIMITY_MARGIN
CONFIRMATION_FRAMES_FOR_RELEASE = utils.CONFIRMATION_FRAMES_FOR_RELEASE

SAVE_RELEASE_FRAMES = True
RELEASE_FRAME_OUTPUT_DIR = "outputs/release_frames_ccs_fit"  # Different dir
if SAVE_RELEASE_FRAMES and not os.path.exists(RELEASE_FRAME_OUTPUT_DIR):
    os.makedirs(RELEASE_FRAME_OUTPUT_DIR)

FRAMES_FOR_NAIVE_ESTIMATION = 5  # Still used for naive, fitting can use more
BASKETBALL_REAL_DIAMETER_M = 0.24
GRAVITY_ACCELERATION = 9.81  # m/s^2. Assuming camera Yc positive is "down" physically.

# --- Processing Mode Selection ---
UNDISTORT_POINTS_AFTER_DETECTION = True
DETECT_ON_FULLY_UNDISTORTED_FRAME = False

# --- END NEW ---


# ... (rest of setup: model loading, video input/output as before) ...
try:
    ball_model = YOLO('Yolo-Weights/model2.pt')
    print("Ball detection custom model loaded successfully.")
except Exception as e:
    print(f"Error loading ball_model: {e}")
    exit()
try:
    pose_model = YOLO('yolov8n-pose.pt')
    print("Pose estimation model (yolov8n-pose.pt) loaded successfully.")
except Exception as e:
    print(f"Error loading pose_model: {e}")
    exit()

video_path = "footage/freethrows2.mp4"  # Ensure this path is correct
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path} with any backend.")
        exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)  # Renamed to avoid conflict
if video_fps == 0 or video_fps > 1000:
    print(f"Warning: Video FPS reported as {video_fps}, defaulting to 30 FPS for output.")
    video_fps = 30

output_filename_suffix = "_ccs_trajectory_fit"
output_path = f"outputs/freethrow_speed_arc_5frames{output_filename_suffix}.avi"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), video_fps, (frame_width, frame_height))
if not out.isOpened():
    print(f"Error: Could not open video writer for {output_path}")
    cap.release()
    exit()

print(f"Processing video: {video_path} (FPS: {video_fps})")
print(f"Outputting to: {output_path}")

TARGET_BALL_CLASS_NAME = "ball"
id_of_ball = 0

# --- State Variables ---
release_detected_in_shot = False
release_frame_info_dict = None
potential_release_buffer = []
trajectory_points_pixels_undistorted = []

trajectory_points_ccs = []
Zc_at_release = None

estimated_speed_mps_naive = 0.0
estimated_angle_deg_elev_ccs_naive = 0.0

# --- NEW: Variables for fitted results ---
fitted_speed_mps = 0.0
fitted_angle_deg_elev_ccs = 0.0
fitted_V0_ccs = None  # Store the fitted initial velocity vector
MAX_FRAMES_FOR_FIT = 30  # Use up to this many frames after release for fitting
# ---
map1, map2 = None, None
if DETECT_ON_FULLY_UNDISTORTED_FRAME:
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (frame_width, frame_height), cv2.CV_16SC2)

frame_count = 0
shot_count = 0

while cap.isOpened():
    ret, frame_orig = cap.read()
    if not ret: break
    frame_count += 1
    if frame_count % 30 == 0: print(f"Processing frame {frame_count}...")

    # ... (frame_to_detect_on, ball & pose detection, point undistortion logic as before) ...
    frame_to_detect_on = frame_orig
    current_frame_annotated = frame_orig.copy()
    if DETECT_ON_FULLY_UNDISTORTED_FRAME and map1 is not None:
        frame_to_detect_on = cv2.remap(frame_orig, map1, map2, interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)
        current_frame_annotated = frame_to_detect_on.copy()

    ball_results_list = ball_model(frame_to_detect_on, conf=0.6, verbose=False, classes=[id_of_ball])
    ball_bbox_from_detection = utils.get_ball_bbox(ball_results_list, id_of_ball)

    ball_center_undistorted_px = None
    if ball_bbox_from_detection is not None:
        center_x_detected = (ball_bbox_from_detection[0] + ball_bbox_from_detection[2]) / 2.0
        center_y_detected = (ball_bbox_from_detection[1] + ball_bbox_from_detection[3]) / 2.0
        if UNDISTORT_POINTS_AFTER_DETECTION and not DETECT_ON_FULLY_UNDISTORTED_FRAME:
            dist_pts = np.array([[[center_x_detected, center_y_detected]]], dtype=np.float32)
            undist_pts_px = cv2.fisheye.undistortPoints(dist_pts, K, D, P=K)
            if undist_pts_px is not None: ball_center_undistorted_px = (
            int(round(undist_pts_px[0][0][0])), int(round(undist_pts_px[0][0][1])))
        else:
            ball_center_undistorted_px = (int(round(center_x_detected)), int(round(center_y_detected)))

    pose_results_list = pose_model(frame_to_detect_on, conf=0.35, verbose=False)
    shooter_wrist_pos_processed = None
    if ball_bbox_from_detection is not None:
        shooter_wrist_pos_detected, _ = utils.get_likely_shooter_wrist_and_person_idx(
            pose_results_list, ball_bbox_from_detection, SHOOTING_WRIST_KP_INDEX,
            MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
        )
        if shooter_wrist_pos_detected is not None:
            if UNDISTORT_POINTS_AFTER_DETECTION and not DETECT_ON_FULLY_UNDISTORTED_FRAME:
                wrist_dist_pts = np.array([[[shooter_wrist_pos_detected[0], shooter_wrist_pos_detected[1]]]],
                                          dtype=np.float32)
                wrist_undist_px = cv2.fisheye.undistortPoints(wrist_dist_pts, K, D, P=K)
                if wrist_undist_px is not None: shooter_wrist_pos_processed = (
                int(round(wrist_undist_px[0][0][0])), int(round(wrist_undist_px[0][0][1])))
            else:
                shooter_wrist_pos_processed = (int(shooter_wrist_pos_detected[0]), int(shooter_wrist_pos_detected[1]))

    # Release Detection (mostly same, but reset fitted_V0_ccs too)
    if not release_detected_in_shot and ball_center_undistorted_px is not None and shooter_wrist_pos_processed is not None and ball_bbox_from_detection is not None:
        wrist_in_ball_bbox = utils.is_point_inside_bbox(shooter_wrist_pos_processed, ball_bbox_from_detection,
                                                        margin=WRIST_BALL_PROXIMITY_MARGIN)
        if not wrist_in_ball_bbox:
            # ... (potential_release_buffer append logic) ...
            potential_release_buffer.append({
                "frame_no": frame_count,
                "ball_center_undistorted_px": ball_center_undistorted_px,
                "ball_bbox_detection": ball_bbox_from_detection,
                "original_frame_at_this_moment": frame_orig.copy()
            })
            if len(potential_release_buffer) > CONFIRMATION_FRAMES_FOR_RELEASE: potential_release_buffer.pop(0)
            if len(potential_release_buffer) == CONFIRMATION_FRAMES_FOR_RELEASE:
                release_frame_info_dict = potential_release_buffer[0]
                release_detected_in_shot = True
                shot_count += 1
                estimated_speed_mps_naive = 0.0  # Reset naive
                # ... (reset other naive angle vars)
                fitted_speed_mps = 0.0  # Reset fitted
                fitted_angle_deg_elev_ccs = 0.0  # Reset fitted
                fitted_V0_ccs = None  # Reset fitted

                trajectory_points_pixels_undistorted = []
                trajectory_points_ccs = []
                Zc_at_release = None

                u_release, v_release = release_frame_info_dict['ball_center_undistorted_px']
                trajectory_points_pixels_undistorted.append((u_release, v_release))

                rel_ball_bbox_det = release_frame_info_dict['ball_bbox_detection']
                apparent_diameter_px_release = ((rel_ball_bbox_det[2] - rel_ball_bbox_det[0]) + (
                            rel_ball_bbox_det[3] - rel_ball_bbox_det[1])) / 2.0
                if apparent_diameter_px_release > 0:
                    avg_focal_length_px = (fx + fy) / 2.0
                    Zc_at_release = (BASKETBALL_REAL_DIAMETER_M * avg_focal_length_px) / apparent_diameter_px_release
                else:
                    Zc_at_release = 1.0

                xn_release = (u_release - cx) / fx
                yn_release = (v_release - cy) / fy
                Xc_release = xn_release * Zc_at_release
                Yc_release = yn_release * Zc_at_release
                Pc_release = np.array([Xc_release, Yc_release, Zc_at_release])
                trajectory_points_ccs.append({'frame': release_frame_info_dict['frame_no'], 'Pc': Pc_release})

                print(f"RELEASE DETECTED (Shot {shot_count}) at frame: {release_frame_info_dict['frame_no']}")
                # ... (print Zc, Pc_release) ...
                if SAVE_RELEASE_FRAMES:  # ... (save frame logic) ...
                    pass
        else:  # Wrist back in ball
            potential_release_buffer = []
            if release_detected_in_shot:  # If a shot was in progress and is now "cancelled"
                print(f"Shot {shot_count} release cancelled (wrist back in ball proximity).")
            release_detected_in_shot = False

    # Store Trajectory & Perform Estimations
    if release_detected_in_shot and ball_center_undistorted_px is not None and Zc_at_release is not None:
        current_u, current_v = ball_center_undistorted_px
        if not trajectory_points_pixels_undistorted or (current_u, current_v) != trajectory_points_pixels_undistorted[
            -1]:
            trajectory_points_pixels_undistorted.append((current_u, current_v))
            xn_current = (current_u - cx) / fx
            yn_current = (current_v - cy) / fy
            Xc_current = xn_current * Zc_at_release
            Yc_current = yn_current * Zc_at_release
            Pc_current = np.array([Xc_current, Yc_current, Zc_at_release])
            trajectory_points_ccs.append({'frame': frame_count, 'Pc': Pc_current})
            print(f"CCS point size: {len(trajectory_points_ccs)}")

            # Naive Estimation (as before, for comparison)
            if estimated_speed_mps_naive == 0.0 and len(trajectory_points_ccs) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
                # ... (naive estimation logic using trajectory_points_ccs[0] and trajectory_points_ccs[FRAMES_FOR_NAIVE_ESTIMATION]) ...
                # For brevity, assuming it's there and works.
                pass

            # --- NEW: Trajectory Fitting ---
            # Perform fit once enough points are collected, or at the end of a segment if ball is lost
            # For now, let's try fitting when we have a certain number of points if not already fitted.
            if fitted_V0_ccs is None and len(trajectory_points_ccs) >= MAX_FRAMES_FOR_FIT:  # Or some other condition
                print(
                    f"Attempting trajectory fit for Shot {shot_count} with {len(trajectory_points_ccs)} CCS points...")
                segment_to_fit = trajectory_points_ccs[:MAX_FRAMES_FOR_FIT]  # Use a segment

                V0_fit, P0_fit, pcov_fit = utils.fit_trajectory_ccs(segment_to_fit, video_fps)

                if V0_fit is not None:
                    fitted_V0_ccs = V0_fit
                    fitted_speed_mps = np.linalg.norm(fitted_V0_ccs)

                    # Elevation angle from camera's XZ plane
                    # Using -V0_fit[1] because positive Yc is assumed physically down for gravity model
                    magnitude_in_XZ_plane_fit = math.sqrt(fitted_V0_ccs[0] ** 2 + fitted_V0_ccs[2] ** 2)
                    if magnitude_in_XZ_plane_fit > 1e-6:  # Avoid division by zero if Vcx and Vcz are zero
                        fitted_angle_rad_elev_ccs = math.atan2(-fitted_V0_ccs[1], magnitude_in_XZ_plane_fit)
                        fitted_angle_deg_elev_ccs = math.degrees(fitted_angle_rad_elev_ccs)
                    else:  # Purely vertical shot in camera Y
                        fitted_angle_deg_elev_ccs = 90.0 if -fitted_V0_ccs[1] > 0 else -90.0 if -fitted_V0_ccs[
                            1] < 0 else 0.0

                    print(f"  TRAJECTORY FIT (Shot {shot_count}, {len(segment_to_fit)} pts):")
                    print(f"    Fitted V0_ccs (m/s): {fitted_V0_ccs.round(2)}")
                    print(f"    Fitted P0_ccs (m): {P0_fit.round(3)} (should be close to first point)")
                    print(f"    Fitted Speed: {fitted_speed_mps:.2f} m/s")
                    print(f"    Fitted Elev Angle: {fitted_angle_deg_elev_ccs:.2f} deg")
                else:
                    print(f"  Trajectory fit failed for Shot {shot_count}.")
            # --- END NEW ---

    # --- Annotations (on current_frame_annotated) ---
    # ... (ball bbox, undistorted center, pose, wrist as before) ...
    if ball_bbox_from_detection is not None: cv2.rectangle(current_frame_annotated, (
    int(ball_bbox_from_detection[0]), int(ball_bbox_from_detection[1])), (int(ball_bbox_from_detection[2]),
                                                                          int(ball_bbox_from_detection[3])),
                                                           (255, 0, 0), 1)
    if ball_center_undistorted_px is not None: cv2.circle(current_frame_annotated, ball_center_undistorted_px, 5,
                                                          (0, 0, 255), -1)
    if pose_results_list and len(pose_results_list) > 0: current_frame_annotated = pose_results_list[0].plot(
        img=current_frame_annotated, boxes=False, labels=False)
    if shooter_wrist_pos_processed is not None: cv2.circle(current_frame_annotated, shooter_wrist_pos_processed, 7,
                                                           (0, 255, 255), -1)

    if release_detected_in_shot and len(trajectory_points_pixels_undistorted) >= 2:
        points_np = np.array(trajectory_points_pixels_undistorted, dtype=np.int32)
        cv2.polylines(current_frame_annotated, [points_np], isClosed=False, color=(0, 255, 0), thickness=2)

    if release_detected_in_shot:
        if release_frame_info_dict:
            cv2.putText(current_frame_annotated, f"S{shot_count} Rls:Fr{release_frame_info_dict['frame_no']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if trajectory_points_pixels_undistorted:
            cv2.circle(current_frame_annotated, trajectory_points_pixels_undistorted[0], 15, (0, 0, 255), 2)

        # Display Naive estimates (if calculated, for comparison)
        # if estimated_speed_mps_naive > 0: ...

        # Display Fitted estimates
        if fitted_speed_mps > 0:
            cv2.putText(current_frame_annotated, f"Fit Speed: {fitted_speed_mps:.2f}m/s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)  # Orange for fitted
            cv2.putText(current_frame_annotated, f"Fit Elev: {fitted_angle_deg_elev_ccs:.1f}deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow("CCS Trajectory Fit", current_frame_annotated)
    out.write(current_frame_annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished processing. Output saved to {output_path}")