import math
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0" # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100" # Try to prioritize FFMPEG
import numpy as np
import cv2
from ultralytics import YOLO
from testmodel import utils  # Assuming your utils.py contains the helper functions

# --- Global Constants and Configuration ---
# Camera Intrinsics
K = np.array([
    [1134.4909447935638, 0.0, 968.5293267526777],
    [0.0, 1133.2194561973472, 553.0426928367185],
    [0.0, 0.0, 1.0]
])
D = np.array([
    0.43466848378748985, -0.6373157421587791, 2.297946857510941, -2.0968503612499836
])

# Utils Constants (ensure these are defined in utils.py or here)
SHOOTING_WRIST_KP_INDEX = utils.SHOOTING_WRIST_KP_INDEX
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
WRIST_BALL_PROXIMITY_MARGIN = utils.WRIST_BALL_PROXIMITY_MARGIN
CONFIRMATION_FRAMES_FOR_RELEASE = utils.CONFIRMATION_FRAMES_FOR_RELEASE

# Script Configuration
SAVE_RELEASE_FRAMES = True
# Determine the base directory of the project dynamically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Adjust if script location changes
RELEASE_FRAME_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "undistortion", "release_frames_refactored_debug")

PRE_RELEASE_BALL_CONF = 0.35
POST_RELEASE_BALL_CONF = 0.25  # Confidence for initial detection post-release
MAX_PIXEL_DISPLACEMENT_POST_RELEASE = 100
FRAMES_FOR_NAIVE_ESTIMATION = 5
BASKETBALL_REAL_RADIUS_M = 0.12
TARGET_BALL_CLASS_NAME = "ball"

# Processing Mode (Set one to True)
UNDISTORT_POINTS_AFTER_DETECTION = True
DETECT_ON_FULLY_UNDISTORTED_FRAME = True  # Set to True or False as needed for your test run

#MANUAL RELEASE OVERRIDE
MANUAL_RELEASE_FRAME_OVERRIDE = 180  # Set to an integer frame number to override, e.g., 150
MANUAL_RELEASE_SHOT_NUMBER_TARGET = 1 # Which shot number this override applies to (1st shot, 2nd shot, etc.)


# --- Helper Function for Undistorting a Single Point ---
def undistort_point(point_xy, K_matrix, D_coeffs, is_fisheye=True):
    if point_xy is None:
        return None
    distorted_np = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    if is_fisheye:
        undistorted_np = cv2.fisheye.undistortPoints(distorted_np, K_matrix, D_coeffs, P=K_matrix)
    else:
        undistorted_np = cv2.undistortPoints(distorted_np, K_matrix, D_coeffs, P=K_matrix)

    if undistorted_np is not None and len(undistorted_np) > 0:
        return (int(round(undistorted_np[0][0][0])), int(round(undistorted_np[0][0][1])))
    # print("      undistort_point: Undistortion failed or returned empty.")
    return None


# --- Refactored Functions ---
def preprocess_frame(frame_orig, K_matrix, D_coeffs, map1_val, map2_val, do_full_undistort):
    if do_full_undistort and map1_val is not None and map2_val is not None:
        return cv2.remap(frame_orig, map1_val, map2_val, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return frame_orig


def get_processed_detections(current_frame_num, frame_to_detect_on_val, ball_model_obj, pose_model_obj,
                             id_ball_class_val,
                             current_ball_conf_val, K_matrix, D_coeffs_val,
                             do_undistort_points, is_full_frame_undistorted_val,
                             current_trajectory_pixels, max_pixel_disp_low_conf_val,
                             is_release_detected_val):  # Added is_release_detected_val for context
    """
    Runs detections, processes ball/wrist points including optional undistortion and proximity check.
    """
    ball_results = ball_model_obj(frame_to_detect_on_val, conf=current_ball_conf_val, verbose=False,
                                  classes=[id_ball_class_val])
    ball_bbox_detected_val = utils.get_ball_bbox(ball_results,
                                                 id_ball_class_val)  # This gets highest conf above current_ball_conf_val

    ball_center_final_processed = None
    ball_bbox_for_logic_final = ball_bbox_detected_val  # Initially assume detected bbox is for logic
    used_low_conf_heuristic_flag = False  # In this dynamic conf setup, this flag is less distinct

    print(f"  Frame {current_frame_num}: get_processed - Initial ball_bbox_detected_val: {ball_bbox_detected_val} (conf_thresh={current_ball_conf_val:.2f})")

    if ball_bbox_detected_val is not None:
        center_x_det = (ball_bbox_detected_val[0] + ball_bbox_detected_val[2]) / 2.0
        center_y_det = (ball_bbox_detected_val[1] + ball_bbox_detected_val[3]) / 2.0

        temp_ball_center_after_undistort = None
        if do_undistort_points and not is_full_frame_undistorted_val:
            temp_ball_center_after_undistort = undistort_point((center_x_det, center_y_det), K_matrix, D_coeffs_val)
            # if temp_ball_center_after_undistort is None: print(f"    Frame {current_frame_num}: Ball center undistortion failed.")
        else:
            temp_ball_center_after_undistort = (int(round(center_x_det)), int(round(center_y_det)))

        print(f"    Frame {current_frame_num}: temp_ball_center_after_undistort: {temp_ball_center_after_undistort}")

        if temp_ball_center_after_undistort is not None:
            if is_release_detected_val and len(current_trajectory_pixels) > 0:
                # This is a post-release frame, and we have a previous point to check against
                last_known = current_trajectory_pixels[-1]
                dist_sq = (temp_ball_center_after_undistort[0] - last_known[0]) ** 2 + \
                          (temp_ball_center_after_undistort[1] - last_known[1]) ** 2
                print(f"    Frame {current_frame_num}: Post-release proximity check: dist_sq={dist_sq:.1f} vs threshold_sq={max_pixel_disp_low_conf_val**2:.1f}")
                if dist_sq <= max_pixel_disp_low_conf_val ** 2:
                    ball_center_final_processed = temp_ball_center_after_undistort  # Accept
                    # 'used_low_conf_heuristic_flag' becomes true if POST_RELEASE_BALL_CONF was used.
                    if current_ball_conf_val == POST_RELEASE_BALL_CONF:
                        used_low_conf_heuristic_flag = True
                else:
                    ball_center_final_processed = None  # Reject by proximity
                    ball_bbox_for_logic_final = None
                    print(f"    Frame {current_frame_num}: REJECTED by proximity.")
            else:  # Pre-release, OR first point after release (no previous point for proximity), OR no temp_ball_center_after_undistort
                ball_center_final_processed = temp_ball_center_after_undistort  # Accept
        else:  # temp_ball_center_after_undistort is None (e.g., undistortion failed)
            ball_bbox_for_logic_final = None  # No valid center, so no valid logic bbox
    else:
        print(f"  Frame {current_frame_num}: No ball detected at conf={current_ball_conf_val:.2f}")

    # Pose and Wrist
    pose_results_val = pose_model_obj(frame_to_detect_on_val, conf=0.5, verbose=False) #0.35 works okayish
    shooter_wrist_final_processed = None
    if ball_bbox_for_logic_final is not None:
        wrist_detected_val, _ = utils.get_likely_shooter_wrist_and_person_idx(
            pose_results_val, ball_bbox_for_logic_final, SHOOTING_WRIST_KP_INDEX,
            MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
        )
        if wrist_detected_val is not None:
            if do_undistort_points and not is_full_frame_undistorted_val:
                shooter_wrist_final_processed = undistort_point(wrist_detected_val, K_matrix, D_coeffs_val)
                # if shooter_wrist_final_processed is None: print(f"    Frame {current_frame_num}: Wrist undistortion failed.")
            else:
                shooter_wrist_final_processed = (int(wrist_detected_val[0]), int(wrist_detected_val[1]))

    # print(f"  Frame {current_frame_num}: get_processed returns: ball_center={ball_center_final_processed}, ball_bbox={ball_bbox_for_logic_final}, wrist={shooter_wrist_final_processed}")
    return ball_center_final_processed, ball_bbox_for_logic_final, shooter_wrist_final_processed, used_low_conf_heuristic_flag


def check_for_release(frame_num_val, ball_center_p, ball_bbox_l, wrist_pos_p,
                      current_release_state, release_buffer_list, frame_orig_val,
                      manual_override_frame, manual_override_shot_target,
                      current_shot_count):
    global release_frame_info_dict, shot_count
    global estimated_speed_mps, estimated_angle_deg, trajectory_points_pixels, trajectory_points_scaled
    global origin_x_pxl, origin_y_pxl, ball_radius_pxl_at_release

    new_release_state = current_release_state  # Start with current state
    forced_release_this_frame = False

    # --- Manual Override Logic ---
    if not current_release_state and \
            manual_override_frame is not None and \
            current_shot_count + 1 == manual_override_shot_target and \
            frame_num_val == manual_override_frame:

        if ball_center_p is not None and ball_bbox_l is not None:
            print(f"--- MANUAL RELEASE OVERRIDE at frame {frame_num_val} for shot {manual_override_shot_target} ---")
            release_frame_info_dict = {
                "frame_no": frame_num_val,
                "ball_center_processed": ball_center_p,
                "ball_bbox_detection": ball_bbox_l,
                "original_frame_at_this_moment": frame_orig_val.copy()
            }
            new_release_state = True
            shot_count += 1
            # Reset trajectory and estimation variables
            estimated_speed_mps, estimated_angle_deg = 0.0, 0.0
            trajectory_points_pixels.clear();
            trajectory_points_scaled.clear()
            origin_x_pxl, origin_y_pxl = release_frame_info_dict['ball_center_processed']
            rel_bbox = release_frame_info_dict['ball_bbox_detection']
            width_px = rel_bbox[2] - rel_bbox[0];
            height_px = rel_bbox[3] - rel_bbox[1]
            ball_radius_pxl_at_release = (width_px + height_px) / 4.0
            if ball_radius_pxl_at_release < 1: ball_radius_pxl_at_release = 1.0
            trajectory_points_pixels.append((origin_x_pxl, origin_y_pxl))
            trajectory_points_scaled.append({'x': 0.0, 'y': 0.0, 'frame': release_frame_info_dict['frame_no']})
            print(f"MANUAL RELEASE (Shot {shot_count}) at frame: {release_frame_info_dict['frame_no']}")

            if SAVE_RELEASE_FRAMES:
                # Use the original frame that was stored when this event was added to buffer
                frame_to_save = release_frame_info_dict["original_frame_at_this_moment"]
                # Annotate this specific frame for saving
                annotated_release_frame_for_save = frame_to_save.copy()
                cv2.circle(annotated_release_frame_for_save, (origin_x_pxl, origin_y_pxl), 15, (0, 0, 255),
                           2)  # Release point
                if wrist_pos_p:  # Draw wrist used for logic if available
                    cv2.circle(annotated_release_frame_for_save, wrist_pos_p, 7, (0, 255, 255),
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

            forced_release_this_frame = True
        else:
            print(f"--- MANUAL RELEASE OVERRIDE SKIPPED at frame {frame_num_val}: Ball data not available. ---")

    # --- Automatic Release Detection Logic ---
    # Only run if not already in a released state from this frame's manual override
    # AND not already in a released state from previous frames (unless we allow re-triggering)
    if not new_release_state and not forced_release_this_frame and \
            ball_center_p is not None and wrist_pos_p is not None and ball_bbox_l is not None:
        wrist_in_ball = utils.is_point_inside_bbox(wrist_pos_p, ball_bbox_l, margin=WRIST_BALL_PROXIMITY_MARGIN)
        print(f"Frame: {frame_num_val}, wrist in ball: {wrist_in_ball}")
        if not wrist_in_ball:
            release_buffer_list.append({
                "frame_no": frame_num_val,
                "ball_center_processed": ball_center_p,
                "ball_bbox_detection": ball_bbox_l,
                "original_frame_at_this_moment": frame_orig_val.copy()
            })
            if len(release_buffer_list) > CONFIRMATION_FRAMES_FOR_RELEASE:
                release_buffer_list.pop(0)
            if len(release_buffer_list) == CONFIRMATION_FRAMES_FOR_RELEASE:
                release_frame_info_dict = release_buffer_list[0]
                new_release_state = True
                shot_count += 1
                # Reset trajectory and estimation variables
                estimated_speed_mps, estimated_angle_deg = 0.0, 0.0
                trajectory_points_pixels.clear();
                trajectory_points_scaled.clear()
                origin_x_pxl, origin_y_pxl = release_frame_info_dict['ball_center_processed']
                rel_bbox = release_frame_info_dict['ball_bbox_detection']
                width_px = rel_bbox[2] - rel_bbox[0];
                height_px = rel_bbox[3] - rel_bbox[1]
                ball_radius_pxl_at_release = (width_px + height_px) / 4.0
                if ball_radius_pxl_at_release < 1: ball_radius_pxl_at_release = 1.0
                trajectory_points_pixels.append((origin_x_pxl, origin_y_pxl))
                trajectory_points_scaled.append({'x': 0.0, 'y': 0.0, 'frame': release_frame_info_dict['frame_no']})
                print(f"AUTOMATIC RELEASE (Shot {shot_count}) at frame: {release_frame_info_dict['frame_no']}")
                # ... (Save frame logic as before) ...
        else:  # Wrist is in ball
            release_buffer_list.clear()
            # If it was previously thought to be released (current_release_state was True coming in)
            # and now wrist is back, this is a "cancellation" or end of the detected shot.
            # We should allow `new_release_state` to become `False` if `current_release_state` was true.
            # However, the global `release_detected_in_shot` in the main loop controls the overall phase.
            # This function's `new_release_state` return primarily signals if a *new* release *just occurred*.
            if current_release_state:  # It was released, but now wrist is back.
                print(f"  Frame {frame_num_val}: Wrist back in ball proximity for Shot {shot_count}. Clearing buffer.")
                # For now, we don't toggle new_release_state back to False here based on this alone.
                # The global `release_detected_in_shot` would be reset outside if needed for multiple shots.
                pass


    # This handles the case where a release was previously detected (current_release_state = True)
    # and the ball is now lost (ball_center_p is None). We want to maintain the release state.
    elif current_release_state and ball_center_p is None:
        # new_release_state is already True (from current_release_state), so do nothing.
        pass

    return new_release_state, release_buffer_list


def update_trajectory_and_estimate(current_release_state_val, ball_center_p_val, frame_num_val,
                                   traj_pix_list_val, traj_scaled_list_val,
                                   orig_x, orig_y, radius_release,
                                   video_fps_val):  # Removed current_est_speed, current_est_angle as they are global
    global estimated_speed_mps, estimated_angle_deg  # Indicate globals are used/modified

    if current_release_state_val and ball_center_p_val is not None:
        if not traj_pix_list_val or ball_center_p_val != traj_pix_list_val[-1]:
            traj_pix_list_val.append(ball_center_p_val)
            scaled_x = (ball_center_p_val[0] - orig_x) / radius_release
            scaled_y = (orig_y - ball_center_p_val[1]) / radius_release
            traj_scaled_list_val.append({'x': scaled_x, 'y': scaled_y, 'frame': frame_num_val})
            print(f"Scaled x: {scaled_x}, y: {scaled_y}")

            if estimated_speed_mps == 0.0 and len(traj_scaled_list_val) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
                p_start_idx, p_end_idx = 0, FRAMES_FOR_NAIVE_ESTIMATION
                if p_end_idx < len(traj_scaled_list_val):
                    # ... (naive estimation logic as before, sets global estimated_speed_mps, estimated_angle_deg) ...
                    p_start = traj_scaled_list_val[p_start_idx]
                    p_end = traj_scaled_list_val[p_end_idx]
                    d_scaled_x = p_end['x'] - p_start['x']
                    d_scaled_y = p_end['y'] - p_start['y']
                    real_dx_m = d_scaled_x * BASKETBALL_REAL_RADIUS_M
                    real_dy_m = d_scaled_y * BASKETBALL_REAL_RADIUS_M
                    d_frames = p_end['frame'] - p_start['frame']
                    if video_fps_val > 0 and d_frames > 0:
                        d_t_s = d_frames / video_fps_val
                        avg_vx_mps = real_dx_m / d_t_s
                        avg_vy_mps = real_dy_m / d_t_s
                        estimated_speed_mps = math.sqrt(avg_vx_mps ** 2 + avg_vy_mps ** 2)
                        estimated_angle_deg = math.degrees(math.atan2(avg_vy_mps, avg_vx_mps))
                        print(
                            f"  NAIVE ESTIMATION (Shot {shot_count}): Speed={estimated_speed_mps:.2f}m/s, Angle={estimated_angle_deg:.1f}deg")


def draw_annotations_on_frame(frame_to_annotate, ball_bbox_d, ball_center_p, wrist_pos_p,
                              is_release_detected_val, release_info_dict_val, traj_pix_list_val,
                              shot_num_val, est_speed_val, est_angle_val, used_low_conf_flag):
    # ... (drawing logic as before, ensure variable names match) ...
    # Drawing logic using the passed parameters
    if ball_bbox_d is not None:
        left, top, right, bottom = [int(c) for c in ball_bbox_d]
        color = (0, 165, 255) if used_low_conf_flag else (255, 0, 0)  # Orange for low-conf heuristic use
        cv2.rectangle(frame_to_annotate, (left, top), (right, bottom), color, 1)
        # label = f"Ball (LowConfH)" if used_low_conf_flag else f"Ball"
        # cv2.putText(frame_to_annotate, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)

    if ball_center_p is not None:
        cv2.circle(frame_to_annotate, ball_center_p, 5, (0, 0, 255), -1)

    if wrist_pos_p is not None:
        cv2.circle(frame_to_annotate, tuple(wrist_pos_p), 7, (0, 255, 255), -1)

    if is_release_detected_val and len(traj_pix_list_val) >= 2:
        points_np = np.array(traj_pix_list_val, dtype=np.int32)
        cv2.polylines(frame_to_annotate, [points_np], isClosed=False, color=(0, 255, 0), thickness=2)

    if is_release_detected_val:
        if release_info_dict_val:
            cv2.putText(frame_to_annotate, f"S{shot_num_val} Rls:Fr{release_info_dict_val['frame_no']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if traj_pix_list_val and traj_pix_list_val[0] is not None:  # Check if first point exists
            cv2.circle(frame_to_annotate, traj_pix_list_val[0], 15, (0, 0, 255), 2)
        if est_speed_val > 0:
            cv2.putText(frame_to_annotate, f"Speed: {est_speed_val:.2f}m/s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(frame_to_annotate, f"Angle: {est_angle_val:.1f}deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    return frame_to_annotate


# --- Main Processing Loop ---
if __name__ == "__main__":
    os.makedirs(RELEASE_FRAME_OUTPUT_DIR, exist_ok=True)  # Ensure output dir for release frames exists

    MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "Yolo-Weights")
    BALL_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "model2.pt")
    # POSE_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "yolov8n-pose.pt") # Assuming pose model is in Yolo-Weights too
    POSE_MODEL_PATH = 'yolov8n-pose.pt'  # Or if it's downloaded by ultralytics directly

    VIDEO_INPUT_DIR = os.path.join(BASE_DIR, "footage")
    VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "undistortion", "output_video")
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    video_path = os.path.join(VIDEO_INPUT_DIR, "freethrow_arc2.mp4")  # Example, change as needed
    output_filename_suffix = "_refactored_debug"
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"freethrow_arc2{output_filename_suffix}.avi")

    ball_model_g = YOLO(BALL_MODEL_PATH)
    pose_model_g = YOLO(POSE_MODEL_PATH)
    cap_g = cv2.VideoCapture(video_path)
    if not cap_g.isOpened(): cap_g = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
    if not cap_g.isOpened(): exit(f"Error opening video: {video_path}")

    frame_width_g = int(cap_g.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps_g = cap_g.get(cv2.CAP_PROP_FPS)
    if video_fps_g == 0 or video_fps_g > 1000: video_fps_g = 30.0

    out_g = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), video_fps_g, (frame_width_g, frame_height_g))
    if not out_g.isOpened(): exit(f"Error opening video writer: {output_path}")

    id_of_ball_g = 0

    map1_g, map2_g = None, None
    if DETECT_ON_FULLY_UNDISTORTED_FRAME:
        map1_g, map2_g = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (frame_width_g, frame_height_g),
                                                             cv2.CV_16SC2)

    frame_count = 0;
    shot_count = 0
    release_detected_in_shot = False;
    release_frame_info_dict = None
    potential_release_buffer = [];
    trajectory_points_pixels = []
    origin_x_pxl, origin_y_pxl = 0, 0;
    ball_radius_pxl_at_release = 1.0
    trajectory_points_scaled = [];
    estimated_speed_mps = 0.0;
    estimated_angle_deg = 0.0

    print(f"Processing video: {video_path} (FPS: {video_fps_g})")
    print(f"Outputting to: {output_path}")

    while cap_g.isOpened():
        ret, frame_original_main = cap_g.read()
        if not ret: break
        frame_count += 1
        if frame_count % 10 == 0:
            print(
                f"Frame {frame_count}, Shot: {shot_count}, Release: {release_detected_in_shot}, Traj Px Pts: {len(trajectory_points_pixels)}, Traj Scaled Pts: {len(trajectory_points_scaled)}")

        current_frame_to_detect = preprocess_frame(frame_original_main, K, D, map1_g, map2_g,
                                                   DETECT_ON_FULLY_UNDISTORTED_FRAME)
        annotated_frame = current_frame_to_detect.copy()

        active_conf_thresh = POST_RELEASE_BALL_CONF if release_detected_in_shot else PRE_RELEASE_BALL_CONF

        ball_center_p, ball_bbox_l, shooter_wrist_p, used_low_conf_h = \
            get_processed_detections(frame_count, current_frame_to_detect, ball_model_g, pose_model_g, id_of_ball_g,
                                     active_conf_thresh, K, D,
                                     UNDISTORT_POINTS_AFTER_DETECTION, DETECT_ON_FULLY_UNDISTORTED_FRAME,
                                     trajectory_points_pixels, MAX_PIXEL_DISPLACEMENT_POST_RELEASE,
                                     release_detected_in_shot)  # Pass current release state

        release_detected_in_shot, potential_release_buffer = check_for_release(frame_count, ball_center_p, ball_bbox_l, shooter_wrist_p,
                              release_detected_in_shot, potential_release_buffer, frame_original_main,
                              MANUAL_RELEASE_FRAME_OVERRIDE, MANUAL_RELEASE_SHOT_NUMBER_TARGET, shot_count)

        update_trajectory_and_estimate(release_detected_in_shot, ball_center_p, frame_count,
                                       trajectory_points_pixels, trajectory_points_scaled,
                                       origin_x_pxl, origin_y_pxl, ball_radius_pxl_at_release,
                                       video_fps_g)  # Removed est_speed, est_angle as they are global

        annotated_frame = draw_annotations_on_frame(annotated_frame, ball_bbox_l, ball_center_p,
                                                    shooter_wrist_p, release_detected_in_shot,
                                                    release_frame_info_dict, trajectory_points_pixels,
                                                    shot_count, estimated_speed_mps, estimated_angle_deg,
                                                    used_low_conf_h)

        cv2.imshow("Refactored Basketball Analysis", annotated_frame)
        out_g.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): print("Processing stopped by user."); break

    cap_g.release()
    out_g.release()
    cv2.destroyAllWindows()
    print(f"Finished processing. Output saved to {output_path}")