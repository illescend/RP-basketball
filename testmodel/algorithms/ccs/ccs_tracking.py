import math
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0" # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100" # Try to prioritize FFMPEG
import numpy as np
import cv2
from ultralytics import YOLO
from testmodel import utils  # Assuming your utils.py is in 'testmodel' directory

# --- Global Constants and Configuration ---
# Camera Intrinsics
K = np.array([
    [1134.4909447935638, 0.0, 968.5293267526777],
    [0.0, 1133.2194561973472, 553.0426928367185],
    [0.0, 0.0, 1.0]
])
D = np.array([  # Fisheye
    0.43466848378748985, -0.6373157421587791, 2.297946857510941, -2.0968503612499836
])
fx = K[0, 0];
fy = K[1, 1];
cx = K[0, 2];
cy = K[1, 2]

# Utils Constants
SHOOTING_WRIST_KP_INDEX = utils.SHOOTING_WRIST_KP_INDEX
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
WRIST_BALL_PROXIMITY_MARGIN = utils.WRIST_BALL_PROXIMITY_MARGIN
CONFIRMATION_FRAMES_FOR_RELEASE = utils.CONFIRMATION_FRAMES_FOR_RELEASE

# Script Configuration
SAVE_RELEASE_FRAMES = True
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RELEASE_FRAME_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "ccs",
                                        "release_frames_refactored_ccs")  # CCS specific dir

PRE_RELEASE_BALL_CONF = 0.6
POST_RELEASE_BALL_CONF = 0.25
MAX_PIXEL_DISPLACEMENT_POST_RELEASE = 100  # Max pixel distance for low-conf undistorted point
FRAMES_FOR_NAIVE_ESTIMATION = 5
BASKETBALL_REAL_DIAMETER_M = 0.24
TARGET_BALL_CLASS_NAME = "ball"

# Processing Mode
UNDISTORT_POINTS_AFTER_DETECTION = True  # Recommended for CCS: detect on original, undistort points
DETECT_ON_FULLY_UNDISTORTED_FRAME = True


# --- Helper Function for Undistorting a Single Point (Copied from your undistorted.py) ---
def undistort_point(point_xy, K_matrix, D_coeffs, is_fisheye=True):
    if point_xy is None: return None
    distorted_np = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    if is_fisheye:
        undistorted_np = cv2.fisheye.undistortPoints(distorted_np, K_matrix, D_coeffs, P=K_matrix)
    else:
        undistorted_np = cv2.undistortPoints(distorted_np, K_matrix, D_coeffs, P=K_matrix)
    if undistorted_np is not None: return (int(round(undistorted_np[0][0][0])), int(round(undistorted_np[0][0][1])))
    return None


# --- Refactored Functions (Adapted for CCS) ---
def preprocess_frame_ccs(frame_orig, K_matrix, D_coeffs_val, map1_val, map2_val, do_full_undistort_val):
    if do_full_undistort_val and map1_val is not None and map2_val is not None:
        return cv2.remap(frame_orig, map1_val, map2_val, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return frame_orig


def get_processed_detections_ccs(current_frame_num_val, frame_to_detect_on_val, ball_model_obj, pose_model_obj,
                                 id_ball_class_val, current_ball_conf_val, K_matrix_val, D_coeffs_val,
                                 do_undistort_points_val, is_full_frame_undistorted_already,
                                 current_trajectory_pixels_undistorted_val,  # Used for proximity check
                                 max_pixel_disp_val, is_release_detected_currently):
    """
    Runs detections, processes ball/wrist points including optional undistortion and proximity check.
    Returns:
        ball_center_undistorted_px_final: (u,v) final accepted undistorted ball center in pixels
        ball_bbox_for_logic_val: bbox used for logic (e.g. IOU, radius at release), from detection frame
        shooter_wrist_undistorted_px_final: (u,v) final accepted undistorted wrist in pixels
        used_low_conf_heuristic_val_flag: bool
    """
    ball_results = ball_model_obj(frame_to_detect_on_val, conf=current_ball_conf_val, verbose=False,
                                  classes=[id_ball_class_val])
    ball_bbox_detected = utils.get_ball_bbox(ball_results, id_ball_class_val)

    ball_center_undistorted_px_final = None
    ball_bbox_for_logic_val = ball_bbox_detected  # Default to this, might be nulled
    used_low_conf_heuristic_val_flag = False

    if ball_bbox_detected is not None:
        center_x_det = (ball_bbox_detected[0] + ball_bbox_detected[2]) / 2.0
        center_y_det = (ball_bbox_detected[1] + ball_bbox_detected[3]) / 2.0

        # Get the processed (potentially undistorted) center of the current detection
        temp_ball_center_processed_px = None
        if do_undistort_points_val and not is_full_frame_undistorted_already:
            temp_ball_center_processed_px = undistort_point((center_x_det, center_y_det), K_matrix_val, D_coeffs_val)
        else:
            temp_ball_center_processed_px = (int(round(center_x_det)), int(round(center_y_det)))

        if temp_ball_center_processed_px is not None:
            if is_release_detected_currently and len(current_trajectory_pixels_undistorted_val) > 0:
                last_known_undist_px = current_trajectory_pixels_undistorted_val[-1]
                dist_sq = (temp_ball_center_processed_px[0] - last_known_undist_px[0]) ** 2 + \
                          (temp_ball_center_processed_px[1] - last_known_undist_px[1]) ** 2
                if dist_sq <= max_pixel_disp_val ** 2:
                    ball_center_undistorted_px_final = temp_ball_center_processed_px
                    if current_ball_conf_val == POST_RELEASE_BALL_CONF:  # Heuristic used if low conf was active
                        used_low_conf_heuristic_val_flag = True
                else:  # Rejected by proximity
                    ball_bbox_for_logic_val = None  # Nullify if center is rejected
            else:  # Pre-release or first point post-release (no previous to check against)
                ball_center_undistorted_px_final = temp_ball_center_processed_px
        else:  # Undistortion failed for ball center
            ball_bbox_for_logic_val = None
    # else: No ball detected at current_ball_conf_val

    # Pose and Wrist
    pose_results_val = pose_model_obj(frame_to_detect_on_val, conf=0.35, verbose=False)
    shooter_wrist_undistorted_px_final = None
    if ball_bbox_for_logic_val is not None:  # Only proceed if we have a valid ball bbox for context
        wrist_detected_val, _ = utils.get_likely_shooter_wrist_and_person_idx(
            pose_results_val, ball_bbox_for_logic_val, SHOOTING_WRIST_KP_INDEX,
            MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
        )
        if wrist_detected_val is not None:
            if do_undistort_points_val and not is_full_frame_undistorted_already:
                shooter_wrist_undistorted_px_final = undistort_point(wrist_detected_val, K_matrix_val, D_coeffs_val)
            else:
                shooter_wrist_undistorted_px_final = (int(wrist_detected_val[0]), int(wrist_detected_val[1]))

    return ball_center_undistorted_px_final, ball_bbox_for_logic_val, shooter_wrist_undistorted_px_final, used_low_conf_heuristic_val_flag


def check_for_release_ccs(frame_num_val, ball_center_undist_px, ball_bbox_l_val, wrist_pos_undist_px,
                          current_release_state_val, release_buffer_list_val, frame_orig_val):
    global release_frame_info_dict, shot_count, Zc_at_release  # Globals modified
    global trajectory_points_pixels_undistorted, trajectory_points_ccs  # Globals reset
    global estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs  # Globals reset

    new_release_state_val = current_release_state_val
    if not current_release_state_val and ball_center_undist_px is not None and \
            wrist_pos_undist_px is not None and ball_bbox_l_val is not None:
        wrist_in_ball = utils.is_point_inside_bbox(wrist_pos_undist_px, ball_bbox_l_val,
                                                   margin=WRIST_BALL_PROXIMITY_MARGIN)
        if not wrist_in_ball:
            release_buffer_list_val.append({
                "frame_no": frame_num_val,
                "ball_center_undistorted_px": ball_center_undist_px,
                "ball_bbox_detection": ball_bbox_l_val,
                "original_frame_at_this_moment": frame_orig_val.copy()
            })
            if len(release_buffer_list_val) > CONFIRMATION_FRAMES_FOR_RELEASE: release_buffer_list_val.pop(0)
            if len(release_buffer_list_val) == CONFIRMATION_FRAMES_FOR_RELEASE:
                release_frame_info_dict = release_buffer_list_val[0]
                new_release_state_val = True
                shot_count += 1
                estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs = 0.0, 0.0, 0.0
                trajectory_points_pixels_undistorted.clear();
                trajectory_points_ccs.clear()
                Zc_at_release = None

                u_release, v_release = release_frame_info_dict['ball_center_undistorted_px']
                trajectory_points_pixels_undistorted.append((u_release, v_release))

                rel_bbox_det = release_frame_info_dict['ball_bbox_detection']
                app_diam_px_rel = ((rel_bbox_det[2] - rel_bbox_det[0]) + (rel_bbox_det[3] - rel_bbox_det[1])) / 2.0
                if app_diam_px_rel > 0:
                    avg_focal_px = (fx + fy) / 2.0
                    Zc_at_release = (BASKETBALL_REAL_DIAMETER_M * avg_focal_px) / app_diam_px_rel
                else:
                    Zc_at_release = 1.0  # Fallback

                xn_rel = (u_release - cx) / fx;
                yn_rel = (v_release - cy) / fy
                Xc_rel = xn_rel * Zc_at_release;
                Yc_rel = yn_rel * Zc_at_release
                Pc_rel = np.array([Xc_rel, Yc_rel, Zc_at_release])
                trajectory_points_ccs.append({'frame': release_frame_info_dict['frame_no'], 'Pc': Pc_rel})

                print(f"RELEASE DETECTED (Shot {shot_count}) at frame: {release_frame_info_dict['frame_no']}")
                print(f"  Zc_at_release: {Zc_at_release:.3f}m, Pc_release (CCS): {Pc_rel.round(3)}")
                if SAVE_RELEASE_FRAMES:  # Save frame logic...
                    pass
        else:
            release_buffer_list_val.clear()
            if current_release_state_val: print(f"  Frame {frame_num_val}: Shot {shot_count} release cancelled.")
            new_release_state_val = False
    elif current_release_state_val and ball_center_undist_px is None:  # Post-release but ball lost
        pass  # Keep release_detected_in_shot True
    return new_release_state_val, release_buffer_list_val


def update_trajectory_and_estimate_ccs(current_release_state_val, ball_center_undist_px_val, frame_num_val,
                                       traj_pix_list_undist_val, traj_ccs_list_val,
                                       current_Zc_at_release, video_fps_val):
    global estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs  # Globals

    if current_release_state_val and ball_center_undist_px_val is not None and current_Zc_at_release is not None:
        u_curr, v_curr = ball_center_undist_px_val
        if not traj_pix_list_undist_val or (u_curr, v_curr) != traj_pix_list_undist_val[-1]:
            traj_pix_list_undist_val.append((u_curr, v_curr))

            xn_curr = (u_curr - cx) / fx;
            yn_curr = (v_curr - cy) / fy
            Xc_curr = xn_curr * current_Zc_at_release  # Assume Zc = Zc_at_release
            Yc_curr = yn_curr * current_Zc_at_release
            Pc_curr = np.array([Xc_curr, Yc_curr, current_Zc_at_release])
            traj_ccs_list_val.append({'frame': frame_num_val, 'Pc': Pc_curr})
            print(f"frame: {frame_num_val}, Pc: {Pc_curr}")

            if estimated_speed_mps == 0.0 and len(traj_ccs_list_val) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
                p_start_ccs_info = traj_ccs_list_val[0]
                p_end_ccs_info = traj_ccs_list_val[FRAMES_FOR_NAIVE_ESTIMATION]
                Pc_s = p_start_ccs_info['Pc'];
                Pc_e = p_end_ccs_info['Pc']
                delta_Pc_val = Pc_e - Pc_s
                d_frames = p_end_ccs_info['frame'] - p_start_ccs_info['frame']
                if video_fps_val > 0 and d_frames > 0:
                    d_t_s = d_frames / video_fps_val
                    Vc_avg_val = delta_Pc_val / d_t_s
                    estimated_speed_mps = np.linalg.norm(Vc_avg_val)
                    estimated_angle_deg_XY_ccs = math.degrees(math.atan2(-Vc_avg_val[1], Vc_avg_val[0]))
                    mag_XZ = math.sqrt(Vc_avg_val[0] ** 2 + Vc_avg_val[2] ** 2)
                    estimated_angle_deg_elev_ccs = math.degrees(math.atan2(-Vc_avg_val[1], mag_XZ))
                    print(
                        f"  NAIVE CCS (Shot {shot_count}): Speed={estimated_speed_mps:.2f}m/s, Elev={estimated_angle_deg_elev_ccs:.1f}deg")


def draw_annotations_on_frame_ccs(frame_to_annotate_val, ball_bbox_d_val, ball_center_undist_px_val,
                                  wrist_pos_undist_px_val,
                                  is_release_detected_val, release_info_dict_val, traj_pix_list_undist_val,
                                  shot_num_val, est_speed_val, est_angle_elev_val, used_low_conf_flag_val):
    # Drawing logic using the passed parameters, similar to undistorted.py version
    if ball_bbox_d_val is not None:
        l, t, r, b = [int(c) for c in ball_bbox_d_val]
        color = (0, 165, 255) if used_low_conf_flag_val else (255, 0, 0)
        cv2.rectangle(frame_to_annotate_val, (l, t), (r, b), color, 1)
    if ball_center_undist_px_val is not None: cv2.circle(frame_to_annotate_val, ball_center_undist_px_val, 5,
                                                         (0, 0, 255), -1)
    if wrist_pos_undist_px_val is not None: cv2.circle(frame_to_annotate_val, tuple(wrist_pos_undist_px_val), 7,
                                                       (0, 255, 255), -1)
    if is_release_detected_val and len(traj_pix_list_undist_val) >= 2:
        pts_np = np.array(traj_pix_list_undist_val, dtype=np.int32)
        cv2.polylines(frame_to_annotate_val, [pts_np], False, (0, 255, 0), 2)
    if is_release_detected_val:
        if release_info_dict_val: cv2.putText(frame_to_annotate_val,
                                              f"S{shot_num_val} Rls:Fr{release_info_dict_val['frame_no']}", (10, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if traj_pix_list_undist_val and traj_pix_list_undist_val[0] is not None: cv2.circle(frame_to_annotate_val,
                                                                                            traj_pix_list_undist_val[0],
                                                                                            15, (0, 0, 255), 2)
        if est_speed_val > 0:
            cv2.putText(frame_to_annotate_val, f"Speed(CCS): {est_speed_val:.2f}m/s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(frame_to_annotate_val, f"Elev(CCS): {est_angle_elev_val:.1f}deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    return frame_to_annotate_val


# --- Main Processing Loop ---
if __name__ == "__main__":
    os.makedirs(RELEASE_FRAME_OUTPUT_DIR, exist_ok=True)

    MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "Yolo-Weights")
    BALL_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "model2.pt")
    # POSE_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "yolov8n-pose.pt") # Assuming pose model is in Yolo-Weights too
    POSE_MODEL_PATH = 'yolov8n-pose.pt'  # Or if it's downloaded by ultralytics directly

    VIDEO_INPUT_DIR = os.path.join(BASE_DIR, "footage")
    VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "ccs", "output_video")
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    video_path = os.path.join(VIDEO_INPUT_DIR, "freethrow_arc2.mp4")  # Example, change as needed
    output_filename_suffix = "_refactored_debug"
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"freethrow_arc2{output_filename_suffix}.avi")

    ball_model_g = YOLO(BALL_MODEL_PATH)  # Global models
    pose_model_g = YOLO(POSE_MODEL_PATH)

    cap_g = cv2.VideoCapture(video_path)  # Global cap
    if not cap_g.isOpened(): cap_g = cv2.VideoCapture(video_path, cv2.CAP_MSMF)  # Fallback
    if not cap_g.isOpened(): exit(f"Error opening video: {video_path}")

    frame_width_g = int(cap_g.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps_g = cap_g.get(cv2.CAP_PROP_FPS)
    if video_fps_g == 0 or video_fps_g > 1000: video_fps_g = 30.0

    out_g = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), video_fps_g, (frame_width_g, frame_height_g))
    if not out_g.isOpened(): exit(f"Error opening video writer: {output_path}")

    id_of_ball_g = 0  # Assuming id_of_ball is correctly identified globally

    map1_g, map2_g = None, None
    if DETECT_ON_FULLY_UNDISTORTED_FRAME:
        map1_g, map2_g = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (frame_width_g, frame_height_g),
                                                             cv2.CV_16SC2)

    # --- Main loop state variables (reset per video if processing multiple) ---
    frame_count = 0;
    shot_count = 0
    release_detected_in_shot = False;
    release_frame_info_dict = None
    potential_release_buffer = [];
    trajectory_points_pixels_undistorted = []
    trajectory_points_ccs = [];
    Zc_at_release = None
    estimated_speed_mps = 0.0;
    estimated_angle_deg_XY_ccs = 0.0;
    estimated_angle_deg_elev_ccs = 0.0

    print(f"Processing video: {video_path} (FPS: {video_fps_g})")
    print(f"Outputting to: {output_path}")

    while cap_g.isOpened():
        ret, frame_original_main = cap_g.read()
        if not ret: break
        frame_count += 1
        if frame_count % 10 == 0:
            print(
                f"Frame {frame_count}, Shot: {shot_count}, Release: {release_detected_in_shot}, Traj Px Pts: {len(trajectory_points_pixels_undistorted)}, Traj CCS Pts: {len(trajectory_points_ccs)}")

        current_frame_to_detect = preprocess_frame_ccs(frame_original_main, K, D, map1_g, map2_g,
                                                       DETECT_ON_FULLY_UNDISTORTED_FRAME)
        annotated_frame = current_frame_to_detect.copy()

        active_conf_thresh = POST_RELEASE_BALL_CONF if release_detected_in_shot else PRE_RELEASE_BALL_CONF

        ball_center_undist_px, ball_bbox_l, shooter_wrist_undist_px, used_low_conf_h = \
            get_processed_detections_ccs(frame_count, current_frame_to_detect, ball_model_g, pose_model_g, id_of_ball_g,
                                         active_conf_thresh, K, D,
                                         UNDISTORT_POINTS_AFTER_DETECTION, DETECT_ON_FULLY_UNDISTORTED_FRAME,
                                         trajectory_points_pixels_undistorted, MAX_PIXEL_DISPLACEMENT_POST_RELEASE,
                                         release_detected_in_shot)

        release_detected_in_shot, potential_release_buffer = \
            check_for_release_ccs(frame_count, ball_center_undist_px, ball_bbox_l, shooter_wrist_undist_px,
                                  release_detected_in_shot, potential_release_buffer, frame_original_main)

        # Note: update_trajectory_and_estimate_ccs modifies global estimation variables
        update_trajectory_and_estimate_ccs(release_detected_in_shot, ball_center_undist_px, frame_count,
                                           trajectory_points_pixels_undistorted, trajectory_points_ccs,
                                           Zc_at_release, video_fps_g)

        annotated_frame = draw_annotations_on_frame_ccs(annotated_frame, ball_bbox_l, ball_center_undist_px,
                                                        shooter_wrist_undist_px, release_detected_in_shot,
                                                        release_frame_info_dict, trajectory_points_pixels_undistorted,
                                                        shot_count, estimated_speed_mps, estimated_angle_deg_elev_ccs,
                                                        # Pass elev angle
                                                        used_low_conf_h)

        cv2.imshow("Refactored CCS Analysis", annotated_frame)
        out_g.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): print("Processing stopped by user."); break

    cap_g.release()
    out_g.release()
    cv2.destroyAllWindows()
    print(f"Finished processing. Output saved to {output_path}")