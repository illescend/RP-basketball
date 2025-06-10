import math
import os

os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"  # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100"  # Try to prioritize FFMPEG
import numpy as np
import cv2
from ultralytics import YOLO
from testmodel import utils

# --- Global Constants and Configuration ---
K = np.array([[1134.4909, 0.0, 968.5293], [0.0, 1133.2194, 553.0426], [0.0, 0.0, 1.0]])
D = np.array([0.43466, -0.63731, 2.29794, -2.09685])  # Fisheye
fx = K[0, 0];
fy = K[1, 1];
cx = K[0, 2];
cy = K[1, 2]

SHOOTING_WRIST_KP_INDEX = utils.SHOOTING_WRIST_KP_INDEX
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
WRIST_BALL_PROXIMITY_MARGIN = utils.WRIST_BALL_PROXIMITY_MARGIN
CONFIRMATION_FRAMES_FOR_RELEASE = utils.CONFIRMATION_FRAMES_FOR_RELEASE

SAVE_RELEASE_FRAMES = True
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# RELEASE_FRAME_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "ccs", "release_frames_kf_compare") # Original relative path
RELEASE_FRAME_OUTPUT_DIR = "outputs/release_frames_kf_compare"  # Effective path

PRE_RELEASE_BALL_CONF = 0.35
POST_RELEASE_BALL_CONF = 0.25
MAX_PIXEL_DISPLACEMENT_POST_RELEASE = 100
FRAMES_FOR_NAIVE_ESTIMATION = 5
BASKETBALL_REAL_DIAMETER_M = 0.24
TARGET_BALL_CLASS_NAME = "ball"

MANUAL_RELEASE_FRAME_OVERRIDE = 112  # 180 Set to a frame number or None
MANUAL_RELEASE_SHOT_NUMBER_TARGET = 1  # Which shot number to apply manual override
MAX_CONSECUTIVE_MISSES_KF = 5
KF_MEASUREMENT_NOISE_SCALE = 0.001 #Confidence in pixel detection
KF_PROCESS_NOISE_SCALE = 300  # Increased from 0.1 for more dynamic movement
NOISE_ACCELERATION_VARIANCE = 1536

UNDISTORT_POINTS_AFTER_DETECTION = True
DETECT_ON_FULLY_UNDISTORTED_FRAME = True  # Set to False as per "New" script


def setup_kalman_filter(dt=1.0 / 30, noise_acceleration_variance = 5000):
    kf = cv2.KalmanFilter(4, 2)  # 4 state vars (x,y,vx,vy), 2 measurement vars (x,y)
    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)  # Measures x,y

    # Simplified process noise covariance, q is a general factor
    # Adjust q_pos_factor and q_vel_factor based on expected movement variance
    q_pos_factor = 1.0  # Factor for position variance in process noise
    q_vel_factor = 1.0  # Factor for velocity variance in process noise

    kf.processNoiseCov = np.array([
        [dt ** 4 / 4 * q_pos_factor, 0, dt ** 3 / 2 * q_pos_factor, 0],
        [0, dt ** 4 / 4 * q_pos_factor, 0, dt ** 3 / 2 * q_pos_factor],
        [dt ** 3 / 2 * q_pos_factor, 0, dt ** 2 * q_vel_factor, 0],
        [0, dt ** 3 / 2 * q_pos_factor, 0, dt ** 2 * q_vel_factor]
    ], dtype=np.float32) * noise_acceleration_variance

    #Uncertainty of YOLO detections (change based on noisiness of pixel detections)
    kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], dtype=np.float32) * KF_MEASUREMENT_NOISE_SCALE
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 500  # High initial uncertainty
    return kf


def undistort_point(point_xy, K_matrix, D_coeffs, is_fisheye=True):
    if point_xy is None: return None
    dist_np = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    undist_np = cv2.fisheye.undistortPoints(dist_np, K_matrix, D_coeffs, P=K_matrix) if is_fisheye \
        else cv2.undistortPoints(dist_np, K_matrix, D_coeffs, P=K_matrix)
    if undist_np is not None: return (int(round(undist_np[0][0][0])), int(round(undist_np[0][0][1])))
    return None


def preprocess_frame_ccs(frame_orig, K_matrix, D_coeffs_val, map1_val, map2_val, do_full_undistort_val):
    if do_full_undistort_val and map1_val is not None and map2_val is not None:  # map2_val check added
        return cv2.remap(frame_orig, map1_val, map2_val, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return frame_orig


def get_processed_detections_ccs(current_frame_num_val, frame_to_detect_on_val, ball_model_obj, pose_model_obj,
                                 id_ball_class_val, current_ball_conf_val, K_matrix_val, D_coeffs_val,
                                 do_undistort_points_val, is_full_frame_undistorted_already,
                                 current_kf_trajectory_pixels_val,  # List of dicts {'frame', 'px_coord', 'type'}
                                 max_pixel_disp_val, is_release_detected_currently):
    ball_results = ball_model_obj(frame_to_detect_on_val, conf=current_ball_conf_val, verbose=False,
                                  classes=[id_ball_class_val])
    ball_bbox_detected = utils.get_ball_bbox(ball_results, id_ball_class_val)
    ball_center_undistorted_px_final = None
    ball_bbox_for_logic_val = ball_bbox_detected
    used_low_conf_heuristic_val_flag = False

    if ball_bbox_detected is not None:
        center_x_det = (ball_bbox_detected[0] + ball_bbox_detected[2]) / 2.0
        center_y_det = (ball_bbox_detected[1] + ball_bbox_detected[3]) / 2.0

        temp_ball_center_processed_px = None
        if do_undistort_points_val and not is_full_frame_undistorted_already:
            temp_ball_center_processed_px = undistort_point((center_x_det, center_y_det), K_matrix_val, D_coeffs_val)
        else:
            temp_ball_center_processed_px = (int(round(center_x_det)), int(round(center_y_det)))

        if temp_ball_center_processed_px is not None:
            if is_release_detected_currently and len(current_kf_trajectory_pixels_val) > 0:
                # Use the last known KF point for proximity check
                last_known_kf_px = current_kf_trajectory_pixels_val[-1]['px_coord']
                dist_sq = (temp_ball_center_processed_px[0] - last_known_kf_px[0]) ** 2 + \
                          (temp_ball_center_processed_px[1] - last_known_kf_px[1]) ** 2
                if dist_sq <= max_pixel_disp_val ** 2:
                    ball_center_undistorted_px_final = temp_ball_center_processed_px
                    if current_ball_conf_val == POST_RELEASE_BALL_CONF: used_low_conf_heuristic_val_flag = True
                else:
                    ball_bbox_for_logic_val = None  # Rejected raw detection by proximity to KF track
            else:  # Pre-release or first point post-release (no KF trajectory yet to check against)
                ball_center_undistorted_px_final = temp_ball_center_processed_px
        else:  # Undistortion failed
            ball_bbox_for_logic_val = None
    # else: No ball detected

    pose_results_val = pose_model_obj(frame_to_detect_on_val, conf=0.84, verbose=False)
    shooter_wrist_undistorted_px_final = None
    if ball_bbox_for_logic_val is not None:  # Only proceed if we have a valid ball bbox for context
        wrist_detected_val, _ = utils.get_likely_shooter_wrist_and_person_idx(pose_results_val, ball_bbox_for_logic_val,
                                                                              SHOOTING_WRIST_KP_INDEX,
                                                                              MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION)
        if wrist_detected_val is not None:
            if do_undistort_points_val and not is_full_frame_undistorted_already:
                shooter_wrist_undistorted_px_final = undistort_point(wrist_detected_val, K_matrix_val, D_coeffs_val)
            else:
                shooter_wrist_undistorted_px_final = (int(wrist_detected_val[0]), int(wrist_detected_val[1]))

    return ball_center_undistorted_px_final, ball_bbox_for_logic_val, shooter_wrist_undistorted_px_final, used_low_conf_heuristic_val_flag


def check_for_release_ccs(frame_num_val, ball_center_undist_px, ball_bbox_l_val, wrist_pos_undist_px,
                          current_release_state_val, release_buffer_list_val, frame_orig_val,
                          manual_override_frame_val, manual_override_shot_target_val, current_shot_count_val,
                          kf_obj_to_init):  # kf_obj_to_init is the KalmanFilter object
    # Globals MODIFIED by this function or its helper _init_release_state
    global release_frame_info_dict, shot_count, Zc_at_release, kf_initialized
    global trajectory_points_raw_pixels_undistorted, trajectory_points_kf_pixels_undistorted, trajectory_points_ccs
    global estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs

    new_release_state_val = current_release_state_val
    forced_release_this_frame = False

    # Helper to reset state and init KF. Uses kf_obj_to_init from outer scope (closure).
    def _init_release_state(u_rel, v_rel, current_frame_num_init, bbox_at_release, frame_original_at_release,
                            type_of_release="auto"):
        # Globals MODIFIED by this helper function
        global Zc_at_release, kf_initialized
        # release_frame_info_dict is set by the caller (check_for_release_ccs)
        # shot_count is incremented by the caller
        global trajectory_points_raw_pixels_undistorted, trajectory_points_kf_pixels_undistorted, trajectory_points_ccs
        global estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs

        estimated_speed_mps = 0.0
        estimated_angle_deg_XY_ccs = 0.0
        estimated_angle_deg_elev_ccs = 0.0
        trajectory_points_raw_pixels_undistorted.clear()
        trajectory_points_kf_pixels_undistorted.clear()
        trajectory_points_ccs.clear()

        # Add first point to raw trajectory (this is the release point itself)
        trajectory_points_raw_pixels_undistorted.append(
            {'frame': current_frame_num_init, 'px_coord': (u_rel, v_rel), 'type': f'{type_of_release}_raw_init'})

        app_diam_px_rel = ((bbox_at_release[2] - bbox_at_release[0]) + (bbox_at_release[3] - bbox_at_release[1])) / 2.0
        if app_diam_px_rel > 0:
            avg_focal_px = (fx + fy) / 2.0
            Zc_at_release = (BASKETBALL_REAL_DIAMETER_M * avg_focal_px) / app_diam_px_rel
        else:
            Zc_at_release = 1.0  # Fallback

        xn_rel = (u_rel - cx) / fx;
        yn_rel = (v_rel - cy) / fy
        Pc_rel = np.array([xn_rel * Zc_at_release, yn_rel * Zc_at_release, Zc_at_release])

        kf_obj_to_init.statePost = np.array([u_rel, v_rel, 0, 0], dtype=np.float32).reshape(-1, 1)  # Initial velocity 0
        kf_obj_to_init.errorCovPost = np.eye(4, dtype=np.float32) * 50.0  # Reset error covariance for new track
        kf_initialized = True

        trajectory_points_kf_pixels_undistorted.append(
            {'frame': current_frame_num_init, 'px_coord': (u_rel, v_rel), 'type': f'kf_{type_of_release}_init'})
        trajectory_points_ccs.append(
            {'frame': current_frame_num_init, 'Pc': Pc_rel, 'px_coord_type': f'kf_{type_of_release}_init'})

        print(f"  Release Ball Center (undistorted px): ({u_rel}, {v_rel})")
        print(f"  Est. Zc at Release (m): {Zc_at_release:.3f}")
        print(f"  Release Ball Coords (CCS, m): {Pc_rel.round(3)}. KF Initialized.")
        if SAVE_RELEASE_FRAMES:
            os.makedirs(RELEASE_FRAME_OUTPUT_DIR, exist_ok=True)
            annotated_save = frame_original_at_release.copy()
            cv2.circle(annotated_save, (u_rel, v_rel), 15, (0, 0, 255), 2)
            if wrist_pos_undist_px: cv2.circle(annotated_save, wrist_pos_undist_px, 7, (0, 255, 255), -1)
            mode_txt = "ManOverride" if type_of_release == "manual" else "AutoDetect"
            save_p = os.path.join(RELEASE_FRAME_OUTPUT_DIR,
                                  f"release_s{shot_count}_f{current_frame_num_init}_{mode_txt}.png")
            cv2.imwrite(save_p, annotated_save)
            print(f"    Saved release frame: {save_p}")

    # Manual Release Override Logic
    if not current_release_state_val and manual_override_frame_val is not None and \
            current_shot_count_val + 1 == manual_override_shot_target_val and frame_num_val == manual_override_frame_val:
        if ball_center_undist_px is not None and ball_bbox_l_val is not None:
            shot_count += 1  # Increment global shot_count for this new shot
            release_frame_info_dict = {"frame_no": frame_num_val,
                                       "ball_center_undistorted_px": ball_center_undist_px,
                                       "ball_bbox_detection": ball_bbox_l_val,
                                       "original_frame_at_this_moment": frame_orig_val.copy()}
            new_release_state_val = True
            forced_release_this_frame = True
            _init_release_state(ball_center_undist_px[0], ball_center_undist_px[1], frame_num_val, ball_bbox_l_val,
                                frame_orig_val, "manual")
            print(f"MANUAL RELEASE (Shot {shot_count}) at frame {frame_num_val}.")
        else:
            print(f"Manual override at frame {frame_num_val} skipped: no valid ball data.")

    # Automatic Release Detection Logic
    if not new_release_state_val and not forced_release_this_frame and \
            ball_center_undist_px is not None and wrist_pos_undist_px is not None and ball_bbox_l_val is not None and manual_override_frame_val is None:
        wrist_in_ball = utils.is_point_inside_bbox(wrist_pos_undist_px, ball_bbox_l_val,
                                                   margin=WRIST_BALL_PROXIMITY_MARGIN)
        if not wrist_in_ball:  # Wrist is NOT in ball - potential release
            release_buffer_list_val.append(
                {"frame_no": frame_num_val, "ball_center_undistorted_px": ball_center_undist_px,
                 "ball_bbox_detection": ball_bbox_l_val, "original_frame_at_this_moment": frame_orig_val.copy()})
            if len(release_buffer_list_val) > CONFIRMATION_FRAMES_FOR_RELEASE: release_buffer_list_val.pop(0)
            if len(release_buffer_list_val) == CONFIRMATION_FRAMES_FOR_RELEASE:
                shot_count += 1  # Increment global shot_count for this new shot
                release_frame_info_dict = release_buffer_list_val[0].copy()  # Use the first frame in buffer
                new_release_state_val = True

                u_rel_auto, v_rel_auto = release_frame_info_dict['ball_center_undistorted_px']
                bbox_auto = release_frame_info_dict['ball_bbox_detection']
                frame_orig_auto = release_frame_info_dict['original_frame_at_this_moment']
                _init_release_state(u_rel_auto, v_rel_auto, release_frame_info_dict['frame_no'], bbox_auto,
                                    frame_orig_auto, "auto")
                print(f"AUTO RELEASE (Shot {shot_count}) at frame {release_frame_info_dict['frame_no']}.")
                release_buffer_list_val.clear()  # Clear buffer after confirmed release
        else:  # Wrist is in ball - clear buffer
            release_buffer_list_val.clear()
            if current_release_state_val:  # Was in release state, but wrist came back
                new_release_state_val = False
                print(f"  Frame {frame_num_val}: Shot {shot_count} release cancelled (wrist returned to ball).")
    elif current_release_state_val and ball_center_undist_px is None:
        pass  # Ball lost, but keep release_detected state True for now (KF might predict)

    return new_release_state_val, release_buffer_list_val


def update_trajectory_and_estimate_ccs(current_release_state_val,
                                       ball_point_from_kf_px_dict,  # Dict: {'frame', 'px_coord', 'type'}
                                       current_Zc_at_release, video_fps_val):
    # Globals MODIFIED or used by this function
    global estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs, shot_count
    global trajectory_points_kf_pixels_undistorted, trajectory_points_ccs

    if current_release_state_val and ball_point_from_kf_px_dict is not None and current_Zc_at_release is not None:
        frame_num_val = ball_point_from_kf_px_dict['frame']
        u_kf, v_kf = ball_point_from_kf_px_dict['px_coord']
        point_type = ball_point_from_kf_px_dict['type']

        # Append to KF pixel list if it's a new frame
        if not trajectory_points_kf_pixels_undistorted or \
                trajectory_points_kf_pixels_undistorted[-1]['frame'] != frame_num_val:
            trajectory_points_kf_pixels_undistorted.append(ball_point_from_kf_px_dict)
        elif trajectory_points_kf_pixels_undistorted[-1]['px_coord'] != u_kf or \
                trajectory_points_kf_pixels_undistorted[-1]['px_coord'][
                    1] != v_kf:  # Check if coords changed for same frame
            # This case might occur if KF corrected an initial prediction for the same frame.
            # For simplicity, we'll assume one final KF point per frame.
            # If a more refined update for the same frame is needed, logic could replace the last entry.
            # print(f"Debug: KF point for frame {frame_num_val} updated. Old: {trajectory_points_kf_pixels_undistorted[-1]['px_coord']}, New: {(u_kf, v_kf)}")
            # trajectory_points_kf_pixels_undistorted[-1] = ball_point_from_kf_px_dict # Option to update
            pass

        # Convert KF point to CCS and append to CCS trajectory list
        # Ensure CCS point is added only if KF point for this frame is valid and new to CCS list
        if not trajectory_points_ccs or trajectory_points_ccs[-1]['frame'] != frame_num_val:
            xn_kf = (u_kf - cx) / fx;
            yn_kf = (v_kf - cy) / fy
            Xc_kf = xn_kf * current_Zc_at_release  # Assume Zc = Zc_at_release
            Yc_kf = yn_kf * current_Zc_at_release
            Pc_kf = np.array([Xc_kf, Yc_kf, current_Zc_at_release])
            trajectory_points_ccs.append({'frame': frame_num_val, 'Pc': Pc_kf, 'px_coord_type': point_type})
            # print(f"  Frame {frame_num_val}: Added to CCS. KF Pt: {(u_kf, v_kf)}, Type: {point_type}, Pc: {Pc_kf.round(3)}")

        # Naive parameter estimation (if not already estimated for this shot)
        if estimated_speed_mps == 0.0 and len(trajectory_points_ccs) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
            # Ensure points used for estimation are from the current shot, starting from index 0
            if len(trajectory_points_ccs) > FRAMES_FOR_NAIVE_ESTIMATION:  # Check if enough points
                p_start_ccs_info = trajectory_points_ccs[0]  # First point of the shot
                p_end_ccs_info = trajectory_points_ccs[FRAMES_FOR_NAIVE_ESTIMATION]  # Nth point after start

                Pc_s = p_start_ccs_info['Pc'];
                Pc_e = p_end_ccs_info['Pc']
                delta_Pc_val = Pc_e - Pc_s
                d_frames = p_end_ccs_info['frame'] - p_start_ccs_info['frame']

                if video_fps_val > 0 and d_frames > 0:
                    d_t_s = d_frames / video_fps_val
                    Vc_avg_val = delta_Pc_val / d_t_s
                    estimated_speed_mps = np.linalg.norm(Vc_avg_val)
                    # Corrected angle calculations
                    # XY plane angle (azimuth-like): atan2(dX, dZ) or atan2(dY, dX) depending on convention
                    # For basketball, often interested in launch angle in vertical plane and side angle.
                    # Assuming X is right, Y is down, Z is forward (typical camera coordinates after transform)
                    # Velocity components: Vc_avg_val[0] -> Vx, Vc_avg_val[1] -> Vy, Vc_avg_val[2] -> Vz

                    # Elevation angle: angle with the XZ plane (horizontal plane if camera is level)
                    # Projection onto XZ plane: sqrt(Vx^2 + Vz^2)
                    # Elevation angle: atan2(-Vy, sqrt(Vx^2 + Vz^2)) (negative Vy because Y is down)
                    mag_XZ = math.sqrt(Vc_avg_val[0] ** 2 + Vc_avg_val[2] ** 2)
                    if mag_XZ < 1e-6: mag_XZ = 1e-6  # Avoid division by zero
                    estimated_angle_deg_elev_ccs = math.degrees(math.atan2(-Vc_avg_val[1], mag_XZ))

                    # Side angle (azimuth in XZ plane, relative to Z-axis)
                    # estimated_angle_deg_XY_ccs = math.degrees(math.atan2(Vc_avg_val[0], Vc_avg_val[2])) # Angle from Z axis in XZ plane
                    # If original intent was angle in XY plane view from top:
                    estimated_angle_deg_XY_ccs = math.degrees(
                        math.atan2(Vc_avg_val[1], Vc_avg_val[0]))  # Original interpretation based on var name

                    print(
                        f"  NAIVE CCS (Shot {shot_count}, frames {p_start_ccs_info['frame']}-{p_end_ccs_info['frame']}): Speed={estimated_speed_mps:.2f}m/s, Elev={estimated_angle_deg_elev_ccs:.1f}deg, SideXY={estimated_angle_deg_XY_ccs:.1f}deg")


def draw_annotations_on_frame_ccs(frame_to_annotate_val, current_frame_number_for_drawing,
                                  ball_bbox_d_val, raw_ball_center_undist_px_val,
                                  wrist_pos_undist_px_val, is_release_detected_val,
                                  release_info_dict_val, traj_kf_pix_list_of_dicts,
                                  traj_raw_pix_list_of_dicts, shot_num_val,
                                  est_speed_val, est_angle_elev_val, used_low_conf_flag_val,
                                  kf_current_prediction_px_val=None):  # kf_current_prediction_px_val is the raw KF predict() output for this frame

    # Draw raw detected bbox (from current frame's detection)
    if ball_bbox_d_val is not None:
        l, t, r, b = [int(c) for c in ball_bbox_d_val]
        color_bbox = (0, 165, 255) if used_low_conf_flag_val else (
        255, 100, 100)  # Orange if low conf for this raw detection
        cv2.rectangle(frame_to_annotate_val, (l, t), (r, b), color_bbox, 1)

    # Draw raw detected (undistorted) center for the CURRENT frame
    if raw_ball_center_undist_px_val is not None:
        cv2.circle(frame_to_annotate_val, raw_ball_center_undist_px_val, 4, (255, 0, 255),
                   -1)  # Magenta for raw detection point

    # Draw current KF point (corrected or predicted if no good raw detection was used for correction)
    current_kf_point_to_draw = None
    current_kf_point_type = 'unknown'
    if traj_kf_pix_list_of_dicts:  # Check if list is not empty
        # Find the KF point for the current frame, if it exists in the trajectory list
        kf_point_for_current_frame = next(
            (item for item in reversed(traj_kf_pix_list_of_dicts) if item['frame'] == current_frame_number_for_drawing),
            None)
        if kf_point_for_current_frame:
            current_kf_point_to_draw = kf_point_for_current_frame['px_coord']
            current_kf_point_type = kf_point_for_current_frame['type']
        elif kf_current_prediction_px_val and is_release_detected_val:  # Fallback to raw KF prediction if no KF point was added to traj list for this frame
            current_kf_point_to_draw = kf_current_prediction_px_val
            current_kf_point_type = 'kf_pred_only_not_in_traj'

    if current_kf_point_to_draw:
        color_kf = (0, 255, 0)  # Green for 'kf_corr' or 'kf_init'
        if 'pred' in current_kf_point_type:
            color_kf = (255, 255, 0)  # Cyan for 'kf_pred_gated', 'kf_pred_miss'
        elif current_kf_point_type == 'kf_pred_only_not_in_traj':
            color_kf = (255, 165, 0)  # Orange for pure prediction not added to trajectory

        cv2.circle(frame_to_annotate_val, current_kf_point_to_draw, 6, color_kf, -1)
        # cv2.putText(frame_to_annotate_val, current_kf_point_type, (current_kf_point_to_draw[0]+10, current_kf_point_to_draw[1]), cv2.FONT_HERSHEY_PLAIN, 0.8, color_kf, 1)

    if wrist_pos_undist_px_val is not None: cv2.circle(frame_to_annotate_val, tuple(wrist_pos_undist_px_val), 7,
                                                       (0, 255, 255), -1)  # Yellow for wrist

    # Draw RAW trajectory dots (magenta dots)
    if is_release_detected_val and len(traj_raw_pix_list_of_dicts) >= 1:
        for item_info in traj_raw_pix_list_of_dicts:
            # Draw older points smaller/fainter if desired
            cv2.circle(frame_to_annotate_val, item_info['px_coord'], 2, (200, 0, 200), -1)  # Darker Magenta dots

    # Draw KF trajectory line (main green line)
    if is_release_detected_val and len(traj_kf_pix_list_of_dicts) >= 2:
        pixel_points_for_kf_line = [item['px_coord'] for item in traj_kf_pix_list_of_dicts]
        pts_np_kf = np.array(pixel_points_for_kf_line, dtype=np.int32)
        cv2.polylines(frame_to_annotate_val, [pts_np_kf], False, (0, 255, 0), 2)  # Green line

    # Text annotations
    if is_release_detected_val:
        if release_info_dict_val: cv2.putText(frame_to_annotate_val,
                                              f"S{shot_num_val} Rls:Fr{release_info_dict_val['frame_no']}", (10, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Release marker on first KF point of the KF trajectory
        if traj_kf_pix_list_of_dicts and traj_kf_pix_list_of_dicts[0]['px_coord'] is not None:
            cv2.circle(frame_to_annotate_val, traj_kf_pix_list_of_dicts[0]['px_coord'], 15, (0, 0, 255),
                       2)  # Red circle at release

        if est_speed_val > 0:
            cv2.putText(frame_to_annotate_val, f"Speed(CCS): {est_speed_val:.2f}m/s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(frame_to_annotate_val, f"Elev(CCS): {est_angle_elev_val:.1f}deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
    else:  # Pre-release text
        cv2.putText(frame_to_annotate_val, "Awaiting Release", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2)

    return frame_to_annotate_val


# --- Main Processing Loop ---
if __name__ == "__main__":
    os.makedirs(RELEASE_FRAME_OUTPUT_DIR, exist_ok=True)

    MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "Yolo-Weights")
    BALL_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "model2.pt")
    POSE_MODEL_PATH = 'yolov8n-pose.pt'

    VIDEO_INPUT_DIR = os.path.join(BASE_DIR, "footage")
    VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "ccs", "output_video_kf")  # Changed output dir slightly
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    video_path = os.path.join(VIDEO_INPUT_DIR, "freethrow_2.mp4")
    output_filename_suffix = "_ccs_kf_fixed"
    output_path = os.path.join(VIDEO_OUTPUT_DIR,
                               f"{os.path.splitext(os.path.basename(video_path))[0]}{output_filename_suffix}.avi")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ball_model_g = YOLO(BALL_MODEL_PATH)
    pose_model_g = YOLO(POSE_MODEL_PATH)

    cap_g = cv2.VideoCapture(video_path)
    if not cap_g.isOpened(): cap_g = cv2.VideoCapture(video_path, cv2.CAP_MSMF)  # Fallback
    if not cap_g.isOpened(): exit(f"Error opening video: {video_path}")

    frame_width_g = int(cap_g.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps_g = cap_g.get(cv2.CAP_PROP_FPS)
    if not (0 < video_fps_g < 1000): video_fps_g = 30.0  # More robust FPS check, limit to 200

    out_g = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), video_fps_g, (frame_width_g, frame_height_g))
    if not out_g.isOpened(): exit(f"Error opening video writer: {output_path}")

    id_of_ball_g = None
    for cid, name in ball_model_g.names.items():
        if name.lower() == TARGET_BALL_CLASS_NAME: id_of_ball_g = cid; break
    if id_of_ball_g is None: exit(f"Class '{TARGET_BALL_CLASS_NAME}' ID not found in model.")

    map1_g, map2_g = None, None
    if DETECT_ON_FULLY_UNDISTORTED_FRAME:
        map1_g, map2_g = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (frame_width_g, frame_height_g),
                                                             cv2.CV_16SC2)

    # --- Main loop state variables ---
    frame_count = 0
    shot_count = 0  # Total shots detected in video
    release_detected_in_shot = False  # Is a shot currently being tracked post-release?
    release_frame_info_dict = None  # Info about the confirmed release frame
    potential_release_buffer = []  # Buffer for confirming release event

    trajectory_points_raw_pixels_undistorted = []  # List of {'frame', 'px_coord', 'type'} for raw validated points
    trajectory_points_kf_pixels_undistorted = []  # List of {'frame', 'px_coord', 'type'} for KF points
    trajectory_points_ccs = []  # List of {'frame', 'Pc', 'px_coord_type'} for 3D points in CCS

    Zc_at_release = None  # Estimated depth of ball at release
    estimated_speed_mps = 0.0
    estimated_angle_deg_XY_ccs = 0.0
    estimated_angle_deg_elev_ccs = 0.0

    kf = setup_kalman_filter(dt=1.0 / video_fps_g, noise_acceleration_variance= NOISE_ACCELERATION_VARIANCE)
    kf_initialized = False  # Has KF been initialized for the current shot?
    missed_detections_count_kf = 0  # Consecutive frames KF relied on prediction
    kf_current_predicted_pixel_point_for_drawing = None  # Store raw kf.predict() output for drawing

    print(f"Processing video: {video_path} (FPS: {video_fps_g:.2f})")
    print(f"Outputting to: {output_path}")
    if MANUAL_RELEASE_FRAME_OVERRIDE is not None:
        print(
            f"MANUAL RELEASE OVERRIDE ENABLED: Shot {MANUAL_RELEASE_SHOT_NUMBER_TARGET}, Target Frame {MANUAL_RELEASE_FRAME_OVERRIDE}")

    while cap_g.isOpened():
        ret, frame_original_main = cap_g.read()
        if not ret: break
        frame_count += 1

        if frame_count % 30 == 0:  # Print status every 30 frames
            print(f"Frame {frame_count}, Shot: {shot_count}, ReleaseActive: {release_detected_in_shot}, "
                  f"RawPts: {len(trajectory_points_raw_pixels_undistorted)}, KFPts: {len(trajectory_points_kf_pixels_undistorted)}, "
                  f"CCSPts: {len(trajectory_points_ccs)}, KFinit: {kf_initialized}")

        current_frame_to_detect = preprocess_frame_ccs(frame_original_main, K, D, map1_g, map2_g,
                                                       DETECT_ON_FULLY_UNDISTORTED_FRAME)
        annotated_frame = current_frame_to_detect.copy()

        active_conf_thresh = POST_RELEASE_BALL_CONF if release_detected_in_shot else PRE_RELEASE_BALL_CONF

        ball_center_detected_undist_px, ball_bbox_l, shooter_wrist_undist_px, used_low_conf_h = \
            get_processed_detections_ccs(frame_count, current_frame_to_detect, ball_model_g, pose_model_g, id_of_ball_g,
                                         active_conf_thresh, K, D,
                                         UNDISTORT_POINTS_AFTER_DETECTION, DETECT_ON_FULLY_UNDISTORTED_FRAME,
                                         trajectory_points_kf_pixels_undistorted,  # Pass KF trajectory for gating
                                         MAX_PIXEL_DISPLACEMENT_POST_RELEASE,
                                         release_detected_in_shot)

        prev_release_state = release_detected_in_shot
        release_detected_in_shot, potential_release_buffer = \
            check_for_release_ccs(frame_count, ball_center_detected_undist_px, ball_bbox_l, shooter_wrist_undist_px,
                                  release_detected_in_shot, potential_release_buffer, frame_original_main,
                                  MANUAL_RELEASE_FRAME_OVERRIDE, MANUAL_RELEASE_SHOT_NUMBER_TARGET, shot_count, kf)

        # Handle transitions in release state
        if not prev_release_state and release_detected_in_shot:  # Just detected a new release
            print(f"  Frame {frame_count}: New shot (Shot {shot_count}) detected. KF initialized: {kf_initialized}.")
            missed_detections_count_kf = 0  # Reset for new shot
            # _init_release_state (called by check_for_release_ccs) handles trajectory clears and KF state init.
        elif prev_release_state and not release_detected_in_shot:  # Shot just ended or was cancelled
            print(f"  Frame {frame_count}: Shot {shot_count} tracking stopped.")
            kf_initialized = False  # Stop KF updates until next release
            # Trajectories, Zc_at_release, estimations will be reset by the next _init_release_state

        kf_current_predicted_pixel_point_for_drawing = None
        ball_point_from_kf_for_ccs_update = None  # This will hold the KF point (corrected or predicted) for trajectory

        if release_detected_in_shot and kf_initialized:
            predicted_kf_state = kf.predict()
            kf_current_predicted_pixel_point_for_drawing = (
            int(round(predicted_kf_state[0, 0])), int(round(predicted_kf_state[1, 0])))

            if ball_center_detected_undist_px is not None:  # If there's a raw detection this frame
                # Gate the raw detection against the KF prediction
                dist_sq_to_kf_pred = (ball_center_detected_undist_px[0] - kf_current_predicted_pixel_point_for_drawing[
                    0]) ** 2 + \
                                     (ball_center_detected_undist_px[1] - kf_current_predicted_pixel_point_for_drawing[
                                         1]) ** 2
                KF_CORRECTION_GATE_SQ = (
                                                    MAX_PIXEL_DISPLACEMENT_POST_RELEASE * 1.5) ** 2  # Allow somewhat larger gate for KF correction

                if dist_sq_to_kf_pred <= KF_CORRECTION_GATE_SQ:
                    measurement = np.array(
                        [[float(ball_center_detected_undist_px[0])], [float(ball_center_detected_undist_px[1])]],
                        dtype=np.float32)
                    corrected_kf_state = kf.correct(measurement)
                    ball_point_from_kf_for_ccs_update = {'frame': frame_count,
                                                         'px_coord': (int(round(corrected_kf_state[0, 0])),
                                                                      int(round(corrected_kf_state[1, 0]))),
                                                         'type': 'kf_corr'}
                    missed_detections_count_kf = 0
                else:  # Raw detection too far from KF prediction
                    missed_detections_count_kf += 1
                    if missed_detections_count_kf <= MAX_CONSECUTIVE_MISSES_KF:
                        ball_point_from_kf_for_ccs_update = {'frame': frame_count,
                                                             'px_coord': kf_current_predicted_pixel_point_for_drawing,
                                                             'type': 'kf_pred_gated'}
                    # If max misses exceeded, ball_point_from_kf_for_ccs_update remains None
            else:  # No raw detection this frame
                missed_detections_count_kf += 1
                if missed_detections_count_kf <= MAX_CONSECUTIVE_MISSES_KF:
                    ball_point_from_kf_for_ccs_update = {'frame': frame_count,
                                                         'px_coord': kf_current_predicted_pixel_point_for_drawing,
                                                         'type': 'kf_pred_miss'}
                # If max misses exceeded, ball_point_from_kf_for_ccs_update remains None

            # Update trajectories and estimations using the KF point
            if ball_point_from_kf_for_ccs_update is not None:
                update_trajectory_and_estimate_ccs(
                    release_detected_in_shot,
                    ball_point_from_kf_for_ccs_update,
                    Zc_at_release,
                    video_fps_g
                )
            elif missed_detections_count_kf > MAX_CONSECUTIVE_MISSES_KF:  # Max misses exceeded
                print(
                    f"  Frame {frame_count}: Ball lost for Shot {shot_count} (KF max misses). KF/CCS Trajectory not updated.")
                # Optionally, could set release_detected_in_shot = False here to terminate the shot.
                # release_detected_in_shot = False
                # kf_initialized = False

        # Populate raw trajectory if release is active and raw point exists
        if release_detected_in_shot and ball_center_detected_undist_px is not None:
            if not trajectory_points_raw_pixels_undistorted or \
                    trajectory_points_raw_pixels_undistorted[-1]['frame'] != frame_count:
                trajectory_points_raw_pixels_undistorted.append({
                    'frame': frame_count,
                    'px_coord': ball_center_detected_undist_px,
                    'type': 'raw_low_conf_heur' if used_low_conf_h else 'raw_validated'
                })

        annotated_frame = draw_annotations_on_frame_ccs(
            annotated_frame, frame_count,
            ball_bbox_l, ball_center_detected_undist_px,
            shooter_wrist_undist_px, release_detected_in_shot,
            release_frame_info_dict, trajectory_points_kf_pixels_undistorted,
            trajectory_points_raw_pixels_undistorted, shot_count,
            estimated_speed_mps, estimated_angle_deg_elev_ccs,
            used_low_conf_h, kf_current_predicted_pixel_point_for_drawing
        )

        cv2.imshow("CCS KF Manual Test", annotated_frame)
        out_g.write(annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("Processing stopped by user."); break
        if key == ord('p'):  # Pause
            print(f"Paused at frame {frame_count}. Press any key to continue...")
            cv2.waitKey(-1)

    cap_g.release()
    out_g.release()
    cv2.destroyAllWindows()
    print(f"Finished processing. Output saved to {output_path}")