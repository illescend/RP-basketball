import math
import os

from matplotlib import pyplot as plt

os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"  # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100"  # Try to prioritize FFMPEG
import numpy as np
import cv2
from ultralytics import YOLO
from testmodel.algorithms import utils
from testmodel.algorithms.trajectory_fitting import trajectory_utils as tu

# --- Global Constants and Configuration HERO12 ---
# K = np.array([[1134.4909, 0.0, 968.5293], [0.0, 1133.2194, 553.0426], [0.0, 0.0, 1.0]])
# D = np.array([0.43466, -0.63731, 2.29794, -2.09685])  # Fisheye

# --- Global Constant and Config HERO7
K = np.array([[1006.894800376467,0.0,964.3578313551827],[0.0,994.7402914201482,537.3734258659648],[0.0,0.0, 1.0]])
D = np.array([0.36097005855002745,-0.013703354643132513,0.45553079194926527,-0.32891315019398665])  # Fisheye

fx = K[0, 0];
fy = K[1, 1];
cx = K[0, 2];
cy = K[1, 2]



SHOOTING_WRIST_KP_INDEX = utils.SHOOTING_WRIST_KP_INDEX
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
WRIST_BALL_PROXIMITY_MARGIN = utils.WRIST_BALL_PROXIMITY_MARGIN
CONFIRMATION_FRAMES_FOR_RELEASE = utils.CONFIRMATION_FRAMES_FOR_RELEASE

SHOOTING_ARM_WRIST_IDX = utils.RIGHT_WRIST
SHOOTING_ARM_ELBOW_IDX = utils.RIGHT_ELBOW
SHOOTING_ARM_SHOULDER_IDX = utils.RIGHT_SHOULDER

SAVE_RELEASE_FRAMES = True
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# RELEASE_FRAME_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "ccs", "release_frames_kf_compare") # Original relative path
#RELEASE_FRAME_OUTPUT_DIR = "outputs/release_frames_kf_compare"  # Effective path
RELEASE_FRAME_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "trajectory_fitting", "wcs_release_frames")  # Changed output dir slightly
os.makedirs(RELEASE_FRAME_OUTPUT_DIR, exist_ok=True)

PRE_RELEASE_BALL_CONF = 0.35
POST_RELEASE_BALL_CONF = 0.25
MAX_PIXEL_DISPLACEMENT_POST_RELEASE = 100
FRAMES_FOR_NAIVE_ESTIMATION = 20
BASKETBALL_REAL_DIAMETER_M = 0.24
TARGET_BALL_CLASS_NAME = "ball"

#3D World Points
WCS_FT_LINE_TO_RIM_CENTER_X = 4.191  # m (X-coord of rim center)
WCS_RIM_HEIGHT_Y = 3.048           # m (Y-coord of rim)
WCS_RIM_RADIUS = 0.4572 / 2        # m
WCS_BACKBOARD_OFFSET_FROM_RIM_CENTER = WCS_RIM_RADIUS + 0.15 # m (backboard is behind rim center)
WCS_BACKBOARD_X = WCS_FT_LINE_TO_RIM_CENTER_X + WCS_BACKBOARD_OFFSET_FROM_RIM_CENTER # X-coord of backboard face
WCS_BB_INNER_RECT_WIDTH = 0.61     # m
WCS_BB_INNER_RECT_HEIGHT = 0.45    # m
WCS_BB_INNER_RECT_BOTTOM_Y = WCS_RIM_HEIGHT_Y - WCS_BB_INNER_RECT_HEIGHT # Y-coord of inner rectangle bottom
WCS_BB_INNER_RECT_TOP_Y = WCS_RIM_HEIGHT_Y + (WCS_BB_INNER_RECT_HEIGHT - 0.05)

MANUAL_RELEASE_FRAME_OVERRIDE = None #111 #176  # 180 #111 for freethrow_2
MANUAL_RELEASE_SHOT_NUMBER_TARGET = 1  # Which shot number to apply manual override
MAX_CONSECUTIVE_MISSES_KF = 5
KF_MEASUREMENT_NOISE_SCALE = 0.001 #Confidence in pixel detection
KF_PROCESS_NOISE_SCALE = 300  # Increased from 0.1 for more dynamic movement
NOISE_ACCELERATION_VARIANCE = 1536

UNDISTORT_POINTS_AFTER_DETECTION = True
DETECT_ON_FULLY_UNDISTORTED_FRAME = True  # Set to False as per "New" script

#Plot variables
# Court dimensions for plotting (should align with WCS definitions)
# These are used by the plotting function later.
# WCS_FT_LINE_TO_RIM_CENTER_X, WCS_RIM_HEIGHT_Y are already defined for solvePnP.
# We might need a few more for detailed backboard plotting if desired.
PLOT_RIM_DIAMETER = 0.4572
PLOT_BACKBOARD_X = WCS_BACKBOARD_X # Use the same as solvePnP
PLOT_BACKBOARD_HEIGHT = 1.0668 # Standard backboard height
# Assuming inner rectangle bottom is at rim height for plotting
PLOT_BACKBOARD_RECT_BOTTOM_Y = WCS_RIM_HEIGHT_Y
PLOT_BACKBOARD_RECT_TOP_Y = WCS_RIM_HEIGHT_Y + WCS_BB_INNER_RECT_HEIGHT # From solvePnP setup
# If plotting the full backboard, its bottom might be lower
PLOT_BACKBOARD_BOTTOM_Y = WCS_RIM_HEIGHT_Y - (WCS_BB_INNER_RECT_HEIGHT/2) - (PLOT_BACKBOARD_HEIGHT - WCS_BB_INNER_RECT_HEIGHT)/2

# --- New WCS Globals ---
R_wcs_to_ccs = None
tvec_wcs_to_ccs = None
wcs_setup_done = False

# --- Trajectory Fitting Globals
fitted_params_dict = None # Will store the dictionary returned by the fitting function
fitting_performed_for_shot = 0 # To track which shot was last fitted

def setup_wcs(camera_matrix, dist_coeffs_val):
    global R_wcs_to_ccs, tvec_wcs_to_ccs, wcs_setup_done

    # 2D Image Points (picked from the undistorted frame)
    # These points correspond to the user-provided image and red dots / estimated features.
    # Ensure these pixel coordinates are accurate for *your specific frame/video setup*.

    # #Session 1
    # image_points = np.array([
    #     (800, 790),  # P1: Shooter's left foot
    #     (1368, 412),  # P2: Front-left of rim
    #     (1368, 375),  # P3: Backboard top-left of inner rectangle
    #     (1400, 355)  # P4: Backboard top-right of inner rectangle
    # ], dtype=np.float32)

    #Session 2
    image_points = np.array([
        (827, 827),  # P1: Shooter's left foot
        (1672, 310),  # P2: Front-left of rim
        (1672, 263),  # P3: Backboard top-left of inner rectangle
        (1742, 235)  # P4: Backboard top-right of inner rectangle
    ], dtype=np.float32)

    # 3D World Points (defined in WCS: Origin at FT line center, X right, Y towards basket, Z up)
    world_points = np.array([
        [0.0, 0.0, 0.0],  # P1: Shooter's left foot (world_p1)
        [WCS_FT_LINE_TO_RIM_CENTER_X, WCS_RIM_HEIGHT_Y, WCS_RIM_RADIUS],  # P2: right of rim (from camera pov) (world_p2_rim)
        [WCS_BACKBOARD_X, WCS_BB_INNER_RECT_TOP_Y, -WCS_BB_INNER_RECT_WIDTH/2],  # P3: BB_UL (world_bb_ul)
        [WCS_BACKBOARD_X, WCS_BB_INNER_RECT_TOP_Y, WCS_BB_INNER_RECT_WIDTH/2]  # P4: BB_UR (world_bb_ur)
    ], dtype=np.float32)

    if DETECT_ON_FULLY_UNDISTORTED_FRAME:
        # If points are from an image already undistorted with P=K,
        # then K is the correct matrix and distortion is effectively zero.
        dist_coeffs_for_solvepnp = None  # or np.zeros((4,1))
    else:
        # If points are from the original distorted image
        dist_coeffs_for_solvepnp = dist_coeffs_val

    try:
        success, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix,
                                           dist_coeffs_for_solvepnp,
                                           flags=cv2.SOLVEPNP_SQPNP)  # cv2.SOLVEPNP_SQPNP might be faster if available
        if success:
            R_wcs_to_ccs, _ = cv2.Rodrigues(rvec)
            tvec_wcs_to_ccs = tvec
            wcs_setup_done = True
            print("WCS setup successful. R_wcs_to_ccs and tvec_wcs_to_ccs computed.")
            print("R_wcs_to_ccs:\n", R_wcs_to_ccs)
            print("tvec_wcs_to_ccs:\n", tvec_wcs_to_ccs)
        else:
            print("WCS setup failed (solvePnP returned False).")
            wcs_setup_done = False
    except Exception as e:
        print(f"Error during solvePnP: {e}")
        wcs_setup_done = False
    return wcs_setup_done


def ccs_to_wcs(Pc_ccs):
    if not wcs_setup_done or R_wcs_to_ccs is None or tvec_wcs_to_ccs is None:
        return None
    # Pc_wcs = R_ccs_to_wcs * (Pc_ccs - tvec_wcs_to_ccs_origin_in_ccs)
    # R_ccs_to_wcs = R_wcs_to_ccs.T
    # tvec_ccs_to_wcs_origin_in_wcs = -R_wcs_to_ccs.T @ tvec_wcs_to_ccs
    # Pc_wcs = R_wcs_to_ccs.T @ Pc_ccs + (-R_wcs_to_ccs.T @ tvec_wcs_to_ccs)
    # Pc_wcs = R_wcs_to_ccs.T @ (Pc_ccs - tvec_wcs_to_ccs)

    # Ensure Pc_ccs is a column vector (3,1)
    Pc_ccs_col = Pc_ccs.reshape((3, 1))
    Pc_wcs_col = R_wcs_to_ccs.T @ (Pc_ccs_col - tvec_wcs_to_ccs)
    return Pc_wcs_col.flatten()  # Return as a flat array (1D)

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

    pose_results_val = pose_model_obj(frame_to_detect_on_val, conf=0.35, verbose=False)
    shooter_wrist_undistorted_px_final = None
    if ball_bbox_for_logic_val is not None:  # Only proceed if we have a valid ball bbox for context

        # Original
        # wrist_detected_val, _ = utils.get_likely_shooter_wrist_and_person_idx(pose_results_val, ball_bbox_for_logic_val,
        #                                                                       SHOOTING_WRIST_KP_INDEX,
        #                                                                       MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION)
        print(f"Frame Wrist Detection: {current_frame_num_val}")
        wrist_detected_val, person_idx_val= utils.get_likely_shooter_wrist_and_person_idx(
            pose_results_list=pose_results_val,  # This is the list of YOLO Results objects
            ball_bbox=ball_bbox_for_logic_val,
            shooting_arm_wrist_idx=SHOOTING_ARM_WRIST_IDX,
            shooting_arm_elbow_idx=SHOOTING_ARM_ELBOW_IDX,
            shooting_arm_shoulder_idx=SHOOTING_ARM_SHOULDER_IDX,
            max_dist_threshold=MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION,  # Your existing global
        )

        if wrist_detected_val is not None:
            if do_undistort_points_val and not is_full_frame_undistorted_already:
                shooter_wrist_undistorted_px_final = undistort_point(wrist_detected_val, K_matrix_val, D_coeffs_val)
            else:
                shooter_wrist_undistorted_px_final = (int(wrist_detected_val[0]), int(wrist_detected_val[1]))

    return ball_center_undistorted_px_final, ball_bbox_for_logic_val, shooter_wrist_undistorted_px_final, used_low_conf_heuristic_val_flag


def check_for_release_ccs(frame_num_val, ball_center_undist_px, ball_bbox_l_val, wrist_pos_undist_px,
                          current_release_state_val, release_buffer_list_val, frame_orig_val,
                          manual_override_frame_val, manual_override_shot_target_val, current_shot_count_val,
                          kf_obj_to_init):
    # Globals MODIFIED by this function or its helper _init_release_state
    global release_frame_info_dict, shot_count, Zc_at_release, kf_initialized
    global trajectory_points_raw_pixels_undistorted, trajectory_points_kf_pixels_undistorted, trajectory_points_ccs
    global estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs
    # --- Add WCS related globals that need resetting for a new shot ---
    global trajectory_points_wcs, estimated_speed_mps_wcs, estimated_angle_deg_elev_wcs, estimated_angle_deg_azim_wcs

    new_release_state_val = current_release_state_val
    forced_release_this_frame = False

    def _init_release_state(u_rel, v_rel, current_frame_num_init, bbox_at_release, frame_original_at_release,
                            type_of_release="auto"):
        global Zc_at_release, kf_initialized
        global trajectory_points_raw_pixels_undistorted, trajectory_points_kf_pixels_undistorted, trajectory_points_ccs
        global estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs
        # --- Reset WCS variables ---
        global trajectory_points_wcs, estimated_speed_mps_wcs, estimated_angle_deg_elev_wcs, estimated_angle_deg_azim_wcs
        global fitted_params_dict

        estimated_speed_mps = 0.0
        estimated_angle_deg_XY_ccs = 0.0  # This was side angle in CCS XY plane
        estimated_angle_deg_elev_ccs = 0.0  # This was elevation in CCS

        # WCS estimates reset
        estimated_speed_mps_wcs = 0.0
        estimated_angle_deg_elev_wcs = 0.0
        estimated_angle_deg_azim_wcs = 0.0

        trajectory_points_raw_pixels_undistorted.clear()
        trajectory_points_kf_pixels_undistorted.clear()
        trajectory_points_ccs.clear()
        trajectory_points_wcs.clear()  # Clear WCS trajectory too

        #Reset fitted parameters
        fitted_params_dict = None

        trajectory_points_raw_pixels_undistorted.append(
            {'frame': current_frame_num_init, 'px_coord': (u_rel, v_rel), 'type': f'{type_of_release}_raw_init'})

        app_diam_px_rel = ((bbox_at_release[2] - bbox_at_release[0]) + (bbox_at_release[3] - bbox_at_release[1])) / 2.0
        if app_diam_px_rel > 0:
            avg_focal_px = (fx + fy) / 2.0
            Zc_at_release = (BASKETBALL_REAL_DIAMETER_M * avg_focal_px) / app_diam_px_rel
        else:
            Zc_at_release = 1.0

        xn_rel = (u_rel - cx) / fx;
        yn_rel = (v_rel - cy) / fy
        Pc_rel_ccs = np.array([xn_rel * Zc_at_release, yn_rel * Zc_at_release, Zc_at_release])

        kf_obj_to_init.statePost = np.array([u_rel, v_rel, 0, 0], dtype=np.float32).reshape(-1, 1)
        kf_obj_to_init.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
        kf_initialized = True

        trajectory_points_kf_pixels_undistorted.append(
            {'frame': current_frame_num_init, 'px_coord': (u_rel, v_rel), 'type': f'kf_{type_of_release}_init'})
        trajectory_points_ccs.append(
            {'frame': current_frame_num_init, 'Pc_ccs': Pc_rel_ccs, 'px_coord_type': f'kf_{type_of_release}_init'})

        # --- Add initial point to WCS trajectory ---
        Pc_rel_wcs = ccs_to_wcs(Pc_rel_ccs)
        if Pc_rel_wcs is not None:
            trajectory_points_wcs.append(
                {'frame': current_frame_num_init, 'Pc_wcs': Pc_rel_wcs, 'px_coord_type': f'kf_{type_of_release}_init'})
            print(f"  Release Ball Coords (WCS, m): {Pc_rel_wcs.round(3)}")
        else:
            print("  Could not transform release point to WCS (WCS not setup or transform failed).")

        print(f"  Release Ball Center (undistorted px): ({u_rel}, {v_rel})")
        print(f"  Est. Zc at Release (m): {Zc_at_release:.3f}")
        print(f"  Release Ball Coords (CCS, m): {Pc_rel_ccs.round(3)}. KF Initialized.")

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

    # ... (rest of your check_for_release_ccs logic)
    # Manual Release Override Logic
    if not current_release_state_val and manual_override_frame_val is not None and \
            current_shot_count_val + 1 == manual_override_shot_target_val and frame_num_val == manual_override_frame_val:
        if ball_center_undist_px is not None and ball_bbox_l_val is not None:
            shot_count += 1
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
            ball_center_undist_px is not None and wrist_pos_undist_px is not None and ball_bbox_l_val is not None and manual_override_frame_val is None :
        wrist_in_ball = utils.is_point_inside_bbox(wrist_pos_undist_px, ball_bbox_l_val,
                                                   margin=WRIST_BALL_PROXIMITY_MARGIN)
        if not wrist_in_ball:
            release_buffer_list_val.append(
                {"frame_no": frame_num_val, "ball_center_undistorted_px": ball_center_undist_px,
                 "ball_bbox_detection": ball_bbox_l_val, "original_frame_at_this_moment": frame_orig_val.copy()})
            if len(release_buffer_list_val) > CONFIRMATION_FRAMES_FOR_RELEASE: release_buffer_list_val.pop(0)
            if len(release_buffer_list_val) == CONFIRMATION_FRAMES_FOR_RELEASE:
                shot_count += 1
                release_frame_info_dict = release_buffer_list_val[0].copy()
                new_release_state_val = True

                u_rel_auto, v_rel_auto = release_frame_info_dict['ball_center_undistorted_px']
                bbox_auto = release_frame_info_dict['ball_bbox_detection']
                frame_orig_auto = release_frame_info_dict['original_frame_at_this_moment']
                _init_release_state(u_rel_auto, v_rel_auto, release_frame_info_dict['frame_no'], bbox_auto,
                                    frame_orig_auto, "auto")
                print(f"AUTO RELEASE (Shot {shot_count}) at frame {release_frame_info_dict['frame_no']}.")
                release_buffer_list_val.clear()
        else:
            release_buffer_list_val.clear()
            if current_release_state_val:
                new_release_state_val = False
                print(f"  Frame {frame_num_val}: Shot {shot_count} release cancelled (wrist returned to ball).")
    elif current_release_state_val and ball_center_undist_px is None:
        pass

    return new_release_state_val, release_buffer_list_val


def update_trajectory_and_estimate_ccs(current_release_state_val,
                                       ball_point_from_kf_px_dict,
                                       current_Zc_at_release, video_fps_val):
    # Globals MODIFIED or used
    global estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs, shot_count # CCS estimates
    global trajectory_points_kf_pixels_undistorted, trajectory_points_ccs
    # --- WCS globals ---
    global trajectory_points_wcs, estimated_speed_mps_wcs, estimated_angle_deg_elev_wcs, estimated_angle_deg_azim_wcs

    if current_release_state_val and ball_point_from_kf_px_dict is not None and current_Zc_at_release is not None:
        frame_num_val = ball_point_from_kf_px_dict['frame']
        u_kf, v_kf = ball_point_from_kf_px_dict['px_coord']
        point_type = ball_point_from_kf_px_dict['type']

        # ... (KF pixel trajectory update logic remains the same) ...
        if not trajectory_points_kf_pixels_undistorted or \
                trajectory_points_kf_pixels_undistorted[-1]['frame'] != frame_num_val:
            trajectory_points_kf_pixels_undistorted.append(ball_point_from_kf_px_dict)


        # Convert KF point to CCS and append
        current_Pc_ccs_kf = None
        if not trajectory_points_ccs or trajectory_points_ccs[-1]['frame'] != frame_num_val:
            xn_kf = (u_kf - cx) / fx; yn_kf = (v_kf - cy) / fy
            current_Pc_ccs_kf = np.array([xn_kf * current_Zc_at_release, yn_kf * current_Zc_at_release, current_Zc_at_release])
            trajectory_points_ccs.append({'frame': frame_num_val, 'Pc_ccs': current_Pc_ccs_kf, 'px_coord_type': point_type})

            # --- Transform to WCS and append ---
            if wcs_setup_done and current_Pc_ccs_kf is not None: # Check current_Pc_ccs_kf
                Pc_wcs_kf = ccs_to_wcs(current_Pc_ccs_kf)
                if Pc_wcs_kf is not None:
                    trajectory_points_wcs.append({'frame': frame_num_val, 'Pc_wcs': Pc_wcs_kf, 'px_coord_type': point_type})

        # Naive parameter estimation (CCS - original)
        if estimated_speed_mps == 0.0 and len(trajectory_points_ccs) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
            p_start_ccs_info = trajectory_points_ccs[0]
            p_end_ccs_info = trajectory_points_ccs[FRAMES_FOR_NAIVE_ESTIMATION] # Ensure this index is valid
            Pc_s_ccs = p_start_ccs_info['Pc_ccs']; Pc_e_ccs = p_end_ccs_info['Pc_ccs']
            delta_Pc_ccs_val = Pc_e_ccs - Pc_s_ccs
            d_frames_ccs = p_end_ccs_info['frame'] - p_start_ccs_info['frame']

            if video_fps_val > 0 and d_frames_ccs > 0:
                d_t_s_ccs = d_frames_ccs / video_fps_val
                Vc_avg_ccs_val = delta_Pc_ccs_val / d_t_s_ccs
                estimated_speed_mps = np.linalg.norm(Vc_avg_ccs_val)
                # CCS: X right, Y down, Z forward
                mag_XZ_ccs = math.sqrt(Vc_avg_ccs_val[0] ** 2 + Vc_avg_ccs_val[2] ** 2)
                if mag_XZ_ccs < 1e-6: mag_XZ_ccs = 1e-6
                estimated_angle_deg_elev_ccs = math.degrees(math.atan2(-Vc_avg_ccs_val[1], mag_XZ_ccs)) # -Yc because Yc is down
                estimated_angle_deg_XY_ccs = math.degrees(math.atan2(Vc_avg_ccs_val[0], Vc_avg_ccs_val[2])) # Azimuth in XZ_ccs plane from Z_ccs

        # --- Naive parameter estimation (WCS) ---
        # WCS: X towards hoop, Y up, Z right
        if wcs_setup_done and estimated_speed_mps_wcs == 0.0 and len(trajectory_points_wcs) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
            p_start_wcs_info = trajectory_points_wcs[0]
            p_end_wcs_info = trajectory_points_wcs[FRAMES_FOR_NAIVE_ESTIMATION] # Ensure this index is valid

            Pc_s_wcs = p_start_wcs_info['Pc_wcs']; Pc_e_wcs = p_end_wcs_info['Pc_wcs']
            delta_Pc_wcs_val = Pc_e_wcs - Pc_s_wcs
            d_frames_wcs = p_end_wcs_info['frame'] - p_start_wcs_info['frame']

            if video_fps_val > 0 and d_frames_wcs > 0:
                d_t_s_wcs = d_frames_wcs / video_fps_val
                Vc_avg_wcs_val = delta_Pc_wcs_val / d_t_s_wcs # [VX_wcs, VY_wcs, VZ_wcs]
                estimated_speed_mps_wcs = np.linalg.norm(Vc_avg_wcs_val)

                # Elevation angle (angle with XZ_wcs plane, i.e., the ground plane)
                # VY_wcs is the vertical component, sqrt(VX_wcs^2 + VZ_wcs^2) is projection on XZ_wcs plane.
                mag_XZ_wcs = math.sqrt(Vc_avg_wcs_val[0]**2 + Vc_avg_wcs_val[2]**2)
                if mag_XZ_wcs < 1e-6: mag_XZ_wcs = 1e-6 # Avoid division by zero if shot is purely vertical
                estimated_angle_deg_elev_wcs = math.degrees(math.atan2(Vc_avg_wcs_val[1], mag_XZ_wcs))

                # Azimuth angle (in XZ_wcs plane, angle from X_wcs-axis towards Z_wcs-axis)
                # This represents side-to-side deviation. X_wcs is straight towards hoop.
                # atan2(VZ_wcs, VX_wcs)
                estimated_angle_deg_azim_wcs = math.degrees(math.atan2(Vc_avg_wcs_val[2], Vc_avg_wcs_val[0]))

                print(
                    f"  NAIVE WCS (Shot {shot_count}, frames {p_start_wcs_info['frame']}-{p_end_wcs_info['frame']}): "
                    f"Speed={estimated_speed_mps_wcs:.2f}m/s, Elev(Y/XZ)={estimated_angle_deg_elev_wcs:.1f}deg, Azim(Z/X)={estimated_angle_deg_azim_wcs:.1f}deg")

def draw_annotations_on_frame_ccs(frame_to_annotate_val, current_frame_number_for_drawing,
                                  ball_bbox_d_val, raw_ball_center_undist_px_val,
                                  wrist_pos_undist_px_val, is_release_detected_val,
                                  release_info_dict_val, traj_kf_pix_list_of_dicts,
                                  traj_raw_pix_list_of_dicts, shot_num_val,
                                  est_speed_val_ccs, est_angle_elev_val_ccs, # CCS estimates
                                  # WCS estimates are accessed via global variables in this example
                                  used_low_conf_flag_val,
                                  kf_current_prediction_px_val=None):

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
                                              f"S{shot_num_val} Rls:Fr{release_info_dict_val['frame_no']}",
                                              (10, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if traj_kf_pix_list_of_dicts and traj_kf_pix_list_of_dicts[0]['px_coord'] is not None:
            cv2.circle(frame_to_annotate_val, traj_kf_pix_list_of_dicts[0]['px_coord'], 15, (0, 0, 255),
                       2)  # Release marker

        if est_speed_val_ccs > 0:  # CCS estimates
            cv2.putText(frame_to_annotate_val, f"Speed(CCS): {est_speed_val_ccs:.2f}m/s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(frame_to_annotate_val, f"Elev(CCS): {est_angle_elev_val_ccs:.1f}deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        start_y_offset = 120  # Start WCS text below CCS text
        if wcs_setup_done and estimated_speed_mps_wcs > 0:  # Naive WCS
            cv2.putText(frame_to_annotate_val, f"Speed(WCS-N): {estimated_speed_mps_wcs:.2f}m/s", (10, start_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 100), 2)
            cv2.putText(frame_to_annotate_val, f"Elev(WCS-N): {estimated_angle_deg_elev_wcs:.1f}deg",
                        (10, start_y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 100), 2)
            cv2.putText(frame_to_annotate_val, f"Azim(WCS-N): {estimated_angle_deg_azim_wcs:.1f}deg",
                        (10, start_y_offset + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 100), 2)
            start_y_offset += 90

        # --- Display WCS estimates (using global variables for simplicity here) ---
        global fitted_params_dict  # Use the dictionary
        if fitted_params_dict and fitted_params_dict.get("success") and \
                fitting_performed_for_shot == shot_num_val:  # Check if fit was for current shot

            start_y_offset_fit = 120  # Continue from where naive WCS text left off
            # (Assuming start_y_offset was updated after naive WCS text)

            cv2.putText(frame_to_annotate_val, f"Speed(WCS-Fit): {fitted_params_dict['speed_mps_wcs']:.2f}m/s",
                        (10, start_y_offset_fit),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_to_annotate_val, f"Elev(WCS-Fit): {fitted_params_dict['angle_deg_elev_wcs']:.1f}deg",
                        (10, start_y_offset_fit + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_to_annotate_val, f"Azim(WCS-Fit): {fitted_params_dict['angle_deg_azim_wcs']:.1f}deg",
                        (10, start_y_offset_fit + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        # ... (existing pre-release text) ...
        cv2.putText(frame_to_annotate_val, "Awaiting Release", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        if not wcs_setup_done:
            cv2.putText(frame_to_annotate_val, "WCS NOT READY", (frame_to_annotate_val.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame_to_annotate_val

def fitting_wcs_tracking():
    if shot_count > 0 and wcs_setup_done and len(
            trajectory_points_wcs) >= FRAMES_FOR_NAIVE_ESTIMATION + 1:
        print(f"\nAttempting to fit trajectory for the last processed shot (Shot {shot_count})...")

        initial_p0_guess_fit = trajectory_points_wcs[0]['Pc_wcs']
        initial_v_speed_angle_guess_fit = [
            estimated_speed_mps_wcs if estimated_speed_mps_wcs > 0 else 8.0,
            estimated_angle_deg_elev_wcs if estimated_angle_deg_elev_wcs != 0 else 45.0,
            estimated_angle_deg_azim_wcs if estimated_angle_deg_azim_wcs != 0 else 0.0
        ]

        # Call the fitting function from trajectory_utils
        # The tu.fit_shot_trajectory_parameters should now update globals if you design it that way,
        # or preferably, return a dictionary of fitted parameters.
        # I've modified tu.fit_shot_trajectory_parameters to return a dictionary.

        fitted_params_result = tu.fit_shot_trajectory_parameters(
            trajectory_points_wcs,
            video_fps_g,
            initial_p0_guess_fit,
            initial_v_speed_angle_guess_fit  # Or some other value
        )

        global fitted_params_dict  # Declare we are modifying this global
        global fitting_performed_for_shot
        fitted_params_dict = fitted_params_result  # Store the whole result

        if fitted_params_dict and fitted_params_dict.get("success"):
            fitting_performed_for_shot = shot_count  # Mark which shot was fitted

            print(f"\nPlotting trajectory for Shot {fitting_performed_for_shot})...")

            # Prepare court dimensions from wcs_kf_tracking constants for plotting
            plot_court_dimensions = {
                "ft_to_rim_x": WCS_FT_LINE_TO_RIM_CENTER_X,
                "rim_y": WCS_RIM_HEIGHT_Y,
                "rim_diameter": WCS_RIM_RADIUS * 2,  # tu.plot expects diameter
                "backboard_x": WCS_BACKBOARD_X,
                "backboard_height": 1.0668,  # Standard backboard height
                "bb_inner_rect_height": WCS_BB_INNER_RECT_HEIGHT
            }

            tu.plot_fitted_trajectory_wcs(
                observed_wcs_positions=trajectory_points_wcs,  # Pass the list of dicts
                fitted_P0=fitted_params_dict["P0_wcs"],
                fitted_V0_components=fitted_params_dict["V0_wcs_components"],
                shot_number=fitting_performed_for_shot,
                video_fps=video_fps_g,
                output_dir=VIDEO_OUTPUT_DIR,
                video_basename=os.path.splitext(os.path.basename(video_path))[0],
                court_dims=plot_court_dimensions
            )
        else:
            print(f"Fitting failed for shot {shot_count} or no result, no plot will be generated.")
            if fitted_params_dict: print(f"Fitting message: {fitted_params_dict.get('message')}")

        # Save KF WCS trajectory (for the last shot)
        # ... (this saving logic can remain largely the same) ...
        wcs_kf_traj_output_path = os.path.join(VIDEO_OUTPUT_DIR,
                                               f"{os.path.splitext(os.path.basename(video_path))[0]}_shot{shot_count}_wcs_kf_trajectory.txt")
        with open(wcs_kf_traj_output_path, 'w') as f_out:
            f_out.write("frame,x_wcs,y_wcs,z_wcs,type\n")
            for pt_info in trajectory_points_wcs:
                f_out.write(
                    f"{pt_info['frame']},{pt_info['Pc_wcs'][0]},{pt_info['Pc_wcs'][1]},{pt_info['Pc_wcs'][2]},{pt_info['px_coord_type']}\n")
        print(f"KF WCS trajectory for shot {shot_count} saved to {wcs_kf_traj_output_path}")

        # Save fitted parameters (if fitting was successful for the last shot)
        if fitted_params_dict and fitted_params_dict.get("success") and fitting_performed_for_shot == shot_count:
            fitted_params_output_path = os.path.join(VIDEO_OUTPUT_DIR,
                                                     f"{os.path.splitext(os.path.basename(video_path))[0]}_shot{fitting_performed_for_shot}_fitted_params.txt")
            with open(fitted_params_output_path, 'w') as f_out:
                f_out.write(f"Fitted_P0_wcs: {fitted_params_dict['P0_wcs'].tolist()}\n")
                f_out.write(f"Fitted_V0_wcs_components: {fitted_params_dict['V0_wcs_components'].tolist()}\n")
                f_out.write(f"Fitted_Speed_mps_wcs: {fitted_params_dict['speed_mps_wcs']}\n")
                f_out.write(f"Fitted_Angle_Elev_wcs: {fitted_params_dict['angle_deg_elev_wcs']}\n")
                f_out.write(f"Fitted_Angle_Azim_wcs: {fitted_params_dict['angle_deg_azim_wcs']}\n")
                f_out.write(f"KB_BALL_used: {tu.KB_BALL}\n")  # Access KB_BALL from tu
                f_out.write(f"Initial_P0_Guess: {initial_p0_guess_fit.tolist()}\n")
                f_out.write(f"Initial_V_Angle_Guess: {initial_v_speed_angle_guess_fit}\n")
                f_out.write(f"Fitting_Success_Message: {fitted_params_dict.get('message', 'N/A')}\n")
            print(f"Fitted parameters for shot {fitting_performed_for_shot} saved to {fitted_params_output_path}")
    else:
        # ... (existing else conditions for not fitting) ...
        pass

# --- Main Processing Loop ---
if __name__ == "__main__":
    os.makedirs(RELEASE_FRAME_OUTPUT_DIR, exist_ok=True)

    MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "Yolo-Weights")
    BALL_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "model2.pt")
    POSE_MODEL_PATH = 'yolov8n-pose.pt'

    VIDEO_INPUT_DIR = os.path.join(BASE_DIR, "footage")
    VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "trajectory_fitting", "output_video_wcs_fit")  # Changed output dir slightly
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    video_path = os.path.join(VIDEO_INPUT_DIR, "freethrow_3v2.mp4")
    output_filename_suffix = "_wcs_kf"
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

    # --- Setup WCS ---
    # Call setup_wcs once, K is global. D might be needed if not using fully undistorted frames for solvePnP points
    if not setup_wcs(K, D):  # Pass D in case you change DETECT_ON_FULLY_UNDISTORTED_FRAME
        print("WARNING: WCS setup failed. WCS transformations will not be available.")
        # Decide if you want to exit or continue without WCS
        # exit("Exiting due to WCS setup failure.")

    # --- Main loop state variables ---
    frame_count = 0
    shot_count = 0
    release_detected_in_shot = False
    release_frame_info_dict = None
    potential_release_buffer = []

    trajectory_points_raw_pixels_undistorted = []
    trajectory_points_kf_pixels_undistorted = []
    trajectory_points_ccs = []  # List of {'frame', 'Pc_ccs', ...}
    trajectory_points_wcs = []  # List of {'frame', 'Pc_wcs', ...} # NEW

    Zc_at_release = None
    # CCS estimates
    estimated_speed_mps = 0.0
    estimated_angle_deg_XY_ccs = 0.0  # This was your side angle in CCS XY plane
    estimated_angle_deg_elev_ccs = 0.0
    # WCS estimates
    estimated_speed_mps_wcs = 0.0  # NEW
    estimated_angle_deg_elev_wcs = 0.0  # NEW - Elevation from ground
    estimated_angle_deg_azim_wcs = 0.0  # NEW - Azimuth from Y_wcs (towards basket)

    kf = setup_kalman_filter(dt=1.0 / video_fps_g, noise_acceleration_variance=NOISE_ACCELERATION_VARIANCE)
    kf_initialized = False
    missed_detections_count_kf = 0
    kf_current_predicted_pixel_point_for_drawing = None

    print(f"Processing video: {video_path} (FPS: {video_fps_g:.2f})")
    print(f"Outputting to: {output_path}")
    if MANUAL_RELEASE_FRAME_OVERRIDE is not None:
        print(
            f"MANUAL RELEASE OVERRIDE ENABLED: Shot {MANUAL_RELEASE_SHOT_NUMBER_TARGET}, Target Frame {MANUAL_RELEASE_FRAME_OVERRIDE}")

    # --- Main Loop ---
    while cap_g.isOpened():
        ret, frame_original_main = cap_g.read()
        if not ret: break
        frame_count += 1

        # (Your existing frame processing, detection, release check, KF update logic...)
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}, Shot: {shot_count}, ReleaseActive: {release_detected_in_shot}, "
                  f"RawPts: {len(trajectory_points_raw_pixels_undistorted)}, KFPts: {len(trajectory_points_kf_pixels_undistorted)}, "
                  f"CCSPts: {len(trajectory_points_ccs)}, WCSPts: {len(trajectory_points_wcs)}, KFinit: {kf_initialized}, WCSReady: {wcs_setup_done}")

        current_frame_to_detect = preprocess_frame_ccs(frame_original_main, K, D, map1_g, map2_g,
                                                       DETECT_ON_FULLY_UNDISTORTED_FRAME)
        annotated_frame = current_frame_to_detect.copy()  # Make sure to annotate on the same frame type as points were picked for solvePnP if drawing those points

        active_conf_thresh = POST_RELEASE_BALL_CONF if release_detected_in_shot else PRE_RELEASE_BALL_CONF

        ball_center_detected_undist_px, ball_bbox_l, shooter_wrist_undist_px, used_low_conf_h = \
            get_processed_detections_ccs(frame_count, current_frame_to_detect, ball_model_g, pose_model_g,
                                         id_of_ball_g,
                                         active_conf_thresh, K, D,
                                         UNDISTORT_POINTS_AFTER_DETECTION, DETECT_ON_FULLY_UNDISTORTED_FRAME,
                                         trajectory_points_kf_pixels_undistorted,
                                         MAX_PIXEL_DISPLACEMENT_POST_RELEASE,
                                         release_detected_in_shot)

        prev_release_state = release_detected_in_shot
        release_detected_in_shot, potential_release_buffer = \
            check_for_release_ccs(frame_count, ball_center_detected_undist_px, ball_bbox_l, shooter_wrist_undist_px,
                                  release_detected_in_shot, potential_release_buffer, frame_original_main,
                                  # Pass original frame for saving
                                  MANUAL_RELEASE_FRAME_OVERRIDE, MANUAL_RELEASE_SHOT_NUMBER_TARGET, shot_count, kf)

        if not prev_release_state and release_detected_in_shot:
            print(
                f"  Frame {frame_count}: New shot (Shot {shot_count}) detected. KF initialized: {kf_initialized}.")
            missed_detections_count_kf = 0
        elif prev_release_state and not release_detected_in_shot:
            print(f"  Frame {frame_count}: Shot {shot_count} tracking stopped.")
            kf_initialized = False

        kf_current_predicted_pixel_point_for_drawing = None
        ball_point_from_kf_for_ccs_update = None

        if release_detected_in_shot and kf_initialized:
            predicted_kf_state = kf.predict()
            kf_current_predicted_pixel_point_for_drawing = (
                int(round(predicted_kf_state[0, 0])), int(round(predicted_kf_state[1, 0])))

            if ball_center_detected_undist_px is not None:
                dist_sq_to_kf_pred = (ball_center_detected_undist_px[0] -
                                      kf_current_predicted_pixel_point_for_drawing[0]) ** 2 + \
                                     (ball_center_detected_undist_px[1] -
                                      kf_current_predicted_pixel_point_for_drawing[1]) ** 2
                KF_CORRECTION_GATE_SQ = (MAX_PIXEL_DISPLACEMENT_POST_RELEASE * 1.5) ** 2

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
                else:
                    missed_detections_count_kf += 1
                    if missed_detections_count_kf <= MAX_CONSECUTIVE_MISSES_KF:
                        ball_point_from_kf_for_ccs_update = {'frame': frame_count,
                                                             'px_coord': kf_current_predicted_pixel_point_for_drawing,
                                                             'type': 'kf_pred_gated'}
            else:
                missed_detections_count_kf += 1
                if missed_detections_count_kf <= MAX_CONSECUTIVE_MISSES_KF:
                    ball_point_from_kf_for_ccs_update = {'frame': frame_count,
                                                         'px_coord': kf_current_predicted_pixel_point_for_drawing,
                                                         'type': 'kf_pred_miss'}

            if ball_point_from_kf_for_ccs_update is not None:
                update_trajectory_and_estimate_ccs(
                    release_detected_in_shot,
                    ball_point_from_kf_for_ccs_update,
                    Zc_at_release,
                    video_fps_g
                )
            elif missed_detections_count_kf > MAX_CONSECUTIVE_MISSES_KF:
                print(
                    f"  Frame {frame_count}: Ball lost for Shot {shot_count} (KF max misses). Trajectory not updated.")
                # release_detected_in_shot = False # Optional: terminate shot
                # kf_initialized = False

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
            estimated_speed_mps, estimated_angle_deg_elev_ccs,  # Pass CCS estimates
            used_low_conf_h, kf_current_predicted_pixel_point_for_drawing
        )

        cv2.imshow("CCS WCS KF Test", annotated_frame)
        out_g.write(annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("Processing stopped by user."); break
        if key == ord('p'): print(f"Paused at frame {frame_count}. Press any key to continue..."); cv2.waitKey(-1)

    cap_g.release()
    out_g.release()
    cv2.destroyAllWindows()
    print(f"Finished processing. Output saved to {output_path}")
    fitting_wcs_tracking()

