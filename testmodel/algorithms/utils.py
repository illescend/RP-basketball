import cv2
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# --- Configuration ---
# Pose Keypoint for shooting hand's wrist (COCO format: 9 for left_wrist, 10 for right_wrist)
SHOOTING_WRIST_KP_INDEX = 10  # Assuming right wrist for Curry

# Release Detection Parameters
CONFIRMATION_FRAMES_FOR_RELEASE = 3  # Wrist must be outside ball for this many consecutive frames

# Optional: A small pixel margin to consider wrist "inside" even if slightly outside bbox due to hand size/occlusion
WRIST_BALL_PROXIMITY_MARGIN = 30 # Pixels. If wrist is within this margin of bbox, still considered "close" / potentially "in hand"

#freethrow_1 = 10

# --- NEW: Pose Keypoint Indices (COCO format, used by YOLOv8-Pose) ---
# These are standard and useful to have defined
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# --- Configuration for Geometric Checks ---
MIN_WRIST_CONFIDENCE = 0.9  # Minimum confidence for a wrist keypoint to be considered
MIN_ELBOW_CONFIDENCE = 0.9# Minimum confidence for an elbow keypoint
MIN_SHOULDER_CONFIDENCE = 0.9 # Minimum confidence for a shoulder keypoint

MIN_WRIST_ELBOW_SEPARATION_PIXELS = 10 # Wrist should be at least this far from elbow
                                     # Adjust based on typical arm segment lengths in your video resolution
MAX_SHOULDER_ELBOW_WRIST_ANGLE_DEG = 180 # Angle should be somewhat straight (e.g. > 90-100 deg for extension)
                                        # Max of 165 allows for slight bend, 180 is perfectly straight.
                                        # A very small angle (e.g. < 45 deg) would be a sharply bent arm.
MIN_SHOULDER_ELBOW_WRIST_ANGLE_DEG = 120 # Allow for some reasonable bend, but not fully collapsed

# --- Helper Functions ---
def get_ball_bbox(ball_results_list, target_ball_class_id):
    """Extracts the bounding box of the first detected ball of the target class."""
    if ball_results_list and len(ball_results_list) > 0:
        res = ball_results_list[0]
        if res.boxes and len(res.boxes.data) > 0:
            for box_data in res.boxes.data: # box_data is [x1, y1, x2, y2, conf, class_id]
                if int(box_data[5]) == target_ball_class_id:
                    return box_data[:4].cpu().numpy().astype(int) # Return [x1, y1, x2, y2]
    return None

def get_wrist_coordinates(pose_results_list, person_index=0, wrist_kp_index=SHOOTING_WRIST_KP_INDEX):
    """Extracts the (x, y) coordinates of the specified wrist keypoint for a given person."""
    if pose_results_list and len(pose_results_list) > 0:
        res = pose_results_list[0]
        if res.keypoints and res.keypoints.xy is not None and len(res.keypoints.xy) > person_index:
            keypoints_person = res.keypoints.xy[person_index]
            if len(keypoints_person) > wrist_kp_index:
                wrist_coord = keypoints_person[wrist_kp_index].cpu().numpy()
                # Check if keypoint is detected (often 0,0 if not)
                # A more robust check would be to use res.keypoints.conf if available and threshold it
                if wrist_coord[0] > 0 and wrist_coord[1] > 0: # Simple check for validity
                    return wrist_coord.astype(int)
    return None

def is_point_inside_bbox(point, bbox, margin=0):
    """Checks if a point (x,y) is inside a bounding box [x1,y1,x2,y2] with an optional margin."""
    if point is None or bbox is None:
        return False
    px, py = point
    bx1, by1, bx2, by2 = bbox
    return (bx1 - margin <= px <= bx2 + margin) and \
           (by1 - margin <= py <= by2 + margin)

def calculate_distance_point_to_bbox_edge(point, bbox):
    """Calculates the distance from a point to the closest edge of a bounding box.
       Returns 0 if the point is inside the bbox."""
    if point is None or bbox is None:
        return float('inf')

    px, py = point
    bx1, by1, bx2, by2 = bbox

    # If point is inside, distance is 0
    if bx1 <= px <= bx2 and by1 <= py <= by2:
        return 0.0

    dx = max(0, bx1 - px, px - bx2)
    dy = max(0, by1 - py, py - by2)
    return np.sqrt(dx**2 + dy**2)


def calculate_angle(p1, p2, p3):
    """Calculates the angle at p2 formed by p1-p2-p3.
       p1, p2, p3 are 2D points (x,y) or numpy arrays."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    v1 = p1 - p2
    v2 = p3 - p2

    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    if mag_v1 * mag_v2 == 0:  # Avoid division by zero if points coincide
        return 180.0  # Or 0.0, depending on convention for coincident points

    cos_angle = dot_product / (mag_v1 * mag_v2)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip for numerical stability
    return np.degrees(angle_rad)


def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


# --- NEW CONFIGURATION ---
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = 75  # Pixels. If closest wrist is further than this, ignore.


# Adjust based on ball size and typical shooting poses.
# This value should be larger than WRIST_BALL_PROXIMITY_MARGIN
# but small enough to exclude non-interacting players.

# --- MODIFIED HELPER FUNCTION ---
# def get_likely_shooter_wrist_and_person_idx(pose_results_list, ball_bbox, wrist_kp_index, max_dist_threshold):
#     """
#     Finds the wrist of the person most likely holding/shooting the ball.
#     Returns: (wrist_coordinates, person_index_in_results) or (None, None)
#     """
#     if not pose_results_list or len(pose_results_list) == 0 or ball_bbox is None:
#         return None, None
#
#     ball_center_x = (ball_bbox[0] + ball_bbox[2]) // 2
#     ball_center_y = (ball_bbox[1] + ball_bbox[3]) // 2
#     ball_center = np.array([ball_center_x, ball_center_y])
#
#     min_dist = float('inf')
#     closest_wrist_pos = None
#     shooter_person_idx = -1
#
#     # Iterate through all detected persons in the frame
#     # pose_results_list[0] is the Results object for the current frame
#     # pose_results_list[0].keypoints.xy is a tensor of [num_persons, num_keypoints, 2]
#     if pose_results_list[0].keypoints and pose_results_list[0].keypoints.xy is not None:
#         all_persons_keypoints_xy = pose_results_list[0].keypoints.xy.cpu().numpy()
#
#         for person_idx, person_kps in enumerate(all_persons_keypoints_xy):
#             if len(person_kps) > wrist_kp_index:
#                 wrist_coord_raw = person_kps[wrist_kp_index]
#                 # Check if keypoint is detected (often 0,0 if not, or use confidence if available)
#                 if wrist_coord_raw[0] > 1 and wrist_coord_raw[1] > 1:  # Basic validity check
#                     current_wrist_pos = wrist_coord_raw.astype(int)
#
#                     # Calculate distance from this wrist to ball center
#                     # dist = np.linalg.norm(current_wrist_pos - ball_center)
#                     # OR use your existing distance to bbox edge for consistency in distance metrics:
#                     dist = calculate_distance_point_to_bbox_edge(current_wrist_pos, ball_bbox)
#
#                     if dist < min_dist:
#                         min_dist = dist
#                         closest_wrist_pos = current_wrist_pos
#                         shooter_person_idx = person_idx
#
#     if closest_wrist_pos is not None and min_dist <= max_dist_threshold:
#         return closest_wrist_pos, shooter_person_idx
#     else:
#         # No wrist was close enough to be considered the shooter's
#         if closest_wrist_pos is not None:  # A closest wrist was found, but it was too far
#             # print(f"Closest wrist found at {min_dist}px, but threshold is {max_dist_threshold}px. Ignoring.")
#             pass
#         return None, None
#
MIN_WRIST_CONFIDENCE = 0.6 # Freethrow_1 0.95
MIN_ELBLOW_CONFIDENCE = 0.93
MIN_SHOULDER_CONFIDENCE = 0.93

#Version that checks for confidence threshold
# def get_likely_shooter_wrist_and_person_idx(pose_results_list, ball_bbox, wrist_kp_index,
#                                             max_dist_threshold, min_confidence=MIN_WRIST_CONFIDENCE): # Added min_confidence
#     """
#     Finds the wrist of the person most likely holding/shooting the ball, considering keypoint confidence.
#     Returns: (wrist_coordinates, person_index_in_results) or (None, None)
#     """
#     if not pose_results_list or len(pose_results_list) == 0 or ball_bbox is None:
#         return None, None
#
#     ball_center_x = (ball_bbox[0] + ball_bbox[2]) // 2
#     ball_center_y = (ball_bbox[1] + ball_bbox[3]) // 2
#     # ball_center = np.array([ball_center_x, ball_center_y]) # Not used if using dist to edge
#
#     min_dist = float('inf')
#     closest_wrist_pos = None
#     shooter_person_idx = -1
#
#     if pose_results_list[0].keypoints and \
#        pose_results_list[0].keypoints.xy is not None and \
#        pose_results_list[0].keypoints.conf is not None: # Check for confidence scores
#
#         all_persons_keypoints_xy = pose_results_list[0].keypoints.xy.cpu().numpy()
#         all_persons_keypoints_conf = pose_results_list[0].keypoints.conf.cpu().numpy()
#
#         for person_idx, person_kps_xy in enumerate(all_persons_keypoints_xy):
#             if len(person_kps_xy) > wrist_kp_index and \
#                len(all_persons_keypoints_conf[person_idx]) > wrist_kp_index: # Check conf array bounds
#
#                 wrist_conf = all_persons_keypoints_conf[person_idx][wrist_kp_index]
#
#                 if wrist_conf >= min_confidence: # Check confidence threshold
#                     wrist_coord_raw = person_kps_xy[wrist_kp_index]
#                     if wrist_coord_raw[0] > 1 and wrist_coord_raw[1] > 1:
#                         current_wrist_pos = wrist_coord_raw.astype(int)
#                         dist = calculate_distance_point_to_bbox_edge(current_wrist_pos, ball_bbox)
#
#                         if dist < min_dist:
#                             min_dist = dist
#                             closest_wrist_pos = current_wrist_pos
#                             shooter_person_idx = person_idx
#                 # else:
#                     # print(f"Person {person_idx} wrist confidence {wrist_conf:.2f} below threshold {min_confidence}")
#
#
#     if closest_wrist_pos is not None and min_dist <= max_dist_threshold:
#         return closest_wrist_pos, shooter_person_idx
#     else:
#         # ... (existing else logic for no close wrist)
#         return None, None

def get_likely_shooter_wrist_and_person_idx(
    pose_results_list, ball_bbox,
    shooting_arm_wrist_idx, shooting_arm_elbow_idx, shooting_arm_shoulder_idx, # NEW: specific arm keypoints
    max_dist_threshold,
    min_kp_confidence_dict=None # NEW: dict for confidences {wrist:0.3, elbow:0.25, shoulder:0.2}
):
    """
    Finds the wrist of the person most likely holding/shooting the ball,
    incorporating geometric constraints and keypoint confidences.

    Args:
        pose_results_list: Output from YOLO pose model.
        ball_bbox: Bounding box of the ball [x1, y1, x2, y2].
        shooting_arm_wrist_idx: Index of the shooting arm's wrist (e.g., RIGHT_WRIST).
        shooting_arm_elbow_idx: Index of the shooting arm's elbow.
        shooting_arm_shoulder_idx: Index of the shooting arm's shoulder.
        max_dist_threshold: Max pixel distance from wrist to ball bbox edge to be considered.
        min_kp_confidence_dict: Dictionary with min confidences for 'wrist', 'elbow', 'shoulder'.
                                Defaults if None.

    Returns:
        (wrist_coordinates, person_index_in_results, wrist_confidence) or (None, None, None)
    """
    if not pose_results_list or len(pose_results_list) == 0 or ball_bbox is None:
        return None, None, None

    if min_kp_confidence_dict is None:
        min_kp_confidence_dict = {
            'wrist': MIN_WRIST_CONFIDENCE,
            'elbow': MIN_ELBOW_CONFIDENCE,
            'shoulder': MIN_SHOULDER_CONFIDENCE
        }

    min_dist_to_ball = float('inf')
    best_shooter_candidate = None # Stores (wrist_pos, person_idx, wrist_conf)

    if pose_results_list[0].keypoints and \
       pose_results_list[0].keypoints.xy is not None and \
       pose_results_list[0].keypoints.conf is not None:

        all_persons_kps_xy = pose_results_list[0].keypoints.xy.cpu().numpy()
        all_persons_kps_conf = pose_results_list[0].keypoints.conf.cpu().numpy()

        for person_idx, person_kps_xy in enumerate(all_persons_kps_xy):
            person_kps_conf = all_persons_kps_conf[person_idx]

            # Check if all required keypoints for the shooting arm are present and have sufficient confidence
            wrist_conf = person_kps_conf[shooting_arm_wrist_idx] if len(person_kps_conf) > shooting_arm_wrist_idx else 0
            elbow_conf = person_kps_conf[shooting_arm_elbow_idx] if len(person_kps_conf) > shooting_arm_elbow_idx else 0
            shoulder_conf = person_kps_conf[shooting_arm_shoulder_idx] if len(person_kps_conf) > shooting_arm_shoulder_idx else 0

            if not (wrist_conf >= min_kp_confidence_dict['wrist'] and \
                    elbow_conf >= min_kp_confidence_dict['elbow'] and \
                    shoulder_conf >= min_kp_confidence_dict['shoulder']):
                # print(f"Person {person_idx}: Insufficient keypoint confidence for shooting arm. WristC:{wrist_conf:.2f} ElbowC:{elbow_conf:.2f} ShoulderC:{shoulder_conf:.2f}")
                continue # Skip this person if essential keypoints are missing/low confidence

            wrist_pos = person_kps_xy[shooting_arm_wrist_idx].astype(int)
            elbow_pos = person_kps_xy[shooting_arm_elbow_idx].astype(int)
            shoulder_pos = person_kps_xy[shooting_arm_shoulder_idx].astype(int)

            # --- Geometric Constraint 1: Wrist-Elbow Separation ---
            dist_wrist_elbow = calculate_distance(wrist_pos, elbow_pos)
            if dist_wrist_elbow < MIN_WRIST_ELBOW_SEPARATION_PIXELS:
                # print(f"Person {person_idx}: Wrist too close to elbow ({dist_wrist_elbow:.1f}px). Possible collapse.")
                continue

            # --- Geometric Constraint 2: Arm Extension Angle (Shoulder-Elbow-Wrist) ---
            # Angle at the elbow. A very small angle means a very bent arm.
            # A large angle (near 180) means an extended arm.
            arm_angle = calculate_angle(shoulder_pos, elbow_pos, wrist_pos)
            if not (MIN_SHOULDER_ELBOW_WRIST_ANGLE_DEG <= arm_angle <= MAX_SHOULDER_ELBOW_WRIST_ANGLE_DEG):
                print(f"Person {person_idx}: Arm angle ({arm_angle:.1f}deg) out of range [{MIN_SHOULDER_ELBOW_WRIST_ANGLE_DEG}-{MAX_SHOULDER_ELBOW_WRIST_ANGLE_DEG}].")
                continue

            # If geometric constraints passed, check proximity to ball
            dist_to_ball_edge = calculate_distance_point_to_bbox_edge(wrist_pos, ball_bbox)

            if dist_to_ball_edge < min_dist_to_ball and dist_to_ball_edge <= max_dist_threshold:
                min_dist_to_ball = dist_to_ball_edge
                best_shooter_candidate = (wrist_pos, person_idx, wrist_conf) # Store conf as well

    if best_shooter_candidate:
        #Wrist Pos + Likely Shooter
        # print(f"Selected shooter: Person {best_shooter_candidate[1]}, WristDistToBall: {min_dist_to_ball:.1f}px, WristConf: {best_shooter_candidate[2]:.2f}")
        return best_shooter_candidate[0], best_shooter_candidate[1]
    else:
        return None, None


def undistort_points_fisheye(points_distorted_px, K, D, K_new=None):
    """
    Undistorts 2D pixel points using fisheye model.
    :param points_distorted_px: Nx2 numpy array of distorted pixel points.
    :param K: Original camera intrinsic matrix.
    :param D: Fisheye distortion coefficients (k1,k2,k3,k4).
    :param K_new: Optional new camera matrix for the undistorted points. If None, normalized coords are scaled by original K.
    :return: Nx2 numpy array of undistorted pixel points.
    """
    if points_distorted_px is None or points_distorted_px.size == 0:
        return np.array([])

    # cv2.fisheye.undistortPoints expects shape (N, 1, 2) and float32
    points_reshaped = np.array(points_distorted_px, dtype=np.float32).reshape(-1, 1, 2)

    # P is the new camera matrix for the undistorted points.
    # If K_new is None, we can use K to get points back in a similar pixel scale.
    P_matrix = K_new if K_new is not None else K

    points_undistorted_px = cv2.fisheye.undistortPoints(points_reshaped, K, D, R=np.eye(3), P=P_matrix)

    return points_undistorted_px.reshape(-1, 2)

GRAVITY_ACCELERATION = 9.81

# --- NEW: Trajectory Fitting Function ---
def projectile_motion_model_ccs(t, v_cx0, v_cy0, v_cz0, x0, y0, z0):
    """
    3D projectile motion model in Camera Coordinate System.
    t: time array
    v_cx0, v_cy0, v_cz0: initial velocity components in CCS
    x0, y0, z0: initial position components in CCS (at t=0 relative to start of segment)
    Returns concatenated Xc, Yc, Zc arrays.
    """
    xc = x0 + v_cx0 * t
    # Assuming positive Yc in camera coords corresponds to downward physical direction
    # So gravity acts in positive Yc direction. If Yc is up, use -GRAVITY_ACCELERATION
    yc = y0 + v_cy0 * t + 0.5 * GRAVITY_ACCELERATION * t**2
    #yc = y0 + v_cy0 * t #TODO this is just a test without gravity (only linear section)
    zc = z0 + v_cz0 * t # No gravity along camera's optical axis (Zc)
    return np.concatenate((xc, yc, zc))

def fit_trajectory_ccs(trajectory_points_ccs_list, initial_fps):
    """
    Fits a projectile motion model to the CCS trajectory points.
    trajectory_points_ccs_list: list of dicts [{'frame': frame_no, 'Pc': [Xc, Yc, Zc]}, ...]
    initial_fps: frames per second of the video.
    Returns: (initial_velocity_vector_ccs, initial_position_ccs, pcov), or (None, None, None) if fit fails
    """
    if not trajectory_points_ccs_list or len(trajectory_points_ccs_list) < 3: # Need at least 3 points for a decent fit
        return None, None, None

    frames = np.array([p['frame'] for p in trajectory_points_ccs_list])
    positions = np.array([p['Pc'] for p in trajectory_points_ccs_list]) # Shape (N, 3)

    # Time relative to the first point in the segment
    t_relative = (frames - frames[0]) / initial_fps

    # Initial guess for parameters [vcx0, vcy0, vcz0, x0, y0, z0]
    # x0, y0, z0 are the coordinates of the first point
    x0_guess, y0_guess, z0_guess = positions[0,0], positions[0,1], positions[0,2]

    # Guess initial velocities from first two points (if available)
    if len(t_relative) > 1 and t_relative[1] > 0:
        vcx_guess = (positions[1,0] - x0_guess) / t_relative[1]
        vcy_guess = (positions[1,1] - y0_guess - 0.5 * GRAVITY_ACCELERATION * t_relative[1]**2) / t_relative[1] # crude
        vcz_guess = (positions[1,2] - z0_guess) / t_relative[1]
    else:
        vcx_guess, vcy_guess, vcz_guess = 1.0, -5.0, 0.1 # Generic guesses

    p0 = [vcx_guess, vcy_guess, vcz_guess, x0_guess, y0_guess, z0_guess]

    # `positions.ravel()` flattens the (N,3) array to (3N,) for curve_fit
    try:
        popt, pcov = curve_fit(projectile_motion_model_ccs, t_relative, positions.ravel(), p0=p0, maxfev=5000)
        initial_velocity_ccs = np.array(popt[0:3])
        initial_position_ccs = np.array(popt[3:6]) # This should be very close to positions[0]
        return initial_velocity_ccs, initial_position_ccs, pcov
    except RuntimeError:
        print("Error - curve_fit could not find optimal parameters.")
        return None, None, None
    except ValueError:
        print("Error - ValueError in curve_fit (often due to NaN/inf in data or bounds).")
        return None, None, None
# --- END NEW ---


#Drag fitting
def trajectory_ode_system(t, S, kb_val, grav_vec_ccs):
    """
    Defines the system of ODEs for projectile motion with air drag in CCS.
    S = [Xc, Yc, Zc, Vcx, Vcy, Vcz]  <-- This is the CURRENT STATE of the system at time 't'
    kb_val: drag constant
    grav_vec_ccs: gravity vector [gx_cam, gy_cam, gz_cam] in camera coords
    Returns dS/dt  <-- This is the vector of DERIVATIVES
    """
    Xc, Yc, Zc, Vcx, Vcy, Vcz = S  # Unpack the current state

    # Speed (calculated from current velocities in S)
    speed = np.sqrt(Vcx ** 2 + Vcy ** 2 + Vcz ** 2)

    # Accelerations due to drag (calculated from current velocities in S and speed)
    Acx_drag = -kb_val * Vcx * speed if speed > 1e-6 else 0
    Acy_drag = -kb_val * Vcy * speed if speed > 1e-6 else 0
    Acz_drag = -kb_val * Vcz * speed if speed > 1e-6 else 0

    # Total accelerations (drag + gravity)
    Acx = Acx_drag + grav_vec_ccs[0]
    Acy = Acy_drag + grav_vec_ccs[1]
    Acz = Acz_drag + grav_vec_ccs[2]

    # Now, we define what dS/dt is:
    # The rate of change of position IS velocity
    dXcdt = Vcx
    dYcdt = Vcy
    dZcdt = Vcz

    # The rate of change of velocity IS acceleration
    dVcxdt = Acx
    dVcydt = Acy
    dVczdt = Acz

    return [dXcdt, dYcdt, dZcdt, dVcxdt, dVcydt, dVczdt]  # This is dS/dt
def objective_function(initial_velocities, observed_trajectory_ccs, t_eval, kb_val, grav_vec_ccs, P0_ccs):
    """
    Solves ODE with given initial velocities and calculates error against observed trajectory.
    initial_velocities: [Vcx0, Vcy0, Vcz0]
    observed_trajectory_ccs: Nx3 array of observed [Xc, Yc, Zc] points
    t_eval: Array of time points corresponding to observed_trajectory_ccs (relative to release)
    P0_ccs: Initial position [Xc0, Yc0, Zc0] (from the first point of observed_trajectory_ccs)
    """
    Vcx0, Vcy0, Vcz0 = initial_velocities
    Xc0, Yc0, Zc0 = P0_ccs

    S0 = [Xc0, Yc0, Zc0, Vcx0, Vcy0, Vcz0] # Initial state vector

    # Solve the ODE system
    # t_span = (t_eval[0], t_eval[-1]) # Integrate over the duration of observed points
    # Ensure t_eval is sorted and t_span covers it
    sol = solve_ivp(trajectory_ode_system, (t_eval[0], t_eval[-1]), S0,
                    args=(kb_val, grav_vec_ccs),
                    t_eval=t_eval, dense_output=True, method='RK45')

    if not sol.success:
        # print(f"ODE solver failed for V0={initial_velocities}")
        return np.inf # Return a large error if solver fails

    # Solved trajectory positions [Xc_solved, Yc_solved, Zc_solved]
    # sol.y is shape (6, num_time_points). We need positions (first 3 rows).
    solved_positions = sol.y[:3, :].T # Transpose to get (num_time_points, 3)

    if solved_positions.shape != observed_trajectory_ccs.shape:
        # This can happen if t_eval caused issues. Interpolate if necessary,
        # but for now, ensure t_eval in solve_ivp matches observed points' times.
        # This should ideally not happen if t_eval is correctly set.
        # Forcing interpolation if shapes mismatch (less ideal for optimization):
        # from scipy.interpolate import interp1d
        # print(f"Shape mismatch: Solved {solved_positions.shape}, Obs {observed_trajectory_ccs.shape}. Interpolating.")
        # if len(sol.t) < 2 or len(observed_trajectory_ccs) < 2: return np.inf
        # try:
        #     interp_func_x = interp1d(sol.t, sol.y[0,:], kind='linear', fill_value="extrapolate")
        #     interp_func_y = interp1d(sol.t, sol.y[1,:], kind='linear', fill_value="extrapolate")
        #     interp_func_z = interp1d(sol.t, sol.y[2,:], kind='linear', fill_value="extrapolate")
        #     solved_positions_interp = np.vstack((interp_func_x(t_eval), interp_func_y(t_eval), interp_func_z(t_eval))).T
        #     if solved_positions_interp.shape != observed_trajectory_ccs.shape: return np.inf
        #     error = np.sum((solved_positions_interp - observed_trajectory_ccs)**2)
        # except ValueError as e:
        #     print(f"Interpolation error: {e}")
        #     return np.inf
        print(f"Warning: ODE solution shape {solved_positions.shape} differs from observed {observed_trajectory_ccs.shape}")
        return np.inf # Or handle more gracefully

    error = np.sum((solved_positions - observed_trajectory_ccs)**2)
    # print(f"V0={initial_velocities}, Error={error}") # For debugging
    return error
