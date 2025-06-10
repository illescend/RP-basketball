import cv2
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# --- Configuration ---
# Pose Keypoint for shooting hand's wrist (COCO format: 9 for left_wrist, 10 for right_wrist)
SHOOTING_WRIST_KP_INDEX = 10  # Assuming right wrist for Curry

# Release Detection Parameters
CONFIRMATION_FRAMES_FOR_RELEASE = 5  # Wrist must be outside ball for this many consecutive frames
# Optional: A small pixel margin to consider wrist "inside" even if slightly outside bbox due to hand size/occlusion
WRIST_BALL_PROXIMITY_MARGIN = 20 # Pixels. If wrist is within this margin of bbox, still considered "close" / potentially "in hand"

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


# --- NEW CONFIGURATION ---
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = 75  # Pixels. If closest wrist is further than this, ignore.


# Adjust based on ball size and typical shooting poses.
# This value should be larger than WRIST_BALL_PROXIMITY_MARGIN
# but small enough to exclude non-interacting players.

# --- MODIFIED HELPER FUNCTION ---
def get_likely_shooter_wrist_and_person_idx(pose_results_list, ball_bbox, wrist_kp_index, max_dist_threshold):
    """
    Finds the wrist of the person most likely holding/shooting the ball.
    Returns: (wrist_coordinates, person_index_in_results) or (None, None)
    """
    if not pose_results_list or len(pose_results_list) == 0 or ball_bbox is None:
        return None, None

    ball_center_x = (ball_bbox[0] + ball_bbox[2]) // 2
    ball_center_y = (ball_bbox[1] + ball_bbox[3]) // 2
    ball_center = np.array([ball_center_x, ball_center_y])

    min_dist = float('inf')
    closest_wrist_pos = None
    shooter_person_idx = -1

    # Iterate through all detected persons in the frame
    # pose_results_list[0] is the Results object for the current frame
    # pose_results_list[0].keypoints.xy is a tensor of [num_persons, num_keypoints, 2]
    if pose_results_list[0].keypoints and pose_results_list[0].keypoints.xy is not None:
        all_persons_keypoints_xy = pose_results_list[0].keypoints.xy.cpu().numpy()

        for person_idx, person_kps in enumerate(all_persons_keypoints_xy):
            if len(person_kps) > wrist_kp_index:
                wrist_coord_raw = person_kps[wrist_kp_index]
                # Check if keypoint is detected (often 0,0 if not, or use confidence if available)
                if wrist_coord_raw[0] > 1 and wrist_coord_raw[1] > 1:  # Basic validity check
                    current_wrist_pos = wrist_coord_raw.astype(int)

                    # Calculate distance from this wrist to ball center
                    # dist = np.linalg.norm(current_wrist_pos - ball_center)
                    # OR use your existing distance to bbox edge for consistency in distance metrics:
                    dist = calculate_distance_point_to_bbox_edge(current_wrist_pos, ball_bbox)

                    if dist < min_dist:
                        min_dist = dist
                        closest_wrist_pos = current_wrist_pos
                        shooter_person_idx = person_idx

    if closest_wrist_pos is not None and min_dist <= max_dist_threshold:
        return closest_wrist_pos, shooter_person_idx
    else:
        # No wrist was close enough to be considered the shooter's
        if closest_wrist_pos is not None:  # A closest wrist was found, but it was too far
            # print(f"Closest wrist found at {min_dist}px, but threshold is {max_dist_threshold}px. Ignoring.")
            pass
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
