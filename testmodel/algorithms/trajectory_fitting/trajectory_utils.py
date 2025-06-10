# trajectory_utils.py
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import math
from matplotlib.patches import Circle

# --- Physics and Court Model Constants ---
GRAVITY_ACCELERATION = 9.81
MASS_BALL = 0.6  # kg
RADIUS_BALL = 0.12  # m
CD_BALL = 0.54
RHO_AIR = 1.204  # kg/m^3
CROSS_SECTIONAL_AREA_BALL = math.pi * RADIUS_BALL ** 2
KB_BALL = (CD_BALL * RHO_AIR * CROSS_SECTIONAL_AREA_BALL) / (2 * MASS_BALL)

# --- Court Dimensions for WCS (Origin @ FT line center, X towards hoop, Y up, Z right) ---
# These are defaults, can be overridden if needed by functions using them
WCS_FT_LINE_TO_RIM_CENTER_X_DEFAULT = 4.191
WCS_RIM_HEIGHT_Y_DEFAULT = 3.048
WCS_RIM_DIAMETER_DEFAULT = 0.4572
WCS_BACKBOARD_X_DEFAULT = WCS_FT_LINE_TO_RIM_CENTER_X_DEFAULT + (WCS_RIM_DIAMETER_DEFAULT / 2) + 0.15
WCS_BACKBOARD_HEIGHT_DEFAULT = 1.0668
WCS_BB_INNER_RECT_HEIGHT_DEFAULT = 0.45  # Assuming this is the height of the inner rectangle


# --- Trajectory Simulation ---
def trajectory_ode_system_drag(t, S, kb_val):
    x, y, z, vx, vy, vz = S
    speed_sq = vx ** 2 + vy ** 2 + vz ** 2
    if speed_sq < 1e-9:
        speed = 0.0
        ax_drag, ay_drag, az_drag = 0.0, 0.0, 0.0
    else:
        speed = np.sqrt(speed_sq)
        ax_drag = -kb_val * vx * speed
        ay_drag = -kb_val * vy * speed
        az_drag = -kb_val * vz * speed
    ax = ax_drag
    ay = ay_drag - GRAVITY_ACCELERATION
    az = az_drag
    return [vx, vy, vz, ax, ay, az]


def simulate_projectile_with_drag(P0_wcs, V0_wcs, t_eval_seconds, kb_val_sim=KB_BALL):
    if P0_wcs is None or V0_wcs is None:
        print("Error: P0_wcs or V0_wcs is None in simulate_projectile_with_drag.")
        return None
    S0 = np.concatenate((P0_wcs, V0_wcs))
    try:
        sol = solve_ivp(trajectory_ode_system_drag, (t_eval_seconds[0], t_eval_seconds[-1]), S0,
                        args=(kb_val_sim,), t_eval=t_eval_seconds, method='RK45', rtol=1e-5, atol=1e-7)
        if not sol.success:
            print(f"ODE solver failed! Msg: {sol.message}")
            return None
        return sol.y[:3, :].T
    except Exception as e:
        print(f"Exception during ODE solution: {e}")
        return None


# --- Fitting Functions ---
def objective_function_release_params(params, observed_wcs_positions, observed_time_points, kb_fixed):
    px, py, pz, speed, elev_deg, azim_deg = params
    P0 = np.array([px, py, pz])
    elev_rad = math.radians(elev_deg)
    azim_rad = math.radians(azim_deg)
    Vx = speed * math.cos(elev_rad) * math.cos(azim_rad)
    Vy = speed * math.sin(elev_rad)
    Vz = speed * math.cos(elev_rad) * math.sin(azim_rad)
    V0 = np.array([Vx, Vy, Vz])
    simulated_trajectory = simulate_projectile_with_drag(P0, V0, observed_time_points, kb_fixed)
    if simulated_trajectory is None or len(simulated_trajectory) != len(observed_wcs_positions):
        return 1e9
    error = np.sum((simulated_trajectory - observed_wcs_positions) ** 2)
    return error


def fit_shot_trajectory_parameters(observed_wcs_trajectory_data, fps, initial_P0_guess, initial_V_speed_angle_guess,
                                   min_points_for_fitting=6):
    if not observed_wcs_trajectory_data or len(observed_wcs_trajectory_data) < min_points_for_fitting:
        print(
            f"Not enough WCS trajectory points to fit (need {min_points_for_fitting}, have {len(observed_wcs_trajectory_data)}).")
        return None  # Indicate failure by returning None for fitted_params

    observed_positions = np.array([item['Pc_wcs'] for item in observed_wcs_trajectory_data])
    first_frame = observed_wcs_trajectory_data[0]['frame']
    time_points = np.array([(item['frame'] - first_frame) / fps for item in observed_wcs_trajectory_data])

    p0_x_guess, p0_y_guess, p0_z_guess = initial_P0_guess
    speed_guess, elev_guess, azim_guess = initial_V_speed_angle_guess
    initial_params = [p0_x_guess, p0_y_guess, p0_z_guess, speed_guess, elev_guess, azim_guess]

    print(f"Initial guess for fitting: P0={initial_params[0:3]}, V0(spd,el,az)={initial_params[3:6]}")

    bounds = [(-1.5, 1.5), (1.0, 3.5), (-1.5, 1.5),  # Looser P0 bounds
              (4, 15), (15, 75), (-40, 40)]  # Looser V0 bounds

    result = minimize(objective_function_release_params, initial_params,
                      args=(observed_positions, time_points, KB_BALL),
                      method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-7, 'gtol': 1e-6, 'maxiter': 500, 'disp': False})

    if result.success:
        fitted_params_values = result.x
        fitted_P0 = np.array(fitted_params_values[0:3])
        fitted_speed = fitted_params_values[3]
        fitted_elev_deg = fitted_params_values[4]
        fitted_azim_deg = fitted_params_values[5]

        elev_rad_fit = math.radians(fitted_elev_deg)
        azim_rad_fit = math.radians(fitted_azim_deg)
        Vx_fit = fitted_speed * math.cos(elev_rad_fit) * math.cos(azim_rad_fit)
        Vy_fit = fitted_speed * math.sin(elev_rad_fit)
        Vz_fit = fitted_speed * math.cos(elev_rad_fit) * math.sin(azim_rad_fit)
        fitted_V0_components = np.array([Vx_fit, Vy_fit, Vz_fit])

        print("\n--- Trajectory Fitting Successful ---")
        print(f"  Fitted P0 (WCS): {fitted_P0.round(3)} m")
        print(f"  Fitted V0 (WCS components): {fitted_V0_components.round(3)} m/s")
        print(f"  Fitted Speed: {fitted_speed:.2f} m/s")
        print(f"  Fitted Elevation Angle: {fitted_elev_deg:.1f} deg")
        print(f"  Fitted Azimuth Angle: {fitted_azim_deg:.1f} deg")
        print(f"  Final Objective Value: {result.fun:.4e}")
        return {
            "P0_wcs": fitted_P0,
            "V0_wcs_components": fitted_V0_components,
            "speed_mps_wcs": fitted_speed,
            "angle_deg_elev_wcs": fitted_elev_deg,
            "angle_deg_azim_wcs": fitted_azim_deg,
            "success": True,
            "message": result.message
        }
    else:
        print("\n--- Trajectory Fitting Failed ---")
        print(f"  Message: {result.message}")
        return {"success": False, "message": result.message}


# --- Plotting Utility ---
def plot_fitted_trajectory_wcs(
        observed_wcs_positions,
        fitted_P0, fitted_V0_components,
        shot_number, video_fps,
        output_dir, video_basename,
        court_dims=None  # Optional dictionary to override default court dimensions
):
    if court_dims is None:
        court_dims = {
            "ft_to_rim_x": WCS_FT_LINE_TO_RIM_CENTER_X_DEFAULT,
            "rim_y": WCS_RIM_HEIGHT_Y_DEFAULT,
            "rim_diameter": WCS_RIM_DIAMETER_DEFAULT,
            "backboard_x": WCS_BACKBOARD_X_DEFAULT,
            "backboard_height": WCS_BACKBOARD_HEIGHT_DEFAULT,
            "bb_inner_rect_height": WCS_BB_INNER_RECT_HEIGHT_DEFAULT
        }

    if len(observed_wcs_positions) == 0 or fitted_P0 is None or fitted_V0_components is None:
        print("Cannot plot: Missing observed data or fitted parameters.")
        return

    # Simulate fitted trajectory for plotting
    # Assuming observed_wcs_positions is an array of [frame, Pc_wcs_dict] or similar
    # We need to extract actual time points from observed data if it's not just positions
    # For now, assuming observed_wcs_positions is just the Nx3 array of points.
    # This means we need relative time for simulation.
    # A better way is to pass the full trajectory_points_wcs list

    # Let's assume observed_wcs_positions is the list of dicts: [{'frame': f, 'Pc_wcs': p}, ...]
    observed_xyz_data = np.array([item['Pc_wcs'] for item in observed_wcs_positions if item['Pc_wcs'] is not None])
    if len(observed_xyz_data) == 0:
        print("No valid Pc_wcs found in observed_wcs_positions for plotting.")
        return

    first_obs_frame = observed_wcs_positions[0]['frame']
    last_obs_frame = observed_wcs_positions[-1]['frame']
    t_start_plot = 0.0
    t_end_plot = (last_obs_frame - first_obs_frame) / video_fps + 0.3  # Simulate slightly longer
    time_points_for_plot_sim = np.linspace(t_start_plot, t_end_plot, 150)

    simulated_fitted_trajectory = simulate_projectile_with_drag(
        fitted_P0, fitted_V0_components, time_points_for_plot_sim
    )

    fig_plot = plt.figure(figsize=(12, 8))
    ax_plot = fig_plot.add_subplot(111)

    # Plot court elements
    rim_circle = Circle((court_dims["ft_to_rim_x"], court_dims["rim_y"]), court_dims["rim_diameter"] / 2,
                        edgecolor='orange', facecolor='none', lw=2, linestyle='-')
    ax_plot.add_patch(rim_circle)

    # Calculate backboard Y coordinates for plotting
    bb_rect_bottom_y = court_dims["rim_y"]  # Assuming inner rect bottom at rim height
    bb_full_bottom_y = bb_rect_bottom_y - (court_dims["backboard_height"] - court_dims["bb_inner_rect_height"]) / 2
    bb_full_top_y = bb_rect_bottom_y + court_dims["bb_inner_rect_height"] + (
                court_dims["backboard_height"] - court_dims["bb_inner_rect_height"]) / 2

    ax_plot.plot([court_dims["backboard_x"], court_dims["backboard_x"]],
                 [bb_full_bottom_y, bb_full_top_y],
                 color='gray', lw=3, label='Backboard (Side View)')
    ax_plot.axhline(0, color='black', lw=1)  # Floor

    # Plot observed and simulated trajectories
    ax_plot.scatter(observed_xyz_data[:, 0], observed_xyz_data[:, 1],
                    c='blue', label='Observed WCS Trajectory (from KF)', s=15, zorder=5, alpha=0.7)

    if simulated_fitted_trajectory is not None:
        # Get fitted speed and angles for label (assuming they are available if fitted_P0 and V0 are)
        # This would ideally be passed in or recalculated if not directly available
        speed_label = np.linalg.norm(fitted_V0_components)
        elev_label_rad = math.atan2(fitted_V0_components[1],
                                    math.sqrt(fitted_V0_components[0] ** 2 + fitted_V0_components[2] ** 2))
        azim_label_rad = math.atan2(fitted_V0_components[2], fitted_V0_components[0])
        elev_label_deg = math.degrees(elev_label_rad)
        azim_label_deg = math.degrees(azim_label_rad)

        ax_plot.plot(simulated_fitted_trajectory[:, 0], simulated_fitted_trajectory[:, 1],
                     label=f'Fitted Traj. (Drag Model)\nP0: [{fitted_P0[0]:.2f},{fitted_P0[1]:.2f},{fitted_P0[2]:.2f}]m\nSpeed={speed_label:.2f}m/s, El={elev_label_deg:.1f}°, Az={azim_label_deg:.1f}°',
                     color='red', linestyle='-', linewidth=2, zorder=4)

    ax_plot.set_xlabel('X_wcs: Distance from FT Line towards Hoop (m)')
    ax_plot.set_ylabel('Y_wcs: Height from Ground (m)')
    ax_plot.set_title(f'Shot {shot_number}: Observed vs. Fitted Trajectory (WCS Side View)')
    ax_plot.legend(fontsize='medium', loc='upper right')
    ax_plot.grid(True, linestyle='--', alpha=0.7)

    all_x = list(observed_xyz_data[:, 0])
    all_y = list(observed_xyz_data[:, 1])
    if simulated_fitted_trajectory is not None:
        all_x.extend(simulated_fitted_trajectory[:, 0])
        all_y.extend(simulated_fitted_trajectory[:, 1])

    if all_x and all_y:
        ax_plot.set_xlim(min(min(all_x), -0.5) - 0.2,
                         max(max(all_x), court_dims["ft_to_rim_x"] + court_dims["rim_diameter"] / 2) + 0.5)
        ax_plot.set_ylim(bottom=-0.1, top=max(max(all_y), court_dims["rim_y"] + court_dims["rim_diameter"] / 2) + 0.5)
    else:
        ax_plot.set_xlim(-0.5, court_dims["ft_to_rim_x"] + 1.5)
        ax_plot.set_ylim(0, court_dims["rim_y"] + 2.0)

    ax_plot.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    plot_save_filename = f"{video_basename}_shot{shot_number}_trajectory_plot.png"
    plot_save_path = os.path.join(output_dir, plot_save_filename)
    plt.savefig(plot_save_path)
    print(f"Trajectory plot saved to {plot_save_path}")
    plt.show()