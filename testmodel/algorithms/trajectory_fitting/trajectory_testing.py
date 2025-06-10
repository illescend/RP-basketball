import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from matplotlib.patches import Rectangle, Circle

# --- Constants ---
GRAVITY_ACCELERATION = 9.81
MASS_BALL = 0.6
RADIUS_BALL = 0.12
CD_BALL = 0.54
RHO_AIR = 1.204
CROSS_SECTIONAL_AREA_BALL = math.pi * RADIUS_BALL ** 2
KB_BALL = (CD_BALL * RHO_AIR * CROSS_SECTIONAL_AREA_BALL) / (2 * MASS_BALL)

print(f"Calculated Drag Constant (kb): {KB_BALL:.4f}")

# --- Court Dimensions (WCS: Origin at FT line center, X towards hoop, Y up, Z right) ---
FT_LINE_TO_RIM_CENTER_X = 4.191  # m (approx. 13.75 ft)
RIM_HEIGHT_Y = 3.048  # m (10 ft)
RIM_DIAMETER = 0.4572  # m (18 inches)
BACKBOARD_WIDTH_Z = 1.8288  # m (6 ft)
BACKBOARD_HEIGHT = 1.0668  # m (3.5 ft)
# Backboard X position (assuming rim center is slightly in front of it)
# Typically rim extends 0.15m (6 inches) from backboard
BACKBOARD_X = FT_LINE_TO_RIM_CENTER_X + (RIM_DIAMETER / 2) + 0.15
BACKBOARD_BOTTOM_Y = RIM_HEIGHT_Y - (0.45 / 2)  # Assuming rim center is center of inner square vertically


# BACKBOARD_BOTTOM_Y = 2.9 # Alternative: bottom of backboard is slightly below rim for FIBA
# BACKBOARD_BOTTOM_Y = RIM_HEIGHT_Y - (BACKBOARD_HEIGHT / 2) + 0.225 # If rim is centered on inner square of 0.45m height

# --- Model 1: Simple Projectile Motion (No Air Drag) ---
def simulate_simple_projectile(P0, V0, t_eval):
    x0, y0, z0 = P0;
    vx0, vy0, vz0 = V0
    x = x0 + vx0 * t_eval
    y = y0 + vy0 * t_eval - 0.5 * GRAVITY_ACCELERATION * t_eval ** 2
    z = z0 + vz0 * t_eval
    return np.vstack((x, y, z)).T


# --- Model 2: Projectile Motion with Air Drag (using ODEs) ---
def trajectory_ode_system_drag(t, S, kb_val):
    x, y, z, vx, vy, vz = S
    speed_sq = vx ** 2 + vy ** 2 + vz ** 2
    if speed_sq < 1e-9:
        speed = 0.0; ax_drag, ay_drag, az_drag = 0, 0, 0
    else:
        speed = np.sqrt(speed_sq)
        ax_drag = -kb_val * vx * speed  # CORRECTED DRAG TERM
        ay_drag = -kb_val * vy * speed  # CORRECTED DRAG TERM
        az_drag = -kb_val * vz * speed  # CORRECTED DRAG TERM
    ax = ax_drag;
    ay = ay_drag - GRAVITY_ACCELERATION;
    az = az_drag
    return [vx, vy, vz, ax, ay, az]


def simulate_projectile_with_drag(P0, V0, t_eval, kb_val):
    S0 = np.concatenate((P0, V0))
    sol = solve_ivp(trajectory_ode_system_drag, (t_eval[0], t_eval[-1]), S0,
                    args=(kb_val,), t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)
    if not sol.success: print(f"ODE solver failed! Msg: {sol.message}"); return None
    return sol.y[:3, :].T


# --- Function to Plot Court Elements (2D side view: X-Y plane) ---
def plot_court_elements_xy(ax):
    # Rim (side view is a line, but let's show a circle for visual target)
    rim_circle = Circle((FT_LINE_TO_RIM_CENTER_X, RIM_HEIGHT_Y), RIM_DIAMETER / 2,
                        edgecolor='orange', facecolor='none', lw=2, linestyle='-')
    ax.add_patch(rim_circle)

    # Backboard (side view is a vertical line)
    ax.plot([BACKBOARD_X, BACKBOARD_X],
            [BACKBOARD_BOTTOM_Y, BACKBOARD_BOTTOM_Y + BACKBOARD_HEIGHT],
            color='gray', lw=3, label='Backboard (Side View)')

    # Floor
    ax.axhline(0, color='black', lw=1)


# --- Simulation Parameters ---
# Assume release from free-throw line (X=0), centered (Z=0), at some height
release_height = 2.582  # m (example typical release height)
P_initial = np.array([0.379, 2.479, 0.0])

# Example: Try to make a shot (adjust speed/angle to see different outcomes)
# speed_initial_mps = 7.0  # m/s
# angle_elevation_deg = 52.0 # degrees
speed_initial_mps = 7.61 # From your previous data
angle_elevation_deg = 52.5 # From your previous data
angle_azimuth_deg = 0.0  # Shoot straight

angle_elevation_rad = math.radians(angle_elevation_deg)
angle_azimuth_rad = math.radians(angle_azimuth_deg)
Vxy_initial = speed_initial_mps * math.cos(angle_elevation_rad)
Vx_initial = Vxy_initial * math.cos(angle_azimuth_rad)
Vy_initial = speed_initial_mps * math.sin(angle_elevation_rad)
Vz_initial = Vxy_initial * math.sin(angle_azimuth_rad)
V_initial = np.array([Vx_initial, Vy_initial, Vz_initial])

print(f"Initial Position P0 (rel. to FT line center): {P_initial} m")
print(
    f"Initial Velocity V0: {V_initial.round(3)} m/s (Speed: {speed_initial_mps} m/s, Elev: {angle_elevation_deg} deg)")

t_start = 0.0;
t_end = 2.0;
num_points = 200  # Simulate a bit longer to see full arc
time_points = np.linspace(t_start, t_end, num_points)

# --- Run Simulations ---
trajectory_simple = simulate_simple_projectile(P_initial, V_initial, time_points)
trajectory_drag = simulate_projectile_with_drag(P_initial, V_initial, time_points, KB_BALL)

# --- Plotting ---
fig = plt.figure(figsize=(10, 7))  # Adjusted for single main plot initially
ax1 = fig.add_subplot(111)  # Single plot for X-Y trajectory

# Plot court elements first (so they are in the background)
plot_court_elements_xy(ax1)

# Plot trajectories
ax1.plot(trajectory_simple[:, 0], trajectory_simple[:, 1], label='No Drag', color='blue', linestyle='--')
if trajectory_drag is not None:
    ax1.plot(trajectory_drag[:, 0], trajectory_drag[:, 1], label='With Air Drag', color='red')

ax1.set_xlabel('Horizontal Distance from FT Line (X) (m)')
ax1.set_ylabel('Height (Y) (m)')
ax1.set_title(f'Basketball Free Throw Trajectory (V0={speed_initial_mps:.1f}m/s, Angle={angle_elevation_deg:.1f}deg)')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(bottom=0, top=max(4.5, np.max(
    trajectory_simple[:, 1]) * 1.1 if trajectory_simple is not None else 4.5))  # Ensure rim is visible
ax1.set_xlim(left=-0.5, right=max(5.0, FT_LINE_TO_RIM_CENTER_X + RIM_DIAMETER))  # Ensure rim is visible
ax1.set_aspect('equal', adjustable='box')  # Make X and Y scales equal for realistic arc

plt.tight_layout()
plt.show()

# ... (Comparison printout as before, ensure variable names match if needed) ...
if trajectory_drag is not None:
    print("\n--- Comparison ---")
    max_height_simple = np.max(trajectory_simple[:, 1])
    max_height_drag = np.max(trajectory_drag[:, 1])
    print(f"Max Height (Simple): {max_height_simple:.2f} m")
    print(f"Max Height (Drag):   {max_height_drag:.2f} m")
    # ... (rest of comparison)