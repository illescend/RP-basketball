import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

# --- Constants ---
GRAVITY_ACCELERATION = 9.81  # m/s^2
# Example Ball Properties (can be adjusted)
MASS_BALL = 0.6  # kg
RADIUS_BALL = 0.12  # m
CD_BALL = 0.54  # Drag coefficient for a sphere/basketball
RHO_AIR = 1.204  # kg/m^3
CROSS_SECTIONAL_AREA_BALL = math.pi * RADIUS_BALL ** 2
KB_BALL = (CD_BALL * RHO_AIR * CROSS_SECTIONAL_AREA_BALL) / (2 * MASS_BALL)

print(f"Calculated Drag Constant (kb): {KB_BALL:.4f}")

# --- Court Dimensions (WCS: Origin at FT line center, X towards hoop, Y up, Z right) ---
FT_LINE_TO_RIM_CENTER_X = 4.191  # m (approx. 13.75 ft)
RIM_HEIGHT_Y = 3.048           # m (10 ft)
RIM_DIAMETER = 0.4572          # m (18 inches)
BACKBOARD_WIDTH_Z = 1.8288     # m (6 ft)
BACKBOARD_HEIGHT = 1.0668      # m (3.5 ft)
# Backboard X position (assuming rim center is slightly in front of it)
# Typically rim extends 0.15m (6 inches) from backboard
BACKBOARD_X = FT_LINE_TO_RIM_CENTER_X + (RIM_DIAMETER / 2) + 0.15
BACKBOARD_BOTTOM_Y = RIM_HEIGHT_Y - (0.45 / 2) # Assuming rim center is center of inner square vertically
# BACKBOARD_BOTTOM_Y = 2.9 # Alternative: bottom of backboard is slightly below rim for FIBA
# BACKBOARD_BOTTOM_Y = RIM_HEIGHT_Y - (BACKBOARD_HEIGHT / 2) + 0.225 # If rim is centered on inner square of 0.45m height


# --- Model 1: Simple Projectile Motion (No Air Drag) ---
def simulate_simple_projectile(P0, V0, t_eval):
    """
    Simulates 3D projectile motion without air drag.
    P0: initial position [x0, y0, z0]
    V0: initial velocity [vx0, vy0, vz0]
    t_eval: array of time points to evaluate
    Returns: Nx3 array of positions [x(t), y(t), z(t)]
    """
    x0, y0, z0 = P0
    vx0, vy0, vz0 = V0

    x = x0 + vx0 * t_eval
    # Assuming positive Y is UP for this model, so gravity is negative
    y = y0 + vy0 * t_eval - 0.5 * GRAVITY_ACCELERATION * t_eval ** 2
    z = z0 + vz0 * t_eval

    return np.vstack((x, y, z)).T


# --- Model 2: Projectile Motion with Air Drag (using ODEs) ---
def trajectory_ode_system_drag(t, S, kb_val):
    """
    Defines the system of ODEs for projectile motion with air drag.
    S = [x, y, z, vx, vy, vz]
    kb_val: drag constant
    Returns dS/dt
    """
    x, y, z, vx, vy, vz = S

    speed_sq = vx ** 2 + vy ** 2 + vz ** 2
    if speed_sq < 1e-9:  # If speed is virtually zero, avoid issues with sqrt or division
        speed = 0.0
        ax_drag, ay_drag, az_drag = 0.0, 0.0, 0.0
    else:
        speed = np.sqrt(speed_sq)
        ax_drag = -kb_val * vx * speed
        ay_drag = -kb_val * vy * speed
        az_drag = -kb_val * vz * speed

        # ax_drag = -kb_val * vx * np.sqrt(abs(vx+vy+vz))
        # ay_drag = -kb_val * vy * np.sqrt(abs(vx+vy+vz))
        # az_drag = -kb_val * vz * np.sqrt(abs(vx+vy+vz))

    # Total accelerations
    # Assuming positive Y is UP, so gravity acts in -Y direction
    ax = ax_drag
    ay = ay_drag - GRAVITY_ACCELERATION
    az = az_drag

    return [vx, vy, vz, ax, ay, az]


def simulate_projectile_with_drag(P0, V0, t_eval, kb_val):
    """
    Simulates 3D projectile motion with air drag using solve_ivp.
    P0: initial position [x0, y0, z0]
    V0: initial velocity [vx0, vy0, vz0]
    t_eval: array of time points to evaluate
    kb_val: drag constant
    Returns: Nx3 array of positions [x(t), y(t), z(t)], or None if solver fails
    """
    x0, y0, z0 = P0
    vx0, vy0, vz0 = V0
    S0 = [x0, y0, z0, vx0, vy0, vz0]  # Initial state vector

    sol = solve_ivp(trajectory_ode_system_drag, (t_eval[0], t_eval[-1]), S0,
                    args=(kb_val,),
                    t_eval=t_eval, dense_output=False, method='RK45', rtol=1e-6, atol=1e-8)

    if not sol.success:
        print(f"ODE solver failed for drag model! Message: {sol.message}")
        return None

    # sol.y is shape (6, num_time_points). We need positions (first 3 rows).
    positions_drag = sol.y[:3, :].T
    return positions_drag


# --- Simulation Parameters ---
# Initial conditions (example: typical basketball shot)
P_initial = np.array([0.0, 2.0, 0.0])  # x0, y0 (height), z0 (m) - Release from 2m height
# Let's use a speed and angle similar to your findings
speed_initial_mps = 6  # m/s
angle_elevation_deg = 59.85  # degrees (relative to horizontal)
angle_azimuth_deg = 0  # degrees (let's shoot straight in X-Y plane, so no Z-velocity initially for simplicity of comparison)

# Convert speed and angle to velocity components
# Assuming Y is up, X is forward, Z is sideways
angle_elevation_rad = math.radians(angle_elevation_deg)
angle_azimuth_rad = math.radians(angle_azimuth_deg)

Vxy_initial = speed_initial_mps * math.cos(angle_elevation_rad)  # Speed in the horizontal (XZ) plane

Vx_initial = Vxy_initial * math.cos(angle_azimuth_rad)
Vy_initial = speed_initial_mps * math.sin(angle_elevation_rad)  # Vertical component
Vz_initial = Vxy_initial * math.sin(angle_azimuth_rad)  # Sideways component (0 in this case)

V_initial = np.array([Vx_initial, Vy_initial, Vz_initial])

print(f"Initial Position P0: {P_initial} m")
print(
    f"Initial Velocity V0: {V_initial.round(3)} m/s (Speed: {speed_initial_mps} m/s, Elev: {angle_elevation_deg} deg)")

# Time parameters
t_start = 0.0
t_end = 1.5  # Simulate for 1.5 seconds (adjust as needed)
num_points = 100  # Number of points to simulate and plot
time_points = np.linspace(t_start, t_end, num_points)

# --- Run Simulations ---
# Model 1: Simple Projectile
trajectory_simple = simulate_simple_projectile(P_initial, V_initial, time_points)

# Model 2: Projectile with Air Drag
trajectory_drag = simulate_projectile_with_drag(P_initial, V_initial, time_points, KB_BALL)

# --- Plotting ---
fig = plt.figure(figsize=(12, 8))

# Plotting Y (Height) vs X (Forward Distance)
ax1 = fig.add_subplot(121)  # 1 row, 2 cols, first plot
ax1.plot(trajectory_simple[:, 0], trajectory_simple[:, 1], label='Simple Projectile (No Drag)', color='blue',
         linestyle='--')
if trajectory_drag is not None:
    ax1.plot(trajectory_drag[:, 0], trajectory_drag[:, 1], label='Projectile with Air Drag', color='red')
ax1.set_xlabel('Horizontal Distance X (m)')
ax1.set_ylabel('Height Y (m)')
ax1.set_title('Trajectory: Height vs. Horizontal Distance')
ax1.legend()
ax1.grid(True)
ax1.axhline(0, color='black', lw=0.5)  # Ground line
ax1.set_ylim(bottom=0)  # Start y-axis at ground

# Optional: Plotting Z (Sideways) vs X (Forward Distance) if Vz_initial is non-zero
# For this example, Z will be 0 for simple, and close to 0 for drag if Vz_initial=0
ax2 = fig.add_subplot(122)
ax2.plot(trajectory_simple[:, 0], trajectory_simple[:, 2], label='Simple Projectile (No Drag) - Z', color='blue',
         linestyle='--')
if trajectory_drag is not None:
    ax2.plot(trajectory_drag[:, 0], trajectory_drag[:, 2], label='Projectile with Air Drag - Z', color='red')
ax2.set_xlabel('Horizontal Distance X (m)')
ax2.set_ylabel('Sideways Distance Z (m)')
ax2.set_title('Trajectory: Sideways vs. Horizontal Distance')
ax2.legend()
ax2.grid(True)
ax2.axhline(0, color='black', lw=0.5)
ax2.set_ylim([-0.5, 0.5])  # Zoom in on Z axis if it's mostly zero

plt.tight_layout()
plt.show()

# Print some key values for comparison
if trajectory_drag is not None:
    print("\n--- Comparison ---")
    # Max height
    max_height_simple = np.max(trajectory_simple[:, 1])
    max_height_drag = np.max(trajectory_drag[:, 1])
    print(f"Max Height (Simple): {max_height_simple:.2f} m")
    print(f"Max Height (Drag):   {max_height_drag:.2f} m")

    # Range (horizontal distance when y returns to P_initial[1] or hits ground)
    # For simplicity, let's find when it hits ground (y=0) or initial height.
    # This is a crude way to find range, better to find roots.

    # Time to hit ground (simple model)
    # 0 = y0 + vy0*t - 0.5*g*t^2. Solve quadratic for t.
    # -0.5*g*t^2 + vy0*t + y0 = 0
    # a = -0.5*g, b = vy0, c = y0
    # t = (-b +/- sqrt(b^2 - 4ac)) / 2a
    a_quad = -0.5 * GRAVITY_ACCELERATION
    b_quad = V_initial[1]
    c_quad = P_initial[1]
    discriminant = b_quad ** 2 - 4 * a_quad * c_quad
    if discriminant >= 0:
        t_ground_simple1 = (-b_quad + np.sqrt(discriminant)) / (2 * a_quad)
        t_ground_simple2 = (-b_quad - np.sqrt(discriminant)) / (2 * a_quad)
        t_ground_simple = max(t_ground_simple1, t_ground_simple2)  # Take positive, larger root
        range_simple = P_initial[0] + V_initial[0] * t_ground_simple
        print(f"Approx. Range (Simple, to y=0): {range_simple:.2f} m at t={t_ground_simple:.2f}s")

    # Find when drag model hits ground (y approx 0)
    ground_hit_idx_drag = np.where(trajectory_drag[:, 1] <= 0.01)[0]  # Find first index where height is near 0
    if len(ground_hit_idx_drag) > 0:
        first_ground_hit_idx_drag = ground_hit_idx_drag[0]
        if first_ground_hit_idx_drag > 0:  # Ensure it's not the start point if y0 was 0
            range_drag = trajectory_drag[first_ground_hit_idx_drag, 0]
            time_at_range_drag = time_points[first_ground_hit_idx_drag]
            print(f"Approx. Range (Drag, to y<=0.01): {range_drag:.2f} m at t={time_at_range_drag:.2f}s")
else:
    print("Drag simulation failed, cannot compare ranges.")