import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# -------------------------------
# PARAMETERS AND INITIALIZATION
# -------------------------------

# Simulation Parameters
np.random.seed(42)  # Reproducibility
num_points = 20  # Number of points per scan
vehicle_speed = 1.5  # Vehicle's constant speed (m/s)
time_step = 1.0  # Sampling time step (seconds)

# Doppler velocities for points (radial speeds)
doppler_speeds = np.random.uniform(1.2, 1.5, num_points)  # Variation between 1.2–1.5 m/s

# Vehicle Dimensions for Visualization
vehicle_length = 4.0  # meters
vehicle_width = 2.0   # meters

# Initial Position and Heading of the Vehicle
vehicle_pos = np.array([0.0, 0.0])  # Initial position (x, y) as floats
vehicle_heading = 0  # Initial heading angle (degrees)

# -------------------------------
# STEP 1: GENERATE INITIAL POINT CLOUD (10 Previous Scans)
# -------------------------------

# Generate 10 previous scans around the vehicle
prev_points = np.random.rand(num_points, 2) * 20 - 10  # Points within a 20x20m area

# Plot Initial Scan
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)

# Plot points and vehicle box
plt.scatter(prev_points[:, 0], prev_points[:, 1], c='blue', label='Prev 10 Samples')
plt.plot([-vehicle_width/2, vehicle_width/2, vehicle_width/2, -vehicle_width/2, -vehicle_width/2],
         [-vehicle_length/2, -vehicle_length/2, vehicle_length/2, vehicle_length/2, -vehicle_length/2],
         'k-', label='Vehicle')

plt.title("Step 1: Initial Scan (10 Samples)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.axis('equal')
plt.legend()
plt.grid()

# -------------------------------
# STEP 2: SIMULATE MOTION (Next 10 Scans)
# -------------------------------

# Apply Translation and Small Rotation (Simulated Motion)
translation_1 = np.array([vehicle_speed * time_step, 0])  # Move straight by 1.5m
rotation_1 = R.from_euler('z', 5, degrees=True).as_matrix()[:2, :2]  # Rotate 5 degrees

# Update Vehicle Position
vehicle_pos += translation_1
vehicle_heading += 5  # Update heading by 5 degrees

# Transform previous points based on motion
transformed_points_1 = (rotation_1 @ prev_points.T).T + translation_1

# Generate 10 new detections for next scans
new_points_1 = np.random.rand(num_points, 2) * 20 - 10  # New detections

# Plot Motion Step 1
plt.subplot(1, 3, 2)

# Plot transformed and new points
plt.scatter(transformed_points_1[:, 0], transformed_points_1[:, 1], c='green', label='Transformed Points (Step 1)')
plt.scatter(new_points_1[:, 0], new_points_1[:, 1], c='red', label='New 10 Samples')

# Add Previous-Previous Samples and Dotted Line Connection
plt.scatter(prev_points[:, 0], prev_points[:, 1], c='blue', marker='x', label='Prev-Prev Samples')
for i in range(num_points):
    plt.plot([prev_points[i, 0], transformed_points_1[i, 0]],
             [prev_points[i, 1], transformed_points_1[i, 1]], 'k--')  # Dotted lines

# Plot vehicle box at new position
plt.plot([-vehicle_width/2, vehicle_width/2, vehicle_width/2, -vehicle_width/2, -vehicle_width/2] + vehicle_pos[0],
         [-vehicle_length/2, -vehicle_length/2, vehicle_length/2, vehicle_length/2, -vehicle_length/2],
         'k-', label='Vehicle')

plt.title("Step 2: Motion Estimation (Next 10 Samples)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.axis('equal')
plt.legend()
plt.grid()

# -------------------------------
# STEP 3: SIMULATE SECOND MOTION (20 Total Samples)
# -------------------------------

# Apply Second Translation and Rotation
translation_2 = np.array([vehicle_speed * time_step, 0])  # Move straight by 1.5m again
rotation_2 = R.from_euler('z', 10, degrees=True).as_matrix()[:2, :2]  # Rotate 10 degrees

# Update Vehicle Position
vehicle_pos += translation_2
vehicle_heading += 10  # Update heading by 10 degrees

# Transform previous transformed points and add new detections
transformed_points_2 = (rotation_2 @ transformed_points_1.T).T + translation_2
new_points_2 = np.random.rand(num_points, 2) * 20 - 10  # New detections again

# Plot Motion Step 2
plt.subplot(1, 3, 3)

# Plot transformed and new points
plt.scatter(transformed_points_2[:, 0], transformed_points_2[:, 1], c='green', label='Transformed Points (Step 2)')
plt.scatter(new_points_2[:, 0], new_points_2[:, 1], c='orange', label='New 10 Samples (Step 2)')

# Add Previous Samples (from Step 2) and Dotted Line Connection
plt.scatter(transformed_points_1[:, 0], transformed_points_1[:, 1], c='purple', marker='x', label='Prev-Prev Samples (Step 2)')
for i in range(num_points):
    plt.plot([transformed_points_1[i, 0], transformed_points_2[i, 0]],
             [transformed_points_1[i, 1], transformed_points_2[i, 1]], 'k--')  # Dotted lines

# Vehicle box updated position
plt.plot([-vehicle_width/2, vehicle_width/2, vehicle_width/2, -vehicle_width/2, -vehicle_width/2] + vehicle_pos[0],
         [-vehicle_length/2, -vehicle_length/2, vehicle_length/2, vehicle_length/2, -vehicle_length/2],
         'k-', label='Vehicle')

plt.title("Step 3: Motion Estimation (Final Update)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.axis('equal')
plt.legend()
plt.grid()

# -------------------------------
# DISPLAY PLOTS
# -------------------------------

plt.tight_layout()
plt.show()
