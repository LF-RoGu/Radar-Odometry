import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

# -------------------------------
# PARAMETERS AND INITIALIZATION
# -------------------------------

np.random.seed(42)  # Reproducibility
num_points = 20  # Number of points per scan
vehicle_speed = 1.5  # Vehicle's constant speed (m/s) along positive Y-axis
time_step = 1.0  # Sampling time step (seconds)

# Doppler velocities (negative, simulating approaching targets)
doppler_speeds = np.random.uniform(-1.5, -1.2, num_points)  # Negative values (-1.2 to -1.5 m/s)

# Vehicle dimensions for visualization
vehicle_length = 4.0  # meters
vehicle_width = 2.0   # meters

# Initial position and heading of the vehicle
vehicle_pos = np.array([0.0, 0.0])  # Initial position (x, y) as floats
vehicle_heading = 0  # Initial heading angle (degrees)

# -------------------------------
# STEP 1: INITIAL SETUP WITH CLUSTERING
# -------------------------------

# Generate 10 previous scans (points in front of the vehicle)
prev_points = np.random.rand(num_points, 2) * [20, 10] + [0, 10]  # X in [-10, 10], Y in [10, 20]

# Perform DBSCAN clustering based on spatial position
dbscan = DBSCAN(eps=3, min_samples=2).fit(prev_points)  # eps=3 defines neighborhood radius
cluster_labels = dbscan.labels_  # Cluster IDs

# Plot Initial Scan with Clusters
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)

# Plot points with cluster colors
for cluster_id in np.unique(cluster_labels):
    cluster_points = prev_points[cluster_labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")

# Plot the vehicle
plt.plot([-vehicle_width/2, vehicle_width/2, vehicle_width/2, -vehicle_width/2, -vehicle_width/2],
         [0, 0, vehicle_length, vehicle_length, 0], 'k-', label='Vehicle')

plt.title("Step 1: Initial Clustering (10 Samples)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.axis('equal')
plt.legend()
plt.grid()

# -------------------------------
# STEP 2: SIMULATE MOTION AND UPDATE TRANSFORMATIONS
# -------------------------------

# Compute motion based on Doppler speeds
doppler_based_motion = np.mean(np.abs(doppler_speeds)) * time_step  # Average motion

# Translation and rotation (Step 1)
translation_1 = np.array([0, doppler_based_motion])  # Move forward in Y-axis
rotation_1 = R.from_euler('z', 5, degrees=True).as_matrix()[:2, :2]  # Rotate 5 degrees

# Update vehicle position and heading
vehicle_pos += translation_1
vehicle_heading += 5

# Transform points using rotation and translation
transformed_points_1 = (rotation_1 @ prev_points.T).T + translation_1

# Generate 10 new detections
new_points_1 = np.random.rand(num_points, 2) * [20, 10] + [0, 10]

# Plot Motion Step 1
plt.subplot(1, 3, 2)

# Plot transformed and new points
plt.scatter(transformed_points_1[:, 0], transformed_points_1[:, 1], c='green', label='Transformed Points (Step 1)')
plt.scatter(new_points_1[:, 0], new_points_1[:, 1], c='red', label='New 10 Samples')

# Plot original positions and connect them with dotted lines
plt.scatter(prev_points[:, 0], prev_points[:, 1], c='blue', marker='x', label='Prev-Prev Samples')
for i in range(num_points):
    plt.plot([prev_points[i, 0], transformed_points_1[i, 0]],
             [prev_points[i, 1], transformed_points_1[i, 1]], 'k--')  # Dotted lines for motion

# Vehicle box updated position
plt.plot([-vehicle_width/2, vehicle_width/2, vehicle_width/2, -vehicle_width/2, -vehicle_width/2],
         [vehicle_pos[1], vehicle_pos[1], vehicle_pos[1] + vehicle_length, vehicle_pos[1] + vehicle_length, vehicle_pos[1]],
         'k-', label='Vehicle')

plt.title("Step 2: Motion Estimation (Next 10 Samples)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.axis('equal')
plt.legend()
plt.grid()

# -------------------------------
# STEP 3: FINAL MOTION UPDATE
# -------------------------------

# Translation and rotation (Step 2)
translation_2 = np.array([0, doppler_based_motion])  # Move forward again
rotation_2 = R.from_euler('z', 10, degrees=True).as_matrix()[:2, :2]  # Rotate 10 degrees

# Update vehicle position and heading
vehicle_pos += translation_2
vehicle_heading += 10

# Transform points again
transformed_points_2 = (rotation_2 @ transformed_points_1.T).T + translation_2

# Plot Motion Step 2
plt.subplot(1, 3, 3)

# Plot transformed points
plt.scatter(transformed_points_2[:, 0], transformed_points_2[:, 1], c='green', label='Transformed Points (Step 2)')
plt.scatter(new_points_1[:, 0], new_points_1[:, 1], c='orange', label='New 10 Samples (Step 2)')

# Plot original positions and connect them with dotted lines
plt.scatter(transformed_points_1[:, 0], transformed_points_1[:, 1], c='purple', marker='x', label='Prev-Prev Samples (Step 2)')
for i in range(num_points):
    plt.plot([transformed_points_1[i, 0], transformed_points_2[i, 0]],
             [transformed_points_1[i, 1], transformed_points_2[i, 1]], 'k--')  # Dotted lines for motion

# Vehicle box updated position
plt.plot([-vehicle_width/2, vehicle_width/2, vehicle_width/2, -vehicle_width/2, -vehicle_width/2],
         [vehicle_pos[1], vehicle_pos[1], vehicle_pos[1] + vehicle_length, vehicle_pos[1] + vehicle_length, vehicle_pos[1]],
         'k-', label='Vehicle')

plt.title("Step 3: Final Motion Estimation")
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
