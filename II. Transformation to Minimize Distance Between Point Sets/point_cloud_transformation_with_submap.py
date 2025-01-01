import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# -------------------------------
# Simulate Submap Aggregation
# -------------------------------

# Generate 10 previous scans (submap)
np.random.seed(0)
num_scans = 10  # Number of previous scans
points_per_scan = 7  # Points per scan
submap_points = []

for _ in range(num_scans):
    scan = np.random.rand(points_per_scan, 2) * 10  # Random 2D points in 10x10m
    submap_points.append(scan)

# Merge all previous scans to form the submap
submap = np.vstack(submap_points)  # Combine all scans into one array

# Simulate a new scan (current scan)
current_scan = np.random.rand(points_per_scan, 2) * 10

# -------------------------------
# Apply Transformation
# -------------------------------

# Define transformation: rotation and translation
rotation = R.from_euler('z', 15, degrees=True).as_matrix()[:2, :2]  # Rotate by 15 degrees
translation = np.array([0.5, 1.0])  # Translate by (0.5, 1.0)

# Apply transformation to the current scan
transformed_scan = (rotation @ current_scan.T).T + translation

# -------------------------------
# Visualization of Submap and Transformation
# -------------------------------

plt.figure(figsize=(10, 8))

# Plot the submap
plt.scatter(submap[:, 0], submap[:, 1], label='Submap (Aggregated Scans)', color='lightgray')

# Plot the current scan
plt.scatter(current_scan[:, 0], current_scan[:, 1], label='Current Scan (Before Transformation)', color='blue')

# Plot the transformed scan
plt.scatter(transformed_scan[:, 0], transformed_scan[:, 1], label='Transformed Scan', color='green')

# Draw correspondences (lines between points)
for i in range(points_per_scan):
    plt.plot([current_scan[i, 0], transformed_scan[i, 0]],
             [current_scan[i, 1], transformed_scan[i, 1]], 'k--')

# Labels and grid
plt.title("Submap Aggregation and Point Cloud Transformation")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid()
plt.show()
