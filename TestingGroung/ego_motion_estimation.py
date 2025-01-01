import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R

# -------------------------------
# Step 1: Generate Simulated Point Cloud with Doppler Velocities
# -------------------------------

np.random.seed(42)
num_points = 100  # Total points
doppler_speeds = np.random.uniform(-1, 1, num_points)  # Doppler velocities (-1 to 1 m/s)

# Generate random points for 10 scans (Submap 1)
prev_points = np.random.rand(num_points, 2) * 20

# Generate new scan (Submap 2)
theta = 5  # Small 5-degree rotation
rotation = R.from_euler('z', theta, degrees=True).as_matrix()[:2, :2]
translation = np.array([0.2, 0.1])  # Translation (x, y)

new_points = (rotation @ prev_points.T).T + translation

# -------------------------------
# Step 2: Clustering Points (DBSCAN)
# -------------------------------

clustering = DBSCAN(eps=1.5, min_samples=3).fit(prev_points)
labels = clustering.labels_

# Visualize clustering
plt.figure(figsize=(10, 8))
plt.scatter(prev_points[:, 0], prev_points[:, 1], c=labels, cmap='viridis', label="Previous Scan")
plt.scatter(new_points[:, 0], new_points[:, 1], marker='x', color='red', label="New Scan")
plt.title("Clustering and Motion Estimation")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# Step 3: Estimate Ego Motion
# -------------------------------

# Calculate Centroid Translation
centroid_prev = np.mean(prev_points, axis=0)  # Centroid of previous points
centroid_new = np.mean(new_points, axis=0)  # Centroid of new points
translation_estimated = centroid_new - centroid_prev

# Estimate Rotation (using SVD for better numerical stability)
H = (prev_points - centroid_prev).T @ (new_points - centroid_new)
U, S, Vt = np.linalg.svd(H)
R_est = Vt.T @ U.T  # Optimal rotation matrix

# Calculate Rotation Angle
theta_est = np.arctan2(R_est[1, 0], R_est[0, 0]) * 180 / np.pi  # Convert radians to degrees

# -------------------------------
# Results
# -------------------------------
print(f"Estimated Translation: {translation_estimated}")
print(f"Estimated Rotation (degrees): {theta_est:.2f}")
