import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

# -------------------------------
# PARAMETERS AND INITIALIZATION
# -------------------------------

np.random.seed(42)  # Reproducibility
num_points = 20  # Number of points per frame
vehicle_speed = 1.5  # Vehicle's constant speed (m/s) along positive Y-axis
time_step = 1 / 30.0  # 30 frames per second
total_frames = 30  # Total number of frames to simulate

# Doppler velocities (negative, simulating approaching targets)
doppler_speeds = np.random.uniform(-1.5, -1.2, num_points)  # Negative values (-1.2 to -1.5 m/s)

# Vehicle dimensions for visualization
vehicle_length = 4.0  # meters
vehicle_width = 2.0   # meters

# Initial position and heading of the vehicle
vehicle_pos = np.array([0.0, 0.0])  # Initial position (x, y) as floats
vehicle_heading = 0  # Initial heading angle (degrees)

# -------------------------------
# FUNCTION TO CLUSTER AND PLOT OBJECTS
# -------------------------------

def plot_clusters(points, cluster_labels, title, vehicle_pos, original_points=None, prev_centroids=None, transformed_centroids=None):
    # Perform clustering for the given points
    dbscan = DBSCAN(eps=3, min_samples=2).fit(points)
    cluster_labels = dbscan.labels_
    
    # Plot original points in gray
    if original_points is not None:
        plt.scatter(original_points[:, 0], original_points[:, 1], c='gray', alpha=0.5, label='Original Points')
    
    # Plot clusters
    for cluster_id in np.unique(cluster_labels):
        cluster_points = points[cluster_labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")
        
        # Add centroid and bounding box
        if cluster_id != -1:  # Ignore noise
            centroid = np.mean(cluster_points, axis=0)
            plt.scatter(centroid[0], centroid[1], c='black', marker='x')  # Centroid
            
            # Bounding box (area)
            width = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
            height = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
            plt.gca().add_patch(plt.Rectangle(
                (centroid[0] - width / 2, centroid[1] - height / 2), width, height,
                fill=False, edgecolor='purple', linewidth=1.5
            ))
            
            # Connect previous centroids to transformed centroids with dotted lines
            if prev_centroids is not None and transformed_centroids is not None:
                prev_centroid = prev_centroids[cluster_id]
                transformed_centroid = transformed_centroids[cluster_id]
                plt.plot([prev_centroid[0], transformed_centroid[0]],
                         [prev_centroid[1], transformed_centroid[1]], 'k--')  # Dotted line
    
    # Plot vehicle position
    plt.plot([-vehicle_width/2, vehicle_width/2, vehicle_width/2, -vehicle_width/2, -vehicle_width/2] + vehicle_pos[0],
             [vehicle_pos[1], vehicle_pos[1], vehicle_pos[1] + vehicle_length, vehicle_pos[1] + vehicle_length, vehicle_pos[1]],
             'k-', label='Vehicle')

    # Final plot settings
    plt.title(title)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid()

# -------------------------------
# SIMULATION - FRAME PROCESSING
# -------------------------------

# Generate 10 frames for the initial submap
prev_points = np.random.rand(num_points, 2) * [20, 10] + [0, 10]

plt.figure(figsize=(15, 10))

# -------------------------------
# STEP 1: INITIAL CLUSTERING (First 10 Frames)
# -------------------------------
plt.subplot(1, 3, 1)
plot_clusters(prev_points, None, "Step 1: Initial Clustering", vehicle_pos, original_points=prev_points)

# -------------------------------
# STEP 2: MOTION UPDATE (Next 10 Frames)
# -------------------------------

# Translation and rotation (Step 1)
doppler_based_motion = np.mean(np.abs(doppler_speeds)) * time_step * 10  # 10 frames
translation_1 = np.array([0, doppler_based_motion])
rotation_1 = R.from_euler('z', 5, degrees=True).as_matrix()[:2, :2]

# Update vehicle position
vehicle_pos += translation_1
vehicle_heading += 5

# Transform points for motion
transformed_points_1 = (rotation_1 @ prev_points.T).T + translation_1

# Generate 10 new detections
new_points_1 = np.random.rand(num_points, 2) * [20, 10] + [0, 10]

# Plot second submap
plt.subplot(1, 3, 2)
plot_clusters(transformed_points_1, None, "Step 2: Motion Update (10+10)", vehicle_pos,
              original_points=prev_points, prev_centroids=prev_points, transformed_centroids=transformed_points_1)

# -------------------------------
# STEP 3: FINAL MOTION UPDATE (Last 10 Frames)
# -------------------------------

# Translation and rotation (Step 2)
translation_2 = np.array([0, doppler_based_motion])
rotation_2 = R.from_euler('z', 10, degrees=True).as_matrix()[:2, :2]

# Update vehicle position
vehicle_pos += translation_2
vehicle_heading += 10

# Transform points for the next step
transformed_points_2 = (rotation_2 @ transformed_points_1.T).T + translation_2

# Plot third submap
plt.subplot(1, 3, 3)
plot_clusters(transformed_points_2, None, "Step 3: Final Motion Update (20+10)", vehicle_pos,
              original_points=transformed_points_1, prev_centroids=transformed_points_1, transformed_centroids=transformed_points_2)

# -------------------------------
# DISPLAY PLOTS
# -------------------------------

plt.tight_layout()
plt.show()
