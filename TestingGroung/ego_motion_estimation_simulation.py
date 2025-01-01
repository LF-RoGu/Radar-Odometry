import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from matplotlib.patches import Ellipse

# -------------------------------
# PARAMETERS AND INITIALIZATION
# -------------------------------

np.random.seed(42)  # Reproducibility
frames_per_second = 30  # Simulation at 30 FPS
num_frames_per_submap = 10  # Frames per submap
num_points_range = (2, 7)  # Random number of points per frame
vehicle_speed = 1.5  # Vehicle's constant speed (m/s) along positive Y-axis
time_step = 1 / frames_per_second  # Time step per frame
doppler_speeds = np.random.uniform(-1.5, -1.2, num_points_range[1])  # Fixed range

# Vehicle dimensions for visualization
vehicle_length = 4.0  # meters
vehicle_width = 2.0   # meters

# Initial position and heading of the vehicle
vehicle_pos = np.array([0.0, 0.0])  # Initial position (x, y)
vehicle_heading = 0  # Initial heading angle (degrees)

# -------------------------------
# FUNCTION TO CLUSTER AND PLOT OBJECTS
# -------------------------------

def plot_clusters(points, title, vehicle_pos, prev_points=None, prev_centroids=None):
    # Perform clustering using DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=2).fit(points)  # Adjusted eps and min_samples
    cluster_labels = dbscan.labels_

    # Plot previous points in gray with ellipses
    if prev_points is not None:
        plt.scatter(prev_points[:, 0], prev_points[:, 1], c='gray', alpha=0.5, label='Previous Points')
        
        # Clustering for previous points
        prev_dbscan = DBSCAN(eps=1.5, min_samples=2).fit(prev_points)
        prev_labels = prev_dbscan.labels_

        # Draw ellipses for previous clusters
        for prev_cluster_id in np.unique(prev_labels):
            if prev_cluster_id == -1:  # Ignore noise
                continue

            prev_cluster_points = prev_points[prev_labels == prev_cluster_id]
            prev_centroid = np.mean(prev_cluster_points, axis=0)

            # Draw ellipse
            width = np.max(prev_cluster_points[:, 0]) - np.min(prev_cluster_points[:, 0])
            height = np.max(prev_cluster_points[:, 1]) - np.min(prev_cluster_points[:, 1])
            ellipse = Ellipse(xy=prev_centroid, width=width, height=height, edgecolor='gray', facecolor='none', linestyle='--')
            plt.gca().add_patch(ellipse)

    # Plot current clusters with bounding boxes
    for cluster_id in np.unique(cluster_labels):
        cluster_points = points[cluster_labels == cluster_id]
        if cluster_id == -1:  # Skip noise
            continue

        # Plot cluster points
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")

        # Calculate and plot centroid
        centroid = np.mean(cluster_points, axis=0)
        plt.scatter(centroid[0], centroid[1], c='black', marker='x')  # Centroid marker
        plt.annotate(f"C{cluster_id}", (centroid[0], centroid[1]), color='blue')  # Label

        # Bounding box for current cluster
        width = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
        height = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
        plt.gca().add_patch(plt.Rectangle(
            (centroid[0] - width / 2, centroid[1] - height / 2), width, height,
            fill=False, edgecolor='purple', linewidth=1.5
        ))

        # Classify priority based on size
        size = len(cluster_points)
        if size >= 7:
            priority = 1
        elif size >= 3:
            priority = 2
        else:
            priority = 3
        plt.text(centroid[0], centroid[1], f"P{priority}", color='red', fontsize=10)

        # Dotted lines for motion paths
        if prev_centroids is not None:
            prev_centroid = prev_centroids[cluster_id]
            plt.plot([prev_centroid[0], centroid[0]], [prev_centroid[1], centroid[1]], 'k--')

    # Vehicle box visualization
    plt.plot([-vehicle_width/2, vehicle_width/2, vehicle_width/2, -vehicle_width/2, -vehicle_width/2] + vehicle_pos[0],
             [vehicle_pos[1], vehicle_pos[1], vehicle_pos[1] + vehicle_length, vehicle_pos[1] + vehicle_length, vehicle_pos[1]],
             'k-', label='Vehicle')

    plt.title(title)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid()

# -------------------------------
# GENERATE SUBMAPS (a1, a2, a3)
# -------------------------------

def generate_submap(num_frames, start_distance, end_distance):
    points = []
    for i in range(num_frames):
        num_points = np.random.randint(num_points_range[0], num_points_range[1] + 1)
        distance = np.linspace(start_distance, end_distance, num_frames)[i]
        frame_points = np.random.rand(num_points, 2) * [20, 5] + [0, distance]
        points.append(frame_points)
    return np.vstack(points)

a1 = generate_submap(num_frames_per_submap, 20, 15)
a2 = generate_submap(num_frames_per_submap, 15, 10)
a3 = generate_submap(num_frames_per_submap, 10, 5)

# -------------------------------
# TRANSFORMATIONS AND PLOTS
# -------------------------------

plt.figure(figsize=(15, 10))

# Plot 1: Submap a1
plt.subplot(1, 3, 1)
plot_clusters(a1, "Submap a1", vehicle_pos)

# Plot 2: Submap a1 → a2
plt.subplot(1, 3, 2)
plot_clusters(a2, "Submap a1 → a2", vehicle_pos, prev_points=a1)

# Plot 3: Submap a2 → a3
plt.subplot(1, 3, 3)
plot_clusters(a3, "Submap a2 → a3", vehicle_pos, prev_points=a2)

plt.tight_layout()
plt.show()
