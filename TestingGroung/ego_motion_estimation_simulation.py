import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle, Ellipse

# -------------------------------
# PARAMETERS AND INITIALIZATION
# -------------------------------

np.random.seed(42)  # Reproducibility
frames_per_second = 30  # Simulation at 30 FPS
num_frames = 30  # Total frames
num_frames_per_submap = 10  # Frames per submap
num_points_range = (1, 7)  # Random number of points per frame
vehicle_speed = 1.5  # Vehicle's constant speed (m/s)
time_step = 1 / frames_per_second  # Time step per frame

# Vehicle dimensions
vehicle_length = 4.0  # meters
vehicle_width = 2.0   # meters

# -------------------------------
# FUNCTION: Generate Submaps
# -------------------------------

def generate_submap(start_frame, end_frame, start_distance, end_distance):
    points = []
    for i in range(start_frame, end_frame):
        num_points = np.random.randint(num_points_range[0], num_points_range[1] + 1)
        distance = np.linspace(start_distance, end_distance, num_frames)[i]
        frame_points = np.random.rand(num_points, 2) * [20, 5] + [-10, distance]  # X-axis spans [-10, 10]
        points.append(frame_points)
    return np.vstack(points)

# Generate submaps
a1 = generate_submap(0, 10, 20, 15)
a2 = generate_submap(0, 20, 20, 10)
a3 = generate_submap(0, 30, 20, 5)

# -------------------------------
# FUNCTION: Cluster Points
# -------------------------------

def cluster_points(points):
    """ Perform DBSCAN clustering and filter clusters based on priorities. """
    dbscan = DBSCAN(eps=1.5, min_samples=2).fit(points)
    labels = dbscan.labels_

    clusters = {}
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Ignore noise
            continue
        cluster_points = points[labels == cluster_id]
        size = len(cluster_points)

        # Ignore clusters with <3 points (Priority 3)
        if size < 3:
            continue

        # Store centroid and priority
        centroid = np.mean(cluster_points, axis=0)
        priority = 1 if size >= 7 else 2  # Priority 1 for 7+, Priority 2 for 3-6
        clusters[cluster_id] = {'centroid': centroid, 'priority': priority, 'points': cluster_points}

    return clusters

# Perform clustering for a1, a2, a3
clusters_a1 = cluster_points(a1)
clusters_a2 = cluster_points(a2)
clusters_a3 = cluster_points(a3)

# -------------------------------
# FUNCTION: Transform and Estimate
# -------------------------------

def estimate_positions(clusters_a1, clusters_a2):
    """ Estimate positions in a3 based on transformations from a1 → a2. """
    estimated_positions = {}
    errors = []

    for cid, cluster_a1 in clusters_a1.items():
        if cid in clusters_a2:  # Match clusters by ID
            # Calculate transformation (translation)
            centroid_a1 = cluster_a1['centroid']
            centroid_a2 = clusters_a2[cid]['centroid']
            translation = centroid_a2 - centroid_a1

            # Predict next position
            estimated_position = centroid_a2 + translation  # Continue motion
            estimated_positions[cid] = estimated_position

            # Compute error with actual positions in a3
            if cid in clusters_a3:
                actual_position = clusters_a3[cid]['centroid']
                error = np.linalg.norm(estimated_position - actual_position)
                errors.append(error)

    return estimated_positions, errors

# Estimate positions and errors
estimated_positions, errors = estimate_positions(clusters_a1, clusters_a2)

# -------------------------------
# FUNCTION: Plot Clusters
# -------------------------------

def plot_clusters(clusters, title, original_points=None, estimated_positions=None, vehicle_pos=np.array([0.0, 0.0])):
    plt.figure(figsize=(8, 8))

    # Plot original points
    if original_points is not None:
        plt.scatter(original_points[:, 0], original_points[:, 1], c='gray', alpha=0.5, label='Original Points')

    # Plot clusters with priorities
    for cid, cluster in clusters.items():
        centroid = cluster['centroid']
        plt.scatter(cluster['points'][:, 0], cluster['points'][:, 1], label=f"Cluster {cid}")
        plt.scatter(centroid[0], centroid[1], c='black', marker='x')  # Centroid marker

        # Bounding box
        width = np.max(cluster['points'][:, 0]) - np.min(cluster['points'][:, 0])
        height = np.max(cluster['points'][:, 1]) - np.min(cluster['points'][:, 1])
        plt.gca().add_patch(Rectangle(
            (centroid[0] - width / 2, centroid[1] - height / 2), width, height,
            fill=False, edgecolor='purple', linewidth=1.5
        ))
        plt.text(centroid[0], centroid[1], f"P{cluster['priority']}", color='red')

        # Estimated positions and errors
        if estimated_positions and cid in estimated_positions:
            plt.scatter(estimated_positions[cid][0], estimated_positions[cid][1], c='orange', marker='^', label=f"Est. C{cid}")
            plt.plot([centroid[0], estimated_positions[cid][0]],
                     [centroid[1], estimated_positions[cid][1]], 'k--')

    # Draw the vehicle
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
# PLOT RESULTS
# -------------------------------

plot_clusters(clusters_a1, "Submap a1", a1)
plot_clusters(clusters_a2, "Submap a2", a1, estimated_positions)
plot_clusters(clusters_a3, "Submap a3", a2)

plt.tight_layout()
plt.show()
