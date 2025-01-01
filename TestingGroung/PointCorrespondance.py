import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle

# -------------------------------
# PARAMETERS AND INITIALIZATION
# -------------------------------

np.random.seed(42)
num_points_range = (1, 7)  # Random number of points per frame
num_frames = 30  # Total frames
vehicle_length = 4.0  # meters
vehicle_width = 2.0   # meters

# -------------------------------
# GENERATE POINTS FOR TWO SUBMAPS
# -------------------------------

def generate_submap(start_frame, end_frame, start_distance, end_distance):
    points = []
    for i in range(start_frame, end_frame):
        num_points = np.random.randint(num_points_range[0], num_points_range[1] + 1)
        distance = np.linspace(start_distance, end_distance, num_frames)[i]
        frame_points = np.random.rand(num_points, 2) * [20, 5] + [-10, distance]
        points.append(frame_points)
    return np.vstack(points)

# Submaps
submap_a1 = generate_submap(0, 10, 20, 15)  # First submap
submap_a2 = generate_submap(0, 20, 20, 10)  # Second submap

# -------------------------------
# PERFORM CLUSTERING
# -------------------------------

def cluster_points(points):
    """ Perform DBSCAN clustering. """
    dbscan = DBSCAN(eps=1.5, min_samples=2).fit(points)
    labels = dbscan.labels_

    clusters = {}
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Ignore noise
            continue
        cluster_points = points[labels == cluster_id]
        size = len(cluster_points)
        if size < 3:
            continue  # Skip clusters with less than 3 points
        centroid = np.mean(cluster_points, axis=0)
        clusters[cluster_id] = {'centroid': centroid, 'points': cluster_points}
    return clusters

# Cluster both submaps
clusters_a1 = cluster_points(submap_a1)
clusters_a2 = cluster_points(submap_a2)

# -------------------------------
# FIND POINT CORRESPONDENCES
# -------------------------------

def find_correspondences(clusters_a1, clusters_a2):
    """ Match centroids between clusters using nearest neighbors. """
    correspondences = {}
    centroids_a1 = np.array([clusters_a1[cid]['centroid'] for cid in clusters_a1])
    centroids_a2 = np.array([clusters_a2[cid]['centroid'] for cid in clusters_a2])

    # Compute pairwise distances
    distances = cdist(centroids_a1, centroids_a2)

    # Find nearest neighbors
    for i, cid_a1 in enumerate(clusters_a1.keys()):
        nearest_idx = np.argmin(distances[i])  # Index of nearest neighbor
        cid_a2 = list(clusters_a2.keys())[nearest_idx]
        correspondences[cid_a1] = cid_a2  # Store matched pairs

    return correspondences

# Get correspondences
correspondences = find_correspondences(clusters_a1, clusters_a2)

# -------------------------------
# VISUALIZE POINT CORRESPONDENCES
# -------------------------------

def plot_clusters_and_correspondences(submap_a1, clusters_a1, submap_a2, clusters_a2, correspondences):
    plt.figure(figsize=(10, 8))

    # Plot clusters for a1
    for cid, cluster in clusters_a1.items():
        plt.scatter(cluster['points'][:, 0], cluster['points'][:, 1], label=f"Cluster A1-{cid}", alpha=0.6)

        # Draw bounding box for a1
        x_min, x_max = np.min(cluster['points'][:, 0]), np.max(cluster['points'][:, 0])
        y_min, y_max = np.min(cluster['points'][:, 1]), np.max(cluster['points'][:, 1])
        plt.gca().add_patch(Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=False, edgecolor='blue', linewidth=1.5
        ))

    # Plot clusters for a2
    for cid, cluster in clusters_a2.items():
        plt.scatter(cluster['points'][:, 0], cluster['points'][:, 1], marker='x', label=f"Cluster A2-{cid}", alpha=0.6)

        # Draw bounding box for a2
        x_min, x_max = np.min(cluster['points'][:, 0]), np.max(cluster['points'][:, 0])
        y_min, y_max = np.min(cluster['points'][:, 1]), np.max(cluster['points'][:, 1])
        plt.gca().add_patch(Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=False, edgecolor='green', linewidth=1.5
        ))

    # Draw correspondences
    for cid_a1, cid_a2 in correspondences.items():
        centroid_a1 = clusters_a1[cid_a1]['centroid']
        centroid_a2 = clusters_a2[cid_a2]['centroid']
        plt.plot([centroid_a1[0], centroid_a2[0]], [centroid_a1[1], centroid_a2[1]], 'k--')

    # Draw vehicle
    plt.plot([-vehicle_width/2, vehicle_width/2, vehicle_width/2, -vehicle_width/2, -vehicle_width/2],
             [0, 0, vehicle_length, vehicle_length, 0], 'k-', label="Vehicle")

    plt.title("Point Correspondences with Clusters and Bounding Boxes")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

# Plot results
plot_clusters_and_correspondences(submap_a1, clusters_a1, submap_a2, clusters_a2, correspondences)