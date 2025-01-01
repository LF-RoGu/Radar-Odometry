import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Define source and target points
source_points = np.array([[1, 1], [2, 1], [2, 2], [3, 2]])
rotation = R.from_euler('z', 30, degrees=True).as_matrix()[:2, :2]
translation = np.array([1, 0.5])
transformed_points = (rotation @ source_points.T).T + translation

# Calculate Euclidean distances
distances = np.linalg.norm(transformed_points - source_points, axis=1)

# Plot source and transformed points with distances
plt.figure(figsize=(8, 6))
for i in range(len(source_points)):
    plt.plot([source_points[i, 0], transformed_points[i, 0]],
             [source_points[i, 1], transformed_points[i, 1]], 'k--')  # Lines connecting matches

plt.scatter(source_points[:, 0], source_points[:, 1], label="Source Points", color="blue")
plt.scatter(transformed_points[:, 0], transformed_points[:, 1], label="Transformed Points", color="green")

# Display distances
for i, d in enumerate(distances):
    plt.text((source_points[i, 0] + transformed_points[i, 0]) / 2,
             (source_points[i, 1] + transformed_points[i, 1]) / 2,
             f'{d:.2f}', fontsize=9, color='red')

plt.legend()
plt.title("Euclidean Distances Between Matches")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid()
plt.show()
