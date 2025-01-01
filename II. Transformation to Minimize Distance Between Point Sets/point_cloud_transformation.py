import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Define source and target points
source_points = np.array([[1, 1], [2, 1], [2, 2], [3, 2]])  # Original points
rotation = R.from_euler('z', 30, degrees=True).as_matrix()[:2, :2]  # Rotate 30 degrees
translation = np.array([1, 0.5])  # Translate by (1, 0.5)

# Apply transformation: rotation + translation
transformed_points = (rotation @ source_points.T).T + translation

# Plot source and transformed points
plt.figure(figsize=(8, 6))
plt.scatter(source_points[:, 0], source_points[:, 1], label="Source Points", color="blue")
plt.scatter(transformed_points[:, 0], transformed_points[:, 1], label="Transformed Points", color="green")

# Draw lines between corresponding points
for i in range(len(source_points)):
    plt.plot([source_points[i, 0], transformed_points[i, 0]],
             [source_points[i, 1], transformed_points[i, 1]], 'k--')

# Labels and grid
plt.legend()
plt.title("Transformation of Point Cloud")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid()
plt.show()
