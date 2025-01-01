import numpy as np
import matplotlib.pyplot as plt

# Generate example data for points and Doppler velocities
np.random.seed(42)
points = np.random.rand(10, 2) * 10  # 10 points in 2D space
velocities = np.random.rand(10, 2) * 2 - 1  # Random velocities in x and y directions

# Plot points
plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], c='blue', label='Radar Points')

# Plot velocity vectors
for i in range(len(points)):
    plt.arrow(points[i, 0], points[i, 1], velocities[i, 0], velocities[i, 1], 
              head_width=0.3, head_length=0.4, fc='red', ec='red')

plt.title('Doppler Velocity Vectors for Radar Points')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True)
plt.legend()
plt.show()
