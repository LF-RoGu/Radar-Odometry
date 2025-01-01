import numpy as np

# Simulated point in 3D
point = np.array([1.0, 2.0, 3.0])

# Calculate Jacobian for translation and rotation
J_position = np.eye(3)  # Identity matrix for position
J_rotation = -np.array([[0, -point[2], point[1]],
                        [point[2], 0, -point[0]],
                        [-point[1], point[0], 0]])  # Skew-symmetric matrix

# Display Jacobians
print("Jacobian for Position (Translation):")
print(J_position)

print("\nJacobian for Rotation:")
print(J_rotation)
