import numpy as np

def convert_coordinate_frame(x_A, y_A, points_A, points_B):
    # Convert input data to NumPy arrays
    points_A = np.array(points_A)
    points_B = np.array(points_B)

    # Calculate the translation vector
    translation = np.mean(points_B - points_A, axis=0)

    # Calculate the scaling factor
    magnitudes_A = np.linalg.norm(points_A, axis=1)
    magnitudes_B = np.linalg.norm(points_B, axis=1)
    scaling = np.mean(magnitudes_B / magnitudes_A)

    # Apply the transformation
    x_B = scaling * (x_A - points_A[:, 0].mean()) + translation[0] + points_B[:, 0].mean()
    y_B = scaling * (y_A - points_A[:, 1].mean()) + translation[1] + points_B[:, 1].mean()

    return x_B, y_B
    
points_A = [(81377.35044311438, 49916.90360337597), (81370.40, 49913.81), (81375.16, 49917.01), (81371.85, 49911.62), (81376.60, 49914.82)]
points_B = [(3535.7806390146916, 1779.5797283311367), (3527.96, 1775.78), (3532.70, 1779.04), (3529.45, 1773.63), (3534.15, 1776.87)]

x_A, y_A = 81373.0, 49915.0
x_B, y_B = convert_coordinate_frame(x_A, y_A, points_A, points_B)
print(f"Point in frame A: ({x_A}, {y_A}), Point in frame B: ({x_B:.2f}, {y_B:.2f})")
