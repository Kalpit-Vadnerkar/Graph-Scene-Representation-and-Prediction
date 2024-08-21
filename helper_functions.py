def lanelet2_to_graph_debug(map_data):
    # Display attributes of the LaneletMap object
    print("LaneletMap Attributes:")
    for attr in dir(map_data):
        if not attr.startswith("__") and not callable(getattr(map_data, attr)):
            print(f"  - {attr}: {getattr(map_data, attr)}")

    # Display attributes of the lineStringLayer object
    print("\nlineStringLayer Attributes:")
    for ls in map_data.lineStringLayer:
        if ls.attributes["type"] == "traffic_light" and ls.id == 1686:
            print("Attributes:")
            for attr in dir(ls):
                if not attr.startswith("__") and not callable(getattr(ls, attr)) and attr != "parameters":
                    print(f"  - {attr}: {getattr(ls, attr)}")
            # Print the points of the traffic light LineString
            print("Points of Traffic Light LineString:")
            for i, point in enumerate(ls):
                print(f"  - Point {i}: id={point.id}, x={point.x}, y={point.y}")
            break

    for ll in map_data.laneletLayer:
        if ll.id == 255:
            print("Lanelet Attributes:")
            for attr in dir(ll):
                if not attr.startswith("__") and not callable(getattr(ll, attr)):
                    print(f"  - {attr}: {getattr(ll, attr)}")
            print(f"Lanelet ID: {ll.id}")

            print("Centerline:")
            centerline = ll.centerline
            print(f"  - Number of points: {len(centerline)}")
            for i, point in enumerate(centerline):
                print(f"    - Point {i}: id={point.id}, x={point.x}, y={point.y}")

            print("Left and Right Bounds:")
            left_bound = ll.leftBound
            right_bound = ll.rightBound
            print(f"  - Left Bound: {len(left_bound)} points")
            for i, point in enumerate(left_bound):
                print(f"    - Point {i}: id={point.id}, x={point.x}, y={point.y}")
            print(f"  - Right Bound: {len(right_bound)} points")
            for i, point in enumerate(right_bound):
                print(f"    - Point {i}: id={point.id}, x={point.x}, y={point.y}")

            break  # Just print the first one to avoid cluttering the output
            
class Point:
    """
    Represents a 2D point with x and y coordinates.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        

def get_mid_point(line_string):
    x1, y1 = line_string[0].x, line_string[0].y
    x2, y2 = line_string[1].x, line_string[1].y
    
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    
    return Point(mx, my)

def convert_coordinate_frame(x_A, y_A):
    """
    Converts coordinates from frame A (e.g., geographic) to frame B (e.g., local Cartesian).

    Args:
        x_A: x-coordinate in frame A
        y_A: y-coordinate in frame A
        reference_points: List of tuples, each containing a point in frame A and its corresponding point in frame B. 
                          At least two reference points are required.

    Returns:
        Point: Converted coordinates (x_B, y_B) in frame B
    """

    reference_points = [
    ((81370.40, 49913.81), (3527.96, 1775.78)),
    ((81375.16, 49917.01), (3532.70, 1779.04)),
    ((81371.85, 49911.62), (3529.45, 1773.63)), 
    ((81376.60, 49914.82), (3534.15, 1776.87)),
    # ... Add more reference points if available for better accuracy
    ]
    if len(reference_points) < 2:
        raise ValueError("At least two reference points are required for conversion.")

    (x_A1, y_A1), (x_B1, y_B1) = reference_points[0]
    (x_A2, y_A2), (x_B2, y_B2) = reference_points[1]

    # Calculate scaling factors (consider both x and y differences)
    a = (x_B2 - x_B1) / (x_A2 - x_A1)
    c = (y_B2 - y_B1) / (y_A2 - y_A1)

    # Calculate translation factors (shift of origin)
    b = x_B1 - a * x_A1
    d = y_B1 - c * y_A1

    # Convert coordinates from frame A to frame B using affine transformation
    x_B = a * x_A + b
    y_B = c * y_A + d

    return Point(x_B, y_B)
    


