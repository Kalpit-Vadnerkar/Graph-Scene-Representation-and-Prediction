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
    # Given points in frame A (mrgs) and B (local cartesian)
    x_A1, y_A1 = 81377.35044311438, 49916.90360337597
    x_B1, y_B1 = 3535.7806390146916, 1779.5797283311367

    # Calculate scaling factors
    a = x_B1 / x_A1
    c = y_B1 / y_A1

    # Convert coordinates from frame A to frame B
    x_B = a * x_A
    y_B = c * y_A

    return Point(x_B, y_B)
