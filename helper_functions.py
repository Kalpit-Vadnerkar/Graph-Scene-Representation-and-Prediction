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
            
