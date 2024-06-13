import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from autoware_planning_msgs.msg import LaneletRoute
from autoware_auto_vehicle_msgs.msg import SteeringReport, VelocityReport
from tf2_msgs.msg import TFMessage
from autoware_auto_perception_msgs.msg import TrackedObjects, TrafficSignalArray
from rclpy.serialization import serialize_message
import json  # For encoding data to JSON format

def extract_steering_data(msg):
    return {"timestamp_sec": msg.stamp.sec,
            "timestamp_ns": msg.stamp.nanosec,
            "steering_angle": msg.steering_tire_angle}

def extract_velocity_data(msg):
    return {"timestamp_sec": msg.header.stamp.sec,
            "timestamp_ns": msg.header.stamp.nanosec,
            "longitudinal_velocity": msg.longitudinal_velocity,
            "lateral_velocity": msg.lateral_velocity,
            "yaw_rate": msg.heading_rate,
    }

def extract_tracked_objects_data(msg):
    extracted_objects = []
    num_objects = len(msg.objects)
    is_map_frame = (msg.header.frame_id == "map") 

    for obj in msg.objects:
        position = obj.kinematics.pose_with_covariance.pose.position
        orientation = obj.kinematics.pose_with_covariance.pose.orientation
        linear_velocity = obj.kinematics.twist_with_covariance.twist.linear
        classification = max(obj.classification, key=lambda c: c.probability).label

        extracted_objects.append({
            "x": position.x,
            "y": position.y,
            "z": position.z,
            "orientation_x": orientation.x,
            "orientation_y": orientation.y,
            "orientation_z": orientation.z,
            "orientation_w": orientation.w,
            "linear_velocity_x": linear_velocity.x,
            "linear_velocity_y": linear_velocity.y,
            "linear_velocity_z": linear_velocity.z,
            "classification": classification,
        })

    result = {
        "timestamp_sec": msg.header.stamp.sec,
        "timestamp_ns": msg.header.stamp.nanosec,
        "num_objects": num_objects,
        "is_map_frame": is_map_frame, # Add the frame id information
        "objects": extracted_objects,
    }
    return result

def extract_vehicle_pos(msg):
    for transform in msg.transforms:  # Iterate through the transforms
        if (transform.header.frame_id == "map" and
                transform.child_frame_id == "base_link"):  # Find the relevant transform

            # Extract position
            position = {
                "x": transform.transform.translation.x,
                "y": transform.transform.translation.y,
                "z": transform.transform.translation.z,
            }

            # Extract orientation
            orientation = {
                "x": transform.transform.rotation.x,
                "y": transform.transform.rotation.y,
                "z": transform.transform.rotation.z,
                "w": transform.transform.rotation.w,
            }
            
            result = {
                "timestamp_sec": transform.header.stamp.sec,
                "timestamp_ns": transform.header.stamp.nanosec,
                "position": position,
                "orientation": orientation,
            }
            return result

    # If the specific transform isn't found, return None (or handle it differently as needed)
    return {"Transform not found"}

def extract_traffic_light_data(msg):
    extracted_lights = []
    num_lights = len(msg.signals)

    for signal in msg.signals:
        light = signal.lights[0]  # Assuming only one light per signal for now

        extracted_lights.append({
            "map_primitive_id": signal.map_primitive_id,
            "color": light.color,
            "status": light.status,
            "confidence": light.confidence,
        })

    return {
        "timestamp_sec": msg.header.stamp.sec,
        "timestamp_ns": msg.header.stamp.nanosec,
        "num_lights": num_lights,
        "lights": extracted_lights,
    }


def extract_route(msg):
    route_segments = []
    for segment in msg.segments:
        preferred_primitive_id = segment.preferred_primitive.id if segment.preferred_primitive else None
        primitives = [{"id": primitive.id, "primitive_type": primitive.primitive_type}
                      for primitive in segment.primitives]

        route_segments.append({
            "preferred_primitive_id": preferred_primitive_id,
            "primitives": primitives,
        })

    return {
        "timestamp_sec": msg.header.stamp.sec,
        "timestamp_ns": msg.header.stamp.nanosec,
        "route_segments": route_segments,
    }



bag_filename = "my_ros2_data"
output_format = "jsonl"

rclpy.init()
node = Node("rosbag2_recorder")


# Handle route message (published only once)
route_topic = "/planning/mission_planning/route"

#topic_info = TopicMetadata(
#    name=route_topic,
#    type="autoware_auto_planning_msgs/msg/LaneletRoute",
#    serialization_format=output_format
#)  
#writer.create_topic(topic_info)

def route_callback(msg):
    with open(f"{bag_filename}.{output_format}", "a") as f:
        if output_format == "json":
            json.dump({"topic": route_topic, "data": extract_route(msg)}, f)
            f.write("\n")
    node.destroy_subscription(route_sub)

route_sub = node.create_subscription(
    LaneletRoute,
    route_topic,
    route_callback,
    1 
)

#writer = SequentialWriter()
#storage_options = StorageOptions(uri=bag_filename, storage_id='sqlite3')
#converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='json')  # Change serialization to JSON
#writer.open(storage_options, converter_options)


def make_callback(topic_name, data_extractor=None):
    def callback(msg):
        data_to_write = msg
        if data_extractor:
            data_to_write = data_extractor(msg)

        with open(f"{bag_filename}.jsonl", "a") as f:
            json.dump({"topic": topic_name, "data": data_to_write}, f)  
            f.write("\n")  # Add newline for JSON Lines
    return callback



#def make_callback(t, mt, data_extractor=None):
#    def callback(msg):
#        data_to_write = msg
#        if data_extractor:
#            data_to_write = data_extractor(msg)
#        
#        serialized_msg = json.dumps(data_to_write)  # Serialize as JSON
#
#        writer.write(t, serialized_msg, node.get_clock().now().nanoseconds)
#    return callback


topics_to_record = {
    "/vehicle/status/steering_status": SteeringReport,
    "/vehicle/status/velocity_status": VelocityReport,
    "/tf": TFMessage,
    "/perception/object_recognition/tracking/objects": TrackedObjects,
    "/perception/traffic_light_recognition/traffic_signals": TrafficSignalArray,
}

# Create subscriptions for other topics
for topic, msg_type in topics_to_record.items():
    if topic != route_topic:  # Skip the route topic here
        topic_info = TopicMetadata(
            name=topic,
            type=msg_type.__module__ + "/" + msg_type.__name__,
            serialization_format=output_format
        )
        #writer.create_topic(topic_info)

        data_extractor = None
        if topic == "/vehicle/status/steering_status":
            data_extractor = extract_steering_data
        elif topic == "/vehicle/status/velocity_status":  # Update for velocity
            data_extractor = extract_velocity_data
        elif topic == "/perception/object_recognition/tracking/objects":  # Update for tracked objects
            data_extractor = extract_tracked_objects_data
        elif topic == "/tf":  # Update for vehicle pos
            data_extractor = extract_vehicle_pos
        elif topic == "/perception/traffic_light_recognition/traffic_signals":  # Update for traffic signal
            data_extractor = extract_traffic_light_data

        node.create_subscription(
            msg_type,
            topic,
            make_callback(topic, data_extractor),  
            10
        )

rclpy.spin(node)
