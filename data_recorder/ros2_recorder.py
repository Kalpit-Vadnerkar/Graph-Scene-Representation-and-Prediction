import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from autoware_planning_msgs.msg import LaneletRoute
from autoware_auto_vehicle_msgs.msg import SteeringReport, VelocityReport
from tf2_msgs.msg import TFMessage
from autoware_auto_perception_msgs.msg import TrackedObjects, TrafficSignalArray
from autoware_auto_mapping_msgs.msg import HADMapBin
import lanelet2
import json  # For encoding data to JSON format
from extractor_functions import *
import os

def extract_lanelet2_map(msg):
    map_data = lanelet2.io_handlers.binary_map.BinaryMap()
    map_data.from_bytes(msg.data)
    return map_data.data



# Create a folder to store the JSON files
output_folder = "ros2_data"  
os.makedirs(output_folder, exist_ok=True)

#bag_filename = "my_ros2_data"
#output_format = "jsonl"

rclpy.init()
node = Node("rosbag2_recorder")


# Handle route message (published only once)
route_topic = "/planning/mission_planning/route"

def route_callback(msg):
    filename = os.path.join(output_folder, "route.json") 
    with open(filename, "w") as f:
        json.dump({"topic": "/planning/mission_planning/route", "data": ensure_json_serializable(extract_route(msg))}, f)
    print("Route recorded")
    node.destroy_subscription(route_sub)

route_sub = node.create_subscription(
    LaneletRoute,
    route_topic,
    route_callback,
    1 
)


def make_callback(topic_name, data_extractor=None):
    filename = os.path.join(output_folder, topic_name.replace('/', '_')+'.json')  # Create filename from topic
    def callback(msg):
        data_to_write = msg
        if data_extractor:
            data_to_write = ensure_json_serializable(data_extractor(msg))
        with open(filename, "a") as f:
            json.dump({"data": data_to_write}, f)
            f.write("\n")
    print(filename)
    return callback

topics_to_record = {
    "/vehicle/status/steering_status": SteeringReport,
    "/vehicle/status/velocity_status": VelocityReport,
    "/tf": TFMessage,
    "/perception/object_recognition/tracking/objects": TrackedObjects,
    "/perception/traffic_light_recognition/traffic_signals": TrafficSignalArray,    
}

# Create subscriptions for other topics
for topic, msg_type in topics_to_record.items():
    if topic != "/planning/mission_planning/route":  
        data_extractor = None
        if topic == "/vehicle/status/steering_status":
            data_extractor = extract_steering_data
        elif topic == "/vehicle/status/velocity_status": 
            data_extractor = extract_velocity_data
        elif topic == "/perception/object_recognition/tracking/objects": 
            data_extractor = extract_tracked_objects_data
        elif topic == "/tf": 
            data_extractor = extract_vehicle_pos
        elif topic == "/perception/traffic_light_recognition/traffic_signals": 
            data_extractor = extract_traffic_light_data
        elif topic == "/map/vector_map": 
            data_extractor = extract_lanelet2_map

        node.create_subscription(
            msg_type,
            topic,
            make_callback(topic, data_extractor), 
            10
        )

rclpy.spin(node)
