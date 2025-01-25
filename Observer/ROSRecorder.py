from rclpy.node import Node
from autoware_planning_msgs.msg import LaneletRoute
from autoware_auto_vehicle_msgs.msg import SteeringReport, VelocityReport
from autoware_auto_control_msgs.msg import AckermannControlCommand
from tf2_msgs.msg import TFMessage
from autoware_auto_perception_msgs.msg import TrackedObjects, TrafficSignalArray
import json
import os
from Observer.MessageExtractor import MessageExtractor
from Observer.MessageCleaner import MessageCleaner

class ROSObserver(Node):
    def __init__(self, output_folder = None):
        super().__init__("ros_observer")
        self.output_folder = output_folder
        self.data_extractor = MessageExtractor()
        
        # Create output folder
        if self.output_folder:
            print("Observer running in STORAGE mode")
            os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize subscriptions
        self._init_subscriptions()

    def _init_subscriptions(self):
        # Initialize route subscription (one-time)
        self.route_sub = self.create_subscription(
            LaneletRoute,
            "/planning/mission_planning/route",
            self._route_callback,
            1
        )

        # Initialize other subscriptions
        self.topics_to_record = {
            "/vehicle/status/steering_status": (SteeringReport, self.data_extractor.extract_steering_data),
            "/system/emergency/control_cmd": (AckermannControlCommand, self.data_extractor.extract_control_data),
            "/vehicle/status/velocity_status": (VelocityReport, self.data_extractor.extract_velocity_data),
            "/tf": (TFMessage, self.data_extractor.extract_vehicle_pos),
            "/perception/object_recognition/tracking/objects": (TrackedObjects, self.data_extractor.extract_tracked_objects_data),
            "/perception/traffic_light_recognition/traffic_signals": (TrafficSignalArray, self.data_extractor.extract_traffic_light_data),
        }

        # Create subscriptions for each topic
        for topic, (msg_type, extractor) in self.topics_to_record.items():
            self.create_subscription(
                msg_type,
                topic,
                self._create_callback(topic, extractor),
                10
            )

    def _route_callback(self, msg):
        """Handle route message (published only once)"""
        filename = os.path.join(self.output_folder, "route.json")
        with open(filename, "w") as f:
            json.dump({
                "topic": "/planning/mission_planning/route",
                "data": self.data_extractor.ensure_json_serializable(
                    self.data_extractor.extract_route(msg)
                )
            }, f)
        self.get_logger().info("Route recorded")
        self.destroy_subscription(self.route_sub)

    def _create_callback(self, topic_name, extractor):
        """Create a callback function for a given topic"""
        filename = os.path.join(self.output_folder, topic_name.replace('/', '_') + '.json')
        
        def callback(msg):
            data_to_write = msg
            if extractor:
                data_to_write = self.data_extractor.ensure_json_serializable(extractor(msg))
            with open(filename, "a") as f:
                json.dump({"data": data_to_write}, f)
                f.write("\n")
            
        self.get_logger().info(f'Created callback for: {filename}')
        return callback

class StreamObserver(ROSObserver):
    def __init__(self, output_folder=None, stream_mode=True):
        self.stream_mode = stream_mode
        super().__init__(output_folder)
        self.current_data = {}
        self.data_streamer = None
        self.message_cleaner = MessageCleaner(stream_mode=True)
        print("Observer running in STREAMING mode")
    
    def set_data_streamer(self, streamer):
        self.data_streamer = streamer
    
    def _create_callback(self, topic_name, extractor):
        if self.stream_mode:
            def callback(msg):
                data_to_write = msg
                if extractor:
                    data_to_write = self.data_extractor.ensure_json_serializable(extractor(msg))
                self.current_data[topic_name] = data_to_write
                if self._check_data_complete() and self.data_streamer:
                    # Clean the data before processing
                    cleaned_data = self.message_cleaner.clean_data(self.current_data)
                    if cleaned_data:  # Only process if data passes cleaning
                        self.data_streamer.process_current_data(cleaned_data)
                    self.current_data = {}
        else:
            return super()._create_callback(topic_name, extractor)
        
        return callback
    
    def _check_data_complete(self):
        required_topics = set(self.topics_to_record.keys())
        current_topics = set(self.current_data.keys())
        return required_topics.issubset(current_topics)