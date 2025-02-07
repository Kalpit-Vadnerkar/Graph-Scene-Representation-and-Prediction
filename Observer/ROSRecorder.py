from autoware_planning_msgs.msg import LaneletRoute
from autoware_auto_vehicle_msgs.msg import SteeringReport, VelocityReport
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_perception_msgs.msg import TrackedObjects, TrafficSignalArray

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

import json
import os
from Observer.MessageExtractor import MessageExtractor
from Observer.MessageCleaner import MessageCleaner
from Observer.DataStreamer import DataStreamer

from State_Estimator.StateEstimator import StateEstimator
from Digital_Twin.DigitalTwin import DigitalTwin

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
    def __init__(self, output_folder=None, max_buffer_size=None, fps=None):
        self._callbacks = {}  # Cache callbacks
        super().__init__(output_folder)
    
        self.current_data = {}
        self.data_streamer = None
        self.message_cleaner = None
        self.estimator = None
        self.digital_twin = None
        self.max_buffer_size = max_buffer_size
        self.video_created = False
        self.fps = fps
    
    def set_components(self, cleaner: MessageCleaner, streamer: DataStreamer):
        self.message_cleaner = cleaner
        self.data_streamer = streamer

    def attach(self, estimator: StateEstimator, digital_twin: DigitalTwin):
        self.estimator = estimator   
        self.digital_twin = digital_twin

    def create_prediction_video(self, output_path):
        """Create a video of all stored predictions"""
        if not self.digital_twin:
            print("Digital twin not initialized!")
            return
        
        self.digital_twin.create_video(output_path, self.fps)
        print(f"Video saved to {output_path}")
        
    def _route_callback(self, msg):
        """Handle route message"""
        route_data = self.data_extractor.ensure_json_serializable(
            self.data_extractor.extract_route(msg)
        )
        route = [primitive['id'] for segment in route_data['route_segments'] 
                for primitive in segment['primitives'] 
                if primitive['primitive_type'] == 'lane']
        
        self.estimator.update_route(route)
    
    def _create_callback(self, topic_name, extractor):
        if topic_name in self._callbacks:
            return self._callbacks[topic_name]
            
        def callback(msg):
            data = self.data_extractor.ensure_json_serializable(extractor(msg)) if extractor else msg
            self.current_data[topic_name] = data
            
            if len(self.current_data) == len(self.topics_to_record):
                self._process_complete_data()
                self.current_data.clear()
                
        self._callbacks[topic_name] = callback
        return callback
        
    def _process_complete_data(self):
        if not all([self.data_streamer, self.estimator, self.digital_twin, self.max_buffer_size]):
            return
            
        cleaned_data = self.message_cleaner.clean_data(self.current_data)
        if not cleaned_data:
            return
            
        data_buffer, timestamp = self.data_streamer.process_current_data(cleaned_data)
        if not (data_buffer and timestamp):
            return
            
        state = self.estimator.estimate_state(data_buffer, timestamp)
        if not state:
            return
        
        self.digital_twin.update_state(state)
        
        # Check if we should create the video
        if (len(self.digital_twin.state_history) >= self.max_buffer_size and 
            not self.video_created):
            print(f"\nBuffer full ({self.max_buffer_size} states). Creating video...")
            self.create_prediction_video("trajectory_predictions.mp4")
            self.video_created = True
            # Shutdown after video creation
            rclpy.shutdown()