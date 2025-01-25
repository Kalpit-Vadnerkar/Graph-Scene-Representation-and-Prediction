class MessageExtractor:
    @staticmethod
    def ensure_json_serializable(data):
        """Recursively converts non-serializable types to their string representations."""
        if isinstance(data, (list, tuple)):
            return [MessageExtractor.ensure_json_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {key: MessageExtractor.ensure_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, set):
            return list(data)  # Convert sets to lists
        else:
            return data

    def extract_steering_data(self, msg):
        return {
            "timestamp_sec": msg.stamp.sec,
            "timestamp_ns": msg.stamp.nanosec,
            "steering_angle": msg.steering_tire_angle
        }
            
    def extract_control_data(self, msg):
        return {
            "timestamp_sec": msg.stamp.sec,
            "timestamp_ns": msg.stamp.nanosec,
            "steering_angle": msg.lateral.steering_tire_angle,
            "speed": msg.longitudinal.speed,
            "acceleration": msg.longitudinal.acceleration,
            "jerk": msg.longitudinal.jerk
        }

    def extract_velocity_data(self, msg):
        return {
            "timestamp_sec": msg.header.stamp.sec,
            "timestamp_ns": msg.header.stamp.nanosec,
            "longitudinal_velocity": msg.longitudinal_velocity,
            "lateral_velocity": msg.lateral_velocity,
            "yaw_rate": msg.heading_rate
        }

    def extract_tracked_objects_data(self, msg):
        extracted_objects = []
        num_objects = len(msg.objects)
        is_map_frame = (msg.header.frame_id == "map")

        for obj in msg.objects:
            position = obj.kinematics.pose_with_covariance.pose.position
            orientation = obj.kinematics.pose_with_covariance.pose.orientation
            linear_velocity = obj.kinematics.twist_with_covariance.twist.linear
            classification = obj.classification[0].label

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

        return {
            "timestamp_sec": msg.header.stamp.sec,
            "timestamp_ns": msg.header.stamp.nanosec,
            "num_objects": num_objects,
            "is_map_frame": is_map_frame,
            "objects": extracted_objects,
        }

    def extract_vehicle_pos(self, msg):
        for transform in msg.transforms:
            if (transform.header.frame_id == "map" and
                    transform.child_frame_id == "base_link"):

                position = {
                    "x": transform.transform.translation.x,
                    "y": transform.transform.translation.y,
                    "z": transform.transform.translation.z,
                }

                orientation = {
                    "x": transform.transform.rotation.x,
                    "y": transform.transform.rotation.y,
                    "z": transform.transform.rotation.z,
                    "w": transform.transform.rotation.w,
                }
                
                return {
                    "timestamp_sec": transform.header.stamp.sec,
                    "timestamp_ns": transform.header.stamp.nanosec,
                    "position": position,
                    "orientation": orientation,
                }

        return {"Transform not found"}

    def extract_traffic_light_data(self, msg):
        extracted_lights = []
        num_lights = len(msg.signals)

        for signal in msg.signals:
            light = signal.lights[0]

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

    def extract_route(self, msg):
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