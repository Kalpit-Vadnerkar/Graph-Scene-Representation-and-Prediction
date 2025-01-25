import rclpy
from Data_Curator.config import Config
from Data_Curator.MapProcessor import MapProcessor
from Observer.ROSRecorder import ROSObserver, StreamObserver
from Observer.MessageCleaner import MessageCleaner
from Observer.DataStreamer import DataStreamer

####################################################################################
########################## OFFLINE MODE ############################################

#rclpy.init()
#observer = ROSObserver(output_folder="Recorded_ROS_Data")
#rclpy.spin(observer)
#observer.destroy_node()
#rclpy.shutdown()


#cleaner = MessageCleaner(data_points_per_second=10, max_stopped_duration=3)
#metrics = cleaner.clean_data(input_folder=input("Enter Data Folder Name: "), output_folder="Cleaned_ROS_Data")
#print(f"Data cleaning and velocity filtering completed. Cleaned data stored in 'Cleaned_Dataset' folder.")
#for metric, values in metrics.items():
#    print(f"Minimum {metric}: {values['min']}")
#    print(f"Maximum {metric}: {values['max']}")

####################################################################################
########################## ONLINE MODE ############################################



def initialize_streaming():
    # Initialize ROS node
    rclpy.init()
    
    config = Config()  # Your config class
    map_processor = MapProcessor()
    observer = StreamObserver(stream_mode=True)
    streamer = DataStreamer(config, map_processor)
    observer.set_data_streamer(streamer)
    
    # Initialize ROS node spin 
    rclpy.spin(observer)
    observer.destroy_node()
    rclpy.shutdown()

initialize_streaming()