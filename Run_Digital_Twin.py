import rclpy
from Data_Curator.config import Config
from model_config import CONFIG

from Observer.ROSRecorder import ROSObserver, StreamObserver
from Observer.MessageCleaner import MessageCleaner
from Observer.DataStreamer import DataStreamer

from State_Estimator.MapProcessor import MapProcessor
from State_Estimator.SequenceProcessor import SequenceProcessor
from State_Estimator.StateEstimator import StateEstimator, SequenceProcessor
from Digital_Twin.DigitalTwin import DigitalTwin

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
    rclpy.init()
    
    # Initialize components
    config = Config()
    observer = StreamObserver(max_buffer_size=60)  # Adjust buffer size as needed
    cleaner = MessageCleaner(stream_mode=True)
    streamer = DataStreamer(config)
    observer.set_components(cleaner, streamer)
    
    # Initialize prediction components
    estimator = StateEstimator(config)
    digital_twin = DigitalTwin(CONFIG)  # model_config for NN
    observer.attach(estimator, digital_twin)
    
    try:
        rclpy.spin(observer)
    except KeyboardInterrupt:
        if not observer.video_created:
            print("\nCreating video from collected states...")
            observer.create_prediction_video("trajectory_predictions.mp4")
    finally:
        observer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    initialize_streaming()