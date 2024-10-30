import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.model_utils import load_model, make_predictions

def extract_fault_detection_features(sequence_residuals, window_size=5):
    features = {}
    
    for residual_type, values in sequence_residuals.items():
        values = np.array(values)
        
        # Static features
        features[f'{residual_type}_mean'] = np.mean(values)
        features[f'{residual_type}_std'] = np.std(values)
        features[f'{residual_type}_max'] = np.max(values)
        features[f'{residual_type}_min'] = np.min(values)
        
        # Trend features
        features[f'{residual_type}_trend'] = np.polyfit(np.arange(len(values)), values, 1)[0]
        
        # Moving statistics
        if len(values) >= window_size:
            rolling_mean = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            rolling_std = np.array([np.std(values[i:i+window_size]) 
                                  for i in range(len(values)-window_size+1)])
            
            features[f'{residual_type}_rolling_mean_std'] = np.std(rolling_mean)
            features[f'{residual_type}_rolling_std_mean'] = np.mean(rolling_std)
        
        # Frequency domain features
        if len(values) > 1:
            fft_vals = np.abs(np.fft.fft(values))
            features[f'{residual_type}_fft_max'] = np.max(fft_vals[1:])  # Exclude DC component
            features[f'{residual_type}_fft_mean'] = np.mean(fft_vals[1:])
    
    return features

def calculate_sequence_residuals(ground_truth, predictions, config):
    residuals = {
        'position_X': [],
        'position_Y': [],
        'velocity_X': [],
        'velocity_Y': [],
        'steering': [],
        'acceleration': [],
        'combined': []
    }
    
    for t in range(config['output_seq_len']):
        # Position residuals
        pos_X_error = predictions['position_mean'][t][0] - ground_truth['position'][t][0]
        pos_X_std = np.sqrt(predictions['position_var'][t][0])

        pos_Y_error = predictions['position_mean'][t][1] - ground_truth['position'][t][1]
        pos_Y_std = np.sqrt(predictions['position_var'][t][1])
        
        # Velocity residuals
        vel_X_error = predictions['velocity_mean'][t][0] - ground_truth['velocity'][t][0]
        vel_X_std = np.sqrt(predictions['velocity_var'][t][0])

        vel_Y_error = predictions['velocity_mean'][t][1] - ground_truth['velocity'][t][1]
        vel_Y_std = np.sqrt(predictions['velocity_var'][t][1])
        
        # Steering residuals
        steer_error = abs(
            predictions['steering_mean'][t] - ground_truth['steering'][t]
        )
        steer_std = np.sqrt(predictions['steering_var'][t])
        
        # Acceleration residuals
        accel_error = abs(
            predictions['acceleration_mean'][t] - ground_truth['acceleration'][t]
        )
        accel_std = np.sqrt(predictions['acceleratio_var'][t])

        # Store all residuals
        residuals['position_X'].append(pos_X_error)
        residuals['position_Y'].append(pos_Y_error)
        residuals['velocity_X'].append(vel_X_error)
        residuals['velocity_Y'].append(vel_Y_error)
        residuals['steering'].append(steer_error)
        residuals['acceleration'].append(accel_error)
        
        # Combined residual (weighted sum)
        combined = (
            pos_X_error +
            pos_Y_error +
            vel_X_error +
            vel_Y_error +
            steer_error +
            accel_error
        )
        residuals['combined'].append(combined)
    
    return residuals

def prepare_fault_detection_dataset(config):
    device = config['device']
    model = load_model(config)

    #condition_predictions = {}
    all_features = []
    all_labels = []
    
    for condition in config['conditions']:
        

        data_folder = os.path.join(config['test_data_folder'], condition)
        condition_data = TrajectoryDataset(data_folder, 
                                    position_scaling_factor=config['position_scaling_factor'], 
                                    velocity_scaling_factor=config['velocity_scaling_factor'], 
                                    steering_scaling_factor=config['steering_scaling_factor'], 
                                    acceleration_scaling_factor=config['acceleration_scaling_factor'])

        condition_predictions = make_predictions(model, condition_data, config)


        
        #condition_predictions = predictions[condition]
        
        for i in range(len(condition_data)):
            # Extract ground truth and predictions for this sequence
            ground_truth = {
                'position': np.array([step['position'] for step in condition_data[i]['future']]),
                'velocity': np.array([step['velocity'] for step in condition_data[i]['future']]),
                'steering': np.array([step['steering'] for step in condition_data[i]['future']]),
                'acceleration': np.array([step['acceleration'] for step in condition_data[i]['future']])
            }
            
            # Calculate residuals for this sequence
            sequence_residuals = calculate_sequence_residuals(
                ground_truth, 
                condition_predictions[i],
                config
            )
            
            # Extract features from residuals
            features = extract_fault_detection_features(sequence_residuals)
            all_features.append(features)
            all_labels.append(condition)
    
    # Convert features to DataFrame
    feature_df = pd.DataFrame(all_features)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)
    y = np.array(all_labels)
    
    return X, y, scaler, feature_df.columns.tolist()

def train_fault_detector(X, y):
    # Initialize classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    
    # Train final model
    clf.fit(X, y)
    
    return clf, cv_scores

def analyze_fault_detection_results(clf, X, y, feature_names):
    # Get predictions
    y_pred = clf.predict(X)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results = {
        'confusion_matrix': confusion_matrix(y, y_pred),
        'classification_report': classification_report(y, y_pred),
        'feature_importance': feature_importance
    }
    
    return results

def residuals_analysis(dataset, predictions, config, conditions):
    # Prepare dataset for fault detection
    X, y, scaler, feature_names = prepare_fault_detection_dataset(config)
    
    # Train fault detector
    fault_detector, cv_scores = train_fault_detector(X, y)
    
    # Analyze results
    analysis_results = analyze_fault_detection_results(
        fault_detector, X, y, feature_names
    )
    
    results = {
        'fault_detector': fault_detector,
        'feature_scaler': scaler,
        'feature_names': feature_names,
        'cv_scores': cv_scores,
        'analysis': analysis_results
    }
    
    print(f'Fault detection model trained with cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')
    return results