import numpy as np
from typing import Dict, List, Tuple, Any
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os

from Prediction_Model.TrajectoryDataset import TrajectoryDataset
from Prediction_Model.model_utils import load_model, make_predictions
from Model_Testing.FeatureExtractor import FeatureExtractor
from Model_Testing.ResidualGenerator import ResidualGenerator



class FaultDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_extractor = FeatureExtractor()
        self.residual_calculator = ResidualGenerator()
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def prepare_dataset(self):
        """Prepare dataset for fault detection"""
        device = self.config['device']
        model = load_model(self.config)

        #condition_predictions = {}
        all_features = []
        all_labels = []
        for condition in self.config['conditions']:
            data_folder = os.path.join(self.config['test_data_folder'], condition)
            dataset = TrajectoryDataset(data_folder, 
                                        position_scaling_factor=self.config['position_scaling_factor'], 
                                        velocity_scaling_factor=self.config['velocity_scaling_factor'], 
                                        steering_scaling_factor=self.config['steering_scaling_factor'], 
                                        acceleration_scaling_factor=self.config['acceleration_scaling_factor'])

            predictions = make_predictions(model, dataset, self.config)
        
            for i in range(len(dataset)):
                past, future, graph, graph_bounds = dataset[i]
                # Extract ground truth
                ground_truth = {
                    'position': np.array([step for step in future['position']]),
                    'velocity': np.array([step for step in future['velocity']]),
                    'steering': np.array([step for step in future['steering']]),
                    'acceleration': np.array([step for step in future['acceleration']])
                }
                
                # Calculate residuals
                sequence_residuals = self.residual_calculator.calculate_residuals(
                    ground_truth, predictions[i], self.config
                )
                
                # Extract features
                features = self.feature_extractor.extract_features(sequence_residuals)
                all_features.append(features)
                all_labels.append(condition)
        
        # Convert to DataFrame and scale
        feature_df = pd.DataFrame(all_features)
        self.feature_names = feature_df.columns.tolist()
        
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(feature_df)
        y = np.array(all_labels)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, np.ndarray]:
        """Train the fault detector"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        self.model.fit(X, y)
        return cv_scores
    
    def analyze_results(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze fault detection results"""
        y_pred = self.model.predict(X)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred),
            'feature_importance': feature_importance
        }

def residuals_analysis(config):
    """Main function to perform residuals analysis"""
    
    fault_detector = FaultDetector(config)
    
    # Prepare dataset and train model
    X, y = fault_detector.prepare_dataset()
    cv_scores = fault_detector.train(X, y)
    
    # Analyze results
    analysis_results = fault_detector.analyze_results(X, y)
    
    results = {
        'fault_detector': fault_detector.model,
        'feature_scaler': fault_detector.scaler,
        'feature_names': fault_detector.feature_names,
        'cv_scores': cv_scores,
        'analysis': analysis_results
    }
    
    print(f'Fault detection model trained with cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')
    return results