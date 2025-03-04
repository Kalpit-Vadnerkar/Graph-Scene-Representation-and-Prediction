from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.feature_selection import VarianceThreshold
from typing import List, Dict
import pandas as pd
import numpy as np
import time


class FaultDetector:
    def __init__(self, test_size: float = 0.2, random_state: int = 47, n_estimators: int = 100):
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        
        self.pipeline = Pipeline([
            #('variance_selector', VarianceThreshold(threshold=0.1)),
            ('scaler', RobustScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                max_features='sqrt',
                min_samples_leaf=75
            ))
        ])
        
        self.label_encoder = LabelEncoder()

    def train_and_evaluate(self, features: List[Dict[str, float]], labels: List[str], 
                        residual_gen_time_per_sample: float = 0.0):
        """
        Train and evaluate the fault detection model.
        
        Args:
            features: List of feature dictionaries
            labels: List of condition labels
            residual_gen_time_per_sample: Average time to generate residuals per sample (seconds)
        
        Returns:
            Dict containing evaluation metrics
        """
        # Convert features to DataFrame
        X = pd.DataFrame(features).values
        y_multi = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_multi,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_multi
        )
        
        # Isolate feature extraction steps from classifier
        feature_extraction_pipeline = Pipeline([
            step for step in self.pipeline.steps 
            if step[0] != 'classifier'
        ])
        
        # Measure feature extraction time (training)
        fe_train_start_time = time.time()
        X_train_transformed = feature_extraction_pipeline.fit_transform(X_train)
        fe_train_time = time.time() - fe_train_start_time
        
        # Measure feature extraction time (inference)
        fe_inference_start_time = time.time()
        X_test_transformed = feature_extraction_pipeline.transform(X_test)
        fe_inference_time = time.time() - fe_inference_start_time
        
        # Get classifier from pipeline
        classifier = self.pipeline.named_steps['classifier']
        
        # Measure training time (classifier only)
        train_start_time = time.time()
        classifier.fit(X_train_transformed, y_train)
        classifier_train_time = time.time() - train_start_time
        
        # Total training time
        train_time = fe_train_time + classifier_train_time
        
        # Measure prediction time on test set (classifier only)
        predict_start_time = time.time()
        y_pred = classifier.predict(X_test_transformed)
        classifier_prediction_time = time.time() - predict_start_time
        
        # Total prediction time including feature extraction and residual generation
        # This is the key change - we add residual generation time to the total
        prediction_time = fe_inference_time + classifier_prediction_time
        total_prediction_time = prediction_time + (residual_gen_time_per_sample * len(y_test))
        
        # Get training set predictions for training metrics
        y_train_pred = classifier.predict(X_train_transformed)

        # Calculate metrics
        # Test metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
        
        # Training metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        
        # Class-specific metrics
        class_names = self.label_encoder.inverse_transform(np.unique(y_multi))
        per_class_metrics = {}
        
        for i, class_name in enumerate(class_names):
            class_test_indices = (y_test == i)
            class_train_indices = (y_train == i)
            
            if np.any(class_test_indices):
                per_class_metrics[class_name] = {
                    'test_precision': precision_score(y_test == i, y_pred == i, zero_division=0),
                    'test_recall': recall_score(y_test == i, y_pred == i, zero_division=0),
                    'test_samples': np.sum(class_test_indices),
                    'train_precision': precision_score(y_train == i, y_train_pred == i, zero_division=0),
                    'train_recall': recall_score(y_train == i, y_train_pred == i, zero_division=0),
                    'train_samples': np.sum(class_train_indices)
                }
                
        # Cross-validation metrics
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_results = cross_validate(
            self.pipeline, X, y_multi, 
            cv=cv,
            scoring=['accuracy', 'precision_weighted', 'recall_weighted'],
            return_train_score=True
        )
        
        return {
            'classification_report': classification_report(
                self.label_encoder.inverse_transform(y_test),
                self.label_encoder.inverse_transform(y_pred)
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            
            # Test metrics
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            
            # Training metrics
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            
            # Timing metrics
            'feature_extraction_train_time': fe_train_time,
            'feature_extraction_inference_time': fe_inference_time,
            'classifier_train_time': classifier_train_time,
            'classifier_prediction_time': classifier_prediction_time,
            'avg_feature_extraction_time_per_sample': fe_inference_time / len(y_test) if len(y_test) > 0 else 0,
            'avg_classifier_prediction_time_per_sample': classifier_prediction_time / len(y_test) if len(y_test) > 0 else 0,
            'avg_residual_generation_time_per_sample': residual_gen_time_per_sample,  # Add this new metric
            
            # Total metrics
            'train_time_seconds': train_time,
            'prediction_time_seconds': total_prediction_time,  # Use the new total time that includes residual generation
            'avg_prediction_time_per_sample': total_prediction_time / len(y_test) if len(y_test) > 0 else 0,
            
            # Dataset info
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'feature_count': X.shape[1],
            
            # Cross-validation metrics
            'cv_test_accuracy': np.mean(cv_results['test_accuracy']),
            'cv_test_precision': np.mean(cv_results['test_precision_weighted']),
            'cv_test_recall': np.mean(cv_results['test_recall_weighted']),
            'cv_train_accuracy': np.mean(cv_results['train_accuracy']),
            'cv_train_precision': np.mean(cv_results['train_precision_weighted']),
            'cv_train_recall': np.mean(cv_results['train_recall_weighted']),
            
            # Per-class metrics
            'per_class_metrics': per_class_metrics
        }