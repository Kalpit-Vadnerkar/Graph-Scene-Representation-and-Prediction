from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np

class ResidualClassifier:
    def __init__(self, n_estimators: int = 100, n_splits: int = 5):
        self.n_estimators = n_estimators
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight='balanced'
        )
        
    def prepare_data(self, features: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
        """Convert feature dictionaries to numpy array"""
        df = pd.DataFrame(features)
        
        # Separate sequence tracking columns
        sequence_ids = df['sequence_id'].values
        timestamps = df['timestamp'].values
        
        # Remove tracking columns and convert to feature matrix
        feature_cols = [col for col in df.columns 
                       if col not in ['sequence_id', 'timestamp']]
        X = df[feature_cols].values
        
        return X, sequence_ids, timestamps
        
    def train_and_evaluate(self, 
                          features: List[Dict[str, float]],
                          labels: List[str]) -> Dict[str, Any]:
        """Train and evaluate with proper cross-validation"""
        
        # Prepare data
        X, sequence_ids, timestamps = self.prepare_data(features)
        y = self.label_encoder.fit_transform(labels)
        
        # Initialize group k-fold
        group_kfold = GroupKFold(n_splits=self.n_splits)
        
        # Track results
        fold_results = []
        feature_importance = np.zeros(X.shape[1])
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, sequence_ids)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train classifier
            self.classifier.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.classifier.predict(X_test_scaled)
            
            # Record results
            fold_results.append({
                'fold': fold,
                'classification_report': classification_report(
                    self.label_encoder.inverse_transform(y_test),
                    self.label_encoder.inverse_transform(y_pred)
                ),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            })
            
            # Accumulate feature importance
            feature_importance += self.classifier.feature_importances_
            
        # Average feature importance across folds
        feature_importance /= self.n_splits
        
        # Create feature importance DataFrame
        df = pd.DataFrame(features[0])
        feature_cols = [col for col in df.columns 
                       if col not in ['sequence_id', 'timestamp']]
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'fold_results': fold_results,
            'feature_importance': importance_df
        }