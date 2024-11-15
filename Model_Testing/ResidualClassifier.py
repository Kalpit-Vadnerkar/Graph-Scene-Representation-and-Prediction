from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from collections import Counter

class ResidualClassifier:
    def __init__(self, 
                 n_estimators: int = 100,
                 n_splits: int = 10,
                 test_size: float = 0.2,
                 random_state: int = 47):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        
        # Create pipeline with preprocessing and classifier
        self.pipeline = Pipeline([
            ('variance_selector', VarianceThreshold(threshold=0.1)),
            ('scaler', RobustScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                max_features='sqrt',
                min_samples_leaf=50
            ))
        ])
        
        self.label_encoder = LabelEncoder()
    
    def prepare_data(self, features: List[Dict[str, float]]) -> tuple[np.ndarray, list[str]]:
        df = pd.DataFrame(features)
        
        # Extract feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['sequence_id', 'timestamp']]
        
        return df[feature_cols].values, feature_cols
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets by splitting each condition separately
        """
        # Create a DataFrame with all relevant information
        df = pd.DataFrame({
            'X_index': range(len(X)),
            'label': y
        })
        
        # Print initial data distribution
        print("\nInitial data distribution:")
        for label in np.unique(y):
            condition_mask = df['label'] == label
            n_samples = condition_mask.sum()
            print(f"Condition {self.label_encoder.inverse_transform([label])[0]}: {n_samples} samples")
            
        train_indices, test_indices = train_test_split(
            df.index,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Stratify by condition
        )
        
        # Print distribution information
        print("\nSample distribution in splits:")
        print("Training set:")
        train_counts = Counter(y[train_indices])
        for label, count in train_counts.items():
            print(f"Condition {self.label_encoder.inverse_transform([label])[0]}: {count} samples")
        
        print("\nTest set:")
        test_counts = Counter(y[test_indices])
        for label, count in test_counts.items():
            print(f"Condition {self.label_encoder.inverse_transform([label])[0]}: {count} samples")
        
        return (X[train_indices], X[test_indices],
                y[train_indices], y[test_indices])
    
    def train_and_evaluate(self, 
                          features: List[Dict[str, float]],
                          labels: List[str]) -> Dict[str, Any]:
        """Train and evaluate using both train-test split and cross-validation"""
        
        # Prepare data
        X, feature_names = self.prepare_data(features)
        y = self.label_encoder.fit_transform(labels)
        
        print(f'Number of features: {len(feature_names)}')

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Define cross-validation strategy for training set
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted'
        }
        
        # Perform cross-validation on training set
        cv_results = cross_validate(
            self.pipeline,
            X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True
        )
        
        # Train final model on all training data and evaluate on test set
        self.pipeline.fit(X_train, y_train)
        y_pred_test = self.pipeline.predict(X_test)
        
        # Get feature importances
        feature_importance = np.zeros(len(feature_names))
        selector_mask = self.pipeline.named_steps['variance_selector'].get_support()
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        feature_importance[selector_mask] = importances
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'cv_results': {
                'mean_accuracy': cv_results['test_accuracy'].mean(),
                'std_accuracy': cv_results['test_accuracy'].std(),
                'mean_f1': cv_results['test_f1_weighted'].mean(),
                'std_f1': cv_results['test_f1_weighted'].std(),
                'mean_precision': cv_results['test_precision_weighted'].mean(),
                'std_precision': cv_results['test_precision_weighted'].std(),
                'mean_recall': cv_results['test_recall_weighted'].mean(),
                'std_recall': cv_results['test_recall_weighted'].std()
            },
            'test_results': {
                'classification_report': classification_report(
                    self.label_encoder.inverse_transform(y_test),
                    self.label_encoder.inverse_transform(y_pred_test)
                ),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test),
                'accuracy': (y_test == y_pred_test).mean()
            },
            'feature_importance': importance_df,
            'data_split': {
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        }