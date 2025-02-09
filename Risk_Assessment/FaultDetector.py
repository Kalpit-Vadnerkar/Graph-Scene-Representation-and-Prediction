from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from typing import List, Dict
import pandas as pd


class FaultDetector:
    def __init__(self, test_size: float = 0.2, random_state: int = 47):
        self.test_size = test_size
        self.random_state = random_state
        
        self.pipeline = Pipeline([
            #('variance_selector', VarianceThreshold(threshold=0.1)),
            ('scaler', RobustScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=1000,
                random_state=random_state,
                max_features='sqrt',
                min_samples_leaf=250
            ))
        ])
        
        self.label_encoder = LabelEncoder()
    
    def train_and_evaluate(self, features: List[Dict[str, float]], labels: List[str]):
        X = pd.DataFrame(features).values
        y_multi = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_multi,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_multi
        )
        
        # Train and evaluate
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        
        return {
            'classification_report': classification_report(
                self.label_encoder.inverse_transform(y_test),
                self.label_encoder.inverse_transform(y_pred)
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'accuracy': (y_test == y_pred).mean()
        }