from Risk_Assessment.ResidualGenerator import ResidualFeatures
from Risk_Assessment.FaultDetectionConfig import FEATURE_COMPONENTS
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, List
import numpy as np

class DimensionReductionFeatureExtractor:
    def __init__(self, n_components: Optional[int] = None, scaler_type: str = 'robust'):
        """
        Initialize feature extractor with dimension reduction.
        
        Args:
            n_components: Number of components to keep after dimension reduction.
                        If None, keeps all components.
                        If int, keeps that many components.
            scaler_type: Type of scaling to use before PCA ('standard' or 'robust')
        """
        self.n_components = n_components
        self.scaler_type = scaler_type
        self.pca_models = {}  # Store PCA models for each feature
        self.scalers = {}     # Store scalers for each feature
        self.is_fitted = False
        self.residual_types = None
        
    def _get_scaler(self):
        """Get the appropriate scaler based on scaler_type."""
        if self.scaler_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        return StandardScaler()
    
    def _initialize_models(self, feature_names: List[str]):
        """Initialize PCA models and scalers for each feature."""
        for feature in feature_names:
            if feature not in self.pca_models:
                self.pca_models[feature] = PCA(n_components=self.n_components)
                self.scalers[feature] = self._get_scaler()
    
    def fit(self, all_residuals: List[ResidualFeatures]):
        """
        Fit PCA models on training data, combining residual types into feature dimensions.
        
        Args:
            all_residuals: List of ResidualFeatures objects containing training data
        """
        if not all_residuals:
            raise ValueError("No residuals provided for fitting")

        # Get list of features and residual types from first sample
        features = list(all_residuals[0].residuals.keys())
        self.residual_types = list(all_residuals[0].residuals[features[0]].keys())
        
        # Initialize models
        self._initialize_models(features)
        
        # Collect and combine residuals for each feature
        collected_data = defaultdict(list)
        
        for residual_features in all_residuals:
            for feature in features:
                # Get all residual types for this feature
                feature_data = []
                for residual_type in self.residual_types:
                    values = residual_features.residuals[feature][residual_type]
                    
                    # Handle multidimensional features
                    if len(FEATURE_COMPONENTS[feature]) > 1:
                        # Flatten all dimensions except the first
                        values = values.reshape(values.shape[0], -1)
                    else:
                        # For 1D features, ensure we have a 2D array
                        values = values.reshape(values.shape[0], -1)
                    
                    feature_data.append(values)
                
                # Concatenate all residual types along the feature dimension
                combined_feature_data = np.concatenate(feature_data, axis=1)
                collected_data[feature].append(combined_feature_data)
        
        # Fit PCA and scaler for each feature
        for feature, data_list in collected_data.items():
            if data_list:
                # Stack all samples along first dimension
                stacked_data = np.vstack(data_list)
                
                # Determine number of components
                max_components = min(stacked_data.shape[0], stacked_data.shape[1])
                n_components = min(self.n_components if self.n_components is not None else max_components,
                                max_components)
                
                # Update PCA model with correct number of components
                self.pca_models[feature] = PCA(n_components=n_components)
                
                # Fit and transform with scaler
                scaled_data = self.scalers[feature].fit_transform(stacked_data)
                
                # Fit PCA
                self.pca_models[feature].fit(scaled_data)
        
        self.is_fitted = True
    
    def transform(self, residuals: ResidualFeatures) -> Dict[str, Any]:
        """
        Transform residuals using fitted PCA models.
        
        Args:
            residuals: ResidualFeatures object containing the residuals to transform
            
        Returns:
            Dictionary containing reduced features
        """
        if not self.is_fitted:
            raise ValueError("PCA models not fitted. Call fit() first.")
        
        features = {}
        
        for feature in residuals.residuals.keys():
            # Combine all residual types for this feature
            feature_data = []
            for residual_type in self.residual_types:
                values = residuals.residuals[feature][residual_type]
                
                # Handle multidimensional features
                if len(FEATURE_COMPONENTS[feature]) > 1:
                    values = values.reshape(values.shape[0], -1)
                else:
                    values = values.reshape(values.shape[0], -1)
                
                feature_data.append(values)
            
            # Concatenate all residual types
            combined_feature_data = np.concatenate(feature_data, axis=1)
            
            # Scale the data
            scaled_data = self.scalers[feature].transform(combined_feature_data)
            
            # Apply PCA transformation
            transformed_data = self.pca_models[feature].transform(scaled_data)
            
            # Store each component as a separate feature
            for i in range(transformed_data.shape[1]):
                feature_name = f"{feature}_pc{i+1}"
                features[feature_name] = transformed_data[0, i].item()
            
            # Add explained variance ratio for each component
            explained_var_ratio = self.pca_models[feature].explained_variance_ratio_
            for i, ratio in enumerate(explained_var_ratio):
                feature_name = f"{feature}_explained_var_pc{i+1}"
                features[feature_name] = float(ratio)
        
        return features

    def fit_transform(self, all_residuals: List[ResidualFeatures]) -> List[Dict[str, Any]]:
        """
        Fit the PCA models and transform all residuals.
        
        Args:
            all_residuals: List of ResidualFeatures objects
            
        Returns:
            List of dictionaries containing reduced features
        """
        self.fit(all_residuals)
        return [self.transform(residuals) for residuals in all_residuals]

    def get_feature_importance_by_component(self, classifier) -> Dict[str, Dict[int, float]]:
        """
        Get feature importance grouped by original features and their PCA components.
        
        Args:
            classifier: Trained classifier with feature_importances_ attribute
            
        Returns:
            Dictionary mapping feature names to component importance scores
        """
        if not self.is_fitted:
            raise ValueError("PCA models not fitted. Call fit() first.")
        
        feature_importances = classifier.feature_importances_
        importance_by_feature = defaultdict(dict)
        
        current_idx = 0
        for feature in self.pca_models.keys():
            n_components = self.pca_models[feature].n_components_
            for i in range(n_components):
                if current_idx < len(feature_importances):
                    importance_by_feature[feature][i+1] = feature_importances[current_idx]
                    current_idx += 1
        
        return dict(importance_by_feature)

    def get_cumulative_explained_variance(self) -> Dict[str, np.ndarray]:
        """
        Get cumulative explained variance ratios for each feature.
        
        Returns:
            Dictionary mapping feature to cumulative explained variance ratios
        """
        if not self.is_fitted:
            raise ValueError("PCA models not fitted. Call fit() first.")
        
        cumulative_variance = {}
        for feature, pca_model in self.pca_models.items():
            cumulative_variance[feature] = np.cumsum(pca_model.explained_variance_ratio_)
        
        return cumulative_variance