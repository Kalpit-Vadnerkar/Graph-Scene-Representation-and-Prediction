from Risk_Assessment.ResidualGenerator import ResidualFeatures
from Risk_Assessment.FaultDetectionConfig import FEATURE_COMPONENTS

from typing import Dict, Any, Optional, List
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

class TemporalFeatureExtractor:
    def __init__(self, n_components: Optional[int] = None, scaler_type: str = 'robust'):
        self.n_components = n_components
        self.scaler_type = scaler_type
        self.pca_models = {}
        self.scalers = {}
        self.is_fitted = False
        self.window_size = None
        
    def _extract_temporal_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract temporal features from a sequence before PCA.
        For multi-dimensional features, computes features per dimension then combines them.
        """
        def compute_single_dim_features(seq_1d):
            features = []
            # Statistical features
            features.extend([
                np.mean(seq_1d),
                np.std(seq_1d),
                np.max(seq_1d),
                np.min(seq_1d),
                np.ptp(seq_1d)
            ])
            
            # -------------------- EXTRA FANCY FEATURE EXTRACTION ----------------------

            """ # Temporal features
            features.extend([
                np.mean(np.gradient(seq_1d)),
                np.std(np.diff(seq_1d)),
                *np.percentile(seq_1d, [25, 50, 75])
            ])
            
            # Autocorrelation
            for lag in [1, 2, 3]:
                features.append(np.correlate(seq_1d[:-lag], seq_1d[lag:])[0]) """
            
            # -------------------- EXTRA FANCY FEATURE EXTRACTION ----------------------
            
            return features

        # Handle each dimension separately
        all_features = []
        for dim in range(sequence.shape[1]):
            dim_features = compute_single_dim_features(sequence[:, dim])
            all_features.extend(dim_features)
            
        return np.array(all_features)

    def fit(self, all_residuals: List[ResidualFeatures]):
        if not all_residuals:
            raise ValueError("No residuals provided for fitting")

        features = list(all_residuals[0].residuals.keys())
        residual_types = list(all_residuals[0].residuals[features[0]].keys())
        self.window_size = all_residuals[0].residuals[features[0]][residual_types[0]].shape[0]
        
        # Collect temporal features for each combination
        collected_features = defaultdict(lambda: defaultdict(list))
        
        for residual_features in all_residuals:
            for feature in features:
                for residual_type in residual_types:
                    values = residual_features.residuals[feature][residual_type]
                    
                    # Reshape maintaining temporal dimension
                    if len(FEATURE_COMPONENTS[feature]) > 1:
                        values = values.reshape(values.shape[0], -1)
                    
                    # Extract temporal features
                    temporal_features = self._extract_temporal_features(values)
                    collected_features[feature][residual_type].append(temporal_features)
        
        # Fit PCA on temporal features
        for feature in features:
            self.pca_models[feature] = {}
            self.scalers[feature] = {}
            
            for residual_type in residual_types:
                feature_matrix = np.vstack(collected_features[feature][residual_type])
                
                # Initialize and fit scaler
                self.scalers[feature][residual_type] = (
                    RobustScaler() if self.scaler_type == 'robust' else StandardScaler()
                )
                scaled_features = self.scalers[feature][residual_type].fit_transform(feature_matrix)
                
                # Initialize and fit PCA
                # Each feature now has consistent temporal feature dimensionality
                max_components = min(
                    scaled_features.shape[0],  # Number of samples
                    scaled_features.shape[1]   # Number of temporal features
                )
                
                n_components = min(
                    self.n_components if self.n_components is not None else max_components,
                    max_components
                )
                self.pca_models[feature][residual_type] = PCA(n_components=n_components)
                self.pca_models[feature][residual_type].fit(scaled_features)
        
        self.is_fitted = True

    def transform(self, residuals: ResidualFeatures) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")
        
        features = {}
        
        for feature in residuals.residuals.keys():
            for residual_type in residuals.residuals[feature].keys():
                values = residuals.residuals[feature][residual_type]
                
                if len(FEATURE_COMPONENTS[feature]) > 1:
                    values = values.reshape(values.shape[0], -1)
                
                # Extract temporal features
                temporal_features = self._extract_temporal_features(values)
                temporal_features = temporal_features.reshape(1, -1)
                
                # Scale and transform
                scaled_features = self.scalers[feature][residual_type].transform(temporal_features)
                transformed_features = self.pca_models[feature][residual_type].transform(scaled_features)
                
                # Store components
                for i, value in enumerate(transformed_features.flatten()):
                    features[f"{feature}_{residual_type}_pc{i+1}"] = float(value)
        
        return features

    def fit_transform(self, all_residuals: List[ResidualFeatures]) -> List[Dict[str, float]]:
        """
        Fit the PCA models and transform all residuals.
        
        Args:
            all_residuals: List of ResidualFeatures objects
            
        Returns:
            List of dictionaries containing reduced features with temporal statistics
        """
        self.fit(all_residuals)
        return [self.transform(residuals) for residuals in all_residuals]

    def get_feature_importance_by_component(self, classifier) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Get feature importance grouped by feature, residual type, and their PCA components.
        """
        if not self.is_fitted:
            raise ValueError("PCA models not fitted. Call fit() first.")
        
        feature_importances = classifier.feature_importances_
        importance_by_component = defaultdict(lambda: defaultdict(dict))
        
        current_idx = 0
        for feature in self.pca_models.keys():
            for residual_type in self.pca_models[feature].keys():
                n_components = self.pca_models[feature][residual_type].n_components_
                for i in range(n_components):
                    if current_idx < len(feature_importances):
                        importance_by_component[feature][residual_type][i+1] = \
                            feature_importances[current_idx]
                        current_idx += 1
        
        return dict(importance_by_component)

    def get_cumulative_explained_variance(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get cumulative explained variance ratios for each feature-residual type combination.
        """
        if not self.is_fitted:
            raise ValueError("PCA models not fitted. Call fit() first.")
        
        cumulative_variance = defaultdict(dict)
        for feature, residual_dict in self.pca_models.items():
            for residual_type, pca_model in residual_dict.items():
                cumulative_variance[feature][residual_type] = \
                    np.cumsum(pca_model.explained_variance_ratio_)
        
        return dict(cumulative_variance)



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