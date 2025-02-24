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
        self.keep_raw_sequence = True
        self.pca_models = {}
        self.scalers = {}
        self.is_fitted = False
        self.window_size = None
        
    def _extract_temporal_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract temporal features from a sequence before PCA.
        For multi-dimensional features, performs element-wise addition first.
        Then either keeps the raw summed sequence or computes statistical features.
        
        Args:
            sequence: The input sequence with shape (time_steps, dimensions)
            keep_raw_sequence: If True, return the raw summed sequence without computing stats
        
        Returns:
            np.ndarray: Feature vector
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

        # Always perform element-wise summing for multi-dimensional features
        if sequence.shape[1] > 1:
            # Element-wise addition across dimensions
            combined_signal = np.sum(sequence, axis=1)
        else:
            # For 1D features, just use the sequence directly
            combined_signal = sequence.flatten()
        
        # Now we have the combined_signal, either keep it as-is or compute stats
        if self.keep_raw_sequence:
            # Return the raw summed sequence
            return combined_signal
        else:
            # Compute statistical features from the summed sequence
            return np.array(compute_single_dim_features(combined_signal))
        

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

    def transform_all(self, all_residuals: List[ResidualFeatures]) -> List[Dict[str, float]]:
        """
        Transform all residuals at once using the fitted feature extractor.
        
        Args:
            all_residuals: List of ResidualFeatures objects
            
        Returns:
            List of dictionaries containing reduced features
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")
        
        return [self.transform(residuals) for residuals in all_residuals]