import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy.typing as npt

@dataclass
class ResidualStatistics:
    """Container for residual statistics"""
    error: float
    std: float
    normalized_error: float  # error/std for standardized comparison

@dataclass
class SequenceResiduals:
    """Container for all residuals in a sequence"""
    position_X: List[ResidualStatistics]
    position_Y: List[ResidualStatistics]
    velocity_X: List[ResidualStatistics]
    velocity_Y: List[ResidualStatistics]
    steering: List[ResidualStatistics]
    acceleration: List[ResidualStatistics]
    combined: List[float]

class FeatureExtractor:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def extract_features(self, residuals: SequenceResiduals) -> Dict[str, float]:
        """Extract comprehensive features from residuals including variance analysis"""
        features = {}
        
        for field_name, values in residuals.__dict__.items():
            if field_name == 'combined':
                continue  # Handle combined separately as it's just errors
                
            errors = np.array([v.error for v in values])
            stds = np.array([v.std for v in values])
            norm_errors = np.array([v.normalized_error for v in values])
            
            # Basic statistics
            features.update({
                f'{field_name}_mean_error': np.mean(errors),
                f'{field_name}_std_error': np.std(errors),
                f'{field_name}_max_error': np.max(errors),
                f'{field_name}_mean_std': np.mean(stds),
                f'{field_name}_std_std': np.std(stds),  # Variance of variance
                f'{field_name}_mean_norm_error': np.mean(norm_errors),
                f'{field_name}_std_norm_error': np.std(norm_errors),
            })
            
            # Trend analysis
            features[f'{field_name}_error_trend'] = np.polyfit(np.arange(len(errors)), errors, 1)[0]
            features[f'{field_name}_std_trend'] = np.polyfit(np.arange(len(stds)), stds, 1)[0]
            
            # Rolling statistics
            if len(errors) >= self.window_size:
                #rolling_mean = np.convolve(errors, np.ones(self.window_size)/self.window_size, mode='valid')
                rolling_mean = 1
                rolling_std = np.array([np.std(errors[i:i+self.window_size]) 
                                      for i in range(len(errors)-self.window_size+1)])
                
                features.update({
                    f'{field_name}_rolling_mean_std': np.std(rolling_mean),
                    f'{field_name}_rolling_std_mean': np.mean(rolling_std),
                    f'{field_name}_rolling_std_std': np.std(rolling_std)  # Variance of rolling variance
                })
            
            # Frequency domain features
            if len(errors) > 1:
                fft_errors = np.abs(np.fft.fft(errors))
                fft_stds = np.abs(np.fft.fft(stds))
                
                features.update({
                    f'{field_name}_fft_error_max': np.max(fft_errors[1:]),
                    f'{field_name}_fft_error_mean': np.mean(fft_errors[1:]),
                    f'{field_name}_fft_std_max': np.max(fft_stds[1:]),
                    f'{field_name}_fft_std_mean': np.mean(fft_stds[1:])
                })
        
        # Handle combined residuals
        combined = np.array(residuals.combined)
        features.update({
            'combined_mean': np.mean(combined),
            'combined_std': np.std(combined),
            'combined_max': np.max(combined),
            'combined_trend': np.polyfit(np.arange(len(combined)), combined, 1)[0]
        })
        
        return features