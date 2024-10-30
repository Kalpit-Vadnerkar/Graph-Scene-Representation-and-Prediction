import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy.typing as npt

from Model_Testing.FeatureExtractor import SequenceResiduals, ResidualStatistics, FeatureExtractor

class ResidualGenerator:
    @staticmethod
    def calculate_residuals(ground_truth: Dict[str, np.ndarray], 
                          predictions: Dict[str, np.ndarray],
                          config: Dict[str, Any]) -> SequenceResiduals:
        """Calculate residuals and their statistics for all variables"""
        
        def compute_statistics(error: float, std: float) -> ResidualStatistics:
            return ResidualStatistics(
                error=error,
                std=std,
                normalized_error=error/std if std > 0 else error
            )
        
        sequence_residuals = []
        for t in range(config['output_seq_len']):
            # Position residuals
            pos_X = compute_statistics(
                predictions['position_mean'][t][0] - ground_truth['position'][t][0],
                np.sqrt(predictions['position_var'][t][0])
            )
            pos_Y = compute_statistics(
                predictions['position_mean'][t][1] - ground_truth['position'][t][1],
                np.sqrt(predictions['position_var'][t][1])
            )
            
            # Velocity residuals
            vel_X = compute_statistics(
                predictions['velocity_mean'][t][0] - ground_truth['velocity'][t][0],
                np.sqrt(predictions['velocity_var'][t][0])
            )
            vel_Y = compute_statistics(
                predictions['velocity_mean'][t][1] - ground_truth['velocity'][t][1],
                np.sqrt(predictions['velocity_var'][t][1])
            )
            
            # Control residuals
            steering = compute_statistics(
                predictions['steering_mean'][t] - ground_truth['steering'][t],
                np.sqrt(predictions['steering_var'][t])
            )
            acceleration = compute_statistics(
                predictions['acceleration_mean'][t] - ground_truth['acceleration'][t],
                np.sqrt(predictions['acceleration_var'][t])
            )
            
            # Combined residual (weighted sum of normalized errors)
            combined = (
                pos_X.normalized_error + pos_Y.normalized_error +
                vel_X.normalized_error + vel_Y.normalized_error +
                steering.normalized_error + acceleration.normalized_error
            )
            
            sequence_residuals.append(SequenceResiduals(
                position_X=[pos_X],
                position_Y=[pos_Y],
                velocity_X=[vel_X],
                velocity_Y=[vel_Y],
                steering=[steering],
                acceleration=[acceleration],
                combined=[combined]
            ))
        
        # Combine all timesteps
        return SequenceResiduals(
            position_X=[r.position_X[0] for r in sequence_residuals],
            position_Y=[r.position_Y[0] for r in sequence_residuals],
            velocity_X=[r.velocity_X[0] for r in sequence_residuals],
            velocity_Y=[r.velocity_Y[0] for r in sequence_residuals],
            steering=[r.steering[0] for r in sequence_residuals],
            acceleration=[r.acceleration[0] for r in sequence_residuals],
            combined=[r.combined[0] for r in sequence_residuals]
        )

