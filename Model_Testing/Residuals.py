import numpy as np
from typing import Tuple


class BaseResidual:
    def __init__(self):
        pass
    
    def calculate(self, true_values: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement calculate method")

class RawResidual(BaseResidual):
    def calculate(self, true_values: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
        return pred_mean - true_values

class NormalizedResidual(BaseResidual):
    def calculate(self, true_values: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
        return (pred_mean - true_values) / np.sqrt(pred_var)

class UncertaintyResidual(BaseResidual):
    def calculate(self, true_values: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
        return pred_var

class KLDivergenceResidual(BaseResidual):
    def calculate(self, true_values: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
        return 0.5 * (
            np.log(2 * np.pi * pred_var) + 
            (true_values - pred_mean)**2 / pred_var
        )

class ShewartResidual(BaseResidual):
    def __init__(self, window_size: int = 20):
        super().__init__()
        self.window_size = window_size
        self.control_limits = {}
        
    def calculate_control_limits(self, data: np.ndarray) -> Tuple[float, float, float]:
        center_line = np.mean(data)
        std_dev = np.std(data)
        ucl = center_line + 3 * std_dev
        lcl = center_line - 3 * std_dev
        return center_line, ucl, lcl
    
    def calculate_violation_score(self, value: float, center_line: float, ucl: float, lcl: float) -> float:
        if value > ucl:
            return (value - ucl) / (ucl - center_line)
        elif value < lcl:
            return (lcl - value) / (center_line - lcl)
        return 0.0
    
    def calculate(self, true_values: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
        raw_residuals = pred_mean - true_values
        feature_key = str(hash(str(raw_residuals.shape)))
        
        if feature_key not in self.control_limits:
            cl, ucl, lcl = self.calculate_control_limits(raw_residuals)
            self.control_limits[feature_key] = (cl, ucl, lcl)
        
        cl, ucl, lcl = self.control_limits[feature_key]
        return np.array([
            self.calculate_violation_score(val, cl, ucl, lcl)
            for val in raw_residuals.flatten()
        ]).reshape(raw_residuals.shape)

class CUSUMResidual(BaseResidual):
    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.drift = drift
        self.pos_cusum = {}
        self.neg_cusum = {}
    
    def reset_statistics(self, feature_key: str):
        self.pos_cusum[feature_key] = 0
        self.neg_cusum[feature_key] = 0
    
    def calculate(self, true_values: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
        residuals = pred_mean - true_values
        mean = np.mean(residuals)
        std = np.std(residuals)
        feature_key = str(hash(str(residuals.shape)))
        
        if feature_key not in self.pos_cusum:
            self.reset_statistics(feature_key)
            
        k = self.drift * std / 2
        standardized_residuals = (residuals - mean) / std
        
        cusum_scores = []
        for res in standardized_residuals.flatten():
            self.pos_cusum[feature_key] = max(0, self.pos_cusum[feature_key] + res - k)
            self.neg_cusum[feature_key] = max(0, self.neg_cusum[feature_key] - res - k)
            score = max(self.pos_cusum[feature_key], self.neg_cusum[feature_key])
            cusum_scores.append(score / self.threshold)
            
        return np.array(cusum_scores).reshape(residuals.shape)

class SPRTResidual(BaseResidual):
    def __init__(self, alpha: float = 0.05, beta: float = 0.05, drift: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.drift = drift
        self.sprt_stats = {}
    
    def reset_statistics(self, feature_key: str):
        self.sprt_stats[feature_key] = 0
    
    def calculate(self, true_values: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
        residuals = pred_mean - true_values
        mean = np.mean(residuals)
        std = np.std(residuals)
        feature_key = str(hash(str(residuals.shape)))
        
        if feature_key not in self.sprt_stats:
            self.reset_statistics(feature_key)
            
        standardized_residuals = (residuals - mean) / std
        sprt_scores = []
        
        upper_bound = np.log((1 - self.beta) / self.alpha)
        lower_bound = np.log(self.beta / (1 - self.alpha))
        
        for res in standardized_residuals.flatten():
            llr = (self.drift * res - self.drift**2 / 2) / std
            self.sprt_stats[feature_key] += llr
            
            score = (self.sprt_stats[feature_key] - lower_bound) / (upper_bound - lower_bound)
            sprt_scores.append(score)
            
            if self.sprt_stats[feature_key] >= upper_bound or self.sprt_stats[feature_key] <= lower_bound:
                self.sprt_stats[feature_key] = 0
                
        return np.array(sprt_scores).reshape(residuals.shape)