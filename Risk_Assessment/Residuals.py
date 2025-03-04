import numpy as np
from typing import Tuple
from scipy.stats import norm


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
    def __init__(self, 
                 k: float = 3.0,           # Number of standard deviations for control limits
                 window_size: int = 20,     # Window size for parameter estimation
                 startup_size: int = 10):   # Minimum samples needed before calculating limits
        """
        Initialize Shewart control chart parameters.
        
        Args:
            k (float): Number of standard deviations for control limits
            window_size (int): Window size for parameter estimation
            startup_size (int): Minimum samples needed before calculating valid limits
        """
        super().__init__()
        self.k = k
        self.window_size = window_size
        self.startup_size = min(startup_size, window_size)
        self.buffer = []
    
    def update_buffer(self, value: float):
        """
        Update the sliding window buffer.
        
        Args:
            value (float): New value to add to buffer
        """
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def calculate_control_limits(self):
        """
        Calculate control limits based on current buffer.
        
        Returns:
            tuple: (center_line, upper_control_limit, lower_control_limit)
        """
        if len(self.buffer) < self.startup_size:
            # Not enough data for reliable limits, use standardized limits
            return 0.0, self.k, -self.k
        
        center_line = np.mean(self.buffer)
        std_dev = np.std(self.buffer, ddof=1)  # Use unbiased estimator
        
        ucl = center_line + self.k * std_dev
        lcl = center_line - self.k * std_dev
        
        return center_line, ucl, lcl
    
    def calculate_violation_score(self, 
                                value: float, 
                                center_line: float, 
                                ucl: float, 
                                lcl: float) -> float:
        """
        Calculate normalized violation score.
        
        Args:
            value: Current value
            center_line: Center line of the control chart
            ucl: Upper control limit
            lcl: Lower control limit
            
        Returns:
            float: Normalized violation score
        """
        if value > ucl:
            return (value - ucl) / (ucl - center_line) if ucl != center_line else np.inf
        elif value < lcl:
            return (lcl - value) / (center_line - lcl) if center_line != lcl else np.inf
        return 0.0
    
    def calculate(self, 
                 true_values: np.ndarray, 
                 pred_mean: np.ndarray, 
                 pred_var: np.ndarray) -> np.ndarray:
        """
        Calculate Shewart control chart statistics.
        
        Args:
            true_values: Ground truth values
            pred_mean: Predicted mean values
            pred_var: Predicted variance values
            
        Returns:
            np.ndarray: Violation scores
        """
        # Calculate standardized residuals
        residuals = pred_mean - true_values
        std = np.sqrt(pred_var)
        standardized_residuals = residuals / std
        
        violation_scores = []
        
        # Process each residual sequentially
        for res in standardized_residuals.flatten():
            # Update buffer with new observation
            self.update_buffer(res)
            
            # Calculate control limits based on current buffer
            cl, ucl, lcl = self.calculate_control_limits()
            
            # Calculate violation score
            score = self.calculate_violation_score(res, cl, ucl, lcl)
            violation_scores.append(score)
        
        return np.array(violation_scores).reshape(residuals.shape)

class SPRTResidual(BaseResidual):
    def __init__(self, 
                 mu0: float = 0.0,    # Null hypothesis mean
                 mu1: float = 1.0,    # Alternative hypothesis mean
                 alpha: float = 0.05,  # Type I error probability
                 beta: float = 0.05):  # Type II error probability
        """
        Initialize Sequential Probability Ratio Test (SPRT) parameters.
        
        Args:
            mu0 (float): Mean under null hypothesis H0 (typically 0 for residuals)
            mu1 (float): Mean under alternative hypothesis H1
            alpha (float): Type I error probability (false positive rate)
            beta (float): Type II error probability (false negative rate)
        """
        super().__init__()
        self.mu0 = mu0
        self.mu1 = mu1
        self.alpha = alpha
        self.beta = beta
        
        # Calculate decision boundaries
        self.A = np.log((1 - beta) / alpha)      # Upper boundary
        self.B = np.log(beta / (1 - alpha))      # Lower boundary
        
        # Initialize log-likelihood ratio
        self.llr = 0.0
        
    def reset_statistics(self):
        """Reset the log-likelihood ratio when decision boundaries are crossed."""
        self.llr = 0.0
        
    def calculate_log_likelihood_ratio(self, 
                                     x: float, 
                                     sigma: float) -> float:
        """
        Calculate the log-likelihood ratio for a single observation.
        
        Args:
            x: Observation (residual)
            sigma: Standard deviation
            
        Returns:
            float: Log-likelihood ratio
        """
        return ((self.mu1 - self.mu0) * 
                (2 * x - (self.mu1 + self.mu0)) / 
                (2 * sigma * sigma))
    
    def normalize_sprt_statistic(self, llr_value: float) -> float:
        """
        Normalize the SPRT statistic to [0,1] range.
        
        Args:
            llr_value: Current log-likelihood ratio value
            
        Returns:
            float: Normalized SPRT statistic
        """
        return (llr_value - self.B) / (self.A - self.B)
        
    def calculate(self, 
                 true_values: np.ndarray, 
                 pred_mean: np.ndarray, 
                 pred_var: np.ndarray) -> np.ndarray:
        """
        Calculate SPRT statistics for the residuals.
        
        Args:
            true_values: Ground truth values
            pred_mean: Predicted mean values
            pred_var: Predicted variance values
            
        Returns:
            np.ndarray: Normalized SPRT statistics
        """
        residuals = pred_mean - true_values
        std = np.sqrt(pred_var)  # Use predicted standard deviation
        
        sprt_scores = []
        
        # Process each residual
        for res, s in zip(residuals.flatten(), std.flatten()):
            # Update log-likelihood ratio
            self.llr += self.calculate_log_likelihood_ratio(res, s)
            
            # Normalize the current SPRT statistic
            score = self.normalize_sprt_statistic(self.llr)
            sprt_scores.append(score)
            
            # Reset only when decision boundaries are crossed
            if self.llr >= self.A or self.llr <= self.B:
                self.reset_statistics()
        
        return np.array(sprt_scores).reshape(residuals.shape)


class CUSUMResidual(BaseResidual):
    def __init__(self, k: float = 0.5, h: float = 5.0, target: float = 0.0):
        """
        Initialize CUSUM control chart parameters.
        
        Args:
            k (float): Reference value (slack value), typically set to δ/2 where δ is the shift to detect
            h (float): Decision interval/threshold for detecting changes
            target (float): Target value (usually 0 for residual monitoring)
        """
        super().__init__()
        self.k = k
        self.h = h
        self.target = target
        # Initialize CUSUM statistics
        self.Cp = 0.0  # Upper CUSUM
        self.Cn = 0.0  # Lower CUSUM
    
    def set_delta(self, value):
        self.k - value

    def reset_statistics(self):
        """Reset CUSUM statistics when a significant shift is detected."""
        self.Cp = 0.0
        self.Cn = 0.0
    
    def calculate(self, true_values: np.ndarray, pred_mean: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
        """
        Calculate CUSUM statistics for the residuals.
        
        Args:
            true_values: Ground truth values
            pred_mean: Predicted mean values
            pred_var: Predicted variance values
            
        Returns:
            np.ndarray: CUSUM scores normalized by the decision interval
        """
        # Calculate standardized residuals
        residuals = pred_mean - true_values
        std = np.sqrt(np.mean(pred_var))  # Use predicted variance for standardization
        standardized_residuals = (residuals - self.target) / std
        
        cusum_scores = []
        
        # Calculate CUSUM statistics for each observation
        for res in standardized_residuals.flatten():
            # Upper CUSUM
            self.Cp = max(0, self.Cp + (res - self.k))
            # Lower CUSUM
            self.Cn = max(0, self.Cn + (-res - self.k))
            
            # Calculate normalized score
            score = max(self.Cp, self.Cn) / self.h
            
            # Optional: Reset if a very significant shift is detected
            #if score > 1.0:  # Or some other threshold
            #    self.reset_statistics()
            
            cusum_scores.append(score)
            
        return np.array(cusum_scores).reshape(residuals.shape)
