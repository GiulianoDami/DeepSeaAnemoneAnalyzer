import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def validate_input(data):
    """
    Validates the input data for the DeepSeaAnemoneAnalyzer.
    
    Parameters:
        data (dict or DataFrame): Input data containing spectral and morphological features
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if data is None:
        return False
    
    # Convert to DataFrame if it's a dict
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Check for required columns
    required_columns = ['spectral_signature', 'morphology_features']
    if not all(col in data.columns for col in required_columns):
        return False
    
    # Check that data is not empty
    if len(data) == 0:
        return False
    
    # Validate spectral signature format
    try:
        for idx, row in data.iterrows():
            if not isinstance(row['spectral_signature'], (list, np.ndarray)):
                return False
            # Ensure spectral signature has numeric values
            spectral_array = np.array(row['spectral_signature'])
            if not np.issubdtype(spectral_array.dtype, np.number):
                return False
    except (ValueError, TypeError):
        return False
    
    # Validate morphology features format
    try:
        for idx, row in data.iterrows():
            if not isinstance(row['morphology_features'], (list, np.ndarray)):
                return False
            # Ensure morphology features have numeric values
            morphology_array = np.array(row['morphology_features'])
            if not np.issubdtype(morphology_array.dtype, np.number):
                return False
    except (ValueError, TypeError):
        return False
    
    return True

def adjust_probability_by_depth(prob, depth):
    """
    Adjusts the probability score based on the depth of the sample.
    
    Parameters:
        prob (float): Original probability score (0-1)
        depth (float): Depth in meters
        
    Returns:
        float: Adjusted probability score
    """
    if not 0 <= prob <= 1:
        raise ValueError("Probability must be between 0 and 1")
    
    if depth < 0:
        raise ValueError("Depth cannot be negative")
    
    # Define depth thresholds for adjustment
    shallow_threshold = 1000  # meters
    deep_threshold = 4000     # meters
    
    # Adjust probability based on depth
    if depth <= shallow_threshold:
        # Shallow waters - reduce confidence slightly
        adjusted_prob = prob * 0.8
    elif depth <= deep_threshold:
        # Moderate depths - maintain original probability
        adjusted_prob = prob
    else:
        # Very deep waters - increase confidence for deep-sea organisms
        adjusted_prob = min(1.0, prob * 1.2)
    
    # Ensure probability stays within bounds
    return max(0.0, min(1.0, adjusted_prob))