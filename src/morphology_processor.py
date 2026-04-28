import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MorphologyProcessor:
    """
    Processes morphological features using pattern recognition
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
    def preprocess_morph_data(self, morph_data):
        """
        Preprocess morphological data for analysis
        
        Parameters:
        morph_data (pd.DataFrame): DataFrame containing morphological measurements
        
        Returns:
        np.ndarray: Processed and scaled data
        """
        # Handle missing values
        morph_data = morph_data.fillna(morph_data.mean())
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(morph_data)
        
        return scaled_data
    
    def cluster_morphology(self, morph_data):
        """
        Cluster morphology data to identify patterns
        
        Parameters:
        morph_data (pd.DataFrame): DataFrame containing morphological measurements
        
        Returns:
        np.ndarray: Cluster labels for each sample
        """
        processed_data = self.preprocess_morph_data(morph_data)
        clusters = self.kmeans.fit_predict(processed_data)
        
        return clusters
    
    def extract_features(self, morph_data):
        """
        Extract key morphological features from raw data
        
        Parameters:
        morph_data (pd.DataFrame): DataFrame containing morphological measurements
        
        Returns:
        dict: Dictionary containing extracted features
        """
        if isinstance(morph_data, list):
            morph_data = pd.DataFrame(morph_data)
            
        features = {}
        
        # Basic statistics
        features['mean'] = morph_data.mean().to_dict()
        features['std'] = morph_data.std().to_dict()
        features['max'] = morph_data.max().to_dict()
        features['min'] = morph_data.min().to_dict()
        
        # Shape-related features
        if len(morph_data.columns) >= 2:
            features['area_ratio'] = morph_data.iloc[:, 0] / (morph_data.iloc[:, 1] + 1e-8)
            features['aspect_ratio'] = morph_data.iloc[:, 0] / (morph_data.iloc[:, 1] + 1e-8)
        
        # Cluster assignment
        try:
            features['clusters'] = self.cluster_morphology(morph_data).tolist()
        except:
            features['clusters'] = [0] * len(morph_data)
            
        return features

def extract_features(morph_data):
    """
    Extract key morphological features from raw data
    
    Parameters:
    morph_data (pd.DataFrame): DataFrame containing morphological measurements
    
    Returns:
    dict: Dictionary containing extracted features
    """
    processor = MorphologyProcessor()
    return processor.extract_features(morph_data)