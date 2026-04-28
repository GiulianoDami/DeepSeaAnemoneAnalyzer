import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class DeepSeaAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """Train the classifier with provided data"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
    def predict(self, X):
        """Predict organism classification for given samples"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities for each class"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

def analyze_sample(data):
    """
    Analyze a single deep-sea sample
    
    Parameters:
    data (dict or array-like): Sample data containing spectral and morphological features
    
    Returns:
    dict: Analysis results including classification and probabilities
    """
    # This would typically load a pre-trained model
    # For demonstration, we'll create a mock implementation
    
    # Convert input to numpy array if needed
    if isinstance(data, dict):
        # Assuming standard feature names for deep-sea samples
        features = ['spectral_peak_1', 'spectral_peak_2', 'spectral_peak_3',
                   'morphology_complexity', 'size', 'texture_variance']
        sample_data = [data.get(feat, 0) for feat in features]
    else:
        sample_data = np.array(data)
    
    # Create a mock analyzer instance
    analyzer = DeepSeaAnalyzer()
    
    # Mock training data (in practice, this would be loaded from a dataset)
    mock_features = np.random.rand(100, 6)
    mock_labels = np.random.choice(['anemone', 'sponge', 'other'], 100)
    
    # Train the model
    analyzer.train(mock_features, mock_labels)
    
    # Make prediction
    prediction = analyzer.predict([sample_data])[0]
    probabilities = analyzer.predict_proba([sample_data])[0]
    
    # Get class names
    classes = analyzer.model.classes_
    
    # Create result dictionary
    result = {
        'classification': prediction,
        'probabilities': dict(zip(classes, probabilities))
    }
    
    return result