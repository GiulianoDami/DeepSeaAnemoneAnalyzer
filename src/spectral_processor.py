import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class SpectralProcessor:
    """
    Handles spectral signature analysis for deep-sea organism identification.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_spectrum(self, spectrum):
        """
        Preprocess a spectral signature for analysis.
        
        Args:
            spectrum (array-like): Spectral data
            
        Returns:
            array: Normalized and scaled spectrum
        """
        spectrum = np.array(spectrum).reshape(1, -1)
        return self.scaler.fit_transform(spectrum)[0]
        
    def extract_features(self, spectrum):
        """
        Extract key features from a spectral signature.
        
        Args:
            spectrum (array-like): Spectral data
            
        Returns:
            dict: Dictionary containing spectral features
        """
        spectrum = np.array(spectrum)
        return {
            'mean': np.mean(spectrum),
            'std': np.std(spectrum),
            'max': np.max(spectrum),
            'min': np.min(spectrum),
            'peak_ratio': np.max(spectrum) / (np.mean(spectrum) + 1e-8),
            'slope': np.polyfit(range(len(spectrum)), spectrum, 1)[0] if len(spectrum) > 1 else 0
        }
        
    def compare_spectra(self, sig1, sig2):
        """
        Compare two spectral signatures using multiple metrics.
        
        Args:
            sig1 (array-like): First spectral signature
            sig2 (array-like): Second spectral signature
            
        Returns:
            dict: Comparison results including similarity scores
        """
        # Normalize spectra
        norm_sig1 = self.preprocess_spectrum(sig1)
        norm_sig2 = self.preprocess_spectrum(sig2)
        
        # Cosine similarity
        cos_sim = cosine_similarity([norm_sig1], [norm_sig2])[0][0]
        
        # Euclidean distance
        eucl_dist = np.linalg.norm(norm_sig1 - norm_sig2)
        
        # Correlation coefficient
        corr_coef = np.corrcoef(norm_sig1, norm_sig2)[0, 1]
        
        return {
            'cosine_similarity': float(cos_sim),
            'euclidean_distance': float(eucl_dist),
            'correlation_coefficient': float(corr_coef)
        }

def compare_spectra(sig1, sig2):
    """
    Convenience function to compare two spectral signatures.
    
    Args:
        sig1 (array-like): First spectral signature
        sig2 (array-like): Second spectral signature
        
    Returns:
        dict: Comparison results
    """
    processor = SpectralProcessor()
    return processor.compare_spectra(sig1, sig2)