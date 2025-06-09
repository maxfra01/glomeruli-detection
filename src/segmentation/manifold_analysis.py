import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE, Isomap
import umap
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import seaborn as sns

class ManifoldAnalyzer:
    def __init__(self, model: tf.keras.Model):
        """
        Initialize the ManifoldAnalyzer with a trained model.
        
        Args:
            model: Trained VGG19-based segmentation model
        """
        self.model = model
        self.feature_extractors = {}
        self._setup_feature_extractors()
    
    def _setup_feature_extractors(self):
        """Create models to extract features from different layers."""
        # Get interesting layers for analysis
        target_layers = [
            'block1_conv2',  # Low-level features
            'block3_conv4',  # Mid-level features
            'block5_conv4',  # High-level features
        ]
        
        for layer_name in target_layers:
            layer = self.model.get_layer(layer_name)
            self.feature_extractors[layer_name] = tf.keras.Model(
                inputs=self.model.input,
                outputs=layer.output
            )
    
    def extract_features(self, dataset: tf.data.Dataset, layer_name: str) -> np.ndarray:
        """
        Extract features from a specific layer for the given dataset.
        
        Args:
            dataset: Input dataset
            layer_name: Name of the layer to extract features from
            
        Returns:
            Features array of shape (n_samples, flattened_features)
        """
        features = []
        labels = []
        
        # Get the feature extractor for the specified layer
        feature_extractor = self.feature_extractors[layer_name]
        
        # Extract features
        for images, masks in dataset:
            batch_features = feature_extractor(images)
            batch_features = tf.reshape(batch_features, 
                                     (batch_features.shape[0], -1))
            features.append(batch_features)
            labels.append(masks)
        
        return np.vstack(features), np.vstack(labels)
    
    def apply_dimensionality_reduction(self, 
                                     features: np.ndarray,
                                     method: str = 'tsne',
                                     n_components: int = 2,
                                     **kwargs) -> np.ndarray:
        """
        Apply dimensionality reduction to the features.
        
        Args:
            features: Input features array
            method: One of 'tsne', 'umap', or 'isomap'
            n_components: Number of dimensions to reduce to
            **kwargs: Additional arguments for the reduction method
            
        Returns:
            Reduced features array
        """
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, **kwargs)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=n_components, **kwargs)
        elif method.lower() == 'isomap':
            reducer = Isomap(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return reducer.fit_transform(features)
    
    def visualize_manifold(self,
                          reduced_features: np.ndarray,
                          labels: np.ndarray,
                          title: str,
                          save_path: Optional[str] = None):
        """
        Visualize the reduced dimensional representation.
        
        Args:
            reduced_features: Features after dimensionality reduction
            labels: Corresponding labels/masks
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # For segmentation masks, we'll use the mean mask value as color intensity
        label_intensities = np.mean(labels, axis=(1, 2, 3))
        
        scatter = plt.scatter(reduced_features[:, 0],
                            reduced_features[:, 1],
                            c=label_intensities,
                            cmap='viridis',
                            alpha=0.6)
        
        plt.colorbar(scatter, label='Mean Mask Intensity')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def analyze_layer_manifolds(model: tf.keras.Model,
                          dataset: tf.data.Dataset,
                          output_dir: str = 'manifold_analysis'):
    """
    Analyze and visualize manifolds for different layers of the model.
    
    Args:
        model: Trained segmentation model
        dataset: Dataset to analyze
        output_dir: Directory to save the analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = ManifoldAnalyzer(model)
    
    # Analyze each layer with different manifold learning techniques
    for layer_name in analyzer.feature_extractors.keys():
        # Extract features
        features, labels = analyzer.extract_features(dataset, layer_name)
        
        # Apply different dimensionality reduction techniques
        methods = ['tsne', 'umap', 'isomap']
        for method in methods:
            reduced_features = analyzer.apply_dimensionality_reduction(
                features, method=method
            )
            
            # Visualize and save
            save_path = os.path.join(output_dir, f"{layer_name}_{method}.png")
            analyzer.visualize_manifold(
                reduced_features,
                labels,
                f"{method.upper()} visualization of {layer_name}",
                save_path
            ) 