import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class RobustMacenkoNormalizer:
    """
    Robust implementation of Macenko normalization,
    specifically adapted for PAS staining.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.15,
                 luminosity_threshold: float = 0.8,
                 regularizer: float = 0.01):
        """
        Args:
            alpha: Percentile for robust estimation (default 1.0 = 1% and 99%)
            beta: Optical density threshold
            luminosity_threshold: Threshold to remove background
            regularizer: Regularization term for numerical stability
        """
        self.alpha = alpha
        self.beta = beta
        self.luminosity_threshold = luminosity_threshold
        self.regularizer = regularizer
        self.target_stain_matrix = None
        self.target_max_concentrations = None
        
    def rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to Optical Density (OD)"""
        image = image.astype(np.float64)
        if image.max() > 1.0:
            image = image / 255.0
            
        image = np.maximum(image, 1e-6)
        od = -np.log(image)
        return od
    
    def od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert Optical Density to RGB"""
        rgb = np.exp(-od)
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)
    
    def get_stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract stain matrix using robust Macenko method
        """
        od = self.rgb_to_od(image)
        h, w, c = od.shape
        od_reshaped = od.reshape(-1, 3)
        
        # Remove background (pixels with low OD = white background)
        od_mean = np.mean(od_reshaped, axis=1)
        foreground_mask = od_mean > self.beta
        
        if np.sum(foreground_mask) < 10:
            foreground_mask = od_mean > self.beta * 0.5
            
        if np.sum(foreground_mask) < 10:
            print("Warning: Very few foreground pixels detected")
            foreground_mask = np.ones(len(od_reshaped), dtype=bool)
        
        od_foreground = od_reshaped[foreground_mask]
        
        try:
            od_mean_vec = np.mean(od_foreground, axis=0)
            od_centered = od_foreground - od_mean_vec
            
            pca = PCA(n_components=2)
            pca.fit(od_centered)
            projections = pca.transform(od_centered)
            
            angles = np.arctan2(projections[:, 1], projections[:, 0])
            min_angle = np.percentile(angles, self.alpha)
            max_angle = np.percentile(angles, 100 - self.alpha)
            
            extreme_dir1 = np.array([np.cos(min_angle), np.sin(min_angle)])
            extreme_dir2 = np.array([np.cos(max_angle), np.sin(max_angle)])
            
            stain1 = pca.components_[0] * extreme_dir1[0] + pca.components_[1] * extreme_dir1[1]
            stain2 = pca.components_[0] * extreme_dir2[0] + pca.components_[1] * extreme_dir2[1]
            
            if np.sum(stain1) < 0:
                stain1 = -stain1
            if np.sum(stain2) < 0:
                stain2 = -stain2
                
            stain_matrix = np.array([stain1, stain2])
            
        except Exception as e:
            print(f"PCA failed: {e}, using default stain matrix")
            stain_matrix = np.array([
                [0.65, 0.70, 0.29],  # PAS stain
                [0.07, 0.99, 0.11],  # Counterstain
            ])
        
        try:
            concentrations = np.linalg.lstsq(stain_matrix.T, od_foreground.T, rcond=None)[0]
            max_concentrations = np.percentile(concentrations, 99 - self.alpha, axis=1)
            max_concentrations = np.maximum(max_concentrations, 0.1)
        except:
            max_concentrations = np.array([1.0, 1.0])
        
        return stain_matrix, max_concentrations
    
    def fit(self, target_image: np.ndarray):
        """Fit the normalizer using a target image"""
        self.target_stain_matrix, self.target_max_concentrations = self.get_stain_matrix(target_image)
        return self
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize a new image using the target parameters"""
        if self.target_stain_matrix is None:
            raise ValueError("Normalizer must be fitted first!")
        
        source_stain_matrix, source_max_concentrations = self.get_stain_matrix(image)
        od = self.rgb_to_od(image)
        h, w, c = od.shape
        od_reshaped = od.reshape(-1, 3)
        
        try:
            source_concentrations = np.linalg.lstsq(
                source_stain_matrix.T, od_reshaped.T, rcond=self.regularizer
            )[0]
            
            normalized_concentrations = source_concentrations.copy()
            for i in range(len(source_max_concentrations)):
                if source_max_concentrations[i] > 0:
                    normalized_concentrations[i] *= (
                        self.target_max_concentrations[i] / source_max_concentrations[i]
                    )
            
            normalized_od = self.target_stain_matrix.T @ normalized_concentrations
            normalized_od = normalized_od.T.reshape(h, w, c)
            normalized_image = self.od_to_rgb(normalized_od)
            
        except Exception as e:
            print(f"Normalization failed: {e}, returning original image")
            normalized_image = image.astype(np.uint8)
        
        return normalized_image
    
    def fit_transform(self, images: list, target_idx: int = 0) -> np.ndarray:
        """Fit on one image and normalize all others"""
        self.fit(images[target_idx])
        normalized_images = []
        for i, img in enumerate(images):
            if i == target_idx:
                normalized_images.append(img.astype(np.uint8))
            else:
                norm_img = self.transform(img)
                normalized_images.append(norm_img)
        
        return np.array(normalized_images)

def visualize_normalization(original_images, normalized_images, save_path="normalization_comparison.png"):
    """Visualize normalization results"""
    n_samples = min(4, len(original_images))
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 8))
    
    for i in range(n_samples):
        axes[0, i].imshow(original_images[i])
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(normalized_images[i])
        axes[1, i].set_title(f"Normalized {i}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison saved to {save_path}")
