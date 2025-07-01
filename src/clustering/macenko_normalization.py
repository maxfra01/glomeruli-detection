import numpy as np
import cv2 
from sklearn.decomposition import PCA
from typing import Tuple

class RobustMacenkoNormalizer:
    """
    Implementazione robusta di Macenko normalizzazione
    Specificamente adattata per PAS staining
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.15,
                 luminosity_threshold: float = 0.8,
                 regularizer: float = 0.01):
        """
        Args:
            alpha: Percentile per robust estimation (default 1.0 = 1% e 99%)
            beta: Threshold per optical density
            luminosity_threshold: Soglia per eliminare background
            regularizer: Termine di regolarizzazione per stabilità numerica
        """
        self.alpha = alpha
        self.beta = beta
        self.luminosity_threshold = luminosity_threshold
        self.regularizer = regularizer
        self.target_stain_matrix = None
        self.target_max_concentrations = None
        
    def rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """Converte RGB a Optical Density"""
        # Assicurati che l'immagine sia float e in range [0,1]
        image = image.astype(np.float64)
        if image.max() > 1.0:
            image = image / 255.0
            
        # Evita log(0) aggiungendo small epsilon
        image = np.maximum(image, 1e-6)
        
        # Calcola optical density
        od = -np.log(image)
        return od
    
    def od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Converte Optical Density a RGB"""
        rgb = np.exp(-od)
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)
    
    def get_stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estrae matrice delle colorazioni usando metodo Macenko robusto
        """
        # Converti a OD
        od = self.rgb_to_od(image)
        h, w, c = od.shape
        
        # Reshape per analisi
        od_reshaped = od.reshape(-1, 3)
        
        # Rimuovi background (pixel con bassa OD = sfondo bianco)
        od_mean = np.mean(od_reshaped, axis=1)
        foreground_mask = od_mean > self.beta
        
        if np.sum(foreground_mask) < 10:
            # Se troppo pochi pixel foreground, usa soglia più bassa
            foreground_mask = od_mean > self.beta * 0.5
            
        if np.sum(foreground_mask) < 10:
            # Fallback: usa tutti i pixel
            print("Warning: Very few foreground pixels detected")
            foreground_mask = np.ones(len(od_reshaped), dtype=bool)
        
        od_foreground = od_reshaped[foreground_mask]
        
        # PCA per trovare direzioni principali delle colorazioni
        try:
            # Centra i dati
            od_mean_vec = np.mean(od_foreground, axis=0)
            od_centered = od_foreground - od_mean_vec
            
            # PCA
            pca = PCA(n_components=2)
            pca.fit(od_centered)
            
            # Proiezioni sui primi 2 componenti principali
            projections = pca.transform(od_centered)
            
            # Trova angoli estremi delle proiezioni
            angles = np.arctan2(projections[:, 1], projections[:, 0])
            
            # Usa percentili per robustezza (invece di min/max)
            min_angle = np.percentile(angles, self.alpha)
            max_angle = np.percentile(angles, 100 - self.alpha)
            
            # Calcola direzioni estreme
            extreme_dir1 = np.array([np.cos(min_angle), np.sin(min_angle)])
            extreme_dir2 = np.array([np.cos(max_angle), np.sin(max_angle)])
            
            # Trasforma in spazio originale 3D
            stain1 = pca.components_[0] * extreme_dir1[0] + pca.components_[1] * extreme_dir1[1]
            stain2 = pca.components_[0] * extreme_dir2[0] + pca.components_[1] * extreme_dir2[1]
            
            # Assicurati che abbiano norma positiva
            if np.sum(stain1) < 0:
                stain1 = -stain1
            if np.sum(stain2) < 0:
                stain2 = -stain2
                
            stain_matrix = np.array([stain1, stain2])
            
        except Exception as e:
            print(f"PCA failed: {e}, using default stain matrix")
            # Fallback: matrice default per PAS
            stain_matrix = np.array([
                [0.65, 0.70, 0.29],  # PAS stain
                [0.07, 0.99, 0.11],  # Background/counterstain
            ])
        
        # Calcola concentrazioni massime
        try:
            # Risolvi least squares per ottenere concentrazioni
            concentrations = np.linalg.lstsq(stain_matrix.T, od_foreground.T, rcond=None)[0]
            max_concentrations = np.percentile(concentrations, 99 - self.alpha, axis=1)
            # Assicurati che siano positive
            max_concentrations = np.maximum(max_concentrations, 0.1)
        except:
            max_concentrations = np.array([1.0, 1.0])
        
        return stain_matrix, max_concentrations
    
    def fit(self, target_image: np.ndarray):
        """Fit normalizer su immagine target"""
        self.target_stain_matrix, self.target_max_concentrations = self.get_stain_matrix(target_image)
        return self
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalizza immagine usando parametri target"""
        if self.target_stain_matrix is None:
            raise ValueError("Normalizer must be fitted first!")
        
        # Estrai parametri dell'immagine sorgente
        source_stain_matrix, source_max_concentrations = self.get_stain_matrix(image)
        
        # Converti a OD
        od = self.rgb_to_od(image)
        h, w, c = od.shape
        od_reshaped = od.reshape(-1, 3)
        
        try:
            # Deconvoluzione: ottieni concentrazioni con matrice sorgente
            source_concentrations = np.linalg.lstsq(
                source_stain_matrix.T, od_reshaped.T, rcond=self.regularizer
            )[0]
            
            # Normalizza concentrazioni
            normalized_concentrations = source_concentrations.copy()
            for i in range(len(source_max_concentrations)):
                if source_max_concentrations[i] > 0:
                    normalized_concentrations[i] *= (
                        self.target_max_concentrations[i] / source_max_concentrations[i]
                    )
            
            # Ricostruisci con matrice target
            normalized_od = self.target_stain_matrix.T @ normalized_concentrations
            normalized_od = normalized_od.T.reshape(h, w, c)
            
            # Converti a RGB
            normalized_image = self.od_to_rgb(normalized_od)
            
        except Exception as e:
            print(f"Normalization failed: {e}, returning original image")
            normalized_image = image.astype(np.uint8)
        
        return normalized_image
    
    def fit_transform(self, images: list, target_idx: int = 0) -> np.ndarray:
        """Fit su una immagine e trasforma tutte"""
        # Fit su immagine target
        self.fit(images[target_idx])
        
        # Trasforma tutte le immagini
        normalized_images = []
        for i, img in enumerate(images):
            if i == target_idx:
                normalized_images.append(img.astype(np.uint8))
            else:
                norm_img = self.transform(img)
                normalized_images.append(norm_img)
        
        return np.array(normalized_images)


def macenko(images):
    print("Applying Macenko normalization...")
    try:
        normalizer = RobustMacenkoNormalizer(
            alpha=1.0,  # Usa percentili 1% e 99%
            beta=0.15,  # Soglia OD
            luminosity_threshold=0.8
        )
        
        # Seleziona immagine target (quella con miglior contrasto)
        target_idx = select_best_target_image(images)
        print(f"Using image {target_idx} as normalization target")
        
        normalized_images = normalizer.fit_transform(images, target_idx=target_idx)
        
        print("Macenko normalization completed successfully!")
        
    except Exception as e:
        print(f"Macenko normalization failed: {e}")
        print("Using original images...")
        normalized_images = images

    return normalized_images

def select_best_target_image(images):
    """Seleziona l'immagine con miglior contrasto come target"""
    best_contrast = 0
    best_idx = 0
    
    for i, img in enumerate(images):
        # Calcola contrasto come deviazione standard
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        
        if contrast > best_contrast:
            best_contrast = contrast
            best_idx = i
    
    return best_idx

