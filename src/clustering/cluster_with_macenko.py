import os
from utils import load_glomeruli_images, evaluate_clustering, init_log_file
from visual import plot_cluster_examples
from dim_reduction import reduce_dimensionality
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans, DBSCAN
from tensorflow.keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from macenko_normalization import RobustMacenkoNormalizer, visualize_normalization
import cv2 

def cluster_with_macenko_normalization():
    """Clustering pipeline with Macenko normalization"""
    
    os.makedirs("results_new", exist_ok=True)
    
    NUM_CLUSTERS = 5
    LOG_FILE = f"results_new/log_macenko_{NUM_CLUSTERS}.txt"
    DATA_PATH = "data_clustering/"
    IMG_SIZE = 224
    
    init_log_file(LOG_FILE)
    
    print("Loading images...")
    images = load_glomeruli_images(DATA_PATH, image_size=(IMG_SIZE, IMG_SIZE))
    print(f"Loaded {len(images)} images")
    
    if len(images) == 0:
        print("No images found!")
        return
    
    print("Applying Macenko normalization...")
    try:
        normalizer = RobustMacenkoNormalizer(
            alpha=1.0,       # Use 1st and 99th percentiles
            beta=0.15,       # Optical Density threshold
            luminosity_threshold=0.8
        )
        
        target_idx = select_best_target_image(images)
        print(f"Using image {target_idx} as normalization target")
        
        normalized_images = normalizer.fit_transform(images, target_idx=target_idx)
        
        visualize_normalization(
            images[:4], 
            normalized_images[:4], 
            "results_new/macenko_normalization_comparison.png"
        )
        
        print("Macenko normalization completed successfully!")
        
    except Exception as e:
        print(f"Macenko normalization failed: {e}")
        print("Using original images...")
        normalized_images = images
    
    print("Extracting features with VGG19...")
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer("block4_pool").output)
    
    features = feature_model.predict(normalized_images, batch_size=32, verbose=1)
    features_flat = features.reshape((features.shape[0], -1))
    print(f"Feature shape: {features_flat.shape}")
    
    for method in ["pca", "umap", "isomap"]:
        for n_components in [50, 100, 150]:
            print(f"\nTesting {method.upper()} with {n_components} components...")
            
            try:
                features_reduced = reduce_dimensionality(
                    features_flat, 
                    method=method, 
                    n_components=n_components
                )
                
                # KMeans clustering
                kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=20)
                kmeans_labels = kmeans.fit_predict(features_reduced)
                
                # DBSCAN clustering (unsupervised, no cluster number needed)
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(features_reduced)
                
                # Clustering evaluation
                evaluate_clustering(
                    features_reduced, 
                    kmeans_labels, 
                    method_name=f"KMeans + {method.upper()} (n={n_components})",
                    log_file=LOG_FILE
                )
                
                n = evaluate_clustering(
                    features_reduced, 
                    dbscan_labels, 
                    method_name=f"DBSCAN +  {method.upper()} (n={n_components})",
                    log_file=LOG_FILE
                )
                
                # Cluster visualization
                plot_cluster_examples(
                    images, 
                    kmeans_labels, 
                    n_clusters=NUM_CLUSTERS, 
                    samples_per_cluster=10, 
                    algorithm_name=f"KMeans_{method.upper()}_{n_components}__{NUM_CLUSTERS}"
                )
                
                plot_cluster_examples(
                    images, 
                    dbscan_labels, 
                    n_clusters=n,
                    samples_per_cluster=10, 
                    algorithm_name=f"DBSCAN_{method.upper()}_{n_components}"
                )
                
            except Exception as e:
                print(f"Error with {method} {n_components}: {e}")
                continue

def select_best_target_image(images):
    """Select the image with the highest contrast as normalization target"""
    best_contrast = 0
    best_idx = 0
    
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        
        if contrast > best_contrast:
            best_contrast = contrast
            best_idx = i
    
    return best_idx

if __name__ == "__main__":
    cluster_with_macenko_normalization()
