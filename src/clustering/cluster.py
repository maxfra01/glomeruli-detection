import os
import numpy as np
import cv2
from utils import load_glomeruli_images, evaluate_clustering, init_log_file
from visual import plot_cluster_examples
from dim_reduction import reduce_dimensionality
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from feature_extraction import extract_vgg19_base_features, combine_features, extract_features_from_segmentation_model, extract_morphological_features
from macenko_normalization import macenko

os.makedirs("results", exist_ok=True)

LOG_FILE = "results/log.txt"
DATA_PATH = "data_clustering/"
IMG_SIZE = 384
NUM_CLUSTERS = 3
MODEL_PATH = "./snapshots/model_lr0.0001_bs64_wd0.0001_ar5.keras" 

init_log_file(LOG_FILE)

def load_masks_for_images(data_path, image_size=(224, 224)):
    """
    Load corresponding masks for the glomeruli images
    """
    masks = []
    root_dir = Path(data_path)
    
    for recherche_folder in sorted(root_dir.glob("RECHERCHE*")):
        masks_dir = recherche_folder / "masks"
        
        if not masks_dir.exists():
            print(f"Skipping {recherche_folder.name}: missing 'masks'")
            continue
            
        for mask_path in sorted(masks_dir.glob("*_mask.png")):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_resized = cv2.resize(mask, image_size).astype(np.float32) / 255.0
                masks.append(mask_resized)
    
    return np.array(masks)

def main():
    # Load images and masks
    print("Loading images...")
    images = load_glomeruli_images(DATA_PATH, image_size=(IMG_SIZE, IMG_SIZE))
    print(f"Loaded {len(images)} images")
    
    print("Loading masks...")
    masks = load_masks_for_images(DATA_PATH, image_size=(IMG_SIZE, IMG_SIZE))
    print(f"Loaded {len(masks)} masks")

    norm_images = macenko(images)
    
    # Ensure same number of images and masks
    min_samples = min(len(norm_images), len(masks))
    images = norm_images[:min_samples]
    masks = masks[:min_samples]
    print(f"Using {min_samples} paired samples")
    
    # Extract features from different sources
    print("\n" + "="*50)
    print("FEATURE EXTRACTION")
    print("="*50)
    
    # 1. VGG19 base features
    print("Extracting VGG19 base features...")
    vgg19_features = extract_vgg19_base_features(images)
    
    # 2. Segmentation model features
    print("Extracting segmentation model features...")
    seg_features = extract_features_from_segmentation_model(images, MODEL_PATH)
    
    # 3. Morphological features
    print("Extracting morphological features...")
    morph_features = extract_morphological_features(images, masks)
    
    # Combine all features
    print("\n" + "="*50)
    print("FEATURE COMBINATION")
    print("="*50)
    
    combined_features = combine_features(vgg19_features, seg_features, morph_features)
    
    # Dimensionality reduction and clustering
    print("\n" + "="*50)
    print("DIMENSIONALITY REDUCTION & CLUSTERING")
    print("="*50)
    
    for method in ["pca", "umap", "isomap"]:
        for n_components in [50, 100, 200]:
            print(f"\nProcessing {method.upper()} with {n_components} components...")

            mask = ~np.isnan(combined_features).any(axis=1)
            combined_features_clean = combined_features[mask]

            features_reduced = reduce_dimensionality(
                combined_features, 
                method=method, 
                n_components=n_components
            )

            # Cluster with different algorithms
            kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42).fit(features_reduced)
            gmm = GaussianMixture(n_components=NUM_CLUSTERS, random_state=42).fit(features_reduced)
            dbscan = DBSCAN(eps=0.5, min_samples=5)  

            kmeans_labels = kmeans.predict(features_reduced)
            gmm_labels = gmm.predict(features_reduced)
            dbscan_labels = dbscan.fit_predict(features_reduced)

            # Evaluate clustering
            evaluate_clustering(
                features_reduced, 
                kmeans_labels, 
                method_name=f"KMeans (n={n_components}) + {method.upper()} + Multi-Features"
            )
            evaluate_clustering(
                features_reduced, 
                gmm_labels, 
                method_name=f"GMM (n={n_components}) + {method.upper()} + Multi-Features"
            )
            n = evaluate_clustering(
                    features_reduced, 
                    dbscan_labels, 
                    method_name=f"DBSCAN +  {method.upper()} (n={n_components})",
                    log_file=LOG_FILE
            )

            
            # Plot results
            plot_cluster_examples(
                images, 
                kmeans_labels, 
                n_clusters=NUM_CLUSTERS, 
                samples_per_cluster=3, 
                algorithm_name=f"KMeans_{n_components}_{method.upper()}_MultiFeatures"
            )
            plot_cluster_examples(
                images, 
                gmm_labels, 
                n_clusters=NUM_CLUSTERS, 
                samples_per_cluster=3, 
                algorithm_name=f"GMM_{n_components}_{method.upper()}_MultiFeatures"
            )
            plot_cluster_examples(
                images, 
                dbscan_labels, 
                #n_clusters=NUM_CLUSTERS, 
                n_clusters = n,
                samples_per_cluster=10, 
                algorithm_name=f"DBSCAN_{method.upper()}_{n_components}"
            )


if __name__ == "__main__":
    main()