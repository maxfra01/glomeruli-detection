import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import silhouette_score

# Seed for reproducibility
np.random.seed(42)


def init_log_file(path=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("Clustering Evaluation Log\n")
        f.write("=" * 30 + "\n\n")

def load_glomeruli_images(root_dir, image_size=(224, 224)):
    """
    Load and process glomeruli images from multiple subfolders inside root_dir.
    Each subfolder is expected to have 'images' and 'masks' directories.
    
    Args:
        root_dir (str or Path): Root directory containing RECHERCHEXXX subfolders.
        image_size (tuple): Desired image size (width, height).
        
    Returns:
        np.ndarray: Array of processed images of shape (N, H, W, 3)
    """
    root_dir = Path(root_dir)
    processed_images = []

    for recherche_folder in sorted(root_dir.glob("RECHERCHE*")):
        images_dir = recherche_folder / "images"
        masks_dir = recherche_folder / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            print(f"Skipping {recherche_folder.name}: missing 'images' or 'masks'")
            continue

        for img_path in sorted(images_dir.glob("*.png")):
            mask_path = masks_dir / img_path.name.replace(".png", "_mask.png")
            if not mask_path.exists():
                print(f"Mask not found for {img_path.name} in {recherche_folder.name}, skipping.")
                continue

            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Ensure mask is binary
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Apply mask (black background)
            masked_img = cv2.bitwise_and(img, img, mask=binary_mask)

            # Resize
            masked_img = cv2.resize(masked_img, image_size)
            masked_img = masked_img.astype(np.float32)

            processed_images.append(masked_img)

    return np.array(processed_images)

def evaluate_clustering(features, labels, method_name="Clustering", log_file="results/log.txt"):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    lines = [f"[{method_name}]"]
    lines.append(f"Number of clusters: {n_clusters}")

    if n_clusters <= 1:
        lines.append("Not enough clusters to evaluate.\n")
    else:
        sil_score = silhouette_score(features, labels)
        c_idx = c_index(features, labels)
        d_idx = dunn_index(features, labels)

        lines.append(f"Silhouette Score: {sil_score:.3f}")
        lines.append(f"C-Index: {c_idx:.3f}")
        lines.append(f"Dunn Index: {d_idx:.3f}\n")

    result = "\n".join(lines)
    print(result)

    with open(log_file, "a") as f:
        f.write(result + "\n")
from scipy.spatial.distance import cdist

import numpy as np
from scipy.spatial.distance import pdist, squareform

def c_index(X, labels):
    """
    Calculate the C-Index for clustering.
    
    Args:
        X (array): data points (n_samples, n_features)
        labels (array): cluster labels
        
    Returns:
        float: C-index score (lower is better)
    """
    distances = pdist(X)  # condensed distance matrix
    distances_sorted = np.sort(distances)
    
    n_samples = len(X)
    unique_labels = set(labels)
    
    # Indices of pairs in the same cluster
    same_cluster_pairs = []
    
    # For each cluster, get indices of points
    for label in unique_labels:
        if label == -1:
            continue  # skip noise if any (DBSCAN)
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) < 2:
            continue
        # Distances within cluster
        cluster_dists = pdist(X[cluster_indices])
        same_cluster_pairs.extend(cluster_dists)
    
    sum_same_cluster = np.sum(same_cluster_pairs)
    n_same_cluster = len(same_cluster_pairs)
    
    if n_same_cluster == 0:
        return np.nan
    
    sum_min = np.sum(distances_sorted[:n_same_cluster])
    sum_max = np.sum(distances_sorted[-n_same_cluster:])
    
    c_index_val = (sum_same_cluster - sum_min) / (sum_max - sum_min)
    return c_index_val


def dunn_index(X, labels):
    """
    Calculate the Dunn Index for clustering.
    
    Args:
        X (array): data points (n_samples, n_features)
        labels (array): cluster labels
        
    Returns:
        float: Dunn Index (higher is better)
    """
    unique_labels = set(labels)
    clusters = [X[labels == label] for label in unique_labels if label != -1]
    if len(clusters) < 2:
        return np.nan
    
    # Compute inter-cluster distances (min distance between clusters)
    min_intercluster = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = cdist(clusters[i], clusters[j], metric='euclidean')
            min_dist = np.min(dist)
            if min_dist < min_intercluster:
                min_intercluster = min_dist
    
    # Compute intra-cluster distances (max cluster diameter)
    max_intracluster = 0
    for cluster in clusters:
        dist = pdist(cluster, metric='euclidean')
        if len(dist) == 0:
            continue
        max_diameter = np.max(dist)
        if max_diameter > max_intracluster:
            max_intracluster = max_diameter
    
    if max_intracluster == 0:
        return np.nan
    
    return min_intercluster / max_intracluster
