import numpy as np
import os
import matplotlib.pyplot as plt

def plot_cluster_examples(images, labels, n_clusters=6, samples_per_cluster=3, algorithm_name="KMeans"):
    """
    Plot sample images from each cluster in a grid (samples_per_cluster x n_clusters).
    
    Args:
        images (np.ndarray): Array of images (N, H, W, 3).
        labels (np.ndarray): Cluster labels for each image.
        n_clusters (int): Number of clusters.
        samples_per_cluster (int): Number of random samples per cluster.
    """
    plt.figure(figsize=(3 * n_clusters, 3 * samples_per_cluster))

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            print(f"Cluster {cluster_id} has no samples.")
            continue

        selected_indices = np.random.choice(cluster_indices, size=min(samples_per_cluster, len(cluster_indices)), replace=False)

        for i, idx in enumerate(selected_indices):
            plt_idx = i * n_clusters + cluster_id + 1  # Row-major order
            plt.subplot(samples_per_cluster, n_clusters, plt_idx)
            plt.imshow(images[idx].astype(np.uint8))
            plt.axis("off")
            if i == 0:
                plt.title(f"Cluster {cluster_id}", fontsize=12)

    plt.suptitle(f"Sample Images from Clusters ({algorithm_name})", fontsize=16)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/cluster_examples_{algorithm_name}.png", dpi=300)
    #plt.show()
    plt.close()
