import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from skimage.measure import regionprops, label
from skimage.transform import resize

from dataset import get_dataset
from manifold_analysis import ManifoldAnalyzer

DATA_DIR = "./data/"
MODEL_PATH = "./snapshots/model_lr0.0001_bs64_wd0.0001_ar5.keras"  # modello salvato con tf.keras.models.save_model()
INPUT_SHAPE = (384, 384, 3)
IMAGE_SIZE = (384, 384)
LAYER_NAME = "block3_conv4" # Other possibilities: 'block5_conv3', 'conv2d_4'

#Load dataset
dataset, _, _ = get_dataset(DATA_DIR, crop_size=(384, 384))

#Load model 
if os.path.exists(MODEL_PATH):
    print(f"Caricamento modello da {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.BinaryIoU(name='mean_iou', threshold=0.5)
        ]
    )
else:
    print(f"Modello non trovato in {MODEL_PATH}. Esco.")
    exit(1)

#Initialize ManifoldAnalyzer and setup feature extractor for the specified layer
analyzer = ManifoldAnalyzer(model)
feature_extractor = analyzer.feature_extractors[LAYER_NAME]

# Containers
visual_features = []
morph_features = []

# Iterate on the dataset to extract glomeruli 
# The clustering process is at glomerulus-level not at tile-level
for image, mask in dataset:
    image_np = image.numpy()
    mask_np = np.squeeze(mask.numpy())
    labeled_mask = label(mask_np)

    # Each region corresponds to a glomerulus
    for region in regionprops(labeled_mask):
        minr, minc, maxr, maxc = region.bbox
        glom_img = image_np[minr:maxr, minc:maxc]
        glom_mask = (labeled_mask[minr:maxr, minc:maxc] == region.label).astype(np.uint8)

        # Resize to uniform the input
        glom_img_resized = resize(glom_img, IMAGE_SIZE, preserve_range=True, anti_aliasing=True)
        glom_mask_resized = resize(glom_mask, IMAGE_SIZE, preserve_range=True, anti_aliasing=True)

        # Extract visual features
        glom_tensor = tf.convert_to_tensor(glom_img_resized[np.newaxis, ...], dtype=tf.float32)
        feat = analyzer.extract_from_tensor(glom_tensor, LAYER_NAME)
        visual_features.append(feat.numpy().flatten())

        # Extract morphological features
        props = regionprops(glom_mask_resized.astype(np.uint8))
        if props:
            p = props[0]
            area = p.area
            perimeter = p.perimeter
            eccentricity = p.eccentricity

            # Healty glomeruli tend to have a more compact shape
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            # Sclerosed glomeruli tend to have lower solidity 
            solidity = p.solidity
        else:
            area = perimeter = eccentricity = compactness = solidity = 0
        morph_features.append([area, perimeter, eccentricity, compactness, solidity])

# Preprocessing
vis_features = np.array(visual_features)
morph_features = np.array(morph_features)
combined_features = np.concatenate([vis_features, morph_features], axis=1)
normalized_features = StandardScaler().fit_transform(combined_features)

#Dimensionality reduction (optional, but clustering works better with fewer dimensions)
pca_20 = PCA(n_components=20)
features_20d = pca_20.fit_transform(normalized_features)

#----- Clustering ------
#Method 1: KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(features_20d)

#Method 2: DBSCAN
dbscan = DBSCAN(eps=0.7, min_samples=3)
dbscan_labels = dbscan.fit_predict(features_20d)

#Method 3: Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(features_20d)

#Method 4: Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(features_20d)

# PCA for visualization
features_2d = PCA(n_components=2).fit_transform(features_20d)

# Visualization
def plot_clusters(data, labels, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_clusters(features_2d, kmeans_labels, "KMeans Clustering")
plot_clusters(features_2d, dbscan_labels, "DBSCAN Clustering")
plot_clusters(features_2d, agglo_labels, "Agglomerative Clustering")
plot_clusters(features_2d, gmm_labels, "Gaussian Mixture Clustering")

# Evaluation metrics
print(f"Silhouette Score (KMeans): {silhouette_score(features_20d, kmeans_labels):.3f}")
print(f"Calinski-Harabasz Score (KMeans): {calinski_harabasz_score(features_20d, kmeans_labels):.2f}")

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
if n_clusters >= 2:
    print(f"Silhouette Score (DBSCAN): {silhouette_score(features_20d, dbscan_labels):.3f}")
    print(f"Calinski-Harabasz Score (DBSCAN): {calinski_harabasz_score(features_20d, dbscan_labels):.2f}")
else:
    print("DBSCAN found less than 2 clusters, Silhouette and Calinski-Harabasz Scores cannot be computed.")
    

print(f"Silhouette Score (Agglomerative): {silhouette_score(features_20d, agglo_labels):.3f}")
print(f"Calinski-Harabasz Score (Agglomerative): {calinski_harabasz_score(features_20d, agglo_labels):.2f}")

print(f"Silhouette Score (GMM): {silhouette_score(features_20d, gmm_labels):.3f}")
print(f"Calinski-Harabasz Score (GMM): {calinski_harabasz_score(features_20d, gmm_labels):.2f}")