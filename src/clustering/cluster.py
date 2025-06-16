import os
from utils import load_glomeruli_images, evaluate_clustering, init_log_file
from visual import plot_cluster_examples
from dim_reduction import reduce_dimensionality
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from tensorflow.keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

os.makedirs("results", exist_ok=True)

LOG_FILE = "results/log.txt"
DATA_PATH = "data_clustering/"
IMG_SIZE = 224
NUM_CLUSTERS = 3
MODEL_PATH = "./snapshots/model_lr0.0001_bs64_wd0.0001_ar5.keras" 

init_log_file(LOG_FILE)

images = load_glomeruli_images(DATA_PATH, image_size=(IMG_SIZE, IMG_SIZE)) 
images_proc = preprocess_input(images)  # Preprocess for VGG19
print(images.shape)  # (N, 128, 128, 3)

############################################################################
############################################################################
# Load checkpointed model
#if os.path.exists(MODEL_PATH):
#     print(f"Loading model from {MODEL_PATH} ...")
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# else:
#     print(f"Model not found at {MODEL_PATH}. Exit ...")
#     exit(1)
    
# # Extract deep features
# layer_name = "block1_pool"  # or another, based on what the print shows
# feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
############################################################################
############################################################################

# Load VGG19 without top classifier layers
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# Extract features from the last convolutional block
feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer("block4_pool").output)


features = feature_model.predict(images_proc, batch_size=32, verbose=1)  # (N, H, W, C)
features_flat = features.reshape((features.shape[0], -1))  # Flatten each feature map
print(features_flat.shape)  # (N, H*W*C)

images = load_glomeruli_images(DATA_PATH, image_size=(IMG_SIZE, IMG_SIZE)) 


for method in ["pca", "umap", "isomap"]:
    for n_components in [50, 100, 200]:
        features_reduced = reduce_dimensionality(features_flat, method=method, n_components=n_components)

        # Cluster
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42).fit(features_reduced)
        gmm = GaussianMixture(n_components=NUM_CLUSTERS, random_state=42).fit(features_reduced)

        kmeans_labels = kmeans.predict(features_reduced)
        gmm_labels = gmm.predict(features_reduced)
        

        # Evaluate
        evaluate_clustering(features_reduced, kmeans_labels, method_name=f"KMeans (n={n_components}) + {method.upper()}")
        evaluate_clustering(features_reduced, gmm_labels, method_name=f"GMM (n={n_components}) + {method.upper()}")
        
        plot_cluster_examples(images, kmeans_labels, n_clusters=NUM_CLUSTERS, samples_per_cluster=3, algorithm_name=f"KMeans_{n_components}_{method.upper()}")
        plot_cluster_examples(images, gmm_labels, n_clusters=NUM_CLUSTERS, samples_per_cluster=3, algorithm_name=f"GMM_{n_components}_{method.upper()}")

