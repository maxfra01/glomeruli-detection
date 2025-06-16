from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
import umap

def reduce_dimensionality(features, method="pca", n_components=50, **kwargs):
    """
    Reduce the dimensionality of the features using the specified method.

    Args:
        features (np.ndarray): Feature array of shape (N, D)
        method (str): One of {"pca", "umap", "tsne", "isomap"}
        n_components (int): Number of output dimensions
        kwargs: Additional keyword arguments for the manifold method

    Returns:
        np.ndarray: Transformed features of shape (N, n_components)
    """
    features = StandardScaler().fit_transform(features)

    if method == "pca":
        reducer = PCA(n_components=n_components, **kwargs)
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, **kwargs)
    elif method == "isomap":
        reducer = Isomap(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return reducer.fit_transform(features)