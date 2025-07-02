import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy import stats
from skimage import measure
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.vgg19 import preprocess_input
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from scipy import stats
import os
from umap import UMAP

IMG_SIZE = 384

def extract_basic_shape_features(mask):
    """Extract basic geometric shape features"""
    # Get region properties
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)
    
    if len(regions) == 0:
        return np.zeros(14)  # Return zeros if no regions found
    
    # Take the largest region (should be the glomerulus)
    region = max(regions, key=lambda x: x.area)
    
    # Basic measurements
    area = region.area
    perimeter = region.perimeter
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length
    
    # Derived measurements
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
    extent = region.extent
    solidity = region.solidity
    compactness = perimeter ** 2 / (4 * np.pi * area) if area > 0 else 0
    convexity = region.convex_area / area if area > 0 else 0
    eccentricity = region.eccentricity
    equivalent_diameter = region.equivalent_diameter
    orientation = region.orientation
    
    # Feret diameter (max distance between any two boundary points)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        feret_diameter_max = calculate_feret_diameter(cnt)
    else:
        feret_diameter_max = 0
    
    return np.array([
        area, perimeter, circularity, aspect_ratio, extent, solidity,
        compactness, convexity, eccentricity, equivalent_diameter,
        major_axis, minor_axis, orientation, feret_diameter_max
    ])

def extract_boundary_features(mask):
    """Extract boundary complexity features"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(4)
    
    cnt = max(contours, key=cv2.contourArea)
    
    # Roundness (4π*area/perimeter²)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    roundness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Roughness (perimeter²/area)
    roughness = perimeter ** 2 / area if area > 0 else 0
    
    # Fractal dimension (simplified box-counting)
    fractal_dimension = calculate_fractal_dimension(mask)
    
    # Boundary irregularity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    boundary_irregularity = (hull_area - area) / hull_area if hull_area > 0 else 0
    
    return np.array([roundness, roughness, fractal_dimension, boundary_irregularity])

def extract_intensity_features(image, mask):
    """Extract intensity-based features from the glomerulus region"""
    # Apply mask to get glomerulus pixels only
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    glom_pixels = masked_image[mask > 0]
    
    if len(glom_pixels) == 0:
        return np.zeros(4)
    
    # Convert to grayscale if needed
    if len(glom_pixels.shape) > 1 and glom_pixels.shape[-1] == 3:
        glom_pixels = cv2.cvtColor(glom_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).flatten()
    
    # Statistical moments
    mean_intensity = np.mean(glom_pixels)
    std_intensity = np.std(glom_pixels)
    skewness_intensity = stats.skew(glom_pixels)
    kurtosis_intensity = stats.kurtosis(glom_pixels)
    
    return np.array([mean_intensity, std_intensity, skewness_intensity, kurtosis_intensity])

def extract_texture_features(image, mask):
    """Extract GLCM texture features"""
    # Apply mask and convert to grayscale
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.copy()
    
    gray_masked = cv2.bitwise_and(gray_image, gray_image, mask=mask.astype(np.uint8))
    
    # Calculate GLCM
    try:
        # Normalize to 0-255 and convert to uint8
        if gray_masked.max() == gray_masked.min():
            gray_norm = np.zeros_like(gray_masked, dtype=np.uint8)
        else:
            gray_norm = ((gray_masked - gray_masked.min()) / (gray_masked.max() - gray_masked.min()) * 255).astype(np.uint8)
        
        # Calculate GLCM for different angles
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gray_norm, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Extract Haralick features
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        
    except:
        # Fallback if GLCM calculation fails
        contrast = dissimilarity = homogeneity = energy = 0
    
    return np.array([contrast, dissimilarity, homogeneity, energy])

def calculate_feret_diameter(contour):
    """Calculate maximum Feret diameter"""
    if len(contour) < 2:
        return 0
    
    points = contour.reshape(-1, 2)
    max_dist = 0
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            max_dist = max(max_dist, dist)
    
    return max_dist

def calculate_fractal_dimension(mask):
    """Calculate fractal dimension using box-counting method (simplified)"""
    # Simple fractal dimension approximation
    # Count non-zero pixels at different scales
    scales = [1, 2, 4, 8]
    counts = []
    
    for scale in scales:
        # Downsample the mask
        h, w = mask.shape
        new_h, new_w = h // scale, w // scale
        if new_h > 0 and new_w > 0:
            downsampled = cv2.resize(mask.astype(np.float32), (new_w, new_h))
            count = np.sum(downsampled > 0)
            counts.append(count)
        else:
            counts.append(0)
    
    # Calculate fractal dimension
    if len(counts) > 1 and np.std(counts) > 0:
        log_scales = np.log(scales[:len(counts)])
        log_counts = np.log(np.array(counts) + 1)  # +1 to avoid log(0)
        # Linear regression slope gives fractal dimension
        try:
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            return abs(slope)
        except:
            return 1.0
    else:
        return 1.0

def extract_all_features(image, mask):
    """Extract all morphological features for a single glomerulus"""
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Extract different types of features
    shape_features = extract_basic_shape_features(mask)
    boundary_features = extract_boundary_features(mask)
    intensity_features = extract_intensity_features(image, mask)
    texture_features = extract_texture_features(image, mask)
    
    # Combine all features
    all_features = np.concatenate([
        shape_features, boundary_features, 
        intensity_features, texture_features
    ])
    
    return all_features

def extract_features_from_segmentation_model(images, model_path, layer_name="block4_pool"):
    """
    Extract features from our trained segmentation model
    """
    if os.path.exists(model_path):
        print(f"Loading trained segmentation model from {model_path}...")
        model = load_model(model_path, compile=False)
        
        # Extract features from a specific layer
        try:
            feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        except:
            # If layer doesn't exist, use an intermediate layer from encoder
            print(f"Layer {layer_name} not found, using model output...")
            # For segmentation models, we might want to use encoder features
            # You might need to adjust this based on your model architecture
            feature_model = model
            
        features = feature_model.predict(images, batch_size=32, verbose=1)
        features_flat = features.reshape((features.shape[0], -1))
        
        print(f"Segmentation model features shape: {features_flat.shape}")
        return features_flat
    else:
        print(f"Segmentation model not found at {model_path}")
        return None

def extract_vgg19_base_features(images, layer_name="block4_pool"):
    """
    Extract features from base VGG19 (pretrained on ImageNet)
    """
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
    
    # Preprocess for VGG19
    images_processed = preprocess_input(images.copy())
    
    features = feature_model.predict(images_processed, batch_size=32, verbose=1)
    features_flat = features.reshape((features.shape[0], -1))
    
    print(f"VGG19 base features shape: {features_flat.shape}")
    return features_flat

def extract_morphological_features(images, masks):
    """
    Extract morphological features from all glomeruli
    """
    morph_features = []
    
    print("Extracting morphological features...")
    for i, (image, mask) in enumerate(zip(images, masks)):
        if i % 100 == 0:
            print(f"Processing {i}/{len(images)}...")
        
        # Convert image to uint8 if needed
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
            
        # Convert mask to binary
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        features = extract_all_features(image_uint8, mask_binary)
        morph_features.append(features)
    
    morph_features = np.array(morph_features)
    print(f"Morphological features shape: {morph_features.shape}")
    return morph_features

def combine_features(vgg19_features, seg_features, morph_features, balance_factor=10):
    """
    Combine different types of features with balanced weighting
    """
    print("Combining features...")
    
    # Normalize each feature type separately
    scaler_vgg19 = StandardScaler()
    scaler_seg = StandardScaler()
    scaler_morph = StandardScaler()
    
    umap_vgg19 = UMAP(n_components=75)
    vgg19_reduced = umap_vgg19.fit_transform(vgg19_features)

    vgg19_norm = scaler_vgg19.fit_transform(vgg19_reduced)
    morph_norm = scaler_morph.fit_transform(morph_features)
    
    # Handle segmentation features if available
    if seg_features is not None:
        umap_seg = UMAP(n_components=75)
        seg_reduced = umap_seg.fit_transform(seg_features)

        seg_norm = scaler_seg.fit_transform(seg_reduced)
        # Balance the features by repeating morphological features
        # This gives more weight to domain-specific features
        morph_balanced = np.repeat(morph_norm, balance_factor, axis=1)
        
        # Combine all features
        combined_features = np.concatenate([vgg19_norm, seg_norm, morph_balanced], axis=1)
        print(f"Combined features shape (VGG19 + Segmentation + Morphological): {combined_features.shape}")
    else:
        # Only VGG19 + Morphological
        morph_balanced = np.repeat(morph_norm, balance_factor, axis=1)
        combined_features = np.concatenate([vgg19_norm, morph_balanced], axis=1)
        print(f"Combined features shape (VGG19 + Morphological): {combined_features.shape}")
    
    return combined_features
