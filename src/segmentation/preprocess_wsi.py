import openslide
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import math
from tqdm import tqdm


WSI_PATH = "glomeruli_grading/"
OUTPUT_PATH = "data/"
TILE_SIZE = 2000 # Size of the tiles to extract
STEP_SIZE = 2000 # Step size for sliding window
SKIP_EMPTY = True # Whether to skip tiles with no tissue

def parse_xml(xml_path):
    """
    This function parses the XML file and extracts the coordinates of the annotations.
    It returns a list of elements, each containing a list of coordinates for a glomerulus.
    """
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    
    for annotation in root.findall('.//Annotation'):
        coords = []
        for coord in annotation.find('.//Coordinates').findall('.//Coordinate'):
            x = float(coord.attrib['X'])
            y = float(coord.attrib['Y'])
            coords.append((x, y))
        annotations.append(np.array(coords, dtype=np.int32))
    return annotations

def extract_regions(
    svs_path: Path,
    xml_path: Path,
    output_dir: Path,
    tile_size: int = 1024,
    step_size: int = 1024,
    level: int = 0,
    skip_empty: bool = True
):
    """
    Tiles a whole-slide image (WSI) and generates binary masks from XML annotations.
    Handles edge padding and skips background tiles if specified.

    Args:
        svs_path (Path): Path to the .svs file.
        xml_path (Path): Path to the corresponding XML annotation file.
        output_dir (Path): Directory to save image tiles and mask tiles.
        tile_size (int): Size of each square tile (default: 1024).
        step_size (int): Sliding step between tiles (default: 1024).
        level (int): WSI resolution level to extract from (default: 0).
        skip_empty (bool): Skip tiles with no glomeruli (default: True).
    """
    slide = openslide.OpenSlide(str(svs_path))
    width, height = slide.level_dimensions[level]
    print(f"Width: {width}, Height: {height}")

    annotations = parse_xml(xml_path)
    slide_name = svs_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    full_mask = np.zeros((height, width), dtype=np.uint8) # Empty mask
    for coords in annotations:
        cv2.fillPoly(full_mask, [coords.astype(np.int32)], 255)

    # Number of tiles in each direction
    num_tiles_x = math.ceil(width / step_size)
    num_tiles_y = math.ceil(height / step_size)

    tile_id = 0
    for row in range(num_tiles_y):
        for col in range(num_tiles_x):
            x = col * step_size
            y = row * step_size

            tile_w = min(tile_size, width - x)
            tile_h = min(tile_size, height - y)

            if tile_w <= 0 or tile_h <= 0:
                continue

            # Read region and convert to RGB numpy
            patch = slide.read_region((x, y), level, (tile_w, tile_h)).convert("RGB")
            patch_np = np.array(patch)

            # Crop corresponding mask
            mask_crop = full_mask[y:y + tile_h, x:x + tile_w]

            # Skip tiles with no tissue if specified
            if skip_empty and np.mean(patch_np) > 240:  # Assuming white areas have high mean pixel values
                continue

            # Pad patch and mask if smaller than tile size
            patch_padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            mask_padded = np.zeros((tile_size, tile_size), dtype=np.uint8)
            patch_padded[:tile_h, :tile_w] = patch_np
            mask_padded[:tile_h, :tile_w] = mask_crop

            # Save files
            patch_path = output_dir / f"{slide_name}_tile_{tile_id}.png"
            mask_path = output_dir / f"{slide_name}_tile_{tile_id}_mask.png"
            cv2.imwrite(str(patch_path), patch_padded)
            cv2.imwrite(str(mask_path), mask_padded)

            #print(f"Saved: {patch_path.name}")
            tile_id += 1

     
if __name__ == "__main__":
    
    slides = sorted(Path(WSI_PATH).glob("*.svs"))

    for svs_file in tqdm(slides, desc="Processing WSI files"):
        base_id = svs_file.stem
        xml_file = Path(WSI_PATH) / f"{base_id}.xml"

        if xml_file.exists():
            print(f"Processing: {base_id}")
            case_output = Path(OUTPUT_PATH) / base_id
            extract_regions(svs_file, xml_file, case_output, tile_size=TILE_SIZE, step_size=STEP_SIZE, level=0, skip_empty=SKIP_EMPTY)
        else:
            print(f"XML not found for {base_id}, skipping.")

    files = sorted([
        os.path.relpath(os.path.join(root, f), start=OUTPUT_PATH)
        for root, _, filenames in os.walk(OUTPUT_PATH)
        for f in filenames if f.endswith("mask.png")
    ])
    count_nonzero = 0
    for f in tqdm(files, desc="Counting non-empty masks"):
        mask = cv2.imread(os.path.join(OUTPUT_PATH, f), cv2.IMREAD_GRAYSCALE)
        if np.count_nonzero(mask) > 0:
            count_nonzero += 1
    print(f"Total number of masks: {len(files)}")
    print(f"Number of non-empty masks: {count_nonzero}")
    print(f"Number of empty masks: {len(files) - count_nonzero}")
    print(f"Percentage of empty masks: {(len(files) - count_nonzero) / len(files) * 100:.2f}%")
    print(f"Percentage of non-empty masks: {count_nonzero / len(files) * 100:.2f}%")
    print("Preprocess Done!")