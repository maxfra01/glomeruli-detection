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

def parse_xml(xml_path):
    """
    This function parses the XML file and extracts the coordinates of the annotations.
    It returns a list of elements, each containing a list of coordinates for a glomerulus.
    Each annotation is a tuple of (x, y) values.
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
    slide = openslide.OpenSlide(str(svs_path))
    width, height = slide.level_dimensions[level]

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

            # Skip empty masks if requested
            if skip_empty and np.count_nonzero(mask_crop) == 0:
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

            print(f"Saved: {patch_path.name}")
            tile_id += 1


            
if __name__ == "__main__":
    
    slides = sorted(Path(WSI_PATH).glob("*.svs"))

    for svs_file in tqdm(slides):
        base_id = svs_file.stem
        xml_file = Path(WSI_PATH) / f"{base_id}.xml"

        if xml_file.exists():
            print(f"Processing: {base_id}")
            case_output = Path(OUTPUT_PATH) / base_id
            extract_regions(svs_file, xml_file, case_output, tile_size=2000, step_size=2000, level=0, skip_empty=True)
        else:
            print(f"XML not found for {base_id}, skipping.")
