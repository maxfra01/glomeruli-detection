import openslide
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import os
from tqdm import tqdm

WSI_PATH = "../glomeruli_grading/"
OUTPUT_PATH = "data_clustering/"
LEVEL = 0  # (0 = full res)

def parse_xml(xml_path):
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

def crop_glomeruli(svs_path: Path, xml_path: Path, output_dir: Path, level=0):
    slide = openslide.OpenSlide(str(svs_path))
    annotations = parse_xml(xml_path)
    slide_name = svs_path.stem

    output_img_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    for idx, coords in enumerate(tqdm(annotations, desc=f"{slide_name}")):
        coords = np.array(coords, dtype=np.int32)

        # Bounding box
        x_min, y_min = coords.min(axis=0).astype(int)
        x_max, y_max = coords.max(axis=0).astype(int)

        width = x_max - x_min
        height = y_max - y_min

        # Read patch from slide
        patch = slide.read_region((x_min, y_min), level, (width, height)).convert("RGB")
        patch_np = np.array(patch)

        # Adjust coords to local (crop) coordinates
        local_coords = coords - [x_min, y_min]

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [local_coords], 255)

        # Save
        img_path = output_img_dir / f"{slide_name}_glom_{idx}.png"
        mask_path = output_mask_dir / f"{slide_name}_glom_{idx}_mask.png"

        cv2.imwrite(str(img_path), patch_np)
        cv2.imwrite(str(mask_path), mask)

def main():
    svs_files = sorted(Path(WSI_PATH).glob("*.svs"))
    print(f"Found {len(svs_files)} WSI")

    for svs_file in svs_files:
        xml_file = svs_file.with_suffix(".xml")
        if xml_file.exists():
            print(f"Processing: {svs_file.name}")
            output_case_dir = Path(OUTPUT_PATH) / svs_file.stem
            crop_glomeruli(svs_file, xml_file, output_case_dir, level=LEVEL)
        else:
            print(f"XML missing for {svs_file.name}")

if __name__ == "__main__":
    main()
