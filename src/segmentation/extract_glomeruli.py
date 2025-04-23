import openslide
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_annotations(xml_path):
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

def extract_regions(svs_path, xml_path, output_dir):
    slide = openslide.OpenSlide(svs_path)
    level = 0  # full resolution
    annotations = parse_annotations(xml_path)

    os.makedirs(output_dir, exist_ok=True)

    for i, contour in enumerate(annotations):
        x, y, w, h = cv2.boundingRect(contour)
        region = slide.read_region((x, y), level, (w, h)).convert("RGB")
        region_np = np.array(region)

        contour_offset = contour - [x, y]

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour_offset], 255)

        masked = cv2.bitwise_and(region_np, region_np, mask=mask)

        out_path = os.path.join(output_dir, f"glomerulus_{i:03d}.png")
        cv2.imwrite(out_path, masked)

        print(f"Saved: {out_path}")

def process_all(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    slides = sorted(input_folder.glob("*.svs"))

    for svs_file in slides:
        base_id = svs_file.stem  # es: RECHERCHE-003
        xml_file = input_folder / f"{base_id}.xml"

        if xml_file.exists():
            print(f"Processing: {base_id}")
            case_output = output_folder / base_id
            extract_regions(str(svs_file), str(xml_file), str(case_output))
        else:
            print(f"XML not found for {base_id}, skip.")

if __name__ == "__main__":
    input_folder = "glomeruli_grading/"
    output_folder = "data/"

    process_all(input_folder, output_folder)
