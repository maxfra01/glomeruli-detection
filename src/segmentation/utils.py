import cv2
import numpy as np
from pathlib import Path

def gen_binary_mask(glomerulus_path: Path) -> None:
    """
    Generate a binary mask from the glomerulus image.
    The mask is saved in the same directory as the glomerulus image.
    Args:
        glomerulus_path (Path): Path to the glomerulus image.
    """
    
    base_id = glomerulus_path.stem # es: glomerulus_001
    output_path = glomerulus_path.parent # es: data/RECHERCHE-003
    
    img = cv2.imread(glomerulus_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    output_path = output_path / f"{base_id}_mask.png"
    cv2.imwrite(output_path , mask)

if __name__ == "__main__":
    glomerulus_path = Path("data/RECHERCHE-003/glomerulus_000.png")
    
    gen_binary_mask(glomerulus_path)