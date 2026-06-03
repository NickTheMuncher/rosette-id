"""
Test Cell Segmentation Module

This test file visualizes the cell segmentation results using matplotlib.
It shows the original image with cell outlines and the segmented cells with
individual labels/colors.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cell_segmentation import load_and_validate_images, detect_cells
import config


def main():
    """
    Test the cell segmentation module with visualization.
    """
    # Create output directories if they don't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(config.DATA_OUTPUT_DIR, exist_ok=True)

    # File paths to process
    image_path = os.path.join(config.DATA_DIR, config.IMAGE_FILE)
    files = [image_path]

    # Load images
    imgs = load_and_validate_images(files)

    if not imgs:
        print("No images loaded. Exiting.")
        return

    # Detect and filter cells
    mask, img, valid_cells, cell_properties = detect_cells(
        imgs, config.CELL_DIAMETER, config.MIN_CELL_AREA, config.MAX_CELL_AREA
    )

    # ========================================================================
    print("\n" + "=" * 70)
    print("CELL SEGMENTATION TEST VISUALIZATION")
    print("=" * 70)

    # Create visualization

    # Create one plot (just the segmented cells)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Original image with cell outlines
    ax.imshow(img, cmap="gray")
    ax.contour(mask, levels=np.unique(mask), colors="cyan", linewidths=0.5)
    ax.set_title(f"Cell Segmentation Results\n({len(valid_cells)} cells detected)")
    ax.axis("off")

    plt.tight_layout()

    # Create two plots side by side
    """ fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Original image with cell outlines
    axes[0].imshow(img, cmap='gray')
    axes[0].contour(mask, levels=np.unique(mask), colors='cyan', linewidths=0.5)
    axes[0].set_title(f'Original Image with Cell Outlines\n({len(valid_cells)} cells detected)')
    axes[0].axis('off')
    
    # 2. All cells segmented with individual colors
    axes[1].imshow(mask, cmap='nipy_spectral')
    axes[1].set_title('Cell Segmentation\n(Each color = individual cell)')
    axes[1].axis('off')
    
    plt.tight_layout() """

    # Save to output directory
    output_path = os.path.join(
        config.VISUALIZATION_DIR, "test_cell_segmentation_results.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved as '{output_path}'")
    plt.show()

    # Save data summary
    data_path = os.path.join(config.DATA_OUTPUT_DIR, "test_cell_segmentation_data.txt")
    with open(data_path, "w") as f:
        f.write("CELL SEGMENTATION TEST RESULTS\n")
        f.write(f"{'=' * 80}\n")
        f.write("Parameters:\n")
        f.write(f"  Cell diameter: {config.CELL_DIAMETER}\n")
        f.write(f"  Min cell area: {config.MIN_CELL_AREA}\n")
        f.write(f"  Max cell area: {config.MAX_CELL_AREA}\n")
        f.write("\nResults:\n")
        f.write(f"  Total valid cells: {len(valid_cells)}\n")
        f.write("\nCell Details:\n")
        f.write(f"{'Cell ID':<10} {'Centroid (y,x)':<25} {'Area (pixels)':<15}\n")
        f.write(f"{'-' * 50}\n")

        for cell_id in valid_cells[:20]:  # Show first 20 cells
            props = cell_properties[cell_id]
            centroid_str = f"({props['centroid'][0]:.1f}, {props['centroid'][1]:.1f})"
            f.write(f"{cell_id:<10} {centroid_str:<25} {props['area']:<15}\n")

        if len(valid_cells) > 20:
            f.write(f"... and {len(valid_cells) - 20} more cells\n")

    print(f"Data saved to '{data_path}'")
    print("\n" + "=" * 70)
    print("CELL SEGMENTATION TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
