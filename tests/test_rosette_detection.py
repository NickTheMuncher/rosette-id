"""
Test Rosette Detection Module

This test file visualizes the complete rosette detection results, showing
rosettes with their vertices marked and cells highlighted. Similar to the
original rosette_detection_test.py format.

Author: Rosette Identification Team
Date: November 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cell_segmentation import (
    load_and_validate_images,
    detect_cells,
    extract_cell_boundaries,
)
from src.vertex_detection import find_vertices
from src.rosette_detection import cluster_vertices
import config


def main():
    """
    Test the rosette detection module with visualization.
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

    # Extract cell boundaries
    cell_boundaries = extract_cell_boundaries(valid_cells, cell_properties)

    # Find vertices where cells meet
    vertices = find_vertices(
        valid_cells,
        cell_boundaries,
        mask,
        config.VERTEX_RADIUS,
        config.MIN_CELLS_FOR_ROSETTE,
    )

    # Cluster nearby vertices into rosettes
    rosettes = cluster_vertices(vertices, config.VERTEX_RADIUS)

    num_rosettes = len(rosettes)

    # ========================================================================
    print("\n" + "=" * 70)
    print("ROSETTE DETECTION TEST RESULTS")
    print("=" * 70)
    print(f"Total cells detected: {len(valid_cells)}")
    print(f"Total rosettes found: {num_rosettes}")
    print("=" * 70 + "\n")

    if num_rosettes > 0:
        print("Rosette Details:")
        print("-" * 80)
        print(f"{'#':<5} {'Center (x,y)':<20} {'Num Cells':<12} {'Cell IDs':<30}")
        print("-" * 80)

        for idx, rosette in enumerate(rosettes, 1):
            cell_ids_str = str(rosette["cells"][:5])  # Show first 5 cell IDs
            if len(rosette["cells"]) > 5:
                cell_ids_str = cell_ids_str[:-1] + ", ...]"
            print(
                f"{idx:<5} {str(rosette['location']):<20} {rosette['num_cells']:<12} {cell_ids_str:<30}"
            )

    # ========================================================================
    print("\n" + "=" * 70)
    print("ROSETTE DETECTION TEST VISUALIZATION")
    print("=" * 70)

    # Rosette visualization - cells in rosettes highlighted
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    rosette_mask = np.zeros_like(mask, dtype=bool)

    for rosette in rosettes:
        for cell_id in rosette["cells"]:
            rosette_mask |= cell_properties[cell_id]["mask"]

    # Create RGB visualization
    if len(img.shape) == 3:
        viz_rgb = img.astype(float) / img.max()
    else:
        viz_rgb = np.stack([img, img, img], axis=-1)
        if img.max() > 1:
            viz_rgb = viz_rgb.astype(float) / img.max()

    # Highlight rosette cells in green
    viz_rgb[rosette_mask, 1] = np.clip(viz_rgb[rosette_mask, 1] + 0.5, 0, 1)

    ax.imshow(viz_rgb)
    ax.contour(mask, levels=np.unique(mask), colors="cyan", linewidths=0.5, alpha=0.5)

    # Mark rosette centers with stars and circles
    for rosette in rosettes:
        x, y = rosette["location"]
        ax.plot(x, y, "r*", markersize=5, markeredgewidth=1, markeredgecolor="white")
        circle = plt.Circle(
            (x, y),
            config.VERTEX_RADIUS,
            color="red",
            fill=False,
            linewidth=1,
            linestyle="--",
        )
        ax.add_patch(circle)

    ax.set_title("Rosettes Highlighted\n(Green = rosette cells, Red * = vertices)")
    ax.axis("off")

    # 4 plots: Original with outlines, Segmentation, Rosette Highlight, Rosette Centers
    """ # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. Original image with cell outlines
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].contour(mask, levels=np.unique(mask), colors='cyan', linewidths=0.5)
    axes[0, 0].set_title(f'Original Image with Cell Outlines\n({len(valid_cells)} cells detected)')
    axes[0, 0].axis('off')
    
    # 2. All cells segmented
    axes[0, 1].imshow(mask, cmap='nipy_spectral')
    axes[0, 1].set_title('Cell Segmentation')
    axes[0, 1].axis('off')
    
    # 3. Rosette visualization - cells in rosettes highlighted
    rosette_mask = np.zeros_like(mask, dtype=bool)
    
    for rosette in rosettes:
        for cell_id in rosette['cells']:
            rosette_mask |= cell_properties[cell_id]['mask']
    
    # Create RGB visualization
    if len(img.shape) == 3:
        viz_rgb = img.astype(float) / img.max()
    else:
        viz_rgb = np.stack([img, img, img], axis=-1)
        if img.max() > 1:
            viz_rgb = viz_rgb.astype(float) / img.max()
    
    # Highlight rosette cells in green
    viz_rgb[rosette_mask, 1] = np.clip(viz_rgb[rosette_mask, 1] + 0.5, 0, 1)
    
    axes[1, 0].imshow(viz_rgb)
    axes[1, 0].contour(mask, levels=np.unique(mask), colors='cyan', linewidths=0.5, alpha=0.5)
    
    # Mark rosette centers with stars and circles
    for rosette in rosettes:
        x, y = rosette['location']
        axes[1, 0].plot(x, y, 'r*', markersize=20, markeredgewidth=2, markeredgecolor='white')
        circle = plt.Circle((x, y), config.VERTEX_RADIUS, color='red', fill=False, linewidth=2, linestyle='--')
        axes[1, 0].add_patch(circle)
    
    axes[1, 0].set_title(f'Rosettes Highlighted\n(Green = rosette cells, Red * = vertices)')
    axes[1, 0].axis('off')
    
    # 4. Rosette centers with labels
    axes[1, 1].imshow(img, cmap='gray')
    for idx, rosette in enumerate(rosettes, 1):
        x, y = rosette['location']
        axes[1, 1].plot(x, y, 'ro', markersize=10, markeredgewidth=2, markeredgecolor='white')
        axes[1, 1].text(x, y-25, f"R{idx}\n({rosette['num_cells']} cells)", 
                       color='red', fontsize=8, ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1, 1].set_title(f'Rosette Centers\n({num_rosettes} rosettes)')
    axes[1, 1].axis('off') """

    plt.tight_layout()

    # Save to output directory
    output_path = os.path.join(
        config.VISUALIZATION_DIR, "test_rosette_detection_results.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved as '{output_path}'")
    plt.show()

    # Save detailed data
    data_path = os.path.join(config.DATA_OUTPUT_DIR, "test_rosette_detection_data.txt")
    with open(data_path, "w") as f:
        f.write("ROSETTE DETECTION TEST RESULTS\n")
        f.write(f"{'=' * 80}\n")
        f.write("Parameters:\n")
        f.write(f"  Cell diameter: {config.CELL_DIAMETER}\n")
        f.write(f"  Min cell area: {config.MIN_CELL_AREA}\n")
        f.write(f"  Max cell area: {config.MAX_CELL_AREA}\n")
        f.write(f"  Vertex radius: {config.VERTEX_RADIUS}\n")
        f.write(f"  Min cells for rosette: {config.MIN_CELLS_FOR_ROSETTE}\n")
        f.write("\nResults:\n")
        f.write(f"  Total cells: {len(valid_cells)}\n")
        f.write(f"  Total rosettes: {num_rosettes}\n")
        f.write(f"\n{'=' * 80}\n\n")

        for idx, rosette in enumerate(rosettes, 1):
            f.write(f"Rosette {idx}:\n")
            f.write(f"  Center: {rosette['location']}\n")
            f.write(f"  Number of cells: {rosette['num_cells']}\n")
            f.write(f"  Cell IDs: {rosette['cells']}\n\n")

    print(f"Data saved to '{data_path}'")
    print("\n" + "=" * 70)
    print("ROSETTE DETECTION TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
