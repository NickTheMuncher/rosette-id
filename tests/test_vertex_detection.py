""" "
Test Vertex Detection Module

This test file visualizes ALL vertices where cells meet (3+ cells at a junction).
It displays the original image with small red dots marking every vertex to verify
that the model correctly identifies all cell junctions.
"""

import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cell_segmentation import (
    load_and_validate_images,
    detect_cells,
    extract_cell_boundaries,
)
from src.vertex_detection import find_vertices
import config


def main():
    """
    Test the vertex detection module with visualization of ALL vertices.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(config.DATA_OUTPUT_DIR, exist_ok=True)

    image_path = os.path.join(config.DATA_DIR, config.IMAGE_FILE)
    files = [image_path]

    imgs = load_and_validate_images(files)

    if not imgs:
        print("No images loaded. Exiting.")
        return

    mask, img, valid_cells, cell_properties = detect_cells(
        imgs, config.CELL_DIAMETER, config.MIN_CELL_AREA, config.MAX_CELL_AREA
    )

    cell_boundaries = extract_cell_boundaries(valid_cells, cell_properties)

    vertices = find_vertices(
        valid_cells, cell_boundaries, mask, config.VERTEX_RADIUS, min_cells_for_vertex=3
    )

    # ========================================================================
    print("\n" + "=" * 70)
    print("VERTEX DETECTION TEST RESULTS")
    print("=" * 70)
    print(f"Total cells detected: {len(valid_cells)}")
    print(f"Total vertices found: {len(vertices)}")
    print("=" * 70 + "\n")

    vertex_counts = {}
    for vertex in vertices:
        n = vertex["num_cells"]
        vertex_counts[n] = vertex_counts.get(n, 0) + 1

    print("Vertex distribution by number of cells:")
    for num_cells in sorted(vertex_counts.keys()):
        count = vertex_counts[num_cells]
        print(f"  {num_cells} cells: {count} vertices")

    if len(vertices) > 0:
        print("\nFirst 20 vertices:")
        print("-" * 80)
        print(f"{'#':<5} {'Location (x,y)':<20} {'Num Cells':<12} {'Cell IDs':<30}")
        print("-" * 80)

        for idx, vertex in enumerate(vertices[:20], 1):
            cell_ids_str = str(vertex["cells"][:5])
            if len(vertex["cells"]) > 5:
                cell_ids_str = cell_ids_str[:-1] + ", ...]"
            print(
                f"{idx:<5} {str(vertex['location']):<20} {vertex['num_cells']:<12} {cell_ids_str:<30}"
            )

        if len(vertices) > 20:
            print(f"... and {len(vertices) - 20} more vertices")

    # ========================================================================
    print("\n" + "=" * 70)
    print("VERTEX DETECTION TEST VISUALIZATION")
    print("=" * 70)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.imshow(img, cmap="gray")

    for vertex in vertices:
        x, y = vertex["location"]
        ax.plot(x, y, "r.", markersize=6)

    ax.set_title(
        f"All Vertex Detection Results\n({len(vertices)} vertices where 3+ cells meet)",
        fontsize=14,
        weight="bold",
    )
    ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        config.VISUALIZATION_DIR, "test_vertex_detection_results.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved as '{output_path}'")
    plt.show()

    data_path = os.path.join(config.DATA_OUTPUT_DIR, "test_vertex_detection_data.txt")
    with open(data_path, "w") as f:
        f.write("VERTEX DETECTION TEST RESULTS\n")
        f.write(f"{'=' * 80}\n")
        f.write("Parameters:\n")
        f.write(f"  Cell diameter: {config.CELL_DIAMETER}\n")
        f.write(f"  Min cell area: {config.MIN_CELL_AREA}\n")
        f.write(f"  Max cell area: {config.MAX_CELL_AREA}\n")
        f.write(f"  Vertex radius: {config.VERTEX_RADIUS}\n")
        f.write("  Min cells for vertex: 3\n")
        f.write("\nResults:\n")
        f.write(f"  Total cells: {len(valid_cells)}\n")
        f.write(f"  Total vertices: {len(vertices)}\n")
        f.write("\nVertex distribution:\n")
        for num_cells in sorted(vertex_counts.keys()):
            count = vertex_counts[num_cells]
            f.write(f"  {num_cells} cells: {count} vertices\n")
        f.write(f"\n{'=' * 80}\n\n")

        for idx, vertex in enumerate(vertices, 1):
            f.write(f"Vertex {idx}:\n")
            f.write(f"  Location: {vertex['location']}\n")
            f.write(f"  Number of cells: {vertex['num_cells']}\n")
            f.write(f"  Cell IDs: {vertex['cells']}\n\n")

    print(f"Data saved to '{data_path}'")
    print("\n" + "=" * 70)
    print("VERTEX DETECTION TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
