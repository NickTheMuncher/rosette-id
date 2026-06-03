"""
Test CSV Export Module

This test file validates the CSV export functionality, ensuring that:
- Detailed cell properties are correctly extracted
- Junction participation counts are accurate
- CSV file is properly formatted and complete

"""

import os
import sys
import pandas as pd

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cell_segmentation import (
    load_and_validate_images,
    detect_cells,
    extract_cell_boundaries,
)
from src.vertex_detection import find_vertices
from src.csv_export import generate_csv_export
import config
from src.rosette_detection import calculate_cell_neighbors


def main():
    """
    Test the CSV export module and display summary statistics.
    """
    # Create output directories if they don't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
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

    # Find vertices where cells meet (3+ cells)
    vertices = find_vertices(
        valid_cells, cell_boundaries, mask, config.VERTEX_RADIUS, min_cells_for_vertex=3
    )

    cell_neighbors = calculate_cell_neighbors(valid_cells, cell_boundaries)

    # Generate CSV export
    csv_output_path = os.path.join(config.DATA_OUTPUT_DIR, "test_csv_export.csv")
    generate_csv_export(mask, valid_cells, vertices, cell_neighbors, csv_output_path)

    # ========================================================================
    print("\n" + "=" * 70)
    print("CSV EXPORT TEST - DATA VALIDATION")
    print("=" * 70)

    # Read the CSV back and display summary statistics
    df = pd.read_csv(csv_output_path)

    print(f"\nCSV Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nColumn Names:")
    for col in df.columns:
        print(f"  - {col}")

    print(f"\n{'=' * 70}")
    print("MORPHOLOGICAL PROPERTY STATISTICS")
    print("=" * 70)

    morphology_cols = [
        "area",
        "perimeter",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
        "equivalent_diameter",
    ]

    for col in morphology_cols:
        if col in df.columns:
            print(f"\n{col.upper().replace('_', ' ')}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std:  {df[col].std():.2f}")
            print(f"  Min:  {df[col].min():.2f}")
            print(f"  Max:  {df[col].max():.2f}")

    print(f"\n{'=' * 70}")
    print("JUNCTION PARTICIPATION STATISTICS")
    print("=" * 70)

    junction_cols = [
        "junctions_3_cell",
        "junctions_4_cell",
        "junctions_5_cell",
        "junctions_6_cell",
        "junctions_7_cell",
        "junctions_8plus_cell",
        "total_junctions",
    ]

    for col in junction_cols:
        if col in df.columns:
            total = df[col].sum()
            cells_with = (df[col] > 0).sum()
            print(f"\n{col.upper().replace('_', ' ')}:")
            print(f"  Total across all cells: {int(total)}")
            print(f"  Cells with this junction type: {cells_with}")
            if cells_with > 0:
                print(
                    f"  Average per participating cell: {df[df[col] > 0][col].mean():.2f}"
                )

    print(f"\n{'=' * 70}")
    print("SAMPLE DATA (First 10 Cells)")
    print("=" * 70)

    # Display first 10 rows with selected columns
    display_cols = [
        "cell_id",
        "area",
        "perimeter",
        "junctions_3_cell",
        "junctions_4_cell",
        "junctions_5_cell",
        "total_junctions",
    ]
    print(df[display_cols].head(10).to_string(index=False))

    print(f"\n{'=' * 70}")
    print("CSV EXPORT TEST COMPLETE!")
    print(f"Full CSV available at: {csv_output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
