"""
CSV Export Module

This module handles exporting detailed cell properties and junction analysis
to CSV format. For each cell, it extracts:
- Morphological properties from CellPose/regionprops (area, perimeter, etc.)
- Junction participation counts (3-cell, 4-cell, 5-cell, etc. junctions)

"""

import csv
import numpy as np
from skimage import measure


def get_csv_columns():
    """
    Return CSV columns in stable, logical order.
    """
    return [
        # Cell identification
        "cell_id",
        # Basic morphological properties
        "area",
        "perimeter",
        "centroid_x",
        "centroid_y",
        # Shape properties
        "eccentricity",
        "solidity",
        "extent",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "equivalent_diameter",
        "convex_area",
        "filled_area",
        "euler_number",
        "feret_diameter_max",
        # Neighbor Counts
        "num_neighbors",
        # Junction participation counts
        "junctions_3_cell",
        "junctions_4_cell",
        "junctions_5_cell",
        "junctions_6_cell",
        "junctions_7_cell",
        "junctions_8plus_cell",
        "total_junctions",
    ]


def extract_detailed_cell_properties(mask, valid_cells):
    """
    Extract comprehensive morphological properties for each cell using regionprops.

    Args:
        mask: Segmentation mask array with cell labels
        valid_cells: List of valid cell IDs

    Returns:
        Dictionary mapping cell_id to detailed properties dictionary
    """
    print("\n" + "=" * 70)
    print("EXTRACTING DETAILED CELL PROPERTIES")
    print("=" * 70)

    # Use skimage's regionprops to get comprehensive measurements
    regions = measure.regionprops(mask)

    # Create mapping from label to region object
    region_dict = {region.label: region for region in regions}

    detailed_properties = {}

    for cell_id in valid_cells:
        if cell_id not in region_dict:
            continue

        region = region_dict[cell_id]

        # Extract all relevant properties
        detailed_properties[cell_id] = {
            "cell_id": cell_id,
            "area": region.area,
            "perimeter": region.perimeter,
            "centroid_y": region.centroid[0],
            "centroid_x": region.centroid[1],
            "eccentricity": region.eccentricity,
            "solidity": region.solidity,
            "extent": region.extent,
            "major_axis_length": region.major_axis_length,
            "minor_axis_length": region.minor_axis_length,
            "orientation": region.orientation,
            "equivalent_diameter": region.equivalent_diameter_area,
            "convex_area": region.convex_area,
            "filled_area": region.filled_area,
            "euler_number": region.euler_number,
            "feret_diameter_max": region.feret_diameter_max,
        }

    print(f"Extracted detailed properties for {len(detailed_properties)} cells")

    return detailed_properties


def count_junction_participation(vertices, valid_cells):
    """
    Count how many junctions of each size (3, 4, 5, 6, 7, 8+) each cell participates in.

    Args:
        vertices: List of vertex dictionaries (each has 'cells' and 'num_cells')
        valid_cells: List of valid cell IDs

    Returns:
        Dictionary mapping cell_id to junction count dictionary
    """
    print("\n" + "=" * 70)
    print("COUNTING JUNCTION PARTICIPATION FOR EACH CELL")
    print("=" * 70)

    # Initialize junction counts for each cell
    junction_counts = {}
    for cell_id in valid_cells:
        junction_counts[cell_id] = {
            "junctions_3_cell": 0,
            "junctions_4_cell": 0,
            "junctions_5_cell": 0,
            "junctions_6_cell": 0,
            "junctions_7_cell": 0,
            "junctions_8plus_cell": 0,
            "total_junctions": 0,
        }

    # Count junctions for each cell
    for vertex in vertices:
        num_cells = vertex["num_cells"]
        participating_cells = vertex["cells"]

        # Determine junction category
        if num_cells == 3:
            category = "junctions_3_cell"
        elif num_cells == 4:
            category = "junctions_4_cell"
        elif num_cells == 5:
            category = "junctions_5_cell"
        elif num_cells == 6:
            category = "junctions_6_cell"
        elif num_cells == 7:
            category = "junctions_7_cell"
        else:  # 8 or more
            category = "junctions_8plus_cell"

        # Increment count for each participating cell
        for cell_id in participating_cells:
            if cell_id in junction_counts:
                junction_counts[cell_id][category] += 1
                junction_counts[cell_id]["total_junctions"] += 1

    # Print summary statistics
    total_with_junctions = sum(
        1 for counts in junction_counts.values() if counts["total_junctions"] > 0
    )
    print(
        f"Cells with at least one junction: {total_with_junctions} / {len(valid_cells)}"
    )

    # Count distribution
    junction_types = [
        "junctions_3_cell",
        "junctions_4_cell",
        "junctions_5_cell",
        "junctions_6_cell",
        "junctions_7_cell",
        "junctions_8plus_cell",
    ]

    # Calculate both participation counts and estimated actual junctions
    print("\n" + "-" * 70)
    print("JUNCTION STATISTICS")
    print("-" * 70)
    print(f"{'Junction Type':<20} {'Participations':<18} {'Est. Actual Junctions':<25}")
    print("-" * 70)

    junction_sizes = {
        "junctions_3_cell": 3,
        "junctions_4_cell": 4,
        "junctions_5_cell": 5,
        "junctions_6_cell": 6,
        "junctions_7_cell": 7,
        "junctions_8plus_cell": 10,  # Estimate average size for 8+ category
    }

    for junction_type in junction_types:
        participations = sum(
            counts[junction_type] for counts in junction_counts.values()
        )
        avg_size = junction_sizes[junction_type]
        estimated_actual = participations / avg_size if participations > 0 else 0

        # Format the junction type name for display
        display_name = junction_type.replace("junctions_", "").replace("_cell", "-cell")

        print(
            f"{display_name:<20} {participations:<18} {estimated_actual:>6.0f} junctions"
        )

    print("-" * 70)
    print(
        "\nNote: 'Participations' = sum of all cells participating in each junction type"
    )
    print("      'Est. Actual Junctions' = participations ÷ average junction size")
    print("      (e.g., one 5-cell junction = 5 participations)")

    return junction_counts


def build_csv_rows(detailed_properties, junction_counts, cell_neighbors):
    """
    Build CSV rows (list of dicts) for export.

    Each row represents ONE cell with:
    - Morphological properties (area, perimeter, shape metrics)
    - Junction participation counts: how many junctions of each type this cell participates in

    Note: Junction counts are per-cell participations, not total junctions in the image.
    Example: If cell #5 has junctions_3_cell = 2, it means cell #5 participates in
    2 different 3-cell junctions.

    Args:
        detailed_properties: Dictionary of detailed cell properties
        junction_counts: Dictionary of junction participation counts
        cell_neighbors: Dictionary mapping cell_id to number of neighbors

    Returns:
        Tuple of (columns, rows)
    """
    columns = get_csv_columns()
    rows = []

    for cell_id in sorted(detailed_properties.keys()):
        row_data = {}

        # Add detailed properties
        row_data.update(detailed_properties[cell_id])

        # Add number of neighbors
        row_data["num_neighbors"] = cell_neighbors.get(cell_id, 0)

        # Add junction counts
        if cell_id in junction_counts:
            row_data.update(junction_counts[cell_id])
        else:
            for col in columns:
                if col.startswith("junctions_") or col == "total_junctions":
                    row_data[col] = 0

        # Ensure all columns exist (DictWriter will fill missing with blanks otherwise)
        for col in columns:
            row_data.setdefault(col, "")

        # Coerce numpy scalar types to Python scalars for JSON embedding
        for k, v in list(row_data.items()):
            if isinstance(v, np.generic):
                row_data[k] = v.item()

        rows.append(row_data)

    return columns, rows


def export_rows_to_csv(columns, rows, output_path):
    """
    Write CSV rows to disk.
    """
    print("\n" + "=" * 70)
    print("EXPORTING DATA TO CSV")
    print("=" * 70)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV exported successfully: {output_path}")
    print(f"Total cells exported: {len(rows)}")


def generate_csv_export(mask, valid_cells, vertices, cell_neighbors, output_path):
    """
    Main function to generate CSV export with all cell data.

    Coordinates the extraction of detailed properties, junction counts,
    and CSV export in a single pipeline.

    Args:
        mask: Segmentation mask array with cell labels
        valid_cells: List of valid cell IDs
        vertices: List of vertex dictionaries
        cell_neighbors: Dictionary mapping cell_id to number of neighbors (pre-calculated)
        output_path: Path to output CSV file
    """
    # Extract detailed cell properties
    detailed_properties = extract_detailed_cell_properties(mask, valid_cells)

    # Count junction participation
    junction_counts = count_junction_participation(vertices, valid_cells)

    # Export to CSV
    columns, rows = build_csv_rows(detailed_properties, junction_counts, cell_neighbors)
    export_rows_to_csv(columns, rows, output_path)

    print("\n" + "=" * 70)
    print("CSV EXPORT COMPLETE")
    print("=" * 70)
