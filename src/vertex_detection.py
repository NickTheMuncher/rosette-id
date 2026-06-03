"""
Vertex Detection Module

This module identifies ALL vertices in the image, points where 3 or more cells meet.
Uses the same algorithm as rosette detection but with a lower threshold.
These vertices represent the basic junctions between cells.
"""

import numpy as np


def find_vertices(
    valid_cells, cell_boundaries, mask, vertex_radius, min_cells_for_vertex=5
):
    """
    Identify ALL vertices where multiple cells meet at a common point.

    This function samples points in the image and checks how many cell boundaries
    are within vertex_radius of each point. Uses the exact same algorithm as the
    original rosette detection, just with min_cells_for_vertex = 3.

    Args:
        valid_cells: List of valid cell IDs
        cell_boundaries: Dictionary mapping cell_id to boundary coordinates
        mask: Segmentation mask array
        vertex_radius: Search radius for nearby cells (pixels)
        min_cells_for_vertex: Minimum cells required to form a vertex (default: 3)

    Returns:
        List of vertex dictionaries containing location, cells, and num_cells
    """
    print("\n" + "=" * 70)
    print("STEP 3: IDENTIFYING ALL VERTICES WHERE CELLS MEET")
    print("=" * 70)

    vertices = []

    # Create a grid of candidate vertex points
    y_coords, x_coords = np.where(mask > 0)
    sample_points = np.column_stack([y_coords, x_coords])

    # Sample every Nth point to speed up computation
    sample_stride = max(1, len(sample_points) // 10000)
    sample_points = sample_points[::sample_stride]

    print(f"Searching {len(sample_points)} candidate vertex locations...")
    print(f"Looking for junctions where {min_cells_for_vertex}+ cells meet...")

    # Check each candidate point
    for point in sample_points:
        y, x = point

        # Find all cells within vertex_radius of this point
        cells_near_point = []
        # Track cells that are very close (true convergence)
        cells_very_close = []

        for cell_id in valid_cells:
            boundaries = cell_boundaries[cell_id]

            # Calculate minimum distance from point to cell boundary
            distances = np.sqrt(np.sum((boundaries - point) ** 2, axis=1))
            min_dist = np.min(distances)

            if min_dist <= vertex_radius:
                cells_near_point.append(cell_id)
                if min_dist <= vertex_radius * 0.5:
                    cells_very_close.append(cell_id)

        # STRICTER CRITERION: Require at least min_cells_for_vertex cells AND
        # at least 2 cells must be VERY close (within 50% radius)
        # This filters out edge-touching points and keeps true convergence points
        if len(cells_near_point) >= min_cells_for_vertex and len(cells_very_close) >= 2:
            vertices.append(
                {
                    "location": (x, y),
                    "cells": cells_near_point,
                    "num_cells": len(cells_near_point),
                }
            )

    print(f"Found {len(vertices)} candidate vertex locations")

    # Cluster nearby vertices
    if len(vertices) == 0:
        return []

    print("Clustering nearby vertices...")

    vertex_locations = np.array([v["location"] for v in vertices])

    # Merge nearby vertices
    merge_distance = vertex_radius * 1.0
    merged_vertices = []
    used = set()

    for i, vertex in enumerate(vertices):
        if i in used:
            continue

        # Find all vertices within merge_distance
        loc = vertex_locations[i]
        distances = np.sqrt(np.sum((vertex_locations - loc) ** 2, axis=1))
        nearby_indices = np.where(distances <= merge_distance)[0]

        # Merge all nearby vertices
        merged_cells = set()
        merged_locations = []
        for idx in nearby_indices:
            merged_cells.update(vertices[idx]["cells"])
            merged_locations.append(vertex_locations[idx])
            used.add(idx)

        # Calculate average location
        avg_location = np.mean(merged_locations, axis=0)

        merged_vertices.append(
            {
                "location": tuple(avg_location.astype(int)),
                "cells": list(merged_cells),
                "num_cells": len(merged_cells),
            }
        )

    print(f"After clustering: {len(merged_vertices)} unique vertices")

    return merged_vertices
