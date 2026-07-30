"""
Vertex Detection Module

This module identifies ALL vertices in the image, points where 3 or more cells meet.
Uses the same algorithm as rosette detection but with a lower threshold.
These vertices represent the basic junctions between cells.
"""

import numpy as np

from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize


def find_vertices_topological(mask, valid_cells, vertex_radius, min_cells_for_vertex=5):
    """
    Identify vertices where multiple cells meet, using mask topology
    instead of geometric point-sampling + radius search.

    Approach (all vectorized, no per-point Python loop over cells):
      1. Extract boundary pixels directly from the label mask.
      2. Skeletonize to thin boundaries to single-pixel width.
      3. Convolve to count skeleton-neighbors per pixel; pixels with
         3+ neighboring skeleton branches are junction candidates
         (this is the standard definition of a "branch point" in a
         skeletonized boundary map).
      4. Connected-component label + dilate the junction candidates so
         nearby junction pixels merge directly into one vertex cluster
         (this replaces the separate clustering pass entirely).
      5. For each vertex centroid, look at a small window in the
         *original* label mask to recover which cell ids are actually
         present there (this is the only place we touch per-vertex
         data, and there are only as many vertices as actually exist,
         not thousands of sampled candidate points).

    Args:
        mask: Segmentation mask array (0 = background, >0 = cell id)
        valid_cells: iterable of cell ids considered valid
        vertex_radius: radius (pixels) for the window used to look up
            cell ids in the original mask around each vertex location
        min_cells_for_vertex: minimum distinct valid cells required to
            keep a vertex (default: 5, matching the original default)

    Returns:
        List of vertex dictionaries containing location, cells, and
        num_cells (same shape as the original function's output)
    """
    print("\n" + "=" * 70)
    print("STEP 3 (topological): IDENTIFYING VERTICES FROM MASK BOUNDARIES")
    print("=" * 70)

    valid_cells_set = set(valid_cells)

    pad = int(np.ceil(vertex_radius)) + 2
    padded_mask = np.pad(mask, pad, mode="constant", constant_values=0)
    # 1. Boundary pixels straight from the mask - no precomputed
    #    per-cell boundary coordinate lists needed at all
    boundaries = find_boundaries(padded_mask, mode="inner")

    # 2. Thin to a single-pixel-wide skeleton
    skeleton = skeletonize(boundaries)

    # 3. Count skeleton neighbors per pixel via one convolution over
    #    the whole image
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndi.convolve(
        skeleton.astype(np.uint8), kernel, mode="constant", cval=0
    )

    # Junction candidates: skeleton pixels with 3+ neighboring branches
    junction_candidates = skeleton & (neighbor_count >= 3)

    print(f"Found {int(junction_candidates.sum())} candidate junction pixels")

    if not junction_candidates.any():
        print("No junction candidates found.")
        return []

    # 4. Merge nearby junction pixels into clusters directly via
    #    dilation + connected-component labeling (replaces the
    #    separate O(V^2)/KD-tree clustering pass)
    structure = np.ones((3, 3))
    dilation_iters = max(1, int(round(vertex_radius)))
    dilated = ndi.binary_dilation(
        junction_candidates, structure=structure, iterations=dilation_iters
    )
    labeled_junctions, num_junctions = ndi.label(dilated, structure=structure)

    objs = ndi.find_objects(labeled_junctions)

    centroids = ndi.center_of_mass(
        junction_candidates, labeled_junctions, range(1, num_junctions + 1)
    )

    print(f"After merging: {num_junctions} unique vertex clusters")

    # 5. For each centroid, look at a small window in the original mask
    #    to recover which valid cell labels are actually present there
    vertices = []
    h, w = padded_mask.shape
    margin = int(np.ceil(vertex_radius))

    for i, (cy, cx) in enumerate(centroids, start=1):
        slc = objs[i - 1]

        y0 = max(0, slc[0].start - margin)
        y1 = min(h, slc[0].stop + margin)
        x0 = max(0, slc[1].start - margin)
        x1 = min(w, slc[1].stop + margin)

        window = padded_mask[y0:y1, x0:x1]
        labels_here = np.unique(window)
        labels_here = labels_here[labels_here > 0]  # drop background

        cells_near_point = [int(c) for c in labels_here if c in valid_cells_set]

        if len(cells_near_point) >= min_cells_for_vertex:
            cy_i, cx_i = int(round(cy)), int(round(cx))
            vertices.append(
                {
                    "location": (
                        cx - pad,
                        cy - pad,
                    ),  # (x, y), matching original convention
                    "cells": cells_near_point,
                    "num_cells": len(cells_near_point),
                }
            )

    print(f"Final vertex count: {len(vertices)}")

    return vertices
