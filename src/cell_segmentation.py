"""
Cell Segmentation Module

This module handles loading images and detecting individual cells using CellPose.
It filters cells by size and extracts their properties including centroids, areas,
and binary masks.
"""

import numpy as np
from cellpose import models, io
from cellpose.io import imread
import os
from scipy import ndimage

io.logger_setup()


def load_and_validate_images(file_paths):
    """
    Load images from file paths and validate they exist.

    Args:
        file_paths: List of file paths to image files

    Returns:
        List of loaded image arrays (numpy arrays)
    """
    imgs = []
    for f in file_paths:
        if os.path.exists(f):
            imgs.append(imread(f))
            print(f"Loaded image: {f}")
            print(f"Image shape: {imread(f).shape}")
        else:
            print(f"File not found: {f}")

    return imgs


def detect_cells(imgs, diameter, min_area, max_area):
    """
    Detect and filter individual cells using CellPose segmentation.

    Args:
        imgs: List of image arrays to process
        diameter: Estimated cell diameter for CellPose (pixels)
        min_area: Minimum area threshold for valid cells (pixels)
        max_area: Maximum area threshold for valid cells (pixels)

    Returns:
        Tuple of (mask, img, valid_cells, cell_properties) where:
        - mask: Integer array with cell labels
        - img: Original image
        - valid_cells: List of valid cell IDs
        - cell_properties: Dictionary mapping cell_id to properties (centroid, area, mask)
    """
    print("\n" + "=" * 70)
    print("STEP 1: DETECTING INDIVIDUAL CELLS")
    print("=" * 70)

    # Initialize CellPose model with GPU acceleration
    model = models.CellposeModel(gpu=True, model_type="cyto2", use_bfloat16=False)

    # Run CellPose segmentation
    masks, flows, styles = model.eval(
        imgs, diameter=diameter, channels=[0, 0], flow_threshold=0, do_3D=False
    )

    mask = masks[0]
    img = imgs[0]

    # Get all detected cell IDs
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]  # Remove background (ID 0)

    valid_cells = []
    cell_properties = {}

    # Filter cells by area and compute properties
    for cell_id in cell_ids:
        cell_mask = mask == cell_id
        area = np.sum(cell_mask)

        # Only keep cells within size thresholds
        if min_area <= area <= max_area:
            # Calculate centroid (center of mass)
            centroid = ndimage.center_of_mass(cell_mask)
            valid_cells.append(cell_id)
            cell_properties[cell_id] = {
                "centroid": centroid,
                "area": area,
                "mask": cell_mask,
            }

    print(f"Total objects detected: {len(cell_ids)}")
    print(f"Valid cells (after size filtering): {len(valid_cells)}")

    return mask, img, valid_cells, cell_properties


def extract_cell_boundaries(valid_cells, cell_properties):
    """
    Extract boundary pixels for each cell using morphological erosion.

    Args:
        valid_cells: List of valid cell IDs
        cell_properties: Dictionary with cell properties including masks

    Returns:
        Dictionary mapping cell_id to array of boundary pixel coordinates
    """
    print("\n" + "=" * 70)
    print("STEP 2: FINDING CELL BOUNDARIES AND CONTACT POINTS")
    print("=" * 70)

    from scipy.ndimage import binary_erosion

    cell_boundaries = {}
    for cell_id in valid_cells:
        cell_mask = cell_properties[cell_id]["mask"]

        # Boundary is the difference between mask and its erosion
        eroded = binary_erosion(cell_mask)
        boundary = cell_mask & ~eroded
        boundary_coords = np.column_stack(np.where(boundary))
        cell_boundaries[cell_id] = boundary_coords

    print(f"Extracted boundaries for {len(cell_boundaries)} cells")

    return cell_boundaries
