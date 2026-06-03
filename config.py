"""
Shared Configuration for Rosette Detection Pipeline

This file contains all adjustable parameters and file paths used across
the rosette detection modules and tests.

Author: Rosette Identification Team
Date: November 2025
"""

# ============================================================================
# DETECTION PARAMETERS
# These parameters control the detection sensitivity and filtering criteria
# ============================================================================
CELL_DIAMETER = 30  # Estimated cell diameter for CellPose detection (pixels)
MIN_CELL_AREA = 100  # Minimum area to count as valid cell (pixels)
MAX_CELL_AREA = 5000  # Maximum area to count as valid cell (pixels)
VERTEX_RADIUS = 15  # Search radius for cells meeting at a vertex (pixels)
MIN_CELLS_FOR_ROSETTE = 5  # Minimum cells required to form a rosette

# ============================================================================
# FILE PATHS
# ============================================================================
DATA_DIR = "data/"
IMAGE_FILE = "test_image_1.png"
OUTPUT_DIR = "output/"
VISUALIZATION_DIR = "output/visualizations/"
DATA_OUTPUT_DIR = "output/data/"
