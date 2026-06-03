"""
Rosette Detection Core Modules

This package contains the core functionality for detecting and analyzing
cellular rosettes in microscopy images.

Modules:
- cell_segmentation: Image loading and cell detection using CellPose
- vertex_detection: Identification of vertices where cells meet
- rosette_detection: Clustering of vertices and visualization generation

Author: Rosette Identification Team
Date: November 2025
"""

__version__ = "1.0.0"
__all__ = ["cell_segmentation", "vertex_detection", "rosette_detection"]
