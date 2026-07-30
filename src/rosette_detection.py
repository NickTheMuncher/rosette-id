"""
Rosette Detection Module

This module takes all vertices (from vertex_detection) and identifies rosettes
by filtering for vertices where 5+ cells meet, then clustering nearby vertices
that likely represent the same rosette. It also creates interactive visualizations.
"""

import numpy as np
import json
import base64
from io import BytesIO
from collections import defaultdict
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

import cv2
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely.affinity import scale, rotate
from shapely.strtree import STRtree


def filter_rosette_vertices(vertices, min_cells_for_rosette=5):
    """
    Filter vertices to only include those with enough cells to be rosettes.

    Args:
        vertices: List of all vertex dictionaries
        min_cells_for_rosette: Minimum cells required for a rosette (default: 5)

    Returns:
        List of vertex dictionaries that qualify as potential rosettes
    """
    rosette_vertices = [v for v in vertices if v["num_cells"] >= min_cells_for_rosette]
    print(
        f"Filtered {len(rosette_vertices)} rosette candidates (5+ cells) from {len(vertices)} total vertices"
    )
    return rosette_vertices


def make_cell_polygon(mask, cell_id):
    """
    Build the true polygon outline of a cell from its mask footprint. Used to find the closest points
    between two cells
    """
    cell_mask = (mask == cell_id).astype(np.uint8)
    contours, _ = cv2.findContours(
        cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # guard against noisy/disconnected masks - use the largest contour
    contour = max(contours, key=cv2.contourArea).squeeze(axis=1)  # (N, 2) x,y

    if len(contour) < 3:
        x, y = contour.mean(axis=0)
        return Point(x, y).buffer(0.5)

    polygon = Polygon(contour)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # repair self-intersections from pixel staircasing

    return polygon


def make_cell_elipse(mask, cell_id, scale_factor=1.25):
    """
    Build a bounding elipse for a cell from a cellpose mask
    """

    y, x = np.where(mask == cell_id)
    points = np.stack([x, y], axis=1).astype(float)

    # might be able to use centroid + major/ minor axis instead
    # principal component analysis, finds the natural axis of the cell
    mx, my = points.mean(axis=0)
    centered = points - [mx, my]
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    a = scale_factor * 2 * np.sqrt(eigenvalues[1])  # major axis
    b = scale_factor * 2 * np.sqrt(eigenvalues[0])  # minor axis

    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

    ellipse = rotate(scale(Point(mx, my).buffer(1.0, resolution=64), a, b), angle)

    return ellipse, (mx, my, a, b, angle)


def check_intersection(mask, p1, p2, cell_id, neighbor_id, num_samples=None):
    """
    Check whether the straight line between two cell points passes
    through a third, different cell's mask footprint.
    """
    if num_samples is None:
        # roughly one sample per pixel along the line
        num_samples = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) + 1
        num_samples = max(num_samples, 2)

    xs = np.linspace(p1[0], p2[0], num_samples)
    ys = np.linspace(p1[1], p2[1], num_samples)

    xs_int = np.clip(np.round(xs).astype(int), 0, mask.shape[1] - 1)
    ys_int = np.clip(np.round(ys).astype(int), 0, mask.shape[0] - 1)

    sampled_values = mask[ys_int, xs_int]
    blocking_ids = set(np.unique(sampled_values)) - {0, cell_id, neighbor_id}

    return len(blocking_ids) > 0


def find_cell_neighbors(mask, valid_cells):
    """
    Returns dict mapping each cell_id to a list of neighboring cell_ids, neighbors are defined as cells who's bounding elipses intersect
    """
    candidates_tested = 0
    neighbors_found = 0
    blocked_pairs = 0

    ellipses = {}
    polygons = {}

    for cell_id in valid_cells:
        ellipse, params = make_cell_elipse(mask, cell_id)
        mx, my, a, b, angle = params
        ellipses[cell_id] = ellipse
        polygons[cell_id] = make_cell_polygon(mask, cell_id)

    neighbors = {cell_id: 0 for cell_id in valid_cells}
    # R tree uses bounding boxes to determine if which cells are "worth" comparing against
    # decreases number of cells tested by ruling out cells as who's ellipses cannot intersect (since R tree sorts shapes spatially)
    shapes = list(ellipses.values())
    ids = list(ellipses.keys())
    tree = STRtree(shapes)

    for cell_id, ellipse in ellipses.items():
        candidate_indices = tree.query(ellipse, predicate="intersects")
        for idx in candidate_indices:
            candidates_tested += 1
            neighbor_id = ids[idx]  # convert shapely id to cell_id
            if neighbor_id == cell_id:
                continue
            p1_geom, p2_geom = nearest_points(polygons[cell_id], polygons[neighbor_id])
            p1 = (p1_geom.x, p1_geom.y)
            p2 = (p2_geom.x, p2_geom.y)
            if check_intersection(mask, p1, p2, cell_id, neighbor_id):
                blocked_pairs += 1
                continue
            neighbors[cell_id] += 1
            neighbors_found += 1

    print(f"  ✓ Tested {candidates_tested} candidate pairs")
    print(f"  ✓ Found {neighbors_found} neighbor relationships")
    print(f"  ✓ Rejected {blocked_pairs} pairs due to blockage by a third cell")
    return neighbors


def calculate_cell_vertices(valid_cells, vertices):
    """
    Calculate vertex counts from rosette vertices.

    Args:
        valid_cells: List of valid cell IDs
        vertices: List of vertex dictionaries (rosette vertices only)

    Returns:
        Dictionary mapping cell_id -> number of vertices
    """
    cell_vertex_count = defaultdict(int)

    for vertex in vertices:
        for cell_id in vertex["cells"]:
            cell_vertex_count[cell_id] += 1

    return dict(cell_vertex_count)


def create_base_visualization(
    img, valid_cells, cell_properties, all_vertices, min_cells_for_rosette=5
):
    """
    Create base image with cell outlines and rosette cells highlighted in green.
    Only cells that participate in vertices with min_cells_for_rosette+ cells are highlighted.
    Red dots are NOT drawn here - they will be drawn dynamically in JavaScript.

    Uses PIL to ensure exact pixel-coordinate correspondence between the base
    image and the JavaScript canvas overlay.

    Args:
        img: Original image array
        valid_cells: List of valid cell IDs
        cell_properties: Dictionary with cell properties
        all_vertices: List of all vertex dictionaries (for determining which cells to highlight)
        min_cells_for_rosette: Minimum cells at a vertex to highlight (default: 5)

    Returns:
        Base64-encoded PNG string of the visualization
    """
    # Normalize image to 0-255 range
    if len(img.shape) == 3:
        base_img = img
    else:
        base_img = np.stack([img, img, img], axis=-1)

    if base_img.max() > 1:
        base_img_normalized = (base_img.astype(float) / base_img.max() * 255).astype(
            np.uint8
        )
    else:
        base_img_normalized = (base_img * 255).astype(np.uint8)

    # Convert to PIL Image
    base_pil = Image.fromarray(base_img_normalized)

    # Create transparent overlay layer
    overlay = Image.new("RGBA", base_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw all cell outlines in cyan
    outline_color = (0, 255, 255, 180)  # Cyan with transparency

    for cell_id in valid_cells:
        cell_mask = cell_properties[cell_id]["mask"]

        # Get outline by dilation
        dilated = binary_dilation(cell_mask, iterations=1)
        outline = dilated & ~cell_mask
        ys, xs = np.where(outline)

        # Draw outline pixels
        for y, x in zip(ys, xs):
            overlay.putpixel((x, y), outline_color)

    # Find cells that participate in 5+ cell vertices (not merged rosettes)
    rosette_cells = set()
    for vertex in all_vertices:
        if len(vertex["cells"]) >= min_cells_for_rosette:
            rosette_cells.update(vertex["cells"])

    # Highlight rosette cells in green
    green_color = (0, 255, 0, 76)  # Green with 30% opacity

    for cell_id in rosette_cells:
        if cell_id in cell_properties:
            cell_mask = cell_properties[cell_id]["mask"]
            ys, xs = np.where(cell_mask)

            # Draw filled pixels
            for y, x in zip(ys, xs):
                overlay.putpixel((x, y), green_color)

    # Composite overlay onto base image
    base_pil = base_pil.convert("RGBA")
    final_img = Image.alpha_composite(base_pil, overlay)

    # Convert to base64 for HTML embedding
    buf = BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)
    base_img_base64 = base64.b64encode(buf.read()).decode()

    return base_img_base64


def prepare_interactive_data(
    valid_cells, cell_properties, cell_boundaries, vertices, rosettes, cell_neighbors
):
    """
    - Bounding box prefiltering for speed
    - Uses spatial grid to quickly find candidate neighbors,
      then verifies with actual boundary checking.

    Args:
        valid_cells: List of valid cell IDs
        cell_properties: Dictionary with cell properties
        cell_boundaries: Dictionary mapping cell_id to boundary coordinates
        vertices: List of vertex dictionaries
        rosettes: List of rosette dictionaries
        cell_neighbors: Dictionary mapping cell_id to number of neighbors (pre-calculated)


    Returns:
        Tuple of (cell_pixels, cell_data, rosette_data, cell_to_rosettes)
    """
    import time

    print("\n" + "=" * 70)
    print("PREPARING INTERACTIVE DATA (HYBRID - BOUNDING BOX PREFILTER)")
    print("=" * 70)

    # Cell_neighbors is now passed as parameter (calculated once in app.py)
    print(f"Using pre-calculated neighbor data ({len(cell_neighbors)} cells)")

    # Calculate vertex counts
    print("Calculating cell vertices...")
    cell_vertex_count = defaultdict(int)
    for vertex in vertices:
        for cell_id in vertex["cells"]:
            cell_vertex_count[cell_id] += 1
    cell_vertex_count = dict(cell_vertex_count)

    # Build cell-to-rosettes mapping
    cell_to_rosettes = defaultdict(list)
    for rosette_idx, rosette in enumerate(rosettes):
        for cell_id in rosette["cells"]:
            cell_to_rosettes[cell_id].append(rosette_idx)

    # Prepare pixel data
    print(f"Processing {len(valid_cells)} cells...")
    start = time.time()

    cell_pixels = {}
    cell_data = {}
    batch_size = 200
    total_pixels = 0

    for i in range(0, len(valid_cells), batch_size):
        batch = valid_cells[i : i + batch_size]

        for cell_id in batch:
            ys, xs = np.where(cell_properties[cell_id]["mask"])
            cell_pixels[int(cell_id)] = np.column_stack([ys, xs]).tolist()

            cell_mask = cell_properties[cell_id]["mask"]
            padded = np.pad(cell_mask, 1, mode="constant", constant_values=False)
            boundary = (
                (padded[1:-1, 1:-1] & ~padded[:-2, 1:-1])
                | (padded[1:-1, 1:-1] & ~padded[2:, 1:-1])
                | (padded[1:-1, 1:-1] & ~padded[1:-1, :-2])
                | (padded[1:-1, 1:-1] & ~padded[1:-1, 2:])
            )
            perimeter = int(np.sum(boundary))

            cell_data[int(cell_id)] = {
                "area": int(cell_properties[cell_id]["area"]),
                "perimeter": perimeter,
                "num_neighbors": cell_neighbors.get(cell_id, 0),
                "num_vertices": cell_vertex_count.get(cell_id, 0),
                "in_rosette": cell_id in cell_to_rosettes,
            }

            total_pixels += len(ys)

        processed = min(i + batch_size, len(valid_cells))
        if processed % 400 == 0 or processed == len(valid_cells):
            print(f"  Processed {processed}/{len(valid_cells)} cells...")

    elapsed = time.time() - start
    print(f"✓ Cell processing complete in {elapsed:.1f}s")

    # Prepare rosette data
    rosette_center_to_idx = {
        (int(r["location"][0]), int(r["location"][1])): r_idx
        for r_idx, r in enumerate(rosettes)
    }
    vertex_data = []
    for v_idx, vertex in enumerate(vertices):
        center = (int(vertex["location"][0]), int(vertex["location"][1]))
        vertex_data.append(
            {
                "id": v_idx,
                "cells": [int(c) for c in vertex["cells"]],
                "center": [center[0], center[1]],
                "num_cells": vertex.get("num_cells", len(vertex["cells"])),
                "rosette_idx": rosette_center_to_idx.get(center),
            }
        )

    rosette_data = []
    for idx, rosette in enumerate(rosettes):
        rosette_data.append(
            {
                "id": idx,
                "cells": [int(c) for c in rosette["cells"]],
                "center": [int(rosette["location"][0]), int(rosette["location"][1])],
                "num_cells": rosette["num_cells"],
            }
        )

    print(
        f"✓ Prepared data for {len(cell_pixels)} cells in {len(rosette_data)} rosettes"
    )
    if cell_neighbors:
        neighbor_values = [v for v in cell_neighbors.values() if v > 0]
        if neighbor_values:
            print(f"  - Average neighbors per cell: {np.mean(neighbor_values):.1f}")
            print(
                f"  - Cells with neighbors: {len(neighbor_values)}/{len(valid_cells)}"
            )
    print("=" * 70)

    return cell_pixels, cell_data, vertex_data, rosette_data, cell_to_rosettes


def generate_html_visualization(
    base_img_base64,
    cell_pixels,
    cell_data,
    vertex_data,
    rosette_data,
    cell_to_rosettes,
    num_cells,
    num_rosettes,
    csv_columns=None,
    csv_rows=None,
    csv_filename="cell_data.csv",
):
    """
    Generate interactive HTML visualization file.

    Creates an HTML file with embedded JavaScript that allows users to hover
    over cells and see their properties and associated rosettes. Red dots are
    drawn dynamically and can be removed by clicking.

    Args:
        base_img_base64: Base64-encoded PNG string of base visualization
        cell_pixels: Dictionary mapping cell_id to pixel coordinates
        cell_data: Dictionary mapping cell_id to cell properties
        rosette_data: List of rosette information dictionaries
        cell_to_rosettes: Dictionary mapping cell_id to rosette indices
        num_cells: Total number of valid cells detected
        num_rosettes: Total number of rosettes identified
        csv_columns: Optional list of CSV column names (for client-side export)
        csv_rows: Optional list of CSV row dicts (for client-side export)
        csv_filename: Suggested filename for exported CSV

    Returns:
        String containing complete HTML document
    """
    if csv_columns is None:
        csv_columns = []
    if csv_rows is None:
        csv_rows = []

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Rosette Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #1a1a1a;
            color: #ffffff;
        }}
        #container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #4CAF50;
        }}
        #info {{
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        #canvas-container {{
            position: relative;
            display: inline-block;
            margin: 0 auto;
            display: block;
        }}
        canvas {{
            border: 2px solid #4CAF50;
            cursor: crosshair;
            display: block;
        }}
        #stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 10px;
        }}
        .stat {{
            background-color: #333;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #aaa;
        }}
        #hover-info {{
            background-color: #444;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            min-height: 80px;
        }}
        .cell-property {{
            display: inline-block;
            margin-right: 15px;
            margin-top: 5px;
        }}
        .property-label {{
            color: #aaa;
            font-size: 12px;
        }}
        .property-value {{
            color: #4CAF50;
            font-weight: bold;
        }}
        #actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 12px;
            align-items: center;
        }}
        .btn {{
            background: #4CAF50;
            color: #0b0b0b;
            border: none;
            padding: 10px 12px;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
        }}
        .btn:hover {{
            filter: brightness(1.05);
        }}
        .btn.secondary {{
            background: #2f7bdc;
            color: #ffffff;
        }}
        .btn.ghost {{
            background: transparent;
            border: 1px solid #555;
            color: #ddd;
        }}
        #export-note {{
            color: #aaa;
            font-size: 12px;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <div id="container">
        <h1>Interactive Rosette Visualization</h1>
        <div id="info">
            <div id="stats">
                <div class="stat">
                    <div class="stat-value">{num_cells}</div>
                    <div class="stat-label">Total Cells</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="rosette-count">{num_rosettes}</div>
                    <div class="stat-label">Total Rosettes</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len([c for c in cell_data.values() if c["in_rosette"]])}</div>
                    <div class="stat-label">Cells in Rosettes</div>
                </div>
            </div>
            <div id="legend" style="display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; font-size:12px;">
                <span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;border-radius:50%;background:#4CAF50;display:inline-block;"></span>3-cell</span>
                <span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;border-radius:50%;background:#FFD54F;display:inline-block;"></span>4-cell</span>
                <span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;border-radius:50%;background:#FF9800;display:inline-block;"></span>5-cell</span>
                <span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;border-radius:50%;background:#FF5722;display:inline-block;"></span>6-cell</span>
                <span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;border-radius:50%;background:#E91E63;display:inline-block;"></span>7-cell</span>
                <span style="display:flex;align-items:center;gap:5px;"><span style="width:12px;height:12px;border-radius:50%;background:#9C27B0;display:inline-block;"></span>8+ cell</span>
            </div>
            <div id="actions">
                <button class="btn secondary" id="download-image-btn" type="button">Download edited image (PNG)</button>
                <button class="btn" id="download-csv-btn" type="button">Download updated CSV</button>
                <button class="btn ghost" id="reset-btn" type="button">Reset edits</button>
                <div id="export-note">
                    Exports are downloaded by your browser (the HTML can’t overwrite files on disk).<br>
                    The updated CSV recalculates junction columns based on the current rosette edits.
                </div>
            </div>
            <div id="hover-info">
                <strong>Instructions:</strong> Hover over any cell to see its properties. 
                Rosette cells are shown in green. Click on a red dot to remove that rosette. Click on a grayed area to restore it.
            </div>
        </div>
        
        <div id="canvas-container">
            <canvas id="canvas"></canvas>
        </div>
    </div>

    <script>
        function getJunctionColor(size) {{
            // Color scale by number of participating cells (junction order)
            if (size <= 3) return '#4CAF50';   // green  - normal 3-way junction
            if (size === 4) return '#FFD54F';  // yellow - 4-way
            if (size === 5) return '#FF9800';  // orange - 5-way
            if (size === 6) return '#FF5722';  // deep orange - 6-way
            if (size === 7) return '#E91E63';  // pink/red - 7-way
            return '#9C27B0';                  // purple - 8+ way (higher-order rosette)
        }}
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // Data from Python
        const cellPixels = {json.dumps(cell_pixels)};
        const cellData = {json.dumps(cell_data)};
        const vertexData = {json.dumps(vertex_data)};
        const rosettes = {json.dumps(rosette_data)};
        const cellToRosettes = {json.dumps({int(k): v for k, v in cell_to_rosettes.items()})};
        const csvColumns = {json.dumps(csv_columns)};
        const csvRows = {json.dumps(csv_rows)};
        const originalCsvFilename = {json.dumps(csv_filename)};
        
        // Load base image
        const baseImg = new Image();
        baseImg.src = 'data:image/png;base64,{base_img_base64}';
        
        let currentHighlightedRosettes = new Set();
        let pixelToCellMap = new Map();
        let removedRosettes = new Set(); // Track removed rosettes
        
        baseImg.onload = function() {{
            // Set canvas internal dimensions to match image exactly
            canvas.width = baseImg.width;
            canvas.height = baseImg.height;
            
            // Set canvas CSS display size to match internal dimensions
            canvas.style.width = baseImg.width + 'px';
            canvas.style.height = baseImg.height + 'px';
            
            // Build reverse lookup: pixel -> cell_id
            console.log('Building pixel-to-cell map...');
            for (const [cellId, pixels] of Object.entries(cellPixels)) {{
                for (const [y, x] of pixels) {{
                    const key = `${{x}},${{y}}`;
                    pixelToCellMap.set(key, parseInt(cellId));
                }}
            }}
            console.log(`Mapped ${{pixelToCellMap.size}} pixels`);
            
            // Initial draw
            drawImage();
        }};
        
        function drawImage() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(baseImg, 0, 0);
            
            // Draw gray mask over removed rosettes to hide the green
            if (removedRosettes.size > 0) {{
                ctx.fillStyle = 'rgba(26, 26, 26, 0.85)';  // Dark gray mask matching background
                
                removedRosettes.forEach(rosetteIdx => {{
                    const rosette = rosettes[rosetteIdx];
                    rosette.cells.forEach(cellId => {{
                        const pixels = cellPixels[cellId];
                        if (pixels) {{
                            pixels.forEach(([y, x]) => {{
                                ctx.fillRect(x, y, 1, 1);
                            }});
                        }}
                    }});
                }});
            }}
            // Draw dots for active junctions, colored by participant count
            vertexData.forEach((vertex) => {{
                const rIdx = vertex.rosette_idx;
                if (rIdx !== null && removedRosettes.has(rIdx)) return;  // hide removed rosettes only

                const [cx, cy] = vertex.center;
                ctx.fillStyle = getJunctionColor(vertex.num_cells);
                ctx.beginPath();
                ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
                ctx.fill();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 1;
                ctx.stroke();
            }});
            
            // Draw highlighted rosettes in ORANGE (on hover)
            if (currentHighlightedRosettes.size > 0) {{
                ctx.fillStyle = 'rgba(255, 140, 0, 0.5)';
                
                currentHighlightedRosettes.forEach(rosetteIdx => {{
                    if (removedRosettes.has(rosetteIdx)) return;
                    
                    const rosette = rosettes[rosetteIdx];
                    rosette.cells.forEach(cellId => {{
                        const pixels = cellPixels[cellId];
                        if (pixels) {{
                            pixels.forEach(([y, x]) => {{
                                ctx.fillRect(x, y, 1, 1);
                            }});
                        }}
                    }});
                    
                    // Draw emphasized center marker for hovered rosette
                    const [cx, cy] = rosette.center;
                    
                    ctx.fillStyle = getJunctionColor(rosette.num_cells);
                    ctx.beginPath();
                    ctx.arc(cx, cy, 10, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                    ctx.fillRect(cx + 12, cy - 16, 50, 20);
                    
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 14px Arial';
                    ctx.fillText(`R${{rosetteIdx + 1}}`, cx + 16, cy);
                    
                    ctx.fillStyle = 'rgba(255, 140, 0, 0.5)';
                }});
            }}
        }}

        function downloadBlob(blob, filename) {{
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        }}

        function downloadEditedImage() {{
            // Ensure we export exactly what’s displayed
            drawImage();
            const base = (originalCsvFilename || 'cell_data.csv').replace(/_cell_data\\.csv$/i, '').replace(/\\.csv$/i, '');
            const filename = `${{base}}_edited.png`;
            canvas.toBlob((blob) => {{
                if (!blob) {{
                    alert('Could not export image (canvas toBlob failed).');
                    return;
                }}
                downloadBlob(blob, filename);
            }}, 'image/png');
        }}

        function getJunctionColumnForSize(size) {{
            if (size === 3) return 'junctions_3_cell';
            if (size === 4) return 'junctions_4_cell';
            if (size === 5) return 'junctions_5_cell';
            if (size === 6) return 'junctions_6_cell';
            if (size === 7) return 'junctions_7_cell';
            return 'junctions_8plus_cell';
        }}

        function computeUpdatedJunctionCounts() {{
            const countsByCell = new Map();

            for (const row of csvRows) {{
                const id = parseInt(row.cell_id);
                if (!Number.isFinite(id)) continue;
                countsByCell.set(id, {{
                    junctions_3_cell: 0,
                    junctions_4_cell: 0,
                    junctions_5_cell: 0,
                    junctions_6_cell: 0,
                    junctions_7_cell: 0,
                    junctions_8plus_cell: 0,
                    total_junctions: 0,
                }});
            }}

            rosettes.forEach((rosette, rosetteIdx) => {{
                if (removedRosettes.has(rosetteIdx)) return;
                const size = parseInt(rosette.num_cells);
                const col = getJunctionColumnForSize(size);
                rosette.cells.forEach((cellId) => {{
                    const id = parseInt(cellId);
                    const record = countsByCell.get(id);
                    if (!record) return;
                    record[col] += 1;
                    record.total_junctions += 1;
                }});
            }});

            return countsByCell;
        }}

        function escapeCsvValue(value) {{
            if (value === null || value === undefined) return '';
            const s = String(value);
            if (/[",\\n\\r]/.test(s)) {{
                return '"' + s.replace(/"/g, '""') + '"';
            }}
            return s;
        }}

        function toCsv(columns, rows) {{
            const lines = [];
            lines.push(columns.map(escapeCsvValue).join(','));
            for (const row of rows) {{
                const line = columns.map((c) => escapeCsvValue(row[c])).join(',');
                lines.push(line);
            }}
            return lines.join('\\r\\n');
        }}

        function downloadUpdatedCsv() {{
            if (!csvColumns.length || !csvRows.length) {{
                alert('CSV export data was not embedded in this HTML.');
                return;
            }}

            const countsByCell = computeUpdatedJunctionCounts();

            // Create updated rows (preserve everything except junction columns)
            const updatedRows = csvRows.map((row) => {{
                const id = parseInt(row.cell_id);
                const updated = {{ ...row }};
                const counts = countsByCell.get(id);
                if (counts) {{
                    updated.junctions_3_cell = counts.junctions_3_cell;
                    updated.junctions_4_cell = counts.junctions_4_cell;
                    updated.junctions_5_cell = counts.junctions_5_cell;
                    updated.junctions_6_cell = counts.junctions_6_cell;
                    updated.junctions_7_cell = counts.junctions_7_cell;
                    updated.junctions_8plus_cell = counts.junctions_8plus_cell;
                    updated.total_junctions = counts.total_junctions;
                }} else {{
                    // If a cell isn't in the map, zero out junctions to avoid stale values
                    updated.junctions_3_cell = 0;
                    updated.junctions_4_cell = 0;
                    updated.junctions_5_cell = 0;
                    updated.junctions_6_cell = 0;
                    updated.junctions_7_cell = 0;
                    updated.junctions_8plus_cell = 0;
                    updated.total_junctions = 0;
                }}
                return updated;
            }});

            const csvText = toCsv(csvColumns, updatedRows);
            const base = (originalCsvFilename || 'cell_data.csv').replace(/\\.csv$/i, '');
            const filename = `${{base}}_updated.csv`;
            const blob = new Blob([csvText], {{ type: 'text/csv;charset=utf-8' }});
            downloadBlob(blob, filename);
        }}

        function resetEdits() {{
            removedRosettes = new Set();
            currentHighlightedRosettes = new Set();
            drawImage();
            document.getElementById('rosette-count').textContent = rosettes.length;
            document.getElementById('hover-info').innerHTML =
                '<strong>Instructions:</strong> Hover over any cell to see its properties. Rosette cells are shown in green. Click on a red dot to remove that rosette. Click on a grayed area to restore it.';
        }}
        
        // Handle mouse movement for interactive highlighting
        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor(e.clientX - rect.left);
            const y = Math.floor(e.clientY - rect.top);
            
            const key = `${{x}},${{y}}`;
            const cellId = pixelToCellMap.get(key);
            
            if (cellId && cellData[cellId]) {{
                const data = cellData[cellId];
                const rosetteIndices = cellToRosettes[cellId] || [];
                
                // Filter out removed rosettes
                const activeRosetteIndices = rosetteIndices.filter(idx => !removedRosettes.has(idx));
                const newSet = new Set(activeRosetteIndices);
                
                // Update highlighting if changed
                if (![...newSet].every(r => currentHighlightedRosettes.has(r)) ||
                    ![...currentHighlightedRosettes].every(r => newSet.has(r))) {{
                    currentHighlightedRosettes = newSet;
                    drawImage();
                }}
                
                // Build info display
                let infoHTML = `<strong>Cell ID:</strong> ${{cellId}}<br>`;
                infoHTML += `<div style="margin-top: 8px;">`;
                infoHTML += `<div class="cell-property"><span class="property-label">Area:</span> <span class="property-value">${{data.area}} px²</span></div>`;
                infoHTML += `<div class="cell-property"><span class="property-label">Perimeter:</span> <span class="property-value">${{data.perimeter}} px</span></div>`;
                infoHTML += `<div class="cell-property"><span class="property-label">Neighbors:</span> <span class="property-value">${{data.num_neighbors}}</span></div>`;
                //infoHTML += `<div class="cell-property"><span class="property-label">Vertices:</span> <span class="property-value">${{data.num_vertices}}</span></div>`;
                infoHTML += `</div>`;
                
                if (activeRosetteIndices.length > 0) {{
                    const rosetteInfo = activeRosetteIndices.map(idx => {{
                        const r = rosettes[idx];
                        return `Rosette #${{idx + 1}} (${{r.num_cells}} cells)`;
                    }}).join(', ');
                    infoHTML += `<div style="margin-top: 8px;"><strong>Part of:</strong> ${{rosetteInfo}}</div>`;
                    infoHTML += `<div style="margin-top: 5px; color: #aaa; font-size: 12px;">(Click on red dot to remove rosette)</div>`;
                }}
                
                document.getElementById('hover-info').innerHTML = infoHTML;
            }} else {{
                if (currentHighlightedRosettes.size > 0) {{
                    currentHighlightedRosettes = new Set();
                    drawImage();
                }}
                document.getElementById('hover-info').innerHTML = 
                    '<strong>Instructions:</strong> Hover over any cell to see its properties. Rosette cells are shown in green. Click on a red dot to remove that rosette. Click on a grayed area to restore it.';
            }}
        }});
        
        // Clear highlighting when mouse leaves canvas
        canvas.addEventListener('mouseleave', () => {{
            currentHighlightedRosettes = new Set();
            drawImage();
            document.getElementById('hover-info').innerHTML = 
                '<strong>Instructions:</strong> Hover over any cell to see its properties. Rosette cells are shown in green. Click on a red dot to remove that rosette. Click on a grayed area to restore it.';
        }});
        
        // Handle click to remove/restore rosettes (click on red dot or grayed area)
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor(e.clientX - rect.left);
            const y = Math.floor(e.clientY - rect.top);
            
            // First check if clicking on a removed rosette to restore it
            const key = `${{x}},${{y}}`;
            const cellId = pixelToCellMap.get(key);
            
            if (cellId && cellToRosettes[cellId]) {{
                const rosetteIndices = cellToRosettes[cellId];
                
                // Check if any of this cell's rosettes are removed
                for (const rosetteIdx of rosetteIndices) {{
                    if (removedRosettes.has(rosetteIdx)) {{
                        // Restore this rosette
                        removedRosettes.delete(rosetteIdx);
                        drawImage();
                        
                        // Update rosette count
                        const numRemaining = rosettes.length - removedRosettes.size;
                        document.getElementById('rosette-count').textContent = numRemaining;
                        
                        document.getElementById('hover-info').innerHTML = 
                            `<strong>Rosette #${{rosetteIdx + 1}} restored!</strong> ${{numRemaining}} rosette(s) remaining.`;
                        return;
                    }}
                }}
            }}
            
              // If not clicking on a removed rosette, check if clicking on an active dot to remove it
            for (const vertex of vertexData) {{
                const rIdx = vertex.rosette_idx;
                if (rIdx === null || removedRosettes.has(rIdx)) continue;

                const [cx, cy] = vertex.center;
                const distance = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);

                if (distance <= 10) {{
                    removedRosettes.add(rIdx);
                    currentHighlightedRosettes.delete(rIdx);
                    drawImage();
                    const numRemaining = rosettes.length - removedRosettes.size;
                    document.getElementById('rosette-count').textContent = numRemaining;
                    document.getElementById('hover-info').innerHTML = 
                        `<strong>Rosette #${{rIdx + 1}} removed!</strong> ${{numRemaining}} rosette(s) remaining. (Click on grayed area to restore)`;
                    return;
                }}
            }}
        }});

        document.getElementById('download-image-btn').addEventListener('click', downloadEditedImage);
        document.getElementById('download-csv-btn').addEventListener('click', downloadUpdatedCsv);
        document.getElementById('reset-btn').addEventListener('click', resetEdits);
    </script>
</body>
</html>
"""
    return html_content
