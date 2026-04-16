"""
Interactive Rosette Detection and Visualization - Main Application

This is the main application file that orchestrates the complete rosette detection
workflow by importing and coordinating the modular components:
- cell_segmentation: Loads images and detects individual cells
- vertex_detection: Finds points where multiple cells meet
- rosette_detection: Clusters vertices and creates visualizations
- csv_export: Exports detailed cell properties and junction counts to CSV

Supports both single-image and batch processing modes.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from src.cell_segmentation import load_and_validate_images, detect_cells, extract_cell_boundaries
from src.vertex_detection import find_vertices
from src.rosette_detection import cluster_vertices, create_base_visualization, prepare_interactive_data, generate_html_visualization, calculate_cell_neighbors
from src.csv_export import generate_csv_export

# ============================================================================
# DEFAULT PARAMETERS
# These parameters control the detection sensitivity and filtering criteria
# ============================================================================
DEFAULT_CELL_DIAMETER = 30              # Estimated cell diameter for CellPose detection (pixels)
DEFAULT_MIN_CELL_AREA = 100             # Minimum area to count as valid cell (pixels)
DEFAULT_MAX_CELL_AREA = 5000            # Maximum area to count as valid cell (pixels)
DEFAULT_VERTEX_RADIUS = 15              # Search radius for cells meeting at a vertex (pixels)
DEFAULT_MIN_CELLS_FOR_ROSETTE = 5       # Minimum cells required to form a rosette

DETECTION_PRESETS = {
    # Detect smaller and lower-contrast structures (higher sensitivity)
    'sensitive': {
        'cell_diameter': 36,
        'min_cell_area': 280,
        'max_cell_area': 5000,
        'vertex_radius': 18,
        'min_rosette_cells': 5,
    },
    # Current default behavior
    'balanced': {
        'cell_diameter': DEFAULT_CELL_DIAMETER,
        'min_cell_area': DEFAULT_MIN_CELL_AREA,
        'max_cell_area': DEFAULT_MAX_CELL_AREA,
        'vertex_radius': DEFAULT_VERTEX_RADIUS,
        'min_rosette_cells': DEFAULT_MIN_CELLS_FOR_ROSETTE,
    },
    # Prefer stricter filtering to reduce false positives
    'lenient': {
        'cell_diameter': 38,
        'min_cell_area': 300,
        'max_cell_area': 5200,
        'vertex_radius': 13,
        'min_rosette_cells': 5,
    },
}

# Supported image file extensions
IMAGE_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
# ============================================================================


def list_images_in_folder(folder_path):
    """
    Return sorted list of supported images in a folder.
    """
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    return sorted(image_files)


def parse_simple_text_config(config_path):
    """
    Parse a simple key=value config text file.
    """
    parsed = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                raise ValueError(f"Invalid config line {line_no}: expected key = value")
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip().strip('"').strip("'")
            parsed[key] = value
    return parsed


def parse_bool(value, default=True):
    """
    Parse boolean-like string values.
    """
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


def load_config_from_file(config_path):
    """
    Load and validate processing configuration from text file.
    """
    raw = parse_simple_text_config(config_path)

    mode = raw.get('mode', 'single').strip().lower()
    if mode not in {'single', 'batch'}:
        raise ValueError("mode must be 'single' or 'batch'")
    batch_mode = (mode == 'batch')

    input_path = raw.get('input_path', '').strip()
    if not input_path:
        raise ValueError("input_path is required in the config file")

    if batch_mode:
        if not os.path.isdir(input_path):
            raise ValueError(f"Batch mode requires a valid folder input_path: {input_path}")
        image_paths = list_images_in_folder(input_path)
        if not image_paths:
            raise ValueError(f"No supported image files found in folder: {input_path}")
        output_path = raw.get('output_path', 'output/batch_results').strip()
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'html'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'csv'), exist_ok=True)
        output_dir = None
    else:
        if not os.path.isfile(input_path):
            raise ValueError(f"Single mode requires a valid file input_path: {input_path}")
        image_paths = [input_path]
        input_file = Path(input_path)
        default_output = str(input_file.with_name(f"{input_file.stem}_interactive_viewer.html"))
        output_path = raw.get('output_path', default_output).strip()
        if not output_path.lower().endswith('.html'):
            output_path += '.html'
        output_dir = os.path.dirname(output_path) or '.'
        os.makedirs(output_dir, exist_ok=True)

    preset_key = raw.get('detection_preset', 'balanced').strip().lower()
    if preset_key not in DETECTION_PRESETS:
        valid_presets = ", ".join(sorted(DETECTION_PRESETS.keys()))
        raise ValueError(f"detection_preset must be one of: {valid_presets}")

    preset_values = DETECTION_PRESETS[preset_key]

    try:
        cell_diameter = int(raw.get('cell_diameter', preset_values['cell_diameter']))
        min_cell_area = int(raw.get('min_cell_area', preset_values['min_cell_area']))
        max_cell_area = int(raw.get('max_cell_area', preset_values['max_cell_area']))
        vertex_radius = int(raw.get('vertex_radius', preset_values['vertex_radius']))
        min_rosette_cells = int(raw.get('min_rosette_cells', preset_values['min_rosette_cells']))
    except ValueError as exc:
        raise ValueError(f"Detection parameters must be integers: {exc}") from exc

    if min_cell_area >= max_cell_area:
        raise ValueError("min_cell_area must be less than max_cell_area")

    return {
        'batch_mode': batch_mode,
        'input_path': input_path,
        'image_paths': image_paths,
        'output_path': output_path,
        'output_dir': output_dir,
        'cell_diameter': cell_diameter,
        'min_cell_area': min_cell_area,
        'max_cell_area': max_cell_area,
        'vertex_radius': vertex_radius,
        'min_rosette_cells': min_rosette_cells,
        'detection_preset': preset_key,
        'confirm': parse_bool(raw.get('confirm'), default=True),
    }


def parse_cli_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Rosette detection pipeline with interactive and config-file modes."
    )
    parser.add_argument(
        '--config',
        type=str,
        help="Path to text config file (key=value format)."
    )
    return parser.parse_args()


def get_user_input():
    """
    Prompt user for input file/folder and optional parameters.
    
    Returns:
        Dictionary containing all configuration parameters
    """
    print("\n" + "="*70)
    print("ROSETTE DETECTION - INTERACTIVE SETUP")
    print("="*70)
    
    # Ask if processing single file or batch folder
    print("\nProcessing mode:")
    print("  1. Single image file")
    print("  2. Batch process folder")
    
    while True:
        mode_input = input("\nSelect mode (1 or 2, default: 1): ").strip()
        if mode_input == '' or mode_input == '1':
            batch_mode = False
            break
        elif mode_input == '2':
            batch_mode = True
            break
        else:
            print("Invalid input. Please enter 1 or 2.")
    
    # Get input path
    if batch_mode:
        while True:
            folder_path = input("\nEnter the path to your image folder: ").strip()
            folder_path = folder_path.strip('"').strip("'")
            
            if os.path.isdir(folder_path):
                # Find all image files in the folder
                image_files = list_images_in_folder(folder_path)
                
                if len(image_files) > 0:
                    print(f"✓ Found {len(image_files)} image(s) in folder")
                    for img_file in image_files[:5]:  # Show first 5
                        print(f"  - {os.path.basename(img_file)}")
                    if len(image_files) > 5:
                        print(f"  ... and {len(image_files) - 5} more")
                    break
                else:
                    print(f"✗ No image files found in folder: {folder_path}")
                    print(f"  Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
                    retry = input("Would you like to try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        print("Exiting...")
                        sys.exit(0)
            else:
                print(f"✗ Folder not found: {folder_path}")
                retry = input("Would you like to try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("Exiting...")
                    sys.exit(0)
        
        input_path = folder_path
        image_paths = sorted(image_files)
    else:
        # Single file mode
        while True:
            image_path = input("\nEnter the path to your image file: ").strip()
            image_path = image_path.strip('"').strip("'")
            
            if os.path.exists(image_path):
                print(f"✓ Found image: {image_path}")
                break
            else:
                print(f"✗ File not found: {image_path}")
                retry = input("Would you like to try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("Exiting...")
                    sys.exit(0)
        
        input_path = image_path
        image_paths = [image_path]
    
    # Ask if user wants to customize parameters
    print("\n" + "-"*70)
    customize = input("\nWould you like to customize detection parameters? (y/n, default: n): ").strip().lower()
    
    if customize == 'y':
        print("\nEnter custom parameters (press Enter to use default):")
        
        # Cell diameter
        diameter_input = input(f"  Cell diameter in pixels (default: {DEFAULT_CELL_DIAMETER}): ").strip()
        cell_diameter = int(diameter_input) if diameter_input else DEFAULT_CELL_DIAMETER
        
        # Min cell area
        min_area_input = input(f"  Minimum cell area in pixels (default: {DEFAULT_MIN_CELL_AREA}): ").strip()
        min_cell_area = int(min_area_input) if min_area_input else DEFAULT_MIN_CELL_AREA
        
        # Max cell area
        max_area_input = input(f"  Maximum cell area in pixels (default: {DEFAULT_MAX_CELL_AREA}): ").strip()
        max_cell_area = int(max_area_input) if max_area_input else DEFAULT_MAX_CELL_AREA
        
        # Vertex radius
        vertex_input = input(f"  Vertex search radius in pixels (default: {DEFAULT_VERTEX_RADIUS}): ").strip()
        vertex_radius = int(vertex_input) if vertex_input else DEFAULT_VERTEX_RADIUS
        
        # Min rosette cells
        rosette_input = input(f"  Minimum cells for rosette (default: {DEFAULT_MIN_CELLS_FOR_ROSETTE}): ").strip()
        min_rosette_cells = int(rosette_input) if rosette_input else DEFAULT_MIN_CELLS_FOR_ROSETTE
    else:
        cell_diameter = DEFAULT_CELL_DIAMETER
        min_cell_area = DEFAULT_MIN_CELL_AREA
        max_cell_area = DEFAULT_MAX_CELL_AREA
        vertex_radius = DEFAULT_VERTEX_RADIUS
        min_rosette_cells = DEFAULT_MIN_CELLS_FOR_ROSETTE
    
    # Get output location
    if batch_mode:
        print("\n" + "-"*70)
        output_input = input("\nEnter output folder for results (default: output/batch_results): ").strip()
        output_folder = output_input if output_input else 'output/batch_results'
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'html'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'csv'), exist_ok=True)
        output_path = output_folder
    else:
        print("\n" + "-"*70)
        # Ask for output directory
        output_dir_input = input("\nEnter output directory (default: current directory): ").strip()
        output_dir = output_dir_input if output_dir_input else '.'
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Ask for HTML filename
        output_file_input = input("Enter output HTML filename (default: interactive_rosette_viewer.html): ").strip()
        output_file = output_file_input if output_file_input else 'interactive_rosette_viewer.html'
        
        # Ensure .html extension
        if not output_file.endswith('.html'):
            output_file += '.html'
        
        # Combine directory and filename
        output_path = os.path.join(output_dir, output_file)
    
    return {
        'batch_mode': batch_mode,
        'input_path': input_path,
        'image_paths': image_paths,
        'output_path': output_path,
        'output_dir': output_dir if not batch_mode else None,
        'cell_diameter': cell_diameter,
        'min_cell_area': min_cell_area,
        'max_cell_area': max_cell_area,
        'vertex_radius': vertex_radius,
        'min_rosette_cells': min_rosette_cells,
        'confirm': True,
    }


def process_single_image(image_path, output_html, output_csv, params):
    """
    Process a single image and generate HTML visualization and CSV data.
    
    Args:
        image_path: Path to the image file
        output_html: Path for the output HTML file
        output_csv: Path for the output CSV file
        params: Dictionary with processing parameters (cell_diameter, min_cell_area, etc.)
        
    Returns:
        Dictionary with processing results
    """
    print("\n" + "="*70)
    print(f"Processing: {os.path.basename(image_path)}")
    print("="*70)
    
    # Load images
    imgs = load_and_validate_images([image_path])
    
    if not imgs:
        print(f"ERROR: Failed to load image: {image_path}")
        return {'success': False, 'error': 'Failed to load image'}
    
    # Detect and filter cells
    mask, img, valid_cells, cell_properties = detect_cells(
        imgs, params['cell_diameter'], params['min_cell_area'], params['max_cell_area']
    )
    
    if len(valid_cells) == 0:
        print(f"WARNING: No valid cells detected in {os.path.basename(image_path)}")
        return {'success': False, 'error': 'No valid cells detected'}
    
    # Extract cell boundaries
    cell_boundaries = extract_cell_boundaries(valid_cells, cell_properties)
    
    # Find ALL vertices where cells meet (3+ cells) for comprehensive junction analysis
    all_vertices = find_vertices(
        valid_cells, cell_boundaries, mask, params['vertex_radius'], min_cells_for_vertex=3
    )
    
    # Find rosette vertices (5+ cells) for visualization
    rosette_vertices = find_vertices(
        valid_cells, cell_boundaries, mask, params['vertex_radius'], params['min_rosette_cells']
    )
    
    # Cluster nearby vertices into rosettes
    rosettes = cluster_vertices(rosette_vertices, params['vertex_radius'], params['min_rosette_cells'])
    
    num_rosettes = len(rosettes)
    
    print(f"Results: {len(valid_cells)} cells, {num_rosettes} rosettes")
    
    # Calculate cell neighbors once (used for both visualization and CSV export)
    print("Calculating cell neighbors...")
    cell_neighbors = calculate_cell_neighbors(valid_cells, cell_boundaries)
    
    # Generate base visualization image
    base_img_base64 = create_base_visualization(img, valid_cells, cell_properties, rosettes)
    
    # Prepare data for JavaScript
    print("Creating interactive data...")
    cell_pixels, cell_data, rosette_data, cell_to_rosettes = prepare_interactive_data(
        valid_cells, cell_properties, cell_boundaries, all_vertices, rosettes, cell_neighbors
    )
    
    # Generate HTML file
    html_content = generate_html_visualization(
        base_img_base64, cell_pixels, cell_data, rosette_data, cell_to_rosettes,
        len(valid_cells), num_rosettes
    )
    
    # Save HTML file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Created HTML: {output_html}")
    
    # Generate CSV export using ALL vertices (3+) for complete junction data
    generate_csv_export(mask, valid_cells, all_vertices, cell_neighbors, output_csv)
    
    print(f"✓ Created CSV: {output_csv}")
    
    return {
        'success': True,
        'num_cells': len(valid_cells),
        'num_rosettes': num_rosettes,
        'output_html': output_html,
        'output_csv': output_csv
    }


def main():
    """
    Main execution function for rosette detection and visualization.
    
    Supports both single image processing and batch processing of folders.
    """
    args = parse_cli_args()

    # Get config from file or interactive input
    if args.config:
        print(f"\nLoading config file: {args.config}")
        config = load_config_from_file(args.config)
    else:
        config = get_user_input()
    
    # Display configuration
    print("\n" + "="*70)
    print("ROSETTE DETECTION CONFIGURATION")
    print("="*70)
    if config['batch_mode']:
        print(f"Mode: Batch processing")
        print(f"Input folder: {config['input_path']}")
        print(f"Number of images: {len(config['image_paths'])}")
        print(f"Output folder: {config['output_path']}")
    else:
        print(f"Mode: Single image")
        print(f"Input image: {config['input_path']}")
        print(f"Output directory: {config['output_dir']}")
        print(f"Output HTML file: {os.path.basename(config['output_path'])}")
    
    print(f"Cell diameter: {config['cell_diameter']} pixels")
    print(f"Cell area range: {config['min_cell_area']} - {config['max_cell_area']} pixels")
    print(f"Vertex radius: {config['vertex_radius']} pixels")
    print(f"Min cells for rosette: {config['min_rosette_cells']}")
    print("="*70 + "\n")
    
    # Confirm to proceed
    if config.get('confirm', True):
        proceed = input("Proceed with analysis? (y/n, default: y): ").strip().lower()
        if proceed == 'n':
            print("Analysis cancelled.")
            sys.exit(0)
    
    print("\nStarting analysis...")
    
    if config['batch_mode']:
        # ====================================================================
        # BATCH PROCESSING MODE
        # ====================================================================
        results = []
        successful = 0
        failed = 0
        
        for idx, image_path in enumerate(config['image_paths'], 1):
            print(f"\n[{idx}/{len(config['image_paths'])}]", end=" ")
            
            # Generate output filenames
            base_name = Path(image_path).stem
            output_html = os.path.join(config['output_path'], 'html', f"{base_name}_rosette_viewer.html")
            output_csv = os.path.join(config['output_path'], 'csv', f"{base_name}_cell_data.csv")
            
            try:
                result = process_single_image(image_path, output_html, output_csv, config)
                results.append(result)
                
                if result['success']:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"ERROR processing {os.path.basename(image_path)}: {str(e)}")
                results.append({'success': False, 'error': str(e)})
                failed += 1
        
        # Print summary
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        print(f"Total images: {len(config['image_paths'])}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"\nOutput locations:")
        print(f"  HTML files: {os.path.join(config['output_path'], 'html')}")
        print(f"  CSV files: {os.path.join(config['output_path'], 'csv')}")
        print("="*70 + "\n")
        
    else:
        # ====================================================================
        # SINGLE FILE PROCESSING MODE
        # ====================================================================
        
        # Generate CSV filename in the same directory as the HTML file
        output_dir = os.path.dirname(config['output_path'])
        if not output_dir:  # If no directory (current dir), use '.'
            output_dir = '.'
        base_name = Path(config['output_path']).stem
        csv_output_path = os.path.join(output_dir, f'{base_name}_cell_data.csv')
        
        try:
            result = process_single_image(
                config['input_path'], 
                config['output_path'],
                csv_output_path,
                config
            )
            
            if result['success']:
                print("\n" + "="*70)
                print("ROSETTE DETECTION RESULTS")
                print("="*70)
                print(f"Total cells detected: {result['num_cells']}")
                print(f"Total rosettes found: {result['num_rosettes']}")
                print(f"\n✓ SUCCESS! Outputs created:")
                print(f"  HTML: {result['output_html']}")
                print(f"  CSV:  {result['output_csv']}")
                print(f"\nOpen the HTML file in your web browser to interact with the rosettes!")
                print("="*70 + "\n")
            else:
                print(f"\n✗ Processing failed: {result.get('error', 'Unknown error')}")
                print("\nSuggestions:")
                print(f"  - Try adjusting the cell diameter parameter")
                print(f"  - Check min/max area thresholds")
                print(f"  - Verify image quality and contrast")
                
                retry = input("\nWould you like to run again with different parameters? (y/n): ").strip().lower()
                if retry == 'y':
                    print("\n" + "="*70)
                    print("Restarting analysis...")
                    print("="*70)
                    main()
                    return
                    
        except Exception as e:
            print(f"\n\nERROR: An unexpected error occurred:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nPlease check your input file and parameters.")
            sys.exit(1)


# Execute main function
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: An unexpected error occurred:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your input file and parameters.")
        sys.exit(1)