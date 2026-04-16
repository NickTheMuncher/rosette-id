# Rosette Identification
### **We are using 5 grace days for milestone 5**

This is a project for COMP 333 in collaboration with the Mitchel lab.

## [Project Demonstration Here](https://youtu.be/-Z6dolIHgCA)


## Team Members
- **braeden-falzarano**: Braeden Falzarano
- **zatchliu**: Zachary Liu
- **therevboss**: Oliver Wyner

## [Style Guide](/StyleGuide.md)

---

## What This Project Does

This toolkit automatically detects and identifies **rosettes** in microscopy images of cells. 

**What is a rosette?** A rosette is a cluster of 5 or more cells that meet at a central point (called a vertex), arranged in a radial pattern like the petals of a flower.

**What the program does:**
1. Takes a microscopy image as input
2. Identifies individual cells in the image
3. Finds points where multiple cells meet (vertices)
4. Identifies rosettes (where 5+ cells meet at a vertex)
5. Creates an interactive visualization where you can hover over cells to highlight rosettes

**Output:** An HTML file that you open in your web browser to explore the detected rosettes interactively.

---

## System Requirements

### All Users Need:
- **Python 3.10 or 3.11** installed on your computer
  - To check if you have Python: Open Terminal (Mac) or Command Prompt (Windows) and type `python --version`
  - If you don't have Python, download it from [python.org]([https://www.python.org/downloads/](https://www.python.org/downloads/release/python-3110/))

### GPU Support (for faster processing):
- **macOS with M1/M2/M3 chip**: GPU support is available automatically
- **Windows**: CPU-only mode (GPU not currently supported by CellPose on Windows)

**Note:** GPU makes processing faster, but CPU mode still works - it just takes longer.

---

## Installation Guide

**Important:** We will create a "virtual environment" to keep this project's software separate from other Python projects on your computer. This prevents conflicts between different versions of software.

### For macOS Users (M1/M2/M3 chips)

#### Step 1: Open Terminal
- Press `Command + Space` to open Spotlight
- Type "Terminal" and press Enter

#### Step 2: Navigate to Where You Want the Project

Instead of typing folder paths, let's copy the path from Finder:

1. Open **Finder**
2. Navigate to where you want to put the project (Documents, Desktop, etc.)
3. Right-click (or Control + Click) on the folder
4. Hold down the **Option** key on your keyboard (you'll see "Copy" change to "Copy as Pathname")
5. Click **Copy as Pathname**
6. Go back to Terminal and type `cd ` (with a space after cd)
7. Press `Command + V` to paste the path
8. Press Enter

**Example:** If you copied your Documents folder, you'll see something like this before you press enter:
```bash
cd /Users/yourname/Documents
```
and something like this after you hit enter: 
```bash
braedenfalzarano@Braedens-MacBook-Pro-7 Documents % 
```
**Note:** It won't say exactly "braedenfalzarano@Braedens-MacBook-Pro-7", but it should say something identifying your machine. This is not important or relevant to the previous commands that you ran and should stay the same throughout your use of the terminal.

#### Step 3: Clone (Download) This Project

**Note:** You'll need a GitHub account to clone this repository. If you don't have one, you can create a free account at [github.com](https://github.com).

```bash
git clone https://github.com/YOUR-USERNAME/rosette-identification.git
cd rosette-identification
```

**Note:** YOUR-USERNAME is your GitHub username

**Additional Resources:**
- [Installing and setting up GitHub](https://docs.github.com/en/get-started/git-basics/set-up-git)
- [Cloning a Repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

#### Step 4: Create a Virtual Environment
A virtual environment is like a separate workspace for this project.

```bash
python3.11 -m venv rosette
```

This creates a virtual environment called `rosette` that contains all the software this project needs.

#### Step 5: Activate the Virtual Environment
```bash
source rosette/bin/activate
```

You'll see `(rosette)` appear at the start of your command line. This means you're "inside" the virtual environment.

#### Step 6: Install Required Software

```bash
# Check your Python version
python --version
# If it prints "Python 3.11.SOME_NUMBER," you are all set and ready to proceed
# If not, you may need to check your Python install

# Upgrade pip
pip install --upgrade pip

# Install opencv first
python -m pip install -U pip wheel setuptools
python -m pip install --only-binary=:all: "opencv-python-headless==4.10.0.84"

# Install PyTorch
pip install torch torchvision

# Install CellPose with GPU support
pip install 'cellpose[gui]'

# Install other required packages
pip install numpy==1.24.3 scipy==1.11.4 matplotlib==3.8.2 Pillow==10.1.0

pip install skimage
pip install pandas
```

This may take a few minutes to download and install everything. If it proceeds without any red error messages, you should be all set to go.

#### Step 7: Check for GPU Compatibility
```bash
python -c "from cellpose import core; print(f'GPU available: {core.use_gpu()}')"
```

If you see: `GPU available: True`, your system can use GPUs to increase processing speed.
If not, the code will still run, but may take longer due to using CPUs rather than GPUs.

**Installation complete!** You can now skip to **Running the Rosette Detection**.

---

### For Windows Users

#### Step 1: Open Command Prompt
- Press `Windows Key + R`
- Type `cmd` and press Enter

#### Step 2: Navigate to Where You Want the Project
```cmd
# Go to your Documents folder
cd %USERPROFILE%\Documents

# Or go to your Desktop
cd %USERPROFILE%\Desktop
```

#### Step 3: Clone (Download) This Project
```cmd
git clone https://github.com/YOUR-USERNAME/rosette-identification.git
cd rosette-identification
```

**Note:** YOUR-USERNAME is your GitHub username


#### Step 4: Create a Virtual Environment using miniconda
Link to miniconda installation instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install)

```cmd
conda create -n rosette python=3.10 -y
```

#### Step 5: Activate the Virtual Environment
```cmd
conda activate rosette
```

You'll see `(rosette)` appear at the start of your command line.

#### Step 6: Install Required Software

**Note:** Windows can only use CPU mode (not GPU) for CellPose currently.

```cmd
python -m pip install -U pip wheel setuptools
python -m pip install --only-binary=:all: "opencv-python-headless==4.10.0.84"

# Install pyTorch for CPU use
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install CellPose
pip install "cellpose[gui]"

# Install other packages
pip install numpy==1.24.3 scipy==1.11.4 matplotlib==3.8.2 Pillow==10.1.0
pip install skimage
pip install pandas
```

This may take a few minutes to download and install everything.

#### Step 7: Verify Installation
```cmd
python -c "from cellpose import core; print(f'GPU available: {core.use_gpu()}')"
```

You will likely see: `GPU available: False`

This is normal for Windows - it will use CPU mode as stated earlier.

**Installation complete!** Continue to the next section.

---

## Running the Rosette Detection

### Step 1: Make Sure Your Virtual Environment is Active

You should see `(rosette)` at the start of your command line. If not:

**Mac:**
```bash
source rosette/bin/activate
```

**Windows:**
```cmd
rosette\Scripts\activate
```
### Step 2: Ensure images are in data folder

All the images you want to process into the data folder in the repository

There is already an image 'test_image_1.png' provided in the repo

### Step 3: Run the Detection

**Mac:**
```bash
# To avoid MPS Fallback error (this is necessary on macOS)
export PYTORCH_ENABLE_MPS_FALLBACK=1

python app.py
```

**Windows:**
```cmd
# To avoid MPS Fallback error (just to be safe on Windows)
export PYTORCH_ENABLE_MPS_FALLBACK=1

python app.py
```

### Step 3.5: Follow the Interactive Prompts

The program will guide you through the detection process with a series of prompts:

#### 3.5a. Select Single or Batch Processing
```
Processing mode:
  1. Single image file
  2. Batch process folder

Select mode (1 or 2, default: 1):
```
- Type `1` and then Press **Enter** to use single image processing
- Type `2` to then Press **Enter** to use batch processing


#### 3.5b. Select Your Image or Folder

If you selected `1. Single image file`
```
Enter the path to your image file: 
```

If you selected `2. Batch process folder`
```
Enter the path to your folder: 
```

**Options:**
- Type the full path to your image (e.g., `data/test_image_1.png`)
- Drag and drop the file into the terminal (works on most systems)
- Use a relative path from the project directory

**Example:**
```
Enter the path to your image file: data/my_cells.png
✓ Found image: data/my_cells.png
```

#### 3.5c. Customize Parameters (Optional)

You can press **Enter** to accept a default for any of the prompts

```
Would you like to customize detection parameters? (y/n, default: n):
```

- Type `n` and then Press **Enter** or to use default parameters (recommended for first-time users)
- Type `y` to customize cell diameter, area thresholds, and vertex radius

**If you choose to customize, you'll see:**
```
Enter custom parameters (press Enter to use default):
  Cell diameter in pixels (default: 30): 
  Minimum cell area in pixels (default: 100): 
  Maximum cell area in pixels (default: 5000): 
  Vertex search radius in pixels (default: 15): 
  Minimum cells for rosette (default: 5): 
```
#### 3.5d. Choose Output Path
```
Enter output directory (default: current directory):
```

#### 3.5e. Choose Output Filename (Single mode)
**Note** - If you chose batch upload, then the HTML and CSV outputs files will have the name of the raw image files in the Folder you are processing
```
Enter output HTML filename (default: <input_image_stem>_viewer.html):
```

- Type a custom name (e.g., `my_results.html`) and then Press **Enter**
- If you press **Enter**, the default is based on your input image name (for example, `my_cells_viewer.html`)

#### 3.5f. Confirm and Run
```
======================================================================
ROSETTE DETECTION CONFIGURATION
======================================================================
Input image: data/my_cells.png
Output file: my_cells_viewer.html
Cell diameter: 30 pixels
Cell area range: 100 - 5000 pixels
Vertex radius: 15 pixels
Min cells for rosette: 5
======================================================================

Proceed with analysis? (y/n, default: y):
```

- Type `y` and then press **Enter** or to start the analysis
- Type `n` and then press **Enter** to cancel


### Step 4: Wait for Processing

You'll see progress messages like:
```
==================================================================
STEP 1: DETECTING INDIVIDUAL CELLS
==================================================================
Total objects detected: 1406
Valid cells (after size filtering): 1354

==================================================================
STEP 2: FINDING CELL BOUNDARIES AND CONTACT POINTS
==================================================================
Extracted boundaries for 1354 cells

==================================================================
STEP 3: IDENTIFYING ALL VERTICES WHERE CELLS MEET
==================================================================
...
```
If at any time you want to abort the current processing and start over, enter `Ctrl + C` simultaneously on your keyboard, and this will stop the processing.


**Processing time:**
- Small images (< 1000 cells): ~4 minutes
- Large images (> 1000 cells): 5-10 minutes
- Mac with GPU: Faster
- Windows with CPU: Slower (but it works!)

### Step 5: View Your Results

When processing completes, you'll see something like this (it may look slightly different if you chose batch processing, but in this case, you can find all of the output files in the directory name you entered in step 3.5d under "Running the Rosette Detection"):
```
======================================================================
Interactive visualization created: output/html/name_of_file.html
Open this file in your web browser to interact with the rosettes!
======================================================================

======================================================================
CSV export created: output/csv/name_of_file.csv
======================================================================
```

**Two outputs are created:**

1. **Interactive HTML Visualizations** (`PATH_TO_OUTPUT_FOLDER/html/name_of_file.html`)
   - Find the file `name_of_file.html`
   - Double-click to open it in your web browser (Chrome, Firefox, Safari, Edge, etc)
   - Hover over cells to see properties
   - Click to remove/restore rosettes
   - Download the current edited image from the viewer (**Download edited image (PNG)**)
   - Download an updated CSV from the viewer (**Download updated CSV**)

2. **CSV Data Files** (`PATH_TO_OUTPUT_FOLDER/csv/name_of_file.csv`)
   - Contains detailed cell properties
   - Includes morphological measurements
   - Shows junction participation counts
   - Lists number of neighbors for each cell
   - Open in Excel, Google Sheets, or any CSV reader

---

## Understanding the Output

### The Interactive Visualization

When you open `name_of_file.html`, you'll see:

**Top of page:**
- **Total Cells**: Number of individual cells detected
- **Total Rosettes**: Number of rosettes found
- **Cells in Rosettes**: How many cells are part of rosettes

**The Image:**
- **Cyan outlines**: All cell boundaries
- **Green cells**: Cells that are part of rosettes
- **Red Dots**: Centers of rosettes (vertices)

**Interactive Feature:**
- **Hover your mouse** over any green cell
- That cell's rosette will highlight in **orange**
- The rosette center will get bigger and show a label (R1, R2, etc.)
- Information box shows which cell and rosette you're looking at, as well as the area, perimeter, and number of neighbors for that cell
- Click a red dot to remove a rosette and click a grayed area to restore it
- Use **Download edited image (PNG)** to export what you currently see in the viewer
- Use **Download updated CSV** to export a CSV with recalculated `junctions_*` and `total_junctions` based on current active rosettes

### Understanding the Console Output

When the program runs, it also prints detailed information:

```
Total cells detected: 1354
Total rosettes found: 155

Rosette Details:
#     Center (x,y)         Num Cells    Cell IDs                      
1     (559, 25)            5            [38, 39, 81, 85, 62]
2     (917, 29)            5            [67, 99, 73, 9, 44]
...
```

- **Center (x,y)**: Pixel coordinates of the rosette center
- **Num Cells**: How many cells in that rosette
- **Cell IDs**: Which cells belong to that rosette


---

## CSV Data Export

### What is Exported?

In addition to the interactive visualization, the program automatically generates a detailed CSV file containing comprehensive data for every detected cell.

**Output Location**: `PATH_TO_OUTPUT_FOLDER/csv/name_of_file.csv`

**For each cell, the CSV includes:**
- **Morphological Properties**: Area, perimeter, shape metrics (eccentricity, solidity, etc.)
- **Location Data**: Centroid coordinates, orientation
- **Junction Counts**: How many 3-cell, 4-cell, 5-cell, 6-cell, 7-cell, and 8+ cell junctions each cell participates in

### Exporting Updated CSV from the Interactive Viewer

After removing/restoring rosettes in the HTML viewer, you can click **Download updated CSV**.

- The downloaded CSV keeps the original per-cell morphology/position columns.
- The junction columns (`junctions_3_cell` ... `junctions_8plus_cell`, `total_junctions`) are recalculated from the rosettes currently active in the viewer.
- Browser security does not allow directly overwriting files on disk from the standalone HTML, so exports are downloaded as new files.

### Testing CSV Export

To test the CSV export feature independently:
```bash
python tests/test_csv_export.py
```

This will generate a test CSV and display summary statistics.

---

## Adjusting Detection Parameters

If the detection isn't working well (finding too many or too few rosettes), you can adjust the settings.

### Where to Adjust Settings

Open `config.py` in a text editor. You'll see:

```python
# DETECTION PARAMETERS
CELL_DIAMETER = 30              # Estimated cell diameter for CellPose detection (pixels)
MIN_CELL_AREA = 100             # Minimum area to count as valid cell (pixels)
MAX_CELL_AREA = 5000            # Maximum area to count as valid cell (pixels)
VERTEX_RADIUS = 15              # Search radius for cells meeting at a vertex (pixels)
MIN_CELLS_FOR_ROSETTE = 5       # Minimum cells required to form a rosette
```
Or you can adjust in the terminal (as shown earlier in Step 3.5 of "Running the Rosette Detection")

### What Each Parameter Does

**CELL_DIAMETER** (default: 30)
- How big is a typical cell in your image (in pixels)?
- **Too small** → Might break one cell into multiple pieces
- **Too large** → Might merge multiple cells together
- **How to measure**: Open your image, zoom in, and count pixels across a typical cell

**MIN_CELL_AREA** (default: 100)
- Minimum size to count as a real cell
- **Too small** → Detects noise as cells
- **Too large** → Misses small cells
- Helps filter out debris and artifacts

**MAX_CELL_AREA** (default: 5000)
- Maximum size to count as a single cell
- **Too large** → Might count cell clumps as one cell
- **Too small** → Rejects large cells
- Helps filter out artifacts

**VERTEX_RADIUS** (default: 15)
- How close do cell boundaries need to be to "meet at a point"?
- **Too small** → Misses rosettes where cells don't touch perfectly
- **Too large** → Incorrectly groups distant cells as rosettes
- **Good starting point**: 0.5 to 1.0 times the cell diameter

**MIN_CELLS_FOR_ROSETTE** (default: 5)
- Minimum cells needed to form a rosette
- By most standard definitions, a rosette has 5+ cells

### Example Adjustments

**If you're getting too many false rosettes:**
```python
VERTEX_RADIUS = 10              # Make stricter - cells must be closer
MIN_CELL_AREA = 200             # Ignore smaller debris
```

**If you're missing real rosettes:**
```python
VERTEX_RADIUS = 20              # More lenient - cells can be farther apart
CELL_DIAMETER = 35              # If your cells are larger
```

**After changing settings**, run the program again:
```bash
python app.py
```

---

## Testing Individual Components

You can test each step of the detection process separately to diagnose issues.

### Test 1: Cell Segmentation Only

This shows just the cell detection, without rosette finding.

```bash
python tests/test_cell_segmentation.py
```

**Output:** `output/visualizations/test_cell_segmentation_results.png`

**What to look for:**
- Are all cells outlined in cyan?
- Are cells broken into pieces? (CELL_DIAMETER too small)
- Are multiple cells merged? (CELL_DIAMETER too large)
- Is debris detected as cells? (MIN_CELL_AREA too small)

### Test 2: Vertex Detection

This shows all points where 3+ cells meet.

```bash
python tests/test_vertex_detection.py
```

**Output:** `output/visualizations/test_vertex_detection_results.png`

**What to look for:**
- Red dots should appear where cells touch
- Too many dots? (VERTEX_RADIUS too large)
- Missing obvious junctions? (VERTEX_RADIUS too small)

### Test 3: Complete Rosette Detection

This is the full pipeline with visualization.

```bash
python tests/test_rosette_detection.py
```

**Output:** `output/visualizations/test_rosette_detection_results.png`

**What to look for:**
- Green cells are in rosettes
- Red stars mark rosette centers
- Red circle marks VERTEX_RADIUS
- Compare to what you expect to see

---

## Troubleshooting

### Problem: "Command not found: python"

**Solution:**
- Try `python3` instead of `python`
- Make sure Python is installed: Visit [python.org](https://www.python.org/downloads/)

### Problem: "ModuleNotFoundError: No module named 'cellpose'"

**Solution:**
- Make sure your virtual environment is activated (you should see `(rosette)` in your terminal)
- Run the pip install commands again

### Problem: "zsh: no matches found: cellpose[gui]" (Mac only)

**Solution:**
- Use quotes around the command:
```bash
pip install 'cellpose[gui]'
```

### Problem: Processing is very slow

**Solutions:**
- **If on Mac**: Check that GPU is enabled. Run:
  ```bash
  python -c "from cellpose import core; print(f'GPU available: {core.use_gpu()}')"
  ```
  Should say `True`. If `False`, reinstall cellpose.

- **If on Windows**: This is normal - CPU mode is slower. Be patient, it will complete.

- **For both**: Try processing a smaller region of your image first to test settings

### Problem: Too many/too few rosettes detected

**Solution:**
- See the **Adjusting Detection Parameters** section
- Most common fix: Adjust `VERTEX_RADIUS`
- Start with testing just cell segmentation to make sure cells are detected correctly


---

## Additional Resources

- **CellPose Documentation**: [cellpose.readthedocs.io](https://cellpose.readthedocs.io/)
- **Python Virtual Environments**: [docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)
- **Project Issues**: If you find bugs, report them on our GitHub Issues page

---

## For Developers

See [StyleGuide.md](StyleGuide.md) for code contribution guidelines.

### Project Structure
```
rosette-identification/             # root of the directory
├── app.py                          # Main application entry point
├── config.py                       # Configuration parameters
├── src/
│   ├── cell_segmentation.py        # Cell detection module
│   ├── vertex_detection.py         # Vertex identification module
│   └── rosette_detection.py        # Rosette clustering and visualization
├── tests/
│   ├── test_cell_segmentation.py   # Test cell detection
│   ├── test_vertex_detection.py    # Test vertex detection
│   └── test_rosette_detection.py   # Test complete pipeline
├── data/                           # There is an image and a folder here to test with
└── output/                         # Generated outputs
│   ├── data/                       # Text data outputs
│   └── visualizations/             # Static image outputs
└── Ecad_results_folfer/            # Example output of batch processed folder
    ├── csv/                        # csv data file outputs
    └── html/                       # interactive html file outputs
```
