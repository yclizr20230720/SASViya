# Focus-Exposure Matrix (FEM) Analysis Tool

PS C:\VSMC\Litho_Tools> python FEM.py fem_experimental_data.csv

A comprehensive Python tool for analyzing Focus-Exposure Matrix data in semiconductor lithography, based on the Mack & Byers (2003) improved physics-based model.

## Features

### Analysis Methods
- **Polynomial Fitting** - Traditional polynomial regression method
- **Physics-Based Fitting** - Improved model using lithography physics
- **Outlier Detection** - Automatic identification and removal of outliers
- **Model Comparison** - Side-by-side comparison of fitting methods

### Visualizations
- **Bossung Curves** - CD vs Focus at different exposure doses
- **3D Surface Plots** - Interactive 3D visualization of Focus-Exposure-CD relationship
- **Multi-Angle 3D Views** - Four different viewing angles of the FEM surface
- **Process Window Maps** - Contour plots showing acceptable process regions
- **Residual Analysis** - Model fit quality assessment
- **Noise Robustness** - Model performance under different noise levels

### Key Outputs
- Dose to Size (Es)
- Best Focus (F*)
- Normalized Image Log-Slope (NILS)
- Depth of Focus (DOF)
- Process Window dimensions
- Model fit statistics (χ², σ, residuals)

## Installation

### Requirements
```bash
pip install numpy matplotlib scipy pandas
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Mock Data

First, generate realistic FEM data using the data generator:

```bash
python generate_fem_data.py
```

This creates `fem_experimental_data.csv` with default parameters:
- 7 focus points (-0.3 to 0.3 μm)
- 9 exposure doses (18 to 26 mJ/cm²)
- 130 nm nominal CD
- 2 nm measurement noise

### 2. Run FEM Analysis

#### Basic Usage (Default CSV)
```bash
python FEM.py
```
Automatically loads `fem_experimental_data.csv` if it exists.

#### Specify CSV File
```bash
python FEM.py your_data.csv
```

#### With Custom Parameters
```bash
python FEM.py fem_data.csv --target-cd 90 --tolerance 5
```

#### Force Synthetic Data Mode
```bash
python FEM.py --synthetic
```

#### Get Help
```bash
python FEM.py --help
```

### 3. Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `csv_file` | positional | None | Path to CSV data file |
| `--synthetic` | flag | False | Force synthetic data generation |
| `--target-cd` | float | 130.0 | Target CD value (nm) |
| `--tolerance` | float | 10.0 | CD tolerance (nm) |

## CSV Data Format

Your CSV file must contain these columns:

| Column | Unit | Description | Required |
|--------|------|-------------|----------|
| Focus | μm | Focus offset | Yes |
| Exposure | mJ/cm² | Exposure dose | Yes |
| CD | nm | Critical Dimension | Yes |
| Weight | - | Measurement weight | No (default: 1.0) |

### Example CSV
```csv
Focus,Exposure,CD,Weight
-0.3,18.0,109.3,1.0
-0.2,18.0,105.4,1.0
-0.1,18.0,105.5,1.0
0.0,18.0,106.8,1.0
0.1,18.0,103.7,1.0
...
```

## Output

### Console Output
- Data statistics and summary
- Model fit quality metrics
- Physical parameters (Es, F*, NILS)
- Depth of Focus calculations
- Process window dimensions

### Graphical Output
The tool generates multiple figures:

1. **Model Comparison** - Residual plots for both models
2. **Bossung Curves (Polynomial)** - CD vs Focus curves
3. **Bossung Curves (Physics-Based)** - CD vs Focus curves
4. **3D Surface (Polynomial)** - Single view 3D plot
5. **3D Surface (Physics-Based)** - Single view 3D plot
6. **3D Multi-View (Polynomial)** - Four viewing angles
7. **3D Multi-View (Physics-Based)** - Four viewing angles
8. **Process Window (Polynomial)** - Contour map
9. **Process Window (Physics-Based)** - Contour map
10. **Noise Robustness** - Model comparison (synthetic mode only)
11. **Multi-Noise Comparison** - Performance at different noise levels (synthetic mode only)

## Examples

### Example 1: Analyze Experimental Data
```bash
# Generate mock data
python generate_fem_data.py

# Run analysis
python FEM.py fem_experimental_data.csv
```

### Example 2: Custom Target CD
```bash
python FEM.py my_data.csv --target-cd 90 --tolerance 8
```

### Example 3: Generate Different Mock Data
Edit `generate_fem_data.py` to customize parameters:
```python
df = generate_fem_csv(
    filename='fem_90nm.csv',
    n_focus=9,
    n_exposure=11,
    focus_range=(-0.4, 0.4),
    exposure_range=(25.0, 35.0),
    nominal_cd=90.0,
    dose_to_size=30.0,
    best_focus=-0.05,
    noise_level=1.5
)
```

### Example 4: Synthetic Data Mode
```bash
python FEM.py --synthetic --target-cd 130 --tolerance 10
```

## Understanding the Results

### Model Comparison
- **σ (sigma)**: Standard deviation of residuals - lower is better
- **χ² (chi-squared)**: Goodness of fit metric
- **Improvement**: Percentage improvement of physics-based over polynomial

### Physical Parameters
- **Es (Dose to Size)**: Exposure dose that produces target CD at best focus
- **F* (Best Focus)**: Optimal focus position
- **NILS**: Normalized Image Log-Slope - measure of image contrast

### Process Window
- Green region: CD within target ± tolerance
- Red dashed line: Target CD contour
- Larger window = more robust process

### 3D Visualization
- **Surface**: Fitted model prediction
- **Red points**: Experimental measurements
- **Color scale**: CD values
- Multiple viewing angles help understand the surface topology

## Typical Workflow for Process Engineers

1. **Collect FEM Data**
   - Run wafer with focus-exposure matrix
   - Measure CD at each condition
   - Export to CSV format

2. **Generate CSV File**
   - Format: Focus, Exposure, CD, Weight
   - Include all measurement points

3. **Run Analysis**
   ```bash
   python FEM.py your_fem_data.csv --target-cd 130 --tolerance 10
   ```

4. **Review Results**
   - Check model fit quality (σ values)
   - Identify best focus and dose to size
   - Evaluate process window size
   - Examine 3D surface for anomalies

5. **Make Process Decisions**
   - Set scanner focus based on F*
   - Set exposure dose based on Es
   - Verify DOF meets requirements
   - Confirm process window is adequate

## Troubleshooting

### Issue: "CSV file not found"
- Check file path is correct
- Ensure file is in current directory or provide full path

### Issue: "Failed to load CSV file"
- Verify CSV has required columns: Focus, Exposure, CD
- Check for missing or invalid data
- Ensure numeric values are properly formatted

### Issue: 3D plots not displaying
- Ensure matplotlib is properly installed
- Check that `mpl_toolkits.mplot3d` is available
- Run test: `python test_3d_plot.py`

### Issue: Poor model fit
- Check for outliers in data
- Verify measurement quality
- Consider adjusting model parameters
- Review data collection methodology

## References

- Mack, C. A., & Byers, J. (2003). "Improved Model for Focus-Exposure Matrix Data Analysis"
- Bossung, J. W. (1977). "Projection Printing Characterization"

## File Structure

```
.
├── FEM.py                          # Main analysis tool
├── generate_fem_data.py            # Mock data generator
├── test_3d_plot.py                 # 3D plotting test
├── FEM_README.md                   # This file
├── requirements.txt                # Python dependencies
└── fem_experimental_data.csv       # Generated data (after running generator)
```

## License

This tool is provided for educational and research purposes in semiconductor lithography process development.

## Support

For issues or questions:
1. Check this README
2. Review example usage
3. Verify CSV format
4. Test with synthetic data mode
