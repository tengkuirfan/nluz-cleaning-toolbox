# NLuziaf's Cleaning Toolbox

**Version 0.3.0** - Refactored Release 🎉

A modular, Lego-like toolbox for **data cleaning** and **image preprocessing**, built for fast, reusable workflows. Currently, it is under closed testing and not yet available for public release.

Check demo folder in the source code for example.

## What's New in 0.3.0

- **Refactored Architecture**: Separated into `TabularCleaning` and `ImageCleaning` classes for better organization
- **Operations Logging**: Automatic tracking of all transformations applied
- **Enhanced Error Handling**: Comprehensive validation with helpful error messages
- **Type Hints**: Full type annotations for better IDE support
- **Improved Documentation**: Detailed docstrings for all methods
- **Better Method Chaining**: More intuitive fluent API

## Methods Reference

### TabularCleaning Class Methods

| Method | Description | Parameters | Supported Options |
|--------|-------------|------------|-------------------|
| `replace_symbols()` | Replace specified symbols in columns | `columns`, `symbols=None`, `replacement=""` | Default symbols: `,`, `.`, `!`, `?`, `$`, `%`, `&` |
| `handle_missing()` | Handle missing data using various strategies | `columns`, `method="mean"`, `fill_value=None`, `func=None`, `ref_col=None`, `**kwargs` | `"mean"`, `"median"`, `"drop"`, `"value"`, `"column"`, `"ffill"`, `"bfill"`, `"interpolate"`, `"function"` |
| `detect_outliers_zscore()` | Detect outliers using Z-score method (returns DataFrame of outliers) | `columns`, `threshold=2` | Returns `pd.DataFrame` containing outlier rows |
| `detect_outliers_iqr()` | Detect outliers using IQR method (returns DataFrame of outliers) | `columns`, `k=1.5` | Returns `pd.DataFrame` containing outlier rows |
| `handle_outliers_zscore()` | Handle outliers using Z-score method | `columns`, `threshold=2`, `action="remove"` | Actions: `"remove"`, `"nan"` |
| `handle_outliers_iqr()` | Handle outliers using IQR method | `columns`, `k=1.5`, `action="remove"` | Actions: `"remove"`, `"nan"` |
| `scale()` | Scale columns using various methods | `columns`, `method="standard"` | `"standard"`, `"minmax"`, `"robust"` |
| `binning()` | Create bins for continuous data | `column`, `method="cut"`, `bins=None`, `labels=None`, `q=None`, `right=True`, `include_lowest=True`, `duplicates="raise"` | `"cut"`, `"qcut"`, `"mapping"` |
| `astype()` | Convert columns to specified data type | `columns`, `dtype` | Any valid pandas dtype: `"int"`, `"float"`, `"str"`, `"bool"`, etc. |
| `process_column()` | Apply custom function to existing column or create new column from row data | `column`, `func` | Function takes single value (if column exists) or entire row (if creating new column) |
| `get()` | Return the processed DataFrame | None | Returns `pd.DataFrame` |

### ImageCleaning Class Methods

| Method | Description | Parameters | Supported Options |
|--------|-------------|------------|-------------------|
| `resize()` | Resize all images to specified dimensions | `size=(128, 128)` | Any tuple `(width, height)` |
| `convert_color()` | Convert images to different color modes | `mode="grayscale"` | `"grayscale"`, `"rgb"` |
| `normalize()` | Normalize pixel values | `method="0-1"` | `"0-1"` (scale to [0,1]), `"minus1-1"` (scale to [-1,1]) |
| `denoise()` | Apply denoising filters to images | `method="gaussian"` | `"gaussian"`, `"median"`, `"bilateral"`, `"box"`, `"nl_means"`, `"fastnl"` |
| `process_image()` | Apply custom function to each image | `func` | Any custom function that takes an image and returns processed image |
| `get()` | Return the processed images dictionary | None | Returns `dict` of processed images |

### Usage Notes

**TabularCleaning:**
- All methods return `self` for method chaining (except `get()` and `detect_outliers_*()`)
- Input: `pd.DataFrame` (required parameter in constructor)
- Constructor parameter `copy=True` creates a copy of the DataFrame (default behavior)
- Access `operations_log` attribute to see all transformations applied
- Access `original_shape` to see the original DataFrame dimensions
- Output: Modified `TabularCleaning` instance (use `.get()` to retrieve DataFrame)

**ImageCleaning:**
- All methods return `self` for method chaining (except `get()`)
- Input: `dict` of images, `list` of images, or single `np.ndarray`
- Images are automatically indexed if provided as list or single array
- Output: Modified `ImageCleaning` instance (use `.get()` to retrieve dict of images)
- Images should be in OpenCV format (numpy arrays: 2D grayscale or 3D color)

## Quick Examples

### Tabular Data Cleaning

```python
from nluztoolbox import TabularCleaning
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize the cleaning pipeline
cleaner = TabularCleaning(df, copy=True)

# Detect outliers first (returns DataFrame of outliers)
outliers = cleaner.detect_outliers_zscore(['price', 'quantity'], threshold=2.5)
print(f"Found {len(outliers)} outlier rows")

# Chain multiple operations
result = (cleaner
    .replace_symbols(['text_column'], symbols=[',', '.'], replacement='')
    .handle_missing(['age'], method='median')
    .handle_outliers_iqr(['price'], k=1.5, action='remove')
    .scale(['price', 'quantity'], method='standard')
    .get())

# Check operations log
for log in cleaner.operations_log:
    print(f"{log['operation']}: {log['details']}")
```

### Image Preprocessing

```python
from nluztoolbox import ImageCleaning
import cv2

# Load images
images = {
    'img1': cv2.imread('image1.jpg'),
    'img2': cv2.imread('image2.jpg'),
    'img3': cv2.imread('image3.jpg')
}

# Process images with method chaining
processed = (ImageCleaning(images)
    .resize(size=(224, 224))
    .convert_color(mode='rgb')
    .denoise(method='gaussian')
    .normalize(method='0-1')
    .get())

# Access processed images
for name, img in processed.items():
    print(f"{name}: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
```

### Advanced Usage

```python
from nluztoolbox import TabularCleaning
import pandas as pd

df = pd.DataFrame({
    'age': [18, 25, 35, 45, 55, 65, 75],
    'income': [30000, 50000, 75000, 90000, 85000, 70000, 50000]
})

cleaner = TabularCleaning(df)

# Create age categories
cleaner.binning(
    column='age', 
    method='cut', 
    bins=[0, 30, 60, 100], 
    labels=['Young', 'Middle', 'Senior']
)

# Add custom calculated column
cleaner.process_column(
    column='income_tier',
    func=lambda row: 'High' if row['income'] > 80000 else 'Low'
)

result = cleaner.get()
print(result)
```

## Installation

### From PyPI (when published)
```bash
pip install nluz-cleaning-toolbox
```

### From Source
```bash
git clone https://github.com/tengkuirfan/nluz-cleaning-toolbox.git
cd nluz-cleaning-toolbox
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/tengkuirfan/nluz-cleaning-toolbox.git
cd nluz-cleaning-toolbox
pip install -e ".[dev]"
```

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and changes.

## Migration from v0.2.x

If you're upgrading from version 0.2.x, please see the [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed instructions on updating your code.

**Quick Summary:**
- `DataCleaning` → `TabularCleaning`
- Outlier detection methods now return `pd.DataFrame` instead of `dict`
- Constructor now accepts `copy=True` parameter
- New operations logging feature

## Features

### TabularCleaning
- ✅ Symbol replacement and text cleaning
- ✅ Multiple missing value handling strategies
- ✅ Outlier detection and handling (Z-score and IQR)
- ✅ Feature scaling (Standard, MinMax, Robust)
- ✅ Data binning (cut, qcut, custom mapping)
- ✅ Type conversion
- ✅ Custom column processing
- ✅ Operations logging
- ✅ Method chaining

### ImageCleaning
- ✅ Flexible input (single image, list, or dict)
- ✅ Image resizing
- ✅ Color space conversion
- ✅ Multiple normalization methods
- ✅ Various denoising filters
- ✅ Custom image processing
- ✅ Batch processing
- ✅ Method chaining

## Testing

The package includes comprehensive tests. To run tests:

```bash
# Run test suite
python test_refactored_code.py

# Run demo script
python demo_refactored.py
```

See [TESTING_RESULTS.md](TESTING_RESULTS.md) for detailed testing documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Author

**Tengku Irfan**
- Email: tengku.irfan0278@student.unri.ac.id
- GitHub: [@tengkuirfan](https://github.com/tengkuirfan)

## Acknowledgments

This toolbox was created to streamline data cleaning and image preprocessing workflows for data science and machine learning projects.
