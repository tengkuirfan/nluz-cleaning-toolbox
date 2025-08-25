# NLuziaf's Cleaning Toolbox

A modular, Lego-like toolbox for **data cleaning** and **image preprocessing**, built for fast, reusable workflows. Currently, it is under closed testing and not yet available for public release.

Check demo folder in the source code for example.

## Methods Reference

### DataCleaning Class Methods

| Method | Description | Parameters | Supported Options |
|--------|-------------|------------|-------------------|
| `replace_symbols()` | Replace specified symbols in columns | `columns`, `symbols=None`, `replacement=""` | Default symbols: `,`, `.`, `!`, `?`, `$`, `%`, `&` |
| `handle_missing()` | Handle missing data using various strategies | `columns`, `method="mean"`, `fill_value=None`, `func=None`, `ref_col=None` | `"mean"`, `"median"`, `"drop"`, `"value"`, `"column"`, `"ffill"`, `"bfill"`, `"interpolate"`, `"function"` |
| `detect_outliers_zscore()` | Detect outliers using Z-score method (returns outliers without modifying data) | `columns`, `threshold=2` | Returns dict with outliers and their Z-scores |
| `detect_outliers_iqr()` | Detect outliers using IQR method (returns outliers without modifying data) | `columns`, `k=1.5` | Returns dict with outliers and boundary values |
| `handle_outliers_zscore()` | Handle outliers using Z-score method | `columns`, `threshold=2`, `action="remove"` | Actions: `"remove"`, `"nan"` |
| `handle_outliers_iqr()` | Handle outliers using IQR method | `columns`, `k=1.5`, `action="remove"` | Actions: `"remove"`, `"nan"` |
| `scale()` | Scale columns using various methods | `columns`, `method="standard"` | `"standard"`, `"minmax"`, `"robust"` |
| `binning()` | Create bins for continuous data | `column`, `method="cut"`, `bins=None`, `labels=None`, `q=None` | `"cut"`, `"qcut"`, `"mapping"` |
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

**DataCleaning:**
- All methods return `self` for method chaining (except `get()`)
- Input: `pd.DataFrame`
- Output: Modified `DataCleaning` instance (use `.get()` to retrieve DataFrame)

**ImageCleaning:**
- All methods return `self` for method chaining (except `get()`)
- Input: `dict` of images or `list` of images
- Output: Modified `ImageCleaning` instance (use `.get()` to retrieve images)
- Images should be in OpenCV format (numpy arrays)

## Quick Example

```python
from nluztoolbox import DataCleaning
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize the cleaning pipeline
cleaner = DataCleaning(df)

# Detect outliers first (without modifying data)
outliers_zscore = cleaner.detect_outliers_zscore(['price', 'quantity'], threshold=2.5)
print(f"Found {len(outliers_zscore['price'])} price outliers")

# Chain multiple operations
result = (cleaner
    .replace_symbols(['text_column'], symbols=[',', '.'], replacement='')
    .handle_missing(['age'], method='median')
    .handle_outliers_iqr(['price'], k=1.5, action='remove')
    .scale(['price', 'quantity'], method='standard')
    .get())
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
