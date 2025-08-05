# NLuziaf's Cleaning Toolbox

A modular, Lego-like toolbox for **data cleaning** and **image preprocessing**, built for fast, reusable workflows. Currently, it is under closed testing and not yet available for public release.

## Methods Reference

### DataCleaning Class Methods

| Method | Description | Parameters | Supported Options |
|--------|-------------|------------|-------------------|
| `replace_symbols()` | Replace specified symbols in columns | `columns`, `symbols=None`, `replacement=""` | Default symbols: `,`, `.`, `!`, `?`, `$`, `%`, `&` |
| `handle_missing()` | Handle missing data using various strategies | `columns`, `method="mean"`, `fill_value=None`, `func=None`, `ref_col=None` | `"mean"`, `"median"`, `"mode"`, `"drop"`, `"value"`, `"column"`, `"ffill"`, `"bfill"`, `"interpolate"`, `"function"` |
| `handle_outliers_zscore()` | Handle outliers using Z-score method | `columns`, `threshold=2`, `action="remove"` | Actions: `"remove"`, `"nan"` |
| `handle_outliers_iqr()` | Handle outliers using IQR method | `columns`, `k=1.5`, `action="remove"` | Actions: `"remove"`, `"nan"` |
| `scale()` | Scale columns using various methods | `columns`, `method="standard"` | `"standard"`, `"minmax"`, `"robust"` |
| `binning()` | Create bins for continuous data | `column`, `method="cut"`, `bins=None`, `labels=None`, `q=None` | `"cut"`, `"qcut"`, `"mapping"` |
| `astype()` | Convert columns to specified data type | `columns`, `dtype` | Any valid pandas dtype: `"int"`, `"float"`, `"str"`, `"bool"`, etc. |
| `process_column()` | Apply custom function to a column | `column`, `func` | Any custom function that takes a value and returns processed value |
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

## Quick Start

```python
from nluztoolbox import DataCleaning, ImageCleaning
import pandas as pd

# Data cleaning example
df = pd.DataFrame({'price': ['Rp1,000', 'Rp2,500'], 'rating': [4.5, None]})
cleaner = DataCleaning(df)
clean_df = (cleaner
           .replace_symbols(['price'], symbols=['Rp', ','])
           .handle_missing(['rating'], method='mean')
           .get())
```
