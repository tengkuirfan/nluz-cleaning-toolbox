# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-10-06

### 🎉 Major Refactoring Release

This release represents a significant refactoring of the codebase with breaking changes. The package has been reorganized into two separate classes for better maintainability and clarity.

### ⚠️ Breaking Changes

- **API Change**: `DataCleaning` class has been renamed to `TabularCleaning`
- **Import Change**: Update imports from `from nluztoolbox import DataCleaning` to `from nluztoolbox import TabularCleaning`
- **Outlier Detection**: `detect_outliers_zscore()` and `detect_outliers_iqr()` now return a `pd.DataFrame` of outlier rows instead of a dict

### ✨ Added

- **Operations Logging**: Automatic tracking of all transformations
  - New `operations_log` attribute tracks all operations with details
  - Each log entry includes operation name, details, shape before, and timestamp
  - Access via `cleaner.operations_log` after applying transformations

- **Better Error Handling**: Comprehensive validation
  - Column existence validation with helpful error messages
  - Required parameter validation for each method
  - Type validation for DataFrame and image inputs
  - Empty DataFrame detection

- **Type Hints**: Full type annotations throughout codebase
  - Better IDE support and autocomplete
  - Improved code documentation
  - Easier debugging

- **Enhanced Documentation**: Detailed docstrings for all methods
  - Parameter descriptions
  - Supported options
  - Usage examples in docstrings

### 🔄 Changed

- **Class Restructuring**:
  - `DataCleaning` → `TabularCleaning` (for pandas DataFrames)
  - `ImageCleaning` remains the same but with improved structure
  
- **TabularCleaning Constructor**:
  - Now requires DataFrame as first parameter: `TabularCleaning(df, copy=True)`
  - Added `copy` parameter to control whether to copy the DataFrame (default: True)
  - Validates input is a pandas DataFrame
  - Stores original shape for reference

- **ImageCleaning Constructor**:
  - Now accepts single `np.ndarray` in addition to list and dict
  - Validates image dimensions (must be 2D or 3D)
  - Better type checking and error messages

- **Outlier Detection Methods**:
  - `detect_outliers_zscore()` now returns `pd.DataFrame` instead of dict
  - `detect_outliers_iqr()` now returns `pd.DataFrame` instead of dict
  - Easier to inspect and analyze outliers
  - Can be directly saved or further processed

- **Binning Method**:
  - Added `right`, `include_lowest`, and `duplicates` parameters
  - More control over bin creation
  - Better parameter validation

### 🐛 Fixed

- Fixed missing value handling for non-numeric columns
- Improved scaling validation to check for missing values
- Better error messages when required parameters are missing
- Fixed edge cases in outlier detection

### 📚 Documentation

- Updated README.md with new API examples
- Added comprehensive testing documentation (TESTING_RESULTS.md)
- Created demo scripts showcasing new features
- Updated all code examples to use new class names

### 🧪 Testing

- Added comprehensive test suite (test_refactored_code.py)
- Added demo script (demo_refactored.py)
- All 31 tests passing
- Coverage includes:
  - TabularCleaning: 10 operations tested
  - ImageCleaning: 10 operations tested
  - Error handling: 7 scenarios tested
  - Operations logging verified

### 🔧 Internal

- Separated helper methods with underscore prefix
- Improved code organization and readability
- Consistent error handling patterns
- Better parameter validation

## [0.2.2] - Previous Release

### Added
- Basic data cleaning functionality
- Image preprocessing capabilities
- Method chaining support

### Features
- Symbol replacement
- Missing value handling
- Outlier detection and handling
- Feature scaling
- Data binning
- Image resizing and normalization
- Color space conversion
- Image denoising

---

## Migration Guide (0.2.x → 0.3.0)

### Update Import Statement

**Before:**
```python
from nluztoolbox import DataCleaning
```

**After:**
```python
from nluztoolbox import TabularCleaning
```

### Update Constructor Call

**Before:**
```python
cleaner = DataCleaning(df)
```

**After:**
```python
cleaner = TabularCleaning(df, copy=True)
```

### Update Outlier Detection

**Before:**
```python
outliers = cleaner.detect_outliers_zscore(['price'], threshold=2)
# Returns: {'price': [indices], 'z_scores': [values]}
price_outliers = outliers['price']
```

**After:**
```python
outliers_df = cleaner.detect_outliers_zscore(['price'], threshold=2)
# Returns: DataFrame containing outlier rows
print(f"Found {len(outliers_df)} outliers")
print(outliers_df)
```

### Take Advantage of New Features

```python
# Use operations logging
cleaner = TabularCleaning(df)
cleaner.handle_missing(['age'], method='mean')
cleaner.scale(['income'], method='standard')

# Check what operations were performed
for log in cleaner.operations_log:
    print(f"{log['operation']}: {log['details']}")
```

### ImageCleaning - Now More Flexible

**New: Single Image Support**
```python
import cv2
from nluztoolbox import ImageCleaning

# Can now pass single image directly
img = cv2.imread('image.jpg')
processed = ImageCleaning(img).resize((128, 128)).normalize('0-1').get()
```

---

## Support

For issues, questions, or contributions, please visit:
- GitHub Issues: https://github.com/tengkuirfan/nluz-cleaning-toolbox/issues
- Source Code: https://github.com/tengkuirfan/nluz-cleaning-toolbox
