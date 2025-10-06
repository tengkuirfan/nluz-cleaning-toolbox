# Migration Guide: v0.2.x → v0.3.0

This guide will help you migrate from nluztoolbox v0.2.x to v0.3.0.

## Overview

Version 0.3.0 introduces a major refactoring with breaking changes. The main change is renaming `DataCleaning` to `TabularCleaning` for better clarity and improved functionality.

## Breaking Changes Summary

| Area | v0.2.x | v0.3.0 |
|------|--------|--------|
| Class Name | `DataCleaning` | `TabularCleaning` |
| Import | `from nluztoolbox import DataCleaning` | `from nluztoolbox import TabularCleaning` |
| Outlier Detection Return | `dict` | `pd.DataFrame` |
| Constructor | `DataCleaning(df)` | `TabularCleaning(df, copy=True)` |

## Step-by-Step Migration

### 1. Update Import Statements

**Before (v0.2.x):**
```python
from nluztoolbox import DataCleaning
import pandas as pd

df = pd.read_csv('data.csv')
cleaner = DataCleaning(df)
```

**After (v0.3.0):**
```python
from nluztoolbox import TabularCleaning
import pandas as pd

df = pd.read_csv('data.csv')
cleaner = TabularCleaning(df, copy=True)
```

### 2. Update Outlier Detection Code

The outlier detection methods now return a DataFrame instead of a dictionary.

**Before (v0.2.x):**
```python
outliers = cleaner.detect_outliers_zscore(['price', 'quantity'], threshold=2)
# Returns: {'price': [indices], 'quantity': [indices], 'z_scores': {'price': [values], 'quantity': [values]}}

# Accessing outliers
price_outlier_indices = outliers['price']
print(f"Found {len(price_outlier_indices)} price outliers")
```

**After (v0.3.0):**
```python
outliers_df = cleaner.detect_outliers_zscore(['price', 'quantity'], threshold=2)
# Returns: DataFrame containing all rows with outliers in any of the specified columns

# Accessing outliers
print(f"Found {len(outliers_df)} rows with outliers")
print(outliers_df)

# You can now easily inspect or save the outliers
outliers_df.to_csv('outliers.csv', index=False)
```

### 3. Take Advantage of New Features

#### Operations Logging

**New in v0.3.0:**
```python
cleaner = TabularCleaning(df)

# Perform operations
cleaner.replace_symbols(['price'], symbols=[','])
cleaner.handle_missing(['age'], method='mean')
cleaner.scale(['income'], method='standard')

# View operation history
print("Operations performed:")
for i, log in enumerate(cleaner.operations_log, 1):
    print(f"{i}. {log['operation']}")
    print(f"   Details: {log['details']}")
    print(f"   Shape before: {log['shape_before']}")
    print(f"   Timestamp: {log['timestamp']}")
```

#### Access Original Shape

**New in v0.3.0:**
```python
cleaner = TabularCleaning(df)
print(f"Original shape: {cleaner.original_shape}")

# After operations
cleaner.handle_outliers_iqr(['rating'], k=1.5, action='remove')
print(f"Current shape: {cleaner.get().shape}")
print(f"Removed {cleaner.original_shape[0] - cleaner.get().shape[0]} rows")
```

### 4. ImageCleaning - New Single Image Support

**Before (v0.2.x):**
```python
from nluztoolbox import ImageCleaning
import cv2

# Had to use dict or list
images = {0: cv2.imread('image.jpg')}
cleaner = ImageCleaning(images)
```

**After (v0.3.0):**
```python
from nluztoolbox import ImageCleaning
import cv2

# Can now pass single image directly
img = cv2.imread('image.jpg')
cleaner = ImageCleaning(img)

# Or use list/dict as before
images = [img1, img2, img3]
cleaner = ImageCleaning(images)

# Or dict
images = {'img1': img1, 'img2': img2}
cleaner = ImageCleaning(images)
```

## Common Migration Patterns

### Pattern 1: Basic Data Cleaning Pipeline

**Before (v0.2.x):**
```python
from nluztoolbox import DataCleaning
import pandas as pd

df = pd.read_csv('data.csv')
cleaner = DataCleaning(df)
result = (cleaner
    .replace_symbols(['price'], symbols=[','])
    .handle_missing(['age'], method='median')
    .handle_outliers_iqr(['rating'], k=1.5, action='remove')
    .get())
```

**After (v0.3.0):**
```python
from nluztoolbox import TabularCleaning  # Changed import
import pandas as pd

df = pd.read_csv('data.csv')
cleaner = TabularCleaning(df, copy=True)  # Updated constructor
result = (cleaner
    .replace_symbols(['price'], symbols=[','])
    .handle_missing(['age'], method='median')
    .handle_outliers_iqr(['rating'], k=1.5, action='remove')
    .get())

# Bonus: Check operations log
for log in cleaner.operations_log:
    print(f"Applied: {log['operation']}")
```

### Pattern 2: Outlier Analysis

**Before (v0.2.x):**
```python
from nluztoolbox import DataCleaning
import pandas as pd

df = pd.read_csv('data.csv')
cleaner = DataCleaning(df)

# Detect outliers
outliers = cleaner.detect_outliers_zscore(['price'], threshold=2)
outlier_indices = outliers['price']

# Manually extract outlier rows
outlier_rows = df.loc[outlier_indices]
print(outlier_rows)
```

**After (v0.3.0):**
```python
from nluztoolbox import TabularCleaning  # Changed import
import pandas as pd

df = pd.read_csv('data.csv')
cleaner = TabularCleaning(df, copy=True)  # Updated constructor

# Detect outliers (now returns DataFrame directly)
outliers_df = cleaner.detect_outliers_zscore(['price'], threshold=2)
print(f"Found {len(outliers_df)} outlier rows")
print(outliers_df)

# Can easily save or analyze
outliers_df.to_csv('price_outliers.csv', index=False)
```

### Pattern 3: Image Processing

**No changes needed for ImageCleaning**, but you can now use single images:

**Before (v0.2.x):**
```python
from nluztoolbox import ImageCleaning
import cv2

img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

# Had to wrap in list/dict
images = [img1, img2]
result = (ImageCleaning(images)
    .resize((128, 128))
    .normalize('0-1')
    .get())
```

**After (v0.3.0) - Still works + new option:**
```python
from nluztoolbox import ImageCleaning
import cv2

# Old way still works
img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')
images = [img1, img2]
result = (ImageCleaning(images)
    .resize((128, 128))
    .normalize('0-1')
    .get())

# New way - single image support
single_img = cv2.imread('image.jpg')
processed = (ImageCleaning(single_img)  # Direct single image
    .resize((128, 128))
    .normalize('0-1')
    .get())
```

## Automated Migration Script

You can use this simple script to help automate the migration:

```python
import os
import re

def migrate_file(filepath):
    """Migrate a Python file from v0.2.x to v0.3.0"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace imports
    content = content.replace(
        'from nluztoolbox import DataCleaning',
        'from nluztoolbox import TabularCleaning'
    )
    content = content.replace(
        'import nluztoolbox.DataCleaning',
        'import nluztoolbox.TabularCleaning'
    )
    
    # Replace class instantiation
    content = re.sub(
        r'DataCleaning\s*\(\s*([^)]+)\s*\)',
        r'TabularCleaning(\1, copy=True)',
        content
    )
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Migrated: {filepath}")

# Usage:
# migrate_file('your_script.py')
```

**Note:** This script handles basic cases. Review changes carefully and test thoroughly.

## Checklist

Use this checklist to ensure complete migration:

- [ ] Update all `DataCleaning` imports to `TabularCleaning`
- [ ] Update constructor calls to include `copy=True` parameter
- [ ] Update outlier detection code to handle DataFrame return type
- [ ] Test all data cleaning pipelines
- [ ] Consider using new operations logging feature
- [ ] Update any documentation or comments
- [ ] Run your test suite
- [ ] Review error handling (new error messages may be different)

## Getting Help

If you encounter issues during migration:

1. Check the [CHANGELOG.md](CHANGELOG.md) for detailed changes
2. Review the [README.md](README.md) for updated examples
3. Check [TESTING_RESULTS.md](TESTING_RESULTS.md) for test examples
4. Open an issue on GitHub: https://github.com/tengkuirfan/nluz-cleaning-toolbox/issues

## Benefits of Upgrading

After migration, you'll benefit from:

- ✅ **Better Error Messages**: More helpful validation and error reporting
- ✅ **Operations Logging**: Track all transformations automatically
- ✅ **Type Hints**: Better IDE support and autocomplete
- ✅ **Improved Documentation**: Detailed docstrings for all methods
- ✅ **More Flexibility**: Single image support in ImageCleaning
- ✅ **Better Outlier Handling**: DataFrame return makes outlier analysis easier
- ✅ **Future-Proof**: Better structure for future enhancements

## Timeline

- **v0.2.x**: Will receive critical bug fixes only
- **v0.3.x**: Current stable version with new features
- **Future**: New features will only be added to v0.3.x and beyond

We recommend migrating as soon as possible to benefit from the improvements and receive future updates.
