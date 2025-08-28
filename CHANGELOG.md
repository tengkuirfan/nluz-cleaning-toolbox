# Changelog

## [0.2.2] - 2025-08-28

### Changed
- **BREAKING CHANGE**: Modified `detect_outliers_zscore()` and `detect_outliers_iqr()` methods to return a single DataFrame instead of a dictionary
  - Methods now return a DataFrame containing all outlier rows from the specified columns
  - Simplified return type makes it easier to work with outliers in downstream analysis
  - More intuitive and consistent with pandas conventions

### Improved
- Refactored outlier detection methods to use helper methods for better code organization
- Added `_get_column_outliers_zscore()` and `_get_column_outliers_iqr()` helper methods
- Eliminated code duplication between detection and handling methods
- More concise and maintainable codebase following DRY principles

### Technical Details
- Detection methods now aggregate outliers from all specified columns into a single result
- Handler methods optimized for different actions (remove vs. nan)
- Improved performance by avoiding redundant calculations

## [0.2.1] - 2025-08-25

### Added
- `detect_outliers_zscore()` method to detect outliers using Z-score without modifying data
- `detect_outliers_iqr()` method to detect outliers using IQR without modifying data
- Both detection methods return detailed information about outliers including:
  - Z-scores for outliers (zscore method)
  - Boundary values and IQR calculations (IQR method)

### Changed
- Refactored `handle_outliers_zscore()` to use the new detection method internally
- Refactored `handle_outliers_iqr()` to use the new detection method internally
- Improved code maintainability by following DRY (Don't Repeat Yourself) principle
- Enhanced performance by adding checks to only process columns with actual outliers

### Technical Improvements
- Centralized outlier detection logic to eliminate code duplication
- Better separation of concerns: detection vs. handling of outliers
- More consistent behavior between related methods

## [0.2.0] - Previous Version

### Added
- Initial (actual) release with core data cleaning and image processing functionality
- DataCleaning class with methods for handling missing data, outliers, scaling, and more
- ImageCleaning class with methods for resizing, color conversion, normalization, and denoising
- Method chaining support for fluent API usage
- Comprehensive documentation and examples
