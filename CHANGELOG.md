# Changelog

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
