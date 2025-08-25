import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Union, Optional, Callable, Any, Dict

# ----------------- TABULAR DATA CLEANING -----------------
class DataCleaning:
    def __init__(self, df: pd.DataFrame, copy: bool = True):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        self.df = df.copy() if copy else df
        self.original_shape = df.shape
        self.operations_log = []

    def _validate_columns(self, columns: List[str]) -> None:
        """Validate that columns exist in the dataframe"""
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found: {missing_cols}")

    def _log_operation(self, operation: str, details: str) -> None:
        """Log operations for debugging and tracking"""
        self.operations_log.append({
            'operation': operation,
            'details': details,
            'shape_before': self.df.shape,
            'timestamp': pd.Timestamp.now()
        })

    def replace_symbols(self, columns: List[str], symbols: Optional[List[str]] = None, replacement: str = "") -> 'DataCleaning':
        """
        Replace specified symbols in given columns with a replacement string.
        If replacement is not specified, it defaults to an empty string.
        """
        self._validate_columns(columns)
        if symbols is None:
            symbols = [",", ".", "!", "?", "$", "%", "&"]
        pattern = "[" + "".join(map(lambda s: "\\" + s if s in r"\^$.|?*+()[]{}" else s, symbols)) + "]"
        for col in columns:
            self.df[col] = self.df[col].astype(str).replace(pattern, replacement, regex=True)
        self._log_operation("replace_symbols", f"Replaced symbols {symbols} with '{replacement}' in columns {columns}")
        return self

    def handle_missing(self, 
                      columns: List[str], 
                      method: str = "mean", 
                      fill_value: Optional[Any] = None, 
                      func: Optional[Callable] = None, 
                      ref_col: Optional[str] = None, 
                      **kwargs) -> 'DataCleaning':
        """
        Handle missing data in specified columns using various methods.
        Methods include:
        - "mean": Fill with column mean
        - "median": Fill with column median
        - "drop": Drop rows with missing values in specified columns
        - "value": Fill with a specified value
        - "column": Fill with values from another specified column
        - "ffill": Forward fill missing values
        - "bfill": Backward fill missing values
        - "interpolate": Interpolate missing values
        - "function": Apply a custom function to fill missing values
        """
        self._validate_columns(columns)
        if method == "value" and fill_value is None:
            raise ValueError("fill_value must be provided when method='value'")
        if method == "function" and func is None:
            raise ValueError("func must be provided when method='function'")
        if method == "column":
            if ref_col is None:
                raise ValueError("ref_col must be provided when method='column'")
            self._validate_columns([ref_col])
        
        handlers = {
            "mean": lambda col: self.df[col].fillna(self.df[col].mean()),
            "median": lambda col: self.df[col].fillna(self.df[col].median()),
            "drop": lambda col: self.df.dropna(subset=[col]),
            "value": lambda col: self.df[col].fillna(fill_value),
            "column": lambda col: self.df[col].fillna(self.df[ref_col]),
            "ffill": lambda col: self.df[col].ffill(),
            "bfill": lambda col: self.df[col].bfill(),
            "interpolate": lambda col: self.df[col].interpolate(**kwargs),
            "function": lambda col: self.df[col].apply(lambda x: func(x) if pd.isna(x) else x),
        }
        if method not in handlers:
            raise ValueError(f"Unknown method '{method}' for missing data handling.")
        for col in columns:
            if method == "drop":
                self.df = handlers[method](col)
            else:
                self.df[col] = handlers[method](col)
        self._log_operation("handle_missing", f"Handled missing values in columns {columns} using method '{method}'")
        return self

    def detect_outliers_zscore(self, columns: List[str], threshold: float = 2) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in specified columns using Z-score method.
        Returns a dictionary with column names as keys and DataFrames containing outliers as values.
        """
        self._validate_columns(columns)
        outliers_dict = {}
        
        for col in columns:
            non_null_data = self.df[col].dropna()
            z_scores = np.abs(stats.zscore(non_null_data))
            outlier_indices = non_null_data.index[z_scores >= threshold]
            outliers_dict[col] = self.df.loc[outlier_indices, [col]].copy()
            outliers_dict[col]['z_score'] = z_scores[z_scores >= threshold]
            
        self._log_operation("detect_outliers_zscore", f"Detected outliers in columns {columns} using Z-score method with threshold {threshold}")
        return outliers_dict

    def detect_outliers_iqr(self, columns: List[str], k: float = 1.5) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in specified columns using Interquartile Range (IQR) method.
        Returns a dictionary with column names as keys and DataFrames containing outliers as values.
        """
        self._validate_columns(columns)
        outliers_dict = {}
        
        for col in columns:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            
            mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outliers_dict[col] = self.df.loc[mask, [col]].copy()
            outliers_dict[col]['lower_bound'] = lower_bound
            outliers_dict[col]['upper_bound'] = upper_bound
            outliers_dict[col]['IQR'] = IQR
            
        self._log_operation("detect_outliers_iqr", f"Detected outliers in columns {columns} using IQR method with k={k}")
        return outliers_dict

    def handle_outliers_zscore(self, columns: List[str], threshold: float = 2, action: str = "remove") -> 'DataCleaning':
        """Handle outliers in specified columns using Z-score method."""
        outliers_dict = self.detect_outliers_zscore(columns, threshold)
        
        for col in columns:
            if len(outliers_dict[col]) > 0:
                outlier_indices = outliers_dict[col].index
                if action == "remove":
                    self.df = self.df.drop(outlier_indices)
                elif action == "nan":
                    self.df.loc[outlier_indices, col] = np.nan
                else:
                    raise ValueError(f"Unknown action '{action}' for outlier handling.")
        
        self._log_operation("handle_outliers_zscore", f"Handled outliers in columns {columns} using Z-score method with threshold {threshold}")
        return self

    def handle_outliers_iqr(self, columns: List[str], k: float = 1.5, action: str = "remove") -> 'DataCleaning':
        """Handle outliers in specified columns using Interquartile Range (IQR) method."""
        outliers_dict = self.detect_outliers_iqr(columns, k)
        
        for col in columns:
            if len(outliers_dict[col]) > 0:
                outlier_indices = outliers_dict[col].index
                if action == "remove":
                    self.df = self.df.drop(outlier_indices)
                elif action == "nan":
                    self.df.loc[outlier_indices, col] = np.nan
                else:
                    raise ValueError(f"Unknown action '{action}' for outlier handling.")
        
        self._log_operation("handle_outliers_iqr", f"Handled outliers in columns {columns} using IQR method with k={k}")
        return self
    
    def scale(self, columns: List[str], method: str = "standard") -> 'DataCleaning':
        """
        Scale specified columns using various scaling methods.
        Methods include:
        - "standard": StandardScaler (zero mean, unit variance)
        - "minmax": MinMaxScaler (scales to range [0, 1])
        - "robust": RobustScaler (scales using median and IQR)
        """
        self._validate_columns(columns)
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        if method not in scalers:
            raise ValueError(f"Unknown scaler '{method}'")
        self.df[columns] = scalers[method].fit_transform(self.df[columns])
        self._log_operation("scale", f"Scaled columns {columns} using {method} scaler")
        return self

    def binning(self, column: str, method: str = "cut", bins: Optional[Union[int, List]] = None, 
               labels: Optional[List] = None, q: Optional[int] = None, right: bool = True, 
               include_lowest: bool = True, duplicates: str = "raise") -> 'DataCleaning':
        """Bin a specified column using various binning methods."""
        self._validate_columns([column])
        bin_methods = {
            "cut": lambda: pd.cut(
                self.df[column], bins=bins, labels=labels, 
                right=right, include_lowest=include_lowest
            ),
            "qcut": lambda: pd.qcut(
                self.df[column], q=q, labels=labels, 
                duplicates=duplicates
            ),
            "mapping": lambda: self.df[column].map(bins),
        }
        if method not in bin_methods:
            raise ValueError(f"Unknown binning method '{method}'")
        self.df = self.df.assign(**{column + "_category": bin_methods[method]()})
        self._log_operation("binning", f"Binned column {column} using {method} method")
        return self
    
    def astype(self, columns: List[str], dtype: str) -> 'DataCleaning':
        """Convert specified columns to a given target data type."""
        self._validate_columns(columns)
        for col in columns:
            self.df[col] = self.df[col].astype(dtype)
        self._log_operation("astype", f"Converted columns {columns} to {dtype}")
        return self

    def process_column(self, column: str, func: Callable) -> 'DataCleaning':
        """
        Apply a custom function to a specified column.
        If the column does not exist, create it by applying the function to each row.
        The function should take a single value (or row if column is missing) and return a processed value.
        """
        if column in self.df.columns:
            self.df[column] = self.df[column].apply(func)
        else:
            self.df[column] = self.df.apply(lambda row: func(row), axis=1)
        self._log_operation("process_column", f"Applied custom function to column {column}")
        return self

    def get(self) -> pd.DataFrame:
        return self.df

# ----------------- IMAGE DATA CLEANING -----------------
class ImageCleaning:
    def __init__(self, images: Union[Dict, List, np.ndarray]):
        """
        Initialize ImageCleaning with various input types.
        Images to process. Can be:
        - Dict: {key: image_array, ...}
        - List: [image_array, ...]
        - np.ndarray: Single image array
        """
        if isinstance(images, list):
            self.images = {i: img for i, img in enumerate(images)}
        elif isinstance(images, np.ndarray):
            self.images = {0: images}
        elif isinstance(images, dict):
            self.images = images.copy()
        else:
            raise TypeError("Images must be dict, list, or numpy array")

    def resize(self, size: tuple = (128, 128)) -> 'ImageCleaning':
        """Resize all images to the specified size."""
        self.images = {k: cv2.resize(img, size) for k, img in self.images.items()}
        return self

    def convert_color(self, mode: str = "grayscale") -> 'ImageCleaning':
        """
        Convert all images to a specified color mode.
        Color mode. Options: "grayscale", "rgb"
        """
        converters = {
            "grayscale": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            "rgb": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        }
        if mode not in converters:
            raise ValueError(f"Unknown color mode '{mode}'")
        self.images = {k: converters[mode](img) for k, img in self.images.items()}
        return self

    def normalize(self, method: str = "0-1") -> 'ImageCleaning':
        """
        Normalize all images using specified method.
        Options: "0-1", "minus1-1"
        """
        normalizers = {
            "0-1": lambda img: img.astype("float32") / 255.0,
            "minus1-1": lambda img: (img.astype("float32") / 127.5) - 1,
        }
        if method not in normalizers:
            raise ValueError(f"Unknown normalization method '{method}'")
        self.images = {k: normalizers[method](img) for k, img in self.images.items()}
        return self

    def denoise(self, method: str = "gaussian") -> 'ImageCleaning':
        """
        Denoise all images using specified method.
        Options: "gaussian", "median", "bilateral", "box", "nl_means", "fastnl"
        """
        denoisers = {
            "gaussian": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            "median": lambda img: cv2.medianBlur(img, 5),
            "bilateral": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            "box": lambda img: cv2.blur(img, (5, 5)),
            "nl_means": lambda img: cv2.fastNlMeansDenoising(img, None, 10, 7, 21) if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
                                    else cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
            "fastnl": lambda img: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
        }
        if method not in denoisers:
            raise ValueError(f"Unknown denoise method '{method}'")
        self.images = {k: denoisers[method](img) for k, img in self.images.items()}
        return self

    def process_image(self, func: Callable) -> 'ImageCleaning':
        """
        Apply a custom function to each image.
        Function takes a single image and returns a processed image
        """
        self.images = {k: func(img) for k, img in self.images.items()}
        return self

    def get(self) -> Dict:
        return self.images
