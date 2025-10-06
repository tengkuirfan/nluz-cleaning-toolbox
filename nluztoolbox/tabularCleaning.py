import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Union, Optional, Callable, Any

# ----------------- TABULAR DATA CLEANING -----------------
class TabularCleaning:
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

    def _log_operation(self, operation: str, details: str, shape_before: tuple) -> None:
        """Log operations for debugging and tracking"""
        self.operations_log.append({
            'operation': operation,
            'details': details,
            'shape_before': shape_before,
            'timestamp': pd.Timestamp.now()
        })

    def replace_symbols(self, columns: List[str], symbols: Optional[List[str]] = None, replacement: str = "") -> 'TabularCleaning':
        """
        Replace specified symbols in given columns with a replacement string.
        If replacement is not specified, it defaults to an empty string.
        """
        self._validate_columns(columns)
        if symbols is None:
            symbols = [",", ".", "!", "?", "$", "%", "&"]
        pattern = "[" + "".join(map(lambda s: "\\" + s if s in r"\^$.|?*+()[]{}" else s, symbols)) + "]"
        shape_before = self.df.shape
        for col in columns:
            self.df[col] = self.df[col].astype(str).replace(pattern, replacement, regex=True)
        self._log_operation("replace_symbols", f"Replaced symbols {symbols} with '{replacement}' in columns {columns}", shape_before)
        return self

    def handle_missing(self, 
                      columns: List[str], 
                      method: str = "mean", 
                      fill_value: Optional[Any] = None, 
                      func: Optional[Callable] = None, 
                      ref_col: Optional[str] = None, 
                      **kwargs) -> 'TabularCleaning':
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
        shape_before = self.df.shape
        if method == "drop":
            # Drop rows with missing values in any of the specified columns
            self.df = self.df.dropna(subset=columns)
        else:
            for col in columns:
                self.df[col] = handlers[method](col)
        self._log_operation("handle_missing", f"Handled missing values in columns {columns} using method '{method}'", shape_before)
        return self

    def _get_column_outliers_zscore(self, column: str, threshold: float = 2) -> List[int]:
        """Helper method to get outlier indices for a specific column using Z-score method."""
        non_null_data = self.df[column].dropna()
        z_scores = np.abs(stats.zscore(non_null_data))
        outlier_indices = non_null_data.index[z_scores >= threshold]
        return outlier_indices.tolist()

    def _get_column_outliers_iqr(self, column: str, k: float = 1.5) -> List[int]:
        """Helper method to get outlier indices for a specific column using IQR method."""
        Q1, Q3 = self.df[column].quantile(0.25), self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        outlier_indices = self.df.index[mask]
        return outlier_indices.tolist()

    def detect_outliers_zscore(self, columns: List[str], threshold: float = 2) -> pd.DataFrame:
        """
        Detect outliers in specified columns using Z-score method.
        Returns a DataFrame containing all outliers from the specified columns.
        """
        self._validate_columns(columns)
        shape_before = self.df.shape
        all_outlier_indices = set()
        for col in columns:
            outlier_indices = self._get_column_outliers_zscore(col, threshold)
            all_outlier_indices.update(outlier_indices)
        outliers_df = self.df.loc[list(all_outlier_indices)].copy()
        self._log_operation("detect_outliers_zscore", f"Detected outliers in columns {columns} using Z-score method with threshold {threshold}", shape_before)
        return outliers_df

    def detect_outliers_iqr(self, columns: List[str], k: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in specified columns using Interquartile Range (IQR) method.
        Returns a DataFrame containing all outliers from the specified columns.
        """
        self._validate_columns(columns)
        shape_before = self.df.shape
        all_outlier_indices = set()
        for col in columns:
            outlier_indices = self._get_column_outliers_iqr(col, k)
            all_outlier_indices.update(outlier_indices)
        outliers_df = self.df.loc[list(all_outlier_indices)].copy()
        self._log_operation("detect_outliers_iqr", f"Detected outliers in columns {columns} using IQR method with k={k}", shape_before)
        return outliers_df

    def handle_outliers_zscore(self, columns: List[str], threshold: float = 2, action: str = "remove") -> 'TabularCleaning':
        """Handle outliers in specified columns using Z-score method."""
        self._validate_columns(columns)
        shape_before = self.df.shape
        if action == "remove":
            outliers_df = self.detect_outliers_zscore(columns, threshold)
            if len(outliers_df) > 0:
                self.df = self.df.drop(outliers_df.index).reset_index(drop=True)
        elif action == "nan":
            for col in columns:
                outliers = self._get_column_outliers_zscore(col, threshold)
                if outliers:
                    self.df.loc[outliers, col] = np.nan
        else:
            raise ValueError(f"Unknown action '{action}' for outlier handling.")
        self._log_operation("handle_outliers_zscore", f"Handled outliers in columns {columns} using Z-score method with threshold {threshold}", shape_before)
        return self

    def handle_outliers_iqr(self, columns: List[str], k: float = 1.5, action: str = "remove") -> 'TabularCleaning':
        """Handle outliers in specified columns using Interquartile Range (IQR) method."""
        self._validate_columns(columns)
        shape_before = self.df.shape
        if action == "remove":
            outliers_df = self.detect_outliers_iqr(columns, k)
            if len(outliers_df) > 0:
                self.df = self.df.drop(outliers_df.index).reset_index(drop=True)
        elif action == "nan":
            for col in columns:
                outliers = self._get_column_outliers_iqr(col, k)
                if outliers:
                    self.df.loc[outliers, col] = np.nan
        else:
            raise ValueError(f"Unknown action '{action}' for outlier handling.")
        self._log_operation("handle_outliers_iqr", f"Handled outliers in columns {columns} using IQR method with k={k}", shape_before)
        return self
    
    def scale(self, columns: List[str], method: str = "standard") -> 'TabularCleaning':
        """
        Scale specified columns using various scaling methods.
        Methods include:
        - "standard": StandardScaler (zero mean, unit variance)
        - "minmax": MinMaxScaler (scales to range [0, 1])
        - "robust": RobustScaler (scales using median and IQR)
        
        Note: Columns with missing values will cause errors. Handle missing data first.
        """
        self._validate_columns(columns)
        shape_before = self.df.shape
        # Check for missing values in columns to be scaled
        if self.df[columns].isnull().any().any():
            raise ValueError(f"Columns contain missing values. Please handle missing data before scaling.")
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        if method not in scalers:
            raise ValueError(f"Unknown scaler '{method}'")
        self.df[columns] = scalers[method].fit_transform(self.df[columns])
        self._log_operation("scale", f"Scaled columns {columns} using {method} scaler", shape_before)
        return self

    def binning(self, column: str, method: str = "cut", bins: Optional[Union[int, List]] = None, 
               labels: Optional[List] = None, q: Optional[int] = None, right: bool = True, 
               include_lowest: bool = True, duplicates: str = "raise") -> 'TabularCleaning':
        """Bin a specified column using various binning methods."""
        self._validate_columns([column])
        shape_before = self.df.shape
        
        # Validate required parameters for each method
        if method == "cut" and bins is None:
            raise ValueError("bins parameter must be provided when method='cut'")
        if method == "qcut" and q is None:
            raise ValueError("q parameter must be provided when method='qcut'")
        if method == "mapping" and (bins is None or not isinstance(bins, dict)):
            raise ValueError("bins parameter must be a dict when method='mapping'")
        
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
        self._log_operation("binning", f"Binned column {column} using {method} method", shape_before)
        return self
    
    def astype(self, columns: List[str], dtype: str) -> 'TabularCleaning':
        """Convert specified columns to a given target data type."""
        self._validate_columns(columns)
        shape_before = self.df.shape
        for col in columns:
            self.df[col] = self.df[col].astype(dtype)
        self._log_operation("astype", f"Converted columns {columns} to {dtype}", shape_before)
        return self

    def process_column(self, column: str, func: Callable) -> 'TabularCleaning':
        """
        Apply a custom function to a specified column.
        If the column does not exist, create it by applying the function to each row.
        The function should take a single value (or row if column is missing) and return a processed value.
        """
        shape_before = self.df.shape
        if column in self.df.columns:
            self.df[column] = self.df[column].apply(func)
        else:
            self.df[column] = self.df.apply(lambda row: func(row), axis=1)
        self._log_operation("process_column", f"Applied custom function to column {column}", shape_before)
        return self

    def get(self) -> pd.DataFrame:
        return self.df
