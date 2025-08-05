import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ----------------- TABULAR DATA CLEANING -----------------
class DataCleaning:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def replace_symbols(self, columns, symbols=None, replacement=""):
        """
        Replace specified symbols in given columns with a replacement string.
        If symbols is None, defaults to common punctuation and special characters.
        If replacement is not specified, it defaults to an empty string.
        The symbols are escaped for regex compatibility.
        """
        if symbols is None:
            symbols = [",", ".", "!", "?", "$", "%", "&"]
        pattern = "[" + "".join(map(lambda s: "\\" + s if s in r"\^$.|?*+()[]{}" else s, symbols)) + "]"
        for col in columns:
            self.df[col] = self.df[col].astype(str).replace(pattern, replacement, regex=True)
        return self

    def handle_missing(self, columns, method="mean", fill_value=None, func=None, ref_col=None, **kwargs):
        """
        Handle missing data in specified columns using various methods.
        Methods include:
        - "mean": Fill with column mean
        - "median": Fill with column median
        - "mode": Fill with column mode
        - "drop": Drop rows with missing values in specified columns
        - "value": Fill with a specified value
        - "column": Fill with values from another specified column
        - "ffill": Forward fill missing values
        - "bfill": Backward fill missing values
        - "interpolate": Interpolate missing values
        - "function": Apply a custom function to fill missing values
        """
        handlers = {
            "mean": lambda col: self.df[col].fillna(self.df[col].mean()),
            "median": lambda col: self.df[col].fillna(self.df[col].median()),
            "mode": lambda col: self.df[col].fillna(self.df[col].mode()[0]),
            "drop": lambda col: self.df.dropna(subset=[col]),
            "value": lambda col: self.df[col].fillna(fill_value),
            "column": lambda col: self.df[col].fillna(self.df[ref_col]),
            "ffill": lambda col: self.df[col].fillna(method="ffill"),
            "bfill": lambda col: self.df[col].fillna(method="bfill"),
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
        return self

    def handle_outliers_zscore(self, columns, threshold=2, action="remove"):
        """
        Handle outliers in specified columns using Z-score method.
        - "remove": Remove rows with outliers
        - "nan": Replace outliers with NaN
        """
        for col in columns:
            z = np.abs(stats.zscore(self.df[col].dropna()))
            mask = (z < threshold).reindex(self.df.index, fill_value=False)
            actions = {
                "remove": lambda: self.df[mask],
                "nan": lambda: self.df.assign(**{col: self.df[col].mask(~mask)})
            }
            if action not in actions:
                raise ValueError(f"Unknown action '{action}' for outlier handling.")
            self.df = actions[action]()
        return self

    def handle_outliers_iqr(self, columns, k=1.5, action="remove"):
        """
        Handle outliers in specified columns using Interquartile Range (IQR) method.
        - "remove": Remove rows with outliers
        - "nan": Replace outliers with NaN
        """
        for col in columns:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (self.df[col] >= Q1 - k * IQR) & (self.df[col] <= Q3 + k * IQR)
            actions = {
                "remove": lambda: self.df[mask],
                "nan": lambda: self.df.assign(**{col: self.df[col].mask(~mask)})
            }
            if action not in actions:
                raise ValueError(f"Unknown action '{action}' for outlier handling.")
            self.df = actions[action]()
        return self

    def scale(self, columns, method="standard"):
        """
        Scale specified columns using various scaling methods.
        Methods include:
        - "standard": StandardScaler (zero mean, unit variance)
        - "minmax": MinMaxScaler (scales to range [0, 1])
        - "robust": RobustScaler (scales using median and IQR)
        """
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        if method not in scalers:
            raise ValueError(f"Unknown scaler '{method}'")
        self.df[columns] = scalers[method].fit_transform(self.df[columns])
        return self

    def binning(self, column, method="cut", bins=None, labels=None, q=None, right=True, include_lowest=True, duplicates="raise"):
        """
        Bin a specified column using various binning methods.
        Methods include:
        - "cut": pd.cut for fixed-width bins
        - "qcut": pd.qcut for quantile-based bins
        - "mapping": Map values to specified bins
        """
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
        self.df = self.df.assign(**{column + "Category": bin_methods[method]()})
        return self
    
    def astype(self, columns, dtype):
        """
        Convert specified columns to a given dtype.
        columns: list of column names
        dtype: target data type (e.g., 'float', 'int', 'str')
        """
        for col in columns:
            self.df[col] = self.df[col].astype(dtype)
        return self

    def process_column(self, column, func):
        """
        Apply a custom function to a specified column.
        If the column does not exist, create it by applying the function to each row.
        The function should take a single value (or row if column is missing) and return a processed value.
        """
        if column in self.df.columns:
            self.df[column] = self.df[column].apply(func)
        else:
            self.df[column] = self.df.apply(lambda row: func(row), axis=1)
        return self

    def get(self):
        return self.df

# ----------------- IMAGE DATA CLEANING -----------------
class ImageCleaning:
    def __init__(self, images: dict):
        if isinstance(images, list):
            self.images = {i: img for i, img in enumerate(images)}
        else:
            self.images = images.copy()

    def resize(self, size=(128, 128)):
        """
        Resize all images to the specified size.
        Size should be a tuple (width, height).
        """
        self.images = {k: cv2.resize(img, size) for k, img in self.images.items()}
        return self

    def convert_color(self, mode="grayscale"):
        """
        Convert all images to a specified color mode.
        Supported modes:
        - "grayscale": Convert to grayscale
        - "rgb": Convert to RGB (from BGR)
        """
        converters = {
            "grayscale": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            "rgb": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        }
        if mode not in converters:
            raise ValueError(f"Unknown color mode '{mode}'")
        self.images = {k: converters[mode](img) for k, img in self.images.items()}
        return self

    def normalize(self, method="0-1"):
        """
        Normalize all images using specified method.
        Supported methods:
        - "0-1": Scale pixel values to [0, 1]
        - "minus1-1": Scale pixel values to [-1, 1]
        """
        normalizers = {
            "0-1": lambda img: img.astype("float32") / 255.0,
            "minus1-1": lambda img: (img.astype("float32") / 127.5) - 1,
        }
        if method not in normalizers:
            raise ValueError(f"Unknown normalization method '{method}'")
        self.images = {k: normalizers[method](img) for k, img in self.images.items()}
        return self

    def denoise(self, method="gaussian"):
        """
        Denoise all images using specified method.
        Supported methods:
        - "gaussian": Gaussian blur
        - "median": Median blur
        - "bilateral": Bilateral filter
        - "box": Box blur
        - "nl_means": Non-local means denoising
        - "fastnl": Fast non-local means denoising (colored)
        """
        denoisers = {
            "gaussian": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            "median": lambda img: cv2.medianBlur(img, 5),
            "bilateral": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            "box": lambda img: cv2.blur(img, (5, 5)),
            "nl_means": lambda img: cv2.fastNlMeansDenoising(img, None, 10, 7, 21) if len(img.shape) == 2 
                                    else cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
            "fastnl": lambda img: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
        }
        if method not in denoisers:
            raise ValueError(f"Unknown denoise method '{method}'")
        self.images = {k: denoisers[method](img) for k, img in self.images.items()}
        return self

    def process_image(self, func):
        """
        Apply a custom function to each image.
        The function should take a single image and return a processed image.
        """
        self.images = {k: func(img) for k, img in self.images.items()}
        return self

    def get(self):
        return self.images
