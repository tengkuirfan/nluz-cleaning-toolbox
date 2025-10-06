import cv2
import numpy as np
from typing import List, Union, Callable, Dict

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
            # Validate it's a valid image array (2D or 3D)
            if images.ndim not in [2, 3]:
                raise ValueError("Image array must be 2D (grayscale) or 3D (color)")
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
        Note: "fastnl" only works for color images. Use "nl_means" for automatic selection.
        """
        denoisers = {
            "gaussian": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            "median": lambda img: cv2.medianBlur(img, 5),
            "bilateral": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            "box": lambda img: cv2.blur(img, (5, 5)),
            "nl_means": lambda img: cv2.fastNlMeansDenoising(img, None, 10, 7, 21) if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
                                    else cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
            "fastnl": lambda img: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) if len(img.shape) == 3 and img.shape[2] > 1
                                  else cv2.fastNlMeansDenoising(img, None, 10, 7, 21),
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
