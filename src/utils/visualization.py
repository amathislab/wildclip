# 
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
# 
# Licensed under GNU Lesser General Public License v3.0
# 

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Imports
from PIL import Image, ImageOps


def handle_max_dimensions(img: Image, max_dims: Optional[Union[Tuple, List, int]]) -> Image:
    """Resize an image according to max dimensions preserving ratio

    Args:
        img: Pil Image
        max_dims: Maximum dimensions of the image. If Tuple, specify
            (H,W), similarly for list. If int, the dimension in both
            dimensions. If None, no resize.

    Returns:
        A PIL Image resized according to max_dims
    """

    if max_dims == None:
        return img

    if isinstance(max_dims, int):
        max_dims = (max_dims, max_dims)

    # Get ratio that preserve image ratio so that maximum size is less
    # than required dimensions
    target_width = min(img.width, max_dims[0])
    width_ratio = target_width / img.width
    target_height = int(img.height * width_ratio)
    target_height = min(target_height, max_dims[1])
    height_ratio = target_height / img.height
    target_width = int(img.width * height_ratio)

    img = img.resize((target_width, target_height))

    return img


def open_img(img_path: Union[str, Path], max_dims: Optional[Union[Tuple, List, int]] = None) -> Image:
    """Open an image from filepath

    Open an image with Pillow from given path
    and resize to maximal dimensions accordingly

    Args:
        img_path: Complete path to image
        max_dims: Maximum dimensions of the image. If Tuple, specify
            (H,W), similarly for list. If int, the dimension in both
            dimensions. If None, no resize.

    Returns:
        A PIL Image that has been resized according to max_size
    """

    # Load image
    with open(img_path, "rb") as image_file:
        img_str = image_file.read()
        img = Image.open(io.BytesIO(img_str))

    # Resize if size exceeds max_dims
    if max_dims is not None:
        img = handle_max_dimensions(img, max_dims)

    return img


def add_border_color(img: Image, border_color: str, width: int = 5):
    return ImageOps.expand(img, border=width, fill=border_color)
