"""
AnimeGAN2 DataModule

This module provides the AnimeDataSet class for loading and preprocessing datasets
used in AnimeGAN2 training. It supports loading photo, anime, and smoothed anime images,
and applies optional transformations to the images.
"""

from os import PathLike
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


class AnimeDataSet(Dataset):
    """
    PyTorch Dataset for AnimeGAN2.

    Loads photo, anime, and smoothed anime images from specified directories.
    Applies optional transformations to each image. Returns tuples containing
    the photo, anime, grayscale anime, and smoothed grayscale anime images.
    """

    def __init__(
        self,
        photo_dir: Union[str, PathLike],
        anime_dir: Union[str, PathLike],
        smooth_dir: Union[str, PathLike],
        transform: v2.Compose | None,
    ):
        """
        Args:
            photo_dir (str or PathLike): Directory containing photo images.
            anime_dir (str or PathLike): Directory containing anime images.
            smooth_dir (str or PathLike): Directory containing smoothed anime images.
            transform (v2.Compose or None): Optional transform to be applied on images.
        """
        self.photo_dir = Path(photo_dir)
        self.anime_dir = Path(anime_dir)
        self.smooth_dir = Path(smooth_dir)
        self.photo_files = sorted(self._load_files(self.photo_dir))
        self.anime_files = sorted(self._load_files(self.anime_dir))
        self.smooth_files = sorted(self._load_files(self.smooth_dir))
        self._transform = transform

        self.len_photo = len(self.photo_files)
        self.len_anime = len(self.anime_files)

    def __len__(self) -> int:
        """
        Returns:
            int: The maximum length among photo and anime datasets.
        """
        return max(len(self.photo_files), len(self.anime_files))

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index of the data sample.

        Returns:
            tuple: (photo, anime, anime_gray, smooth_gray) tensors.
        """
        photo_idx = index
        anime_idx = index

        if anime_idx > self.len_anime - 1:
            anime_idx -= self.len_anime * (index // self.len_anime)

        if photo_idx > self.len_photo - 1:
            photo_idx -= self.len_photo * (index // self.len_photo)

        photo = self.load_photo(photo_idx)
        anime, anime_gray = self.load_anime(anime_idx)
        smooth_gray = self.load_anime_smooth(anime_idx)

        return (photo, anime, anime_gray, smooth_gray)

    def load_photo(self, index: int) -> torch.Tensor:
        """
        Loads and transforms a photo image.

        Args:
            index (int): Index of the photo image.

        Returns:
            torch.Tensor: Transformed photo image tensor.
        """
        fpath = self.photo_files[index]
        image = Image.open(fpath).convert("RGB")
        if self._transform:
            image = self._transform(image)
        else:
            image = F.to_tensor(image)
        return image

    def load_anime(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and transforms an anime image and its grayscale version.

        Args:
            index (int): Index of the anime image.

        Returns:
            tuple: (anime RGB tensor, anime grayscale tensor)
        """
        fpath = self.anime_files[index]
        image = Image.open(fpath).convert("RGB")
        if self._transform:
            image = self._transform(image)
        else:
            image = F.to_tensor(image)

        image_gray = Image.open(fpath).convert("L")  # Gray Scale
        if self._transform:
            image_gray = self._transform(image_gray)
        else:
            image_gray = F.to_tensor(image_gray)

        return image, image_gray

    def load_anime_smooth(self, index: int) -> torch.Tensor:
        """
        Loads and transforms a smoothed grayscale anime image.

        Args:
            index (int): Index of the smoothed anime image.

        Returns:
            torch.Tensor: Transformed smoothed grayscale anime image tensor.
        """
        fpath = self.anime_files[index]
        image = Image.open(fpath).convert("L")  # Gray Scale
        if self._transform:
            image = self._transform(image)
        else:
            image = F.to_tensor(image)
        return image

    def _load_files(self, dir_path: Union[str, PathLike]) -> list:
        """
        Recursively loads image file paths from a directory.

        Args:
            dir_path (str or PathLike): Directory path to search for images.

        Returns:
            list: List of image file paths.
        """
        exts = ["*.png", "*.jpg"]
        files = []
        for ext in exts:
            files.extend(list(Path(dir_path).rglob(ext)))
        return files
