# 
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
# 
# Licensed under GNU Lesser General Public License v3.0
# 

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class SerengetiDataset(Dataset):
    """Simple Dataset for the Serengeti data

    Attributes:
        filenames: list of filename paths
        tokenized_captions: array of same size with the tokenized captions corresponding to the filenames. If not provided, returns a value of -1.
        transform_func: transformation to apply after loading images
    """

    def __init__(
        self,
        filenames: List[Path],
        tokenized_captions: Optional[np.ndarray] = None,
        transform_func: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Args:
            filenames: list of filename paths
            tokenized_captions: array of same size with the tokenized captions corresponding to the filenames. If not provided, returns a value of -1.
            transform_func: transformation to apply after loading images
        """

        super().__init__()
        self.filenames = filenames
        self.tokenized_captions = tokenized_captions
        if self.tokenized_captions is None:
            self.tokenized_captions = torch.ones(len(filenames)) * (-1)
        self.transform_func = transform_func

    def set_transform_func(self, transform_func: torch.nn.Module) -> None:
        self.transform_func = transform_func

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx) -> Tuple["torch.Tensor"]:
        """
        Args:
            idx: index from self.filenames
        Returns:
            img: image corresponding to the idx transformed according to self.transform
            tokenized_catpion: tokenized caption corresponding to idx
        """

        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        # Read image as tensor
        image_path = self.filenames[idx]

        with Image.open(image_path) as img:
            if self.transform_func is not None:
                img = self.transform_func(img)
            else:
                img = torchvision.transforms.ToTensor()(img)

        # tokenized caption, weights
        tokenized_caption = self.tokenized_captions[idx]
        tokenized_caption = torch.squeeze(tokenized_caption)

        return img, tokenized_caption


class SerengetiDatasetWithVocabularyReplay(SerengetiDataset):
    """Pytorch Dataset for the Serengeti data for vocabulary replay experiments

    Returns the image, corresponding tokenized caption and pre-computed cosine similarity between the old model image and the replayed vocabulary

    Attributes:
        filenames: list of filename paths
        logits_per_vr_old: pre-computed cosine similarity between the old model image embeddings and the replayed vocabulary
        tokenized_captions: array of same size with the tokenized captions corresponding to the filenames. If not provided, returns a value of -1.
        transform_func: transformation to apply after loading images
    """

    def __init__(
        self,
        filenames: List,
        logits_per_vr_old: np.ndarray,
        tokenized_captions: Optional[np.ndarray] = None,
        transform_func: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Args:
            filenames: list of filename paths
            logits_per_vr_old: pre-computed cosine similarity between the old model image embeddings and the replayed vocabulary
            tokenized_captions: array of same size with the tokenized captions corresponding to the filenames. If not provided, returns a value of -1.
            transform_func: transformation to apply after loading images
        """
        super().__init__(filenames=filenames, tokenized_captions=tokenized_captions, transform_func=transform_func)
        self.logits_per_vr_old = logits_per_vr_old

    def __getitem__(self, idx) -> Tuple["torch.Tensor"]:
        """
        Args:
            idx: index from self.filenames
        Returns:
            img: image corresponding to the idx transformed according to self.transform
            tokenized_catpion: tokenized caption corresponding to idx
            logits_per_vr_old_i: old logits for the vocabulary replay corresponding to idx
        """

        img, tokenized_caption = super().__getitem__(idx)
        logits_per_vr_old_i = self.logits_per_vr_old[idx]

        return img, tokenized_caption, logits_per_vr_old_i
