#
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
#
# Licensed under GNU Lesser General Public License v3.0
#

from random import shuffle
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class SequentialSampler(Sampler[int]):
    """Simple Sequential sampler

    A sampler that returns elements sequentially.

    Attributes:
        data_source: The corresponding dataset to draw samples from
        shuffle: Whether to shuffle samples beforehand or not
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False):
        """
        Args:
            data_source: The corresponding dataset to draw samples from
            shuffle: Whether to shuffle samples beforehand or not
        """
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        """Builds the sampling iterator, shuffle if needed"""

        indices = np.arange(len(self)).tolist()
        if self.shuffle:
            shuffle(indices)
        yield from iter(indices)

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """Simple Random sampler

    Similar to the Sequential sampler but draws a random number of samples.

    Attributes:
        data_source: The corresponding dataset to draw samples from
        num_samples_per_epoch: The number of samples to draw for each epoch
    """

    def __init__(self, data_source: Dataset, num_samples_per_epoch: Optional[int] = None):
        """
        Args:
            data_source: The corresponding dataset to draw samples from
            num_samples_per_epoch: The number of samples to draw
        """
        self.data_source = data_source
        if num_samples_per_epoch is None:
            num_samples_per_epoch = len(data_source)
        self.num_samples_per_epoch = num_samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        """Builds random iterator with fixed length"""

        indices = torch.randint(0, len(self.data_source), (self.num_samples_per_epoch,))
        yield from iter(indices.tolist())

    def __len__(self):
        return self.num_samples_per_epoch


class BalancingSampler(Sampler[int]):
    """Sampler for balancing classes

    Assigns class weights for every sample as inverse of class frequency
    Then select a fixed number of samples with replacement according to the sample weights

    Replacement is usually set to true to draw multiple times the samples from the minority
    class (up sampling) and on average different samples from the majority class (down sampling)

    Attributes:
        labels: array of labels corresponding to the samples
        num_samples_per_epoch: Number of samples to draw for each epoch
        replacement: Whether to do replacement when sampling
    """

    def __init__(self, labels: List[str], num_samples_per_epoch: int, replacement: bool = True):
        self.replacement = replacement
        self.num_samples_per_epoch = num_samples_per_epoch

        # Get number of samples per class in the dataset
        classes, num_sample_per_class = np.unique(labels, return_counts=True)

        # Compute class weight as inverse of frequency and assigns to samples
        class_weights = np.sum(num_sample_per_class) / (num_sample_per_class + 1e-5)
        class_weights_map = {c: w for c, w in zip(classes, class_weights)}

        # Assign a sampling weight to each sample based on its label, between 0 and 1
        self.sample_weights = np.array([class_weights_map[l] for l in labels])
        self.sample_weights /= np.max(self.sample_weights)
        self.sample_weights = torch.Tensor(self.sample_weights)

    def __iter__(self) -> Iterator[int]:
        """Builds random iterator with upsampling strategy"""

        rand_tensor = torch.multinomial(self.sample_weights, self.num_samples_per_epoch, self.replacement)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples_per_epoch


class BalancingSamplerMultiLabelOHE(Sampler[int]):
    """Sampler for balancing classes in a multi label setting

    Assigns class weights for every sample as inverse of class frequency
    Set the sample weight based on the highest label weight
    Then select a fixed number of samples with replacement according to the sample weights

    Replacement is usually set to true to draw multiple times the samples from the minority
    class (up sampling) and on average different samples from the majority class (down sampling)

    Attributes:
        labels_ohe: array of multilabels one hot encoded (ohe) corresponding to the samples
        num_samples_per_epoch: Number of samples to draw for each epoch
        replacement: Whether to do replacement when sampling
    """

    def __init__(self, labels_ohe: "np.array", num_samples_per_epoch: int, replacement: bool = True):
        self.replacement = replacement
        self.num_samples_per_epoch = num_samples_per_epoch

        # Get number of samples per class in the dataset
        labels = [labels_ohe_sample.nonzero()[0].tolist() for labels_ohe_sample in labels_ohe]
        labels_flatten = [l for label in labels for l in label]
        classes, num_sample_per_class = np.unique(labels_flatten, return_counts=True)

        # Compute class weight as inverse of frequency and assigns to samples
        class_weights = np.sum(num_sample_per_class) / (num_sample_per_class + 1e-5)
        class_weights_map = {c: w for c, w in zip(classes, class_weights)}

        # Assign a sampling weight to each sample, based on the rarest label for this sample, between 0 and 1
        self.sample_weights = np.zeros(len(labels))
        for i, sample_label in enumerate(labels):
            max_sample_weight = np.max([class_weights_map[l] for l in sample_label])
            self.sample_weights[i] = max_sample_weight
        self.sample_weights /= np.max(self.sample_weights)
        self.sample_weights = torch.Tensor(self.sample_weights)

    def __iter__(self) -> Iterator[int]:
        """Builds random iterator with upsampling strategy"""

        rand_tensor = torch.multinomial(self.sample_weights, self.num_samples_per_epoch, self.replacement)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples_per_epoch
