# 
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
# 
# Licensed under GNU Lesser General Public License v3.0
# 

""" Implementation of the different losses used in WildCLIP

The loss from CLIP
The loss used during the training of the adapter
The LwF-VR loss
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class ClipLoss(nn.Module):
    """Simple implementation of CLIP loss

    Computes cross-entropy along the image and text dimensions
    Adapted from https://github.com/locuslab/FLYP/blob/d42180645e31c16c50a7b111410c98616c2c2872/clip/loss.py
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
        device: torch.device,
    ) -> torch.float:
        """
        As described in clip paper
        Args:
            logits_per_image: logits along the image dimension
            logits_per_text: logits along the text dimension
            device: torch device
        Returns:
            loss
        """
        n = logits_per_image.shape[0]
        labels = torch.arange(n, dtype=torch.long, device=device)
        loss_i = F.cross_entropy(logits_per_image, labels, reduction="mean")
        loss_t = F.cross_entropy(logits_per_text, labels, reduction="mean")
        loss = (loss_i + loss_t) / 2

        return loss


class AdaptLoss(nn.Module):
    """Adaptation of CLIP loss that only maximizes diagonal elements

    CLIP is trained with a very large batch size and many classes
    Training can be affected when having many false positives per batch (e.g. a caption corresponding to multiple images)

    Maximizing diagonal elements amd minimizing the other reduces the negative effect of false positives
    but it inevitably reduces discrimination between positive and negative classes

    Attributes:
        logit_scale: CLIP logit scaling factor
    """

    def __init__(self, logit_scale: torch.float) -> None:
        """
        Args:
            logit_scale: CLIP logit scaling factor
        """
        super().__init__()
        self.logit_scale = logit_scale

    def forward(
        self,
        logits_per_image: torch.Tensor,
        device: torch.device,
    ) -> torch.float:
        """
        Args:
            logits_per_image: logits along the image dimension
            device: torch device
        """
        # Loss of diagonal elements
        pos_pair_logits = torch.diag(logits_per_image) / self.logit_scale
        target = torch.ones_like(pos_pair_logits).to(device)
        loss = F.mse_loss(pos_pair_logits, target, reduction="mean")

        # Loss of non-diagonal elements
        neg_pair_logits = logits_per_image * (1 - torch.eye(logits_per_image.shape[0]).to(device)) / self.logit_scale
        loss += 0.5 * torch.sum(torch.square(neg_pair_logits))

        return loss


class VRLwR(nn.Module):
    """
    Custom implementation of LwF-VR Loss
    See https://arxiv.org/pdf/2207.09248.pdf
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits_per_vr_new: torch.Tensor, logits_per_vr_old: torch.Tensor) -> torch.float:
        """
        Args:
            logits_per_vr_new: logits along the replayed vocabulary dimension of the current model
            logits_per_vr_old: logits along the replayed vocabulary dimension of the old model

        """

        # Compute softmax along each replayed sentence.
        p_new = F.softmax(logits_per_vr_new, dim=0)

        # Compute cross-entropy between old and new similarity distributions.
        loss = torch.mean(torch.sum(-p_new * F.log_softmax(logits_per_vr_old, dim=0), dim=0))

        return loss
