#
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
#
# Licensed under GNU Lesser General Public License v3.0
#

""" Custom implementation of CLIP-Adapter from Gao et al.
https://github.com/gaopengcuhk/CLIP-Adapter

Clip model from https://github.com/openai/CLIP
"""

from typing import List, Optional, Tuple, Union

import clip

# Imports
import torch
import torch.nn as nn
from clip.model import CLIP
from torch.nn.modules.module import T


def get_model(
    model_name: str = "ViT-B/16", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Load CLIP model from library"""
    model, preprocess = clip.load(model_name, device=device)

    return model, preprocess


def convert_model_to_fp32(model: nn.Module):
    """Converts CLIP model parameters from float16 to float32 when needed"""
    for p in model.parameters():
        p.data = p.data.float()


class MLPAdapter(torch.nn.Module):
    """MLP module of CLIP-Adapter

    Attributes:
        learn_alpha: Whether to learn the residual connection weight (bool)
        alpha: value of the residual connection weight
        adapter: the MLP adapter
    """

    def __init__(
        self,
        embed_dim: int,
        adapter_hidden_channels: int,
        init_alpha: Optional[float] = 0.7,
        learn_alpha: Optional[bool] = False,
    ) -> None:
        """
        Args:
            embed dim: Embedding dimension of the CLIP model
            adapter_hidden_channels: number of neurons in the hidden layer of the MLP
            init_alpha: initial value of alpha
            learn_alpha: whether we want to learn the alpha coefficient
        """

        super().__init__()
        self.learn_alpha = learn_alpha
        self.alpha = nn.Parameter(torch.FloatTensor([init_alpha]), requires_grad=self.learn_alpha)
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, adapter_hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_hidden_channels, embed_dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, embedded_tensor: torch.Tensor) -> torch.nn.Module:
        """
        y = (1-alpha) * x + alpha * MLP(x)
        Args:
           embedded_tensor: output of CLIP encoder
        """

        return self.alpha * self.adapter(embedded_tensor) + (1 - self.alpha) * embedded_tensor

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        """requires_grad does NOT override the self.learn_alpha attribute"""

        self.adapter.requires_grad = requires_grad
        self.alpha.requires_grad = self.learn_alpha


class CLIPAdapter(torch.nn.Module):
    """Implementation of CLIP-Adapter

    Attributes:
        clip_model: CLIP backbone
        adapter: MLP adapter
        use_vision_adapter: Use the MLP adapter after CLIP backbone with skip connection alpha. Otherwise uses CLIP backbone only, which is equivalent to setting alpha to 0.
        return_features_only: returns text and vision features instead of cosine similarities
    """

    def __init__(
        self,
        clip_model: CLIP,
        embed_dim: int,
        adapter_hidden_channels: List[int],
        use_vision_adapter: bool = True,
        learn_alpha: bool = False,
        return_features_only: bool = False,
    ) -> None:
        """
        Args:
            clip_model: CLIP backbone
            embed dim: Embedding dimension of the CLIP model
            adapter_hidden_channels: number of neurons in the hidden layer of the MLP
            use_vision_adapter: Use the MLP adapter after CLIP backbone with skip connection alpha. Otherwise uses CLIP backbone only, which is equivalent to setting alpha to 0.
            init_alpha: initial value of alpha
            learn_alpha: whether we want to learn the alpha coefficient
            return_features_only: returns text and vision features instead of cosine similarities
        """
        super().__init__()

        self.clip_model = clip_model
        if use_vision_adapter:
            init_alpha = 0.7
        else:
            init_alpha = 0

        self.adapter = MLPAdapter(
            embed_dim=embed_dim,
            adapter_hidden_channels=adapter_hidden_channels,
            init_alpha=init_alpha,
            learn_alpha=learn_alpha,
        )
        self.use_vision_adapter = use_vision_adapter
        self.return_features_only = return_features_only

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Args:
            image: batch input to image encoder
            text: batch input to text encoder

        Returns:
            if return_features_only: image and text embeddings, cosine similarities otherwise
        """
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(text)

        if self.use_vision_adapter:
            image_features = self.adapter(image_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if self.return_features_only:
            return image_features, text_features
        else:
            # cosine similarity as logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # shape = [global_batch_size, global_batch_size]
            return logits_per_image, logits_per_text
