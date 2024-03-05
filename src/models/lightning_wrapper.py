#
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
#
# Licensed under GNU Lesser General Public License v3.0
#

""" Different pytorch lightning wrappers to train the models

There is a wrapper to train the vision only architecture, the CLIP-Adapter architecture, and the CLIP-Adapter with VR-LwF loss
"""

from itertools import chain
from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from models.clip_adapter import CLIPAdapter, convert_model_to_fp32
from models.clip_loss import AdaptLoss, ClipLoss, VRLwR
from models.scheduler import CosineLR


class VisionLightningWrapper(pl.LightningModule):
    """Pytorch Lightning Wrapper to train vision only model

    Attributes:
        model: pytorch model
        num_classes: number of output classes
        lock_vision: if true, fine-tunes only the MLP head
        cfg_optim: dictionary with training parameters
        total_steps: total steps per epoch
        loss_func: pytorch loss function
        validation_step_outputs: outputs of validation steps
        test_step_outputs: outputs of test steps
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes,
        lock_vision: bool = True,
    ) -> None:
        """
        Args:
            model: pytorch model
            num_classes: number of output classes
            lock_vision: if true, fine-tunes only the MLP head
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lock_vision = lock_vision

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def prepare(self, cfg_optim: Dict = None, total_steps: int = None, eval=False):
        """
        Args:
            cfg_optim: dictionary with training parameters
            total_steps: total steps per epoch
            eval: if model is in eval mode.
        """
        self.configure_eval_metrics()
        convert_model_to_fp32(self.model)
        if eval == False:
            self.cfg_optim = cfg_optim
            self.total_steps = total_steps
            self.configure_trainable_parameters()
            self.configure_loss_func()

        else:
            self.model.eval()

    def configure_trainable_parameters(self):
        """
        If lock vision is set to true: freezes the model backbone and fine tunes only MLP head
        otherwise, set all weights as trainable
        """
        if self.lock_vision:
            self.model.requires_grad_(False)
            self.model.head.requires_grad_(True)
            self.trainable_parameters = self.model.head.parameters()
        else:
            self.model.requires_grad_(True)
            self.trainable_parameters = self.model.parameters()

    def configure_loss_func(self):
        """Init. loss function"""
        self.loss_func = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        """Defines the optimizer and scheduler from config dict"""

        if self.cfg_optim["optim"] == "sgd":
            self.optimizer = optim.SGD(
                self.trainable_parameters, lr=self.cfg_optim["init_lr"], weight_decay=self.cfg_optim["weight_decay"]
            )
        elif self.cfg_optim["optim"] == "adamw":
            self.optimizer = optim.AdamW(
                self.trainable_parameters,
                lr=self.cfg_optim["init_lr"],
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=self.cfg_optim["weight_decay"],
            )

        if self.cfg_optim["use_cosine_scheduler"]:
            self.scheduler = CosineLR(
                self.optimizer,
                base_lr=self.cfg_optim["init_lr"],
                warmup_length=self.cfg_optim["warmup_length"],
                steps=self.total_steps,
            )
            return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler, "interval": "step"}}
        else:
            return {"optimizer": self.optimizer}

    def forward(self, x):
        """Implements pytorch lightning function

        Args:
            x: tensor of dim (batch, w,h)
        Returns:
            logits
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Implements pytorch lightning function

        Args:
            batch: batch containing inputs and labels
            batch_idx: not used
        Returns:
            loss value
        """

        # Freezes batchnorm during fine-tuning
        # For now in the training step which is not ideal, but it needs to be set to eval, which is overridden by pytorch lightning if done earlier
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        inputs, labels = batch
        preds = self(inputs)

        loss = self.loss_func(preds, labels)
        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Implements pytorch lightning function"""

        loss = self._shared_eval_step(batch)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self, outputs) -> None:
        """Implements pytorch lightning function"""

        loss = torch.stack(self.validation_step_outputs).mean()
        metrics = {"val_loss": loss}

        self.validation_step_outputs.clear()
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        """Implements pytorch lightning function"""

        loss = self._shared_eval_step(batch)
        self.test_step_outputs.append(loss)
        return loss

    def predict_step(self, batch, batch_idx):
        """Implements pytorch lightning function"""

        inputs, _ = batch
        preds = self(inputs)
        return preds

    def _shared_eval_step(self, batch):
        """Called at the end of validation or test step

        Args:
            batch: batch containing inputs and labels
        Returns:
            loss value
        """

        inputs, labels = batch
        preds = self(inputs)
        loss = self.loss_func(preds, labels)
        return loss


class CLIPLightningWrapper(pl.LightningModule):
    """Pytorch Lightning Wrapper to train CLIP-Adapter models

    Attributes:
        model: pytorch model
        lock_vision: if true, fine-tunes only the MLP head
        cfg_optim: dictionary with training parameters
        total_steps: total steps per epoch
        loss_func: pytorch loss function
        text_tokens: for prediciton step only: text tokens to pass through text encoder
        validation_step_outputs: outputs of validation steps
        test_step_outputs: outputs of test steps
    """

    def __init__(
        self,
        model: CLIPAdapter,
        lock_vision: bool = True,
    ) -> None:
        """
        Args:
            model: pytorch model
            lock_vision: if true, fine-tunes only the MLP head
        """
        super().__init__()
        self.model = model
        self.lock_vision = lock_vision
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def prepare(self, cfg_optim: Dict = None, total_steps: int = None, eval=False, use_adapt_loss: bool = False):
        """
        Args:
            cfg_optim: dictionary with training parameters
            total_steps: total steps per epoch
            eval: if model is in eval mode.
            use_adapt_loss: if true, uses the adaptive loss
        """
        convert_model_to_fp32(self.model)
        if eval == False:
            self.cfg_optim = cfg_optim
            self.total_steps = total_steps
            self.configure_trainable_parameters()
            self.configure_loss_func(use_adapt_loss)
        else:
            self.model.eval()

    def configure_trainable_parameters(self):
        """
        use_vision_adapter --> if the samples are passed through the MLP head, if so, we train it
        lock_vision --> to freeze the image backbone
        """
        self.model.clip_model.requires_grad_(False)
        self.trainable_parameters = None

        if self.model.use_vision_adapter and self.lock_vision:
            self.model.adapter.requires_grad_(True)
            self.trainable_parameters = self.model.adapter.parameters()

        if self.model.use_vision_adapter and not self.lock_vision:
            self.model.clip_model.visual.requires_grad_(True)
            self.model.adapter.requires_grad_(True)
            self.trainable_parameters = chain(
                self.model.clip_model.visual.parameters(), self.model.adapter.parameters()
            )

        if not self.model.use_vision_adapter and self.lock_vision:
            pass

        if not self.model.use_vision_adapter and not self.lock_vision:
            self.model.clip_model.visual.requires_grad_(True)
            self.trainable_parameters = self.model.clip_model.visual.parameters()

    def configure_loss_func(self, use_adapt_loss: bool = False):
        """
        Args:
            use_adapt_loss: if true, uses the adaptive loss
        """
        if use_adapt_loss:
            self.loss_func = AdaptLoss(self.model.clip_model.logit_scale.exp())
        else:
            self.loss_func = ClipLoss()

    def configure_optimizers(self):
        """Defines the optimizer and scheduler"""

        if self.cfg_optim["optim"] == "sgd":
            self.optimizer = optim.SGD(
                self.trainable_parameters, lr=self.cfg_optim["init_lr"], weight_decay=self.cfg_optim["weight_decay"]
            )
        elif self.cfg_optim["optim"] == "adamw":
            self.optimizer = optim.AdamW(
                self.trainable_parameters,
                lr=self.cfg_optim["init_lr"],
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=self.cfg_optim["weight_decay"],
            )

        if self.cfg_optim["use_cosine_scheduler"]:
            self.scheduler = CosineLR(
                self.optimizer,
                base_lr=self.cfg_optim["init_lr"],
                eta_min=self.cfg_optim["init_lr"],
                warmup_length=self.cfg_optim["warmup_length"],
                steps=self.total_steps,
            )
            return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler, "interval": "step"}}
        else:
            return {"optimizer": self.optimizer}

    def training_step(self, batch, batch_idx):
        """Implements pytorch lightning function

        Args:
            batch: batch containing inputs and labels
            batch_idx: not used
        Returns:
            loss value
        """

        # Run CLIP in eval mode since our batch size is much smaller than during CLIP training
        for m in self.model.clip_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        images, texts = batch
        logits_per_image, logits_per_text = self(images, texts)
        loss = self.loss_func(logits_per_image, logits_per_text, self.device)

        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}, on_step=False, on_epoch=True)

        return loss

    def forward(self, images, texts):
        """Implements pytorch lightning function"""

        return self.model(images, texts)

    def validation_step(self, batch, batch_idx):
        """Implements pytorch lightning function"""

        loss = self._shared_eval_step(batch)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Implements pytorch lightning function"""

        loss = torch.stack(self.validation_step_outputs).mean()
        metrics = {"val_loss": loss}

        self.validation_step_outputs.clear()  # free memory

        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        """Implements pytorch lightning function"""

        loss = self._shared_eval_step(batch)
        self.test_step_outputs.append(loss)

        return loss

    def predict_step(self, batch, batch_idx):
        """Implements pytorch lightning function"""

        images, _ = batch
        outputs = self(images, self.text_tokens)
        return outputs

    def _register_text_tokens(self, text_tokens: torch.Tensor):
        """To set text tokens as class attributes which is required for prediction step

        Args:
            text_tokens: text already tokenized by clip functions
        """
        self.text_tokens = text_tokens

    def _shared_eval_step(self, batch):
        """Called at the end of validation or test step

        Args:
            batch: batch containing inputs and labels
        Returns:
            loss value
        """

        images, texts = batch
        logits_per_image, logits_per_text = self(images, texts)
        loss = self.loss_func(logits_per_image, logits_per_text, self.device)
        return loss


class CLIPLightningWrapperLwF(CLIPLightningWrapper):
    """Pytorch Lightning Wrapper to train CLIP-Adapter models with VR-LwF loss

    Attributes:
        vr_features: replayed vocabulary passed through text encoder
        loss_vr_lwf: vr-lwf loss function
    """

    def __init__(self, model: CLIPAdapter, vr_features: torch.Tensor, lock_vision: bool = True) -> None:
        """
        Args:
            model: pytorch model
            vr_features: replayed vocabulary passed through text encoder
            lock_vision: if true, fine-tunes only the MLP head
        """
        super().__init__(model=model, lock_vision=lock_vision)
        self.vr_features = vr_features

    def configure_loss_func(self, use_adapt_loss: bool = False):
        """
        Args:
            use_adapt_loss: if true, uses the adapted clip loss
        """
        if use_adapt_loss:
            self.loss_clip = AdaptLoss(self.model.clip_model.logit_scale.exp())
        else:
            self.loss_clip = ClipLoss()
        self.loss_vr_lwf = VRLwR()

    def compute_loss(self, logits_per_image, logits_per_text, logits_per_vr_new, logits_per_vr_old):
        """Computes CLIP loss and VR-LwF loss and averages the two

        Args:
            logits_per_image: cosine similarities between images and texts
            logits_per_text: cosine similarities between texts and images
            logits_per_vr_new: cosine similarities between replayed vocabulary and images passed through new encoder
            logits_per_vr_old: cosine similarities between replayed vocabulary and images passed through old encoder
        """
        clip_loss = self.loss_clip(logits_per_image, logits_per_text, self.device)
        vr_lwf_loss = self.loss_vr_lwf(logits_per_vr_new, logits_per_vr_old)

        return (clip_loss + vr_lwf_loss) / 2

    def _shared_forward(self, images, texts, logits_per_vr_old):
        """Forward step

        Args:
            images: batch images
            texts: batch captions
            logtis_per_vr_old: precomputed cosine similarities between replayed vocabulary and images passed through old encoder
        """
        logits_per_vr_old = logits_per_vr_old.t()

        # CLIP forward pass
        new_image_features, text_features = self(images, texts)

        # cosine similarities between captions and images
        logit_scale = self.model.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * new_image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # cosine similarities between replayed vocabulary and images
        logits_per_vr_new = (logit_scale * new_image_features @ self.vr_features.t()).t()

        return logits_per_image, logits_per_text, logits_per_vr_new, logits_per_vr_old

    def training_step(self, batch, batch_idx):
        """Implements pytorch lightning function"""

        # Run CLIP in eval mode since our batch size is much smaller than during CLIP training
        for m in self.model.clip_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        images, texts, logits_per_vr_old = batch
        logits_per_image, logits_per_text, logits_per_vr_new, logits_per_vr_old = self._shared_forward(
            images, texts, logits_per_vr_old
        )

        loss = self.compute_loss(logits_per_image, logits_per_text, logits_per_vr_new, logits_per_vr_old)

        self.log_dict({"train_loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}, on_step=False, on_epoch=True)

        return loss

    def _shared_eval_step(self, batch):
        """Called at the end of validation or test step"""

        images, texts, logits_per_vr_old = batch
        logits_per_image, logits_per_text, logits_per_vr_new, logits_per_vr_old = self._shared_forward(
            images, texts, logits_per_vr_old
        )

        loss = self.compute_loss(logits_per_image, logits_per_text, logits_per_vr_new, logits_per_vr_old)

        return loss


class CLIPLightningWrapperPrediction(pl.LightningModule):
    """Simple Pytorch Lightning Wrapper to only use the model for prediction
    Useful for predict_clip.py script.

    Attributes:
        model: WildCLIP or CLIP model
    """

    def __init__(
        self,
        model: Union[CLIPAdapter, nn.Module],
    ) -> None:
        """
        Args:
            model: pytorch model
        """
        super().__init__()
        self.model = model

    def forward(self, images, texts):
        """Implements pytorch lightning function"""

        return self.model(images, texts)

    def predict_step(self, batch, batch_idx):
        """Implements pytorch lightning function"""

        outputs = self(batch, self.text_tokens)
        return outputs

    def _register_text_tokens(self, text_tokens: torch.Tensor):
        """To set text tokens as class attributes which is required for prediction step

        Args:
            text_tokens: text already tokenized by clip functions
        """
        self.text_tokens = text_tokens
