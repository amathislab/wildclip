#
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
#
# Licensed under GNU Lesser General Public License v3.0
#

"""
Starting from a pretrained_clip model, this script adapts it to the context of camera traps.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from clip import tokenize
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets.datasets import SerengetiDataset, SerengetiDatasetWithVocabularyReplay
from datasets.sampler import BalancingSampler, SequentialSampler
from models.clip_adapter import CLIPAdapter, get_model
from models.lightning_wrapper import CLIPLightningWrapper, CLIPLightningWrapperLwF
from utils.clip_utils import set_device, set_seed
from utils.lwf_utils import run_logits_per_vr_old, run_replayed_vocabulary_logits
from utils.parse_config_file import parse_transforms, process_config_file


def process_input_csv(
    input_csv: str,
    validation_csv: bool = True,
    label_col: str = None,
    few_shots: bool = False,
    shots: int = None,
    k_shot_col: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Process input CSV

    If multiple captions correspond to an image in label_col, they should be separated with a "; ".
    The resulting dataframes will be exploded with each image-caption entry.

    Args:
        input_csv: path to input annotations
        use_validation: if there is a validation set
        validation_sites_json: entries corresponding to SiteID in input CSV of the validation sites
        label_col: name of the pandas caption column
        few_shots: if few shots setting, only select the K shots
        shots: number of shots to use (predefined)
        k_shot_col: column of the predefined shots

    Returns:
        train image path and caption pairs
        validation image path and caption pairs (empty if use_validation set to false)
        prompt set: unique set of captions (considering both the train and validation sets)
    """

    # Load input annotations
    annotations_train_df = pd.read_csv(input_csv, index_col=0)

    if validation_csv is not None:
        annotations_val_df = pd.read_csv(validation_csv, index_col=0)
    else:
        annotations_val_df = pd.DataFrame(index=None, columns=annotations_train_df.columns)

    if few_shots and shots is not None and k_shot_col is not None:
        annotations_train_df = annotations_train_df[annotations_train_df[k_shot_col] <= shots]

    # one image typically has multiple captions describing it, so we build multiple pairs
    # Captions should
    annotations_train_df[label_col] = annotations_train_df[label_col].str.split("; ")
    annotations_train_df = annotations_train_df.explode(label_col)

    if validation_csv is not None:
        annotations_val_df[label_col] = annotations_val_df[label_col].str.split("; ")
        annotations_val_df = annotations_val_df.explode(label_col)

    if validation_csv is not None:
        train_prompt_set = annotations_train_df[label_col].unique()
        val_prompt_set = annotations_val_df[label_col].unique()
        prompt_set = np.unique(np.concatenate((train_prompt_set, val_prompt_set), axis=0))
    else:
        prompt_set = annotations_train_df[label_col].unique()

    return annotations_train_df, annotations_val_df, prompt_set


def explode_logits_per_vr(
    logits_per_vr_df: pd.DataFrame, annotations_df: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explode logits per vr df to for each image-caption pair
    Args:
        logits_per_vr_df: dataframe containing logits between replayed vocabulary and images
        annotations_df: dataframe of image path -caption pairs
    Returns:
        logits_per_vr: tensor containing logits between replayed vocabulary and images for each image-caption pair
    """

    # Load and merge with datasets
    logits_per_vr_old_col = logits_per_vr_df.columns.tolist()
    annotations_df = annotations_df.merge(logits_per_vr_df, left_index=True, right_index=True)

    # Convert to tensors
    logits_per_vr_old = torch.tensor(annotations_df[logits_per_vr_old_col].values)

    return logits_per_vr_old


def prepare_lightning_wrapper(
    adapted_clip_model: CLIPAdapter,
    cfg_train: Dict,
    lwf_loss: bool = False,
    replayed_vocabulary_features: torch.Tensor = None,
    clip_model_file: Union[Path, str] = None,
    override_adapter: bool = False,
) -> CLIPLightningWrapper:
    """Prepare the pytorch lightning wrapper around the adapted clip model

    Args:
        adapted_clip_model: pytorch model following the CLIP Adapter architecture
        cfg_train: config parameters relative to training
        device: device to load model on
        lwf_loss: if true, loads replayed voc features and class specific wrapper
        path_replayed_vocabulary_features: path to replayed voc features
        clip_model_file: optional path to previous checkpoint
        override_adapter: if true, overrides the previous adapter module

    Returns:
        lightning_model: adapted_clip_model with lightning wrapper
    """

    # Lightning wrapper
    if lwf_loss:
        # Replayed vocabulary embeddings passed through old text encoder
        lightning_model = CLIPLightningWrapperLwF(
            model=adapted_clip_model, lock_vision=cfg_train["lock_vision"], vr_features=replayed_vocabulary_features
        )
    else:
        lightning_model = CLIPLightningWrapper(
            model=adapted_clip_model,
            lock_vision=cfg_train["lock_vision"],
        )

    # Load a previous checkpoint if provided
    if clip_model_file is not None:
        model_dict = torch.load(clip_model_file)

        # In some cases, we want to override the MLP adapter,
        # this is for example when the wrong embedding dimension was provided (going from ResNet to ViT)
        # but did not raise an error during training.
        if override_adapter:
            for key in list(model_dict["state_dict"].keys()):
                if "adapter" in key:
                    del model_dict["state_dict"][key]

        # Strict is set to false since after deletion of the adapter, state_dict keys don't match
        lightning_model.load_state_dict(model_dict["state_dict"], strict=False)

        # If an adapter has already been trained, then we keep alpha value.
        # If not, the alpha parameter was usually set to 0 as an untrainable parameter.
        # In this case, we set it to default 0.7.
        is_alpha_0 = lightning_model.model.adapter.alpha <= 1e-5
        if cfg_train["use_vision_adapter"] and is_alpha_0:
            lightning_model.model.adapter.alpha = torch.nn.Parameter(
                torch.FloatTensor([0.7]), requires_grad=cfg_train["learn_adapter_alpha"]
            )

    return lightning_model


def get_checkpoint_path_few_shots(cfg: Dict, shots: int = None) -> Path:
    """Since few shots experiments are run multiple times, output files have a specific index

    Args:
        cfg: config parameters
        shots: number of shots used during fine-tuning

    Returns:
        checkpoint path name
    """

    if shots is not None:
        chkpt_path = Path(
            cfg["save_dir"] / cfg["experiment_name"] / (cfg["experiment_name"] + "_" + str(shots) + "_shots.pth")
        )
    else:
        chkpt_path = Path(cfg["save_dir"] / cfg["experiment_name"] / (cfg["experiment_name"] + "_all_shots.pth"))

    # we repeat few shot experiments multiple times, so the file name may already exist, which in this case, we increment the ID
    base_path = chkpt_path.parent
    name = chkpt_path.stem

    exp_id = 0
    new_filename = f"{name}_{exp_id}.pth"
    while (base_path / new_filename).exists():
        exp_id += 1
        new_filename = f"{name}_{exp_id}.pth"

    chkpt_path = base_path / new_filename
    return chkpt_path


def fine_tune_clip(
    config_file: Union[str, Path],
    input_csv: Union[str, Path],
    validation_csv: Union[str, Path],
    clip_model_file: Union[str, Path] = None,
    lwf_loss: bool = False,
    path_replayed_vocabulary: Union[str, Path] = None,
    path_replayed_vocabulary_features: Union[str, Path] = None,
    path_logits_per_vr_old_train: Union[str, Path] = None,
    path_logits_per_vr_old_val: Union[str, Path] = None,
    few_shots: bool = False,
    shots: int = None,
    override_adapter: bool = False,
) -> Path:
    """Process to finetune CLIP visual backbone with or without the adapter module

    For consistency purposes, every model follows the CLIP Adapter architecture, even when the adapter module is not finetuned.
    The process is as follows:
        loading data
        building datasets samplers and dataloaders
        loading model
        building the lightning wrapper
        training model
        saving fine-tuned model

    Returns:
        path to fine-tuned model
    """
    # Process configuration file
    cfg, cfg_train, _, cfg_optim = process_config_file(config_file)

    # Set up logger
    logger = TensorBoardLogger(save_dir=cfg["save_dir"], name=cfg["experiment_name"])

    # seed, device
    seed = set_seed()
    device = set_device()

    # Process input CSV
    annotations_train_df, annotations_val_df, prompt_set = process_input_csv(
        input_csv=input_csv,
        validation_csv=validation_csv,
        label_col=cfg["label_col"],
        few_shots=few_shots,
        shots=shots,
        k_shot_col=cfg["k_shot_col"],
    )

    # Tokenize captions
    map_prompt_tokens = {prompt: tokenize(prompt) for prompt in prompt_set}

    # Captions
    train_tokens = annotations_train_df[cfg["label_col"]].map(map_prompt_tokens).values
    train_captions = annotations_train_df[cfg["label_col"]].values
    val_tokens = annotations_val_df[cfg["label_col"]].map(map_prompt_tokens).values
    val_captions = annotations_val_df[cfg["label_col"]].values
    train_filenames = cfg["data_path"] / annotations_train_df["crop_path"].values
    val_filenames = cfg["data_path"] / annotations_val_df["crop_path"].values

    # Prepare VR-LwF loss components
    replayed_voc_features = None
    if lwf_loss:
        if path_logits_per_vr_old_train:
            logits_per_vr_old_train_df = pd.read_csv(path_logits_per_vr_old_train, index_col=0).drop("true", axis=1)
            if path_logits_per_vr_old_val and validation_csv is not None:
                logits_per_vr_old_val_df = pd.read_csv(path_logits_per_vr_old_val, index_col=0).drop("true", axis=1)

        else:
            # We need to compute the cosine similarity with train / val images
            logits_per_vr_old_train_df = run_logits_per_vr_old(
                config_file,
                path_replayed_vocabulary,
                zero_shot_clip=clip_model_file is not None,
                model_file=clip_model_file,
                input_csv=input_csv,
            )
            if validation_csv is not None:
                logits_per_vr_old_val_df = run_logits_per_vr_old(
                    config_file,
                    path_replayed_vocabulary,
                    zero_shot_clip=clip_model_file is not None,
                    model_file=clip_model_file,
                    input_csv=validation_csv,
                )

        # We also need to explode logits per vocabulary replay since we already exploded the columns of
        # of the input CSV for each caption.
        logits_per_vr_old_train = explode_logits_per_vr(logits_per_vr_old_train_df, annotations_train_df)
        if validation_csv is not None:
            logits_per_vr_old_val = explode_logits_per_vr(logits_per_vr_old_val_df, annotations_val_df)

        if path_replayed_vocabulary_features is not None:
            replayed_voc_features = pd.read_csv(path_replayed_vocabulary_features, index_col=0).values
            replayed_voc_features = torch.tensor(replayed_voc_features, dtype=torch.float32).to(device)

        else:
            # We need to compute the embeddings of the replayed vocabulary
            replayed_voc_features = run_replayed_vocabulary_logits(
                config_file=config_file,
                replayed_vocabulary_file=path_replayed_vocabulary,
                zero_shot_clip=clip_model_file is not None,
                model_file=clip_model_file,
                input_csv=input_csv,
            )
            replayed_voc_features = torch.tensor(replayed_voc_features, dtype=torch.float32).to(device)

    # Datasets
    if lwf_loss:
        serengeti_train_dataset = SerengetiDatasetWithVocabularyReplay(
            train_filenames, logits_per_vr_old_train, train_tokens
        )
        serengeti_val_dataset = SerengetiDatasetWithVocabularyReplay(val_filenames, logits_per_vr_old_val, val_tokens)
    else:
        serengeti_train_dataset = SerengetiDataset(train_filenames, train_tokens)
        serengeti_val_dataset = SerengetiDataset(val_filenames, val_tokens)

    # Samplers
    if few_shots:
        train_sampler = SequentialSampler(serengeti_train_dataset, shuffle=True)
    else:
        train_sampler = BalancingSampler(
            train_captions, replacement=True, num_samples_per_epoch=cfg_train["num_samples_per_epoch"]
        )

    if validation_csv is not None:
        val_sampler = BalancingSampler(
            val_captions,
            replacement=True,
            num_samples_per_epoch=int(cfg_train["num_samples_per_epoch"] * cfg_train["validation_set_fraction"]),
        )
    else:
        val_sampler = None

    # Data loaders
    train_dataloader = DataLoader(
        serengeti_train_dataset,
        sampler=train_sampler,
        batch_size=cfg_train["batch_size"],
        num_workers=6,
        pin_memory=False,
    )
    val_dataloader = DataLoader(
        serengeti_val_dataset, batch_size=cfg_train["batch_size"], sampler=val_sampler, num_workers=6, pin_memory=False
    )

    # Create CLIP Adapted model
    clip_model, input_transform = get_model(model_name=cfg["clip_model_name"], device=device)

    if lwf_loss:
        return_features_only = True
    else:
        return_features_only = False

    adapted_clip_model = CLIPAdapter(
        clip_model,
        embed_dim=cfg_train["embed_dim"],
        adapter_hidden_channels=cfg_train["adapter_hidden_channels"],
        use_vision_adapter=cfg_train["use_vision_adapter"],
        learn_alpha=cfg_train["learn_adapter_alpha"],
        return_features_only=return_features_only,
    )

    # Create lightning wrapper
    lightning_model = prepare_lightning_wrapper(
        adapted_clip_model,
        cfg_train,
        lwf_loss,
        replayed_voc_features,
        clip_model_file,
        override_adapter,
    )
    print(lightning_model.model)
    print("Value of alpha:", lightning_model.model.adapter.alpha)

    # Transforms
    train_data_transform = parse_transforms(transform_cfg=cfg_train["transforms"], input_transform=input_transform)
    train_dataloader.dataset.set_transform_func(train_data_transform)
    val_dataloader.dataset.set_transform_func(input_transform)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            every_n_epochs=1,
            save_last=False,
            save_top_k=1,
            monitor="val_loss",
            save_on_train_epoch_end=False,
            save_weights_only=True,
        )
    ]

    # Configures trainable parameters, optimizer and scheduler
    total_steps = len(train_dataloader) * cfg_train["max_num_epochs"]
    lightning_model.prepare(cfg_optim, total_steps)

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        max_epochs=cfg_train["max_num_epochs"],
        min_epochs=cfg_train["min_num_epochs"],
        num_sanity_val_steps=0,
        log_every_n_steps=min(max(len(train_dataloader) // 10, 100), len(train_dataloader)),
    )

    if len(val_dataloader) == 0:
        trainer.fit(model=lightning_model, train_dataloaders=train_dataloader)
    else:
        trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Saves last checkpoint
    if few_shots:
        chkpt_path = get_checkpoint_path_few_shots(cfg, shots)
    else:
        chkpt_path = cfg["save_dir"] / cfg["experiment_name"] / (cfg["experiment_name"] + "_last_ckpt.pth")

    trainer.save_checkpoint(filepath=chkpt_path)

    # Copy config file
    shutil.copy(config_file, Path(chkpt_path).parent)

    return chkpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("-F", "--config_file", type=str, help="YML Configuration file", required=True)
    parser.add_argument("-I", "--input_csv", type=str, help="Input CSV File", required=True)
    parser.add_argument("-V", "--validation_csv", type=str, help="Validation CSV File", required=False)

    # Load from checkpoint
    parser.add_argument(
        "-M", "--clip_model", type=str, help="Path to a pretrained checkpoint", default=None, required=False
    )

    # VR-LwF Loss
    parser.add_argument("--lwf_loss", action="store_true", help="Run with VR-LwF loss")
    parser.add_argument(
        "--path_replayed_vocabulary",
        type=str,
        help="txt file with replayed sentences",
        default=None,
        required=("--lwf_loss" in sys.argv) and ("--path_replayed_vocabulary_features" not in sys.argv),
    )
    parser.add_argument(
        "--path_replayed_vocabulary_features",
        type=str,
        help="CSV file with replayed sentences features",
        default=None,
        required=("--lwf_loss" in sys.argv) and ("--path_replayed_vocabulary" not in sys.argv),
    )
    parser.add_argument(
        "--path_logits_per_vr_old_train",
        type=str,
        help="Precomputed similarities between replayed vocabulary and train images",
        default=None,
        required=("--lwf_loss" in sys.argv) and ("--path_replayed_vocabulary" not in sys.argv),
    )
    parser.add_argument(
        "--path_logits_per_vr_old_val",
        type=str,
        help="Precomputed similarities between replayed vocabulary and validation images",
        default=None,
    )

    # Few-shot fine-tuning (CLIP-Adapter)
    parser.add_argument("--few_shots", action="store_true", help="Few-shot fine-tuning of the CLIP-Adapter module")
    parser.add_argument("-K", "--shots", type=int, choices=[1, 2, 4, 8], help="Number of shot per class", default=None)
    parser.add_argument("--override_adapter", action="store_true", help="Discards adapter module from state dict")

    opt = parser.parse_args()

    fine_tuned_clip_ckpt = fine_tune_clip(
        Path(opt.config_file),
        Path(opt.input_csv),
        opt.validation_csv,
        opt.clip_model,
        opt.lwf_loss,
        opt.path_replayed_vocabulary,
        opt.path_replayed_vocabulary_features,
        opt.path_logits_per_vr_old_train,
        opt.path_logits_per_vr_old_val,
        opt.few_shots,
        opt.shots,
        opt.override_adapter,
    )
