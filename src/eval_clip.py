#
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
#
# Licensed under GNU Lesser General Public License v3.0
#

"""
Evaluate WildCLIP on a given test set and test captions
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from clip import tokenize
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets.datasets import SerengetiDataset
from models.clip_adapter import CLIPAdapter, get_model
from models.lightning_wrapper import CLIPLightningWrapper
from utils.clip_utils import load_prompts, process_features, process_logits, set_device, set_seed
from utils.parse_config_file import process_config_file


def eval_clip(
    config_file: Union[str, Path],
    test_csv: Union[str, Path],
    prompts_file: Union[str, Path],
    zero_shot_clip: bool = False,
    model_file: Union[str, Path] = None,
    return_features: bool = False,
    override_results: bool = True,
) -> Path:
    if zero_shot_clip and model_file is not None:
        print("Inconsistent input: zero-shot CLIP prediction is true but a model file was given.")
        model_file = None
        print("Considering zero-shot CLIP prediction only.")

    # Process configuration file
    cfg, cfg_train, cfg_test, _ = process_config_file(config_file)

    # seed, device
    seed = set_seed()
    device = set_device()

    # Set up logger
    logger = TensorBoardLogger(save_dir=cfg["save_dir"], name=cfg["experiment_name"])

    # test csv
    annotations_test_df = pd.read_csv(test_csv, index_col=0)
    test_filenames = cfg["data_path"] / annotations_test_df.crop_path.values
    test_sample_ids = annotations_test_df.index.values
    test_labels = annotations_test_df[cfg_test["label_col"]].values

    # Dataset, DataLoader
    serengeti_test_dataset = SerengetiDataset(test_filenames)
    test_dataloader = DataLoader(
        serengeti_test_dataset, batch_size=cfg_test["batch_size"], num_workers=6, pin_memory=True
    )

    # Prompts
    prompts_file = Path(prompts_file)
    prompts_dict = load_prompts(prompts_file)
    prompt_set = list(prompts_dict.values())
    text_tokens = torch.cat([tokenize(prompt) for prompt in prompt_set]).to(device)
    text_tokens = torch.squeeze(text_tokens)

    # Get model from config result folder
    clip_model, input_transform = get_model(model_name=cfg["clip_model_name"], device=device)
    serengeti_test_dataset.set_transform_func(input_transform)

    # If we use zero shot CLIP, we skip the vision adapter
    if zero_shot_clip:
        use_vision_adapter = False
    else:
        use_vision_adapter = cfg_train["use_vision_adapter"]

    adapted_clip_model = CLIPAdapter(
        clip_model,
        embed_dim=cfg_train["embed_dim"],
        adapter_hidden_channels=cfg_train["adapter_hidden_channels"],
        use_vision_adapter=use_vision_adapter,
        learn_alpha=False,
        return_features_only=return_features,
    )

    # Lightning wrapper
    lightning_model = CLIPLightningWrapper(model=adapted_clip_model)

    if not zero_shot_clip and model_file is not None:
        model_file = Path(model_file)
        model_dict = torch.load(model_file)
        lightning_model.load_state_dict(model_dict["state_dict"])

    # Add test prompts and prepare model in eval mode
    lightning_model._register_text_tokens(text_tokens)
    lightning_model.prepare(eval=True)

    # Get trainer
    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=-1)

    # Predict
    outputs = trainer.predict(model=lightning_model, dataloaders=test_dataloader, return_predictions=True)

    # output file
    if not zero_shot_clip and model_file is not None:
        output_filename = (
            cfg["save_dir"] / cfg["experiment_name"] / (model_file.stem + "_" + prompts_file.stem + "_predictions.csv")
        )
    else:
        output_filename = (
            cfg["save_dir"]
            / cfg["experiment_name"]
            / (cfg["experiment_name"] + "_clip_zero_shot_" + prompts_file.stem + "_predictions.csv")
        )

    if not output_filename.parent.exists():
        os.makedirs(output_filename.parent)

    if output_filename.exists() and not override_results:
        base, ext = output_filename.name.split(".")
        output_filename = output_filename.parent / (base + f"_new.{ext}")

    # process outputs
    if adapted_clip_model.return_features_only:
        image_features_df, text_features_df = process_features(
            outputs, prompt_set=prompt_set, sample_ids=test_sample_ids, save_filename=output_filename
        )

        return image_features_df, text_features_df

    else:
        predictions_df = process_logits(
            outputs,
            sample_ids=test_sample_ids,
            prompt_set=prompt_set,
            labels=test_labels,
            save_filename=output_filename,
        )

        return predictions_df


if __name__ == "__main__":
    # Read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--config_file", type=str, help="YML Configuration file", required=True)
    parser.add_argument("-I", "--test_csv", type=str, help="Test csv file", required=True)
    parser.add_argument(
        "-C", "--prompts_file", type=str, help="TXT file with one prompt to test per line", required=True
    )
    parser.add_argument("--zero-shot-clip", action="store_true", help="Set to True for zero-shot clip evaluation")
    parser.add_argument(
        "-M", "--model_file", type=str, help="Path to pytorch model file", required="--zero-shot-clip" not in sys.argv
    )
    parser.add_argument(
        "--return-features",
        action="store_true",
        help="If true, returns image and text features, if false, return cosine similarities",
    )

    opt = parser.parse_args()

    output = eval_clip(
        Path(opt.config_file),
        Path(opt.test_csv),
        Path(opt.prompts_file),
        opt.zero_shot_clip,
        opt.model_file,
        opt.return_features,
    )
