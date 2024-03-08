#
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
#
# Licensed under GNU Lesser General Public License v3.0
#

"""
Runs CLIP or a given WildCLIP model file on a folder of images recursively and computes cosine similarity with a given set of queries.
"""
import argparse
import os
import sys
from pathlib import Path
from random import sample
from typing import Union

import torch
from clip import tokenize
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from datasets.datasets import ImageDataset
from models.clip_adapter import CLIPAdapter, get_model
from models.lightning_wrapper import CLIPLightningWrapperPrediction
from utils.clip_utils import load_prompts, process_features, process_logits, set_device, set_seed


def predict_clip(
    image_folder: Union[str, Path],
    prompts_file: Union[str, Path],
    output_folder: Union[str, Path],
    clip_model_name: str = "ViT-B/16",
    num_samples: int = None,
    batch_size: int = 256,
    zero_shot_clip: bool = False,
    model_file: Union[str, Path] = None,
    adapter_hidden_channels: int = 128,
    adapter_embeding_dim: int = 512,
    use_adapter: bool = False,
    return_features: bool = False,
) -> Path:
    if zero_shot_clip and model_file is not None:
        print("Inconsistent input: zero-shot CLIP prediction is true but a model file was given.")
        model_file = None
        print("Considering zero-shot CLIP prediction only.")

    # seed, device
    seed = set_seed()
    device = set_device()

    # test files (recursive)
    test_filenames = [f for f in image_folder.rglob("*") if f.is_file()]
    test_filenames = [
        f for f in test_filenames if f.suffix.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
    ]

    # Sample
    if num_samples is not None:
        test_filenames = sample(test_filenames, k=num_samples)

    # Dataset, DataLoader
    test_dataset = ImageDataset(test_filenames)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=6, pin_memory=True)

    # Prompts
    prompts_file = Path(prompts_file)
    prompts_dict = load_prompts(prompts_file)
    prompt_set = list(prompts_dict.values())
    text_tokens = torch.cat([tokenize(prompt) for prompt in prompt_set]).to(device)

    # Get model from config result folder
    clip_model, input_transform = get_model(model_name=clip_model_name, device=device)
    test_dataset.set_transform_func(input_transform)

    # Prepare model
    if not zero_shot_clip and model_file is not None:
        model_file = Path(model_file)
        model_ckpt = torch.load(model_file)

        adapted_clip_model = CLIPAdapter(
            clip_model,
            embed_dim=adapter_embeding_dim,
            adapter_hidden_channels=adapter_hidden_channels,
            use_vision_adapter=use_adapter,
            learn_alpha=False,
            return_features_only=return_features,
        )

        lightning_model = CLIPLightningWrapperPrediction(model=adapted_clip_model)
        lightning_model.load_state_dict(model_ckpt["state_dict"])
    else:
        lightning_model = CLIPLightningWrapperPrediction(model=clip_model)

    # Add test prompts and prepare model in eval mode
    lightning_model._register_text_tokens(text_tokens)
    lightning_model.model.eval()

    # Get trainer
    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=-1, logger=False)

    # Predict
    outputs = trainer.predict(model=lightning_model, dataloaders=test_dataloader, return_predictions=True)

    # output file
    if not zero_shot_clip and model_file is not None:
        output_filename = output_folder / (str(model_file.stem) + "_" + str(prompts_file.stem) + "_predictions.csv")
    else:
        output_filename = output_folder / ("clip_zero_shot_" + str(prompts_file.stem) + "_predictions.csv")

    if not output_filename.parent.exists():
        os.makedirs(output_filename.parent)

    # process outputs
    if return_features:
        image_features_df, text_features_df = process_features(
            outputs, sample_ids=test_filenames, prompt_set=prompt_set, save_filename=output_filename
        )

        return image_features_df, text_features_df

    else:
        predictions_df = process_logits(
            outputs, sample_ids=test_filenames, prompt_set=prompt_set, labels=None, save_filename=output_filename
        )

        return predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I",
        "--image_folder",
        type=str,
        help="Folder containing images to test. It will be processed recursively",
        required=True,
    )

    parser.add_argument(
        "-C", "--prompts_file", type=str, help="TXT file with one prompt to test per line", required=True
    )
    parser.add_argument("-O", "--output_folder", type=str, help="Folder to save results", required=True)
    parser.add_argument("--clip_model_name", type=str, help="CLIP Model Name", default="ViT-B/16")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=256)
    parser.add_argument("--num_samples", type=int, help="Number of samples to randomly process", default=None)
    parser.add_argument("--zero-shot-clip", action="store_true", help="Set to True for zero-shot clip evaluation")
    parser.add_argument(
        "-M",
        "--model_file",
        type=str,
        help="Path to a wildclip model file",
        required="--zero-shot-clip" not in sys.argv,
    )
    parser.add_argument(
        "--adapter_hidden_channels",
        type=str,
        help="Number of hidden channels of the wildclip model",
        required="--model_file" in sys.argv,
        default=128,
    )
    parser.add_argument(
        "--adapter_embeding_dim",
        type=str,
        help="Embeding dimension of the wildclip model",
        required="--model_file" in sys.argv,
        default=512,
    )
    parser.add_argument("--use_adapter", action="store_true", help="Use the adapter of the wildclip model")
    parser.add_argument(
        "--return-features",
        action="store_true",
        help="If true, returns image and text features in different csv files, if false, return cosine similarities",
    )

    opt = parser.parse_args()

    output = predict_clip(
        Path(opt.image_folder),
        Path(opt.prompts_file),
        Path(opt.output_folder),
        opt.clip_model_name,
        opt.num_samples,
        opt.batch_size,
        opt.zero_shot_clip,
        opt.model_file,
        opt.adapter_hidden_channels,
        opt.adapter_embeding_dim,
        opt.use_adapter,
        opt.return_features,
    )
