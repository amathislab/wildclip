# 
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
# 
# Licensed under GNU Lesser General Public License v3.0
# 

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch


# GPU or CPU computation
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device


# Random state
def set_seed(seed=None):
    if not seed:
        seed = sum([int(ord(c)) for c in "wildai"])
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)

    return seed


def load_prompts(*prompt_files: Tuple[Union[Path, str]]) -> Dict:
    """
    Load prompts from files
    One prompt per line

    Returns:
        Dictionary with prompt as values and id as key
    """

    all_prompts = []

    for prompt_file in prompt_files:
        with open(prompt_file, "r") as f:
            data = f.read()
            prompts = data.splitlines()
            all_prompts += prompts

    # Remove duplicates
    all_prompts = list(set(all_prompts))

    prompt_dict = {n: c for n, c in enumerate(all_prompts)}

    return prompt_dict


def process_logits(
    outputs: "torch.Tensor",
    labels: "np.array",
    prompt_set: List,
    sample_ids: Union["np.array", List],
    save_filename: Union[str, Path] = None,
) -> "pd.DataFrame":
    """Process output from forward pass of species classifier

    Args:
        outputs: logits for all batches
        labels: labels corresponding to each output
        prompt_set: prompts for which the cosine similarity was computed. Must be unique.
        sample_ids: sample id corresponding to each output
        save_filename: Path to save processed outputs. If none, does not save the file

    Returns:
        Dataframe containing the cosine similarity for each prompt, and image labels, and sample ids as index
    """

    columns = [
        *prompt_set,
        "true",
    ]
    prediction_df = pd.DataFrame(index=sample_ids, columns=columns)

    # Init. holders
    logits = np.concatenate([b[0].tolist() for b in outputs], axis=0)
    prediction_df[[*prompt_set]] = logits
    prediction_df["true"] = labels

    # Save to CSV
    if save_filename is not None:
        prediction_df.to_csv(save_filename)

    return prediction_df


def process_features(
    outputs: "torch.Tensor",
    prompt_set: List,
    sample_ids: Union[np.array, List],
    save_filename: Union[str, Path] = None,
) -> "pd.DataFrame":
    """Save features in two csv files

    Args:
        outputs: features for all batches
        prompt_set: prompts for which the cosine similarity was computed.
        sample_ids: sample id corresponding to each output
        save_filename: Path to save processed outputs, will be appended. If none, does not save the file

    Returns:
        Dataframe containing the embeddings of each images
        Dataframe containing the embeddings for each prompt of the prompt set
    """

    image_features = np.concatenate([b[0].tolist() for b in outputs], axis=0)

    # The same for all batches, so we just keep the first
    text_features = np.array(outputs[0][1].tolist())
    image_features_df = pd.DataFrame(index=sample_ids, data=image_features)
    text_features_df = pd.DataFrame(index=prompt_set, data=text_features)

    if save_filename is not None:
        output_file_images = Path(save_filename).parent / (Path(save_filename).stem + "_img_features.csv")
        output_file_text = Path(save_filename).parent / (Path(save_filename).stem + "_txt_features.csv")
        image_features_df.to_csv(output_file_images)
        text_features_df.to_csv(output_file_text)

    return image_features_df, text_features_df
