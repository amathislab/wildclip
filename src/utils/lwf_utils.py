# 
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
# 
# Licensed under GNU Lesser General Public License v3.0
# 

""" Pre-run replayed vocabulary on a given model
"""
import torch

from eval_clip import eval_clip


def run_replayed_vocabulary_logits(config_file, replayed_vocabulary_file, zero_shot_clip, model_file, input_csv):
    """Returns the text embeddings of the replayed vocabulary.

    Since we return text embeddings only, input_csv can be anything.
    """

    _, text_features_df = eval_clip(
        config_file=config_file,
        test_csv=input_csv,
        prompts_file=replayed_vocabulary_file,
        zero_shot_clip=zero_shot_clip,
        model_file=model_file,
        return_features=True,
    )

    text_features = torch.tensor(text_features_df.values)

    return text_features


def run_logits_per_vr_old(config_file, replayed_vocabulary_file, zero_shot_clip, model_file, input_csv):
    """Returns the cosine similarly between text embeddings of the replayed vocabulary
    and image embeddings of the input images.
    """

    predictions_df = eval_clip(
        config_file=config_file,
        test_csv=input_csv,
        prompts_file=replayed_vocabulary_file,
        zero_shot_clip=zero_shot_clip,
        model_file=model_file,
        return_features=False,
        override_results=False,
    )

    prediction_df = predictions_df.drop("true", axis=1)

    return prediction_df
