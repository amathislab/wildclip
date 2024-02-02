# 
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
# 
# Licensed under GNU Lesser General Public License v3.0
# 

import argparse
from pathlib import Path
from pprint import pprint
from typing import Dict, Tuple

import torchvision
import yaml
from torch.nn import ModuleList
from torchvision import transforms

transforms_dict = {
    "random_resize": transforms.RandomResizedCrop(380, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
    "random_vertical_flip": transforms.RandomVerticalFlip(p=1.0),
    "random_horizontal_flip": transforms.RandomHorizontalFlip(p=1.0),
    "gaussian_blur": transforms.GaussianBlur(9, sigma=(0.1, 2.0)),
    "random_grayscale": transforms.RandomGrayscale(p=1.0),
    "color_jitter": transforms.ColorJitter(brightness=0.1, hue=0.2, saturation=0.2, contrast=0),
}


def parse_config_file(config_file: Path) -> Dict:
    """Parse YML config file

    Args:
        config_file: path to yml config file
    Returns:
        dictionary containing the data
    """
    assert config_file.exists(), f"{config_file} config file not found."

    with open(config_file, "rb") as f:
        config_data = yaml.safe_load(f)

    # Check file integrity
    global_cfg = config_data["global"]
    train_cfg = config_data["train"]
    optim_cfg = train_cfg["optim"]

    # experiment name:
    global_cfg["experiment_name"] = config_file.stem

    # Input files
    global_cfg["save_dir"] = Path(global_cfg["save_dir"])
    assert global_cfg["save_dir"].exists(), f"{global_cfg['save_dir']} output directory does not exist"

    global_cfg["data_path"] = Path(global_cfg["data_path"])
    assert global_cfg["data_path"].exists(), f"{global_cfg['data_path']} does not exists"

    optim_cfg["weight_decay"] = float(optim_cfg["weight_decay"])
    optim_cfg["init_lr"] = float(optim_cfg["init_lr"])
    train_cfg["validation_set_fraction"] = float(train_cfg["validation_set_fraction"])

    return config_data


def process_config_file(config_file: str) -> Tuple[Dict, Dict, Dict]:
    """Process and parse config file

    Args:
        config_file: path to config file

    Returns:
        cfg: global config parameters
        cfg_train: config parameters relative to training
        cfg_optim: config parameters relative to training optimization
    """
    config_file = Path(config_file)
    config_dict = parse_config_file(config_file)

    cfg_train = config_dict["train"]
    cfg_test = config_dict["test"]
    cfg = config_dict["global"]
    cfg_optim = cfg_train["optim"]

    return cfg, cfg_train, cfg_test, cfg_optim


def parse_transforms(
    transform_cfg: Dict,
    input_transform: "torchvision.transforms" = None,
) -> "torchvision.transforms":
    """Returns a chained transformation from the dictionary of transforms defined in the config file"""

    list_geom_transforms = [
        transforms_dict[transform]
        for transform in transform_cfg["geometric"].keys()
        if transform_cfg["geometric"][transform]
    ]
    list_color_transforms = [
        transforms_dict[transform] for transform in transform_cfg["color"].keys() if transform_cfg["color"][transform]
    ]

    transforms_geom = transforms.RandomApply(ModuleList(list_geom_transforms), p=transform_cfg["p_geometric"])
    transforms_color = transforms.RandomApply(ModuleList(list_color_transforms), p=transform_cfg["p_color"])

    if input_transform == None:
        transforms_chain = transforms.Compose(
            [
                transforms_geom,
                transforms_color,
            ]
        )
    else:
        transforms_chain = transforms.Compose([transforms_geom, transforms_color, input_transform])

    return transforms_chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--config_file", type=str, help="YML Configuration file")

    opt = parser.parse_args()

    config_file = Path(opt.config_file)
    assert config_file.exists()

    config_dict = parse_config_file(config_file)
    pprint(config_dict, indent=2)
