import os
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from dinov2.factory import dinov2_factory
from dinov2.layers import MemEffAttention
from dinov2.layers import NestedTensorBlock as Block

from .models.compvit import CompViT

CONFIG_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "configs"


def compvit_factory(
    model_name: Literal["compvits14", "compvitb14", "compvitl14", "compvitg14"],
    **kwargs
):
    config_path = CONFIG_PATH / "compvit_dinov2.yaml"
    # Loads the default configuration.
    conf = OmegaConf.load(config_path)
    # kwargs can overwrite the default config. This allows for overriding config defaults.
    conf = OmegaConf.merge(conf[model_name], kwargs)

    return (
        CompViT(block_fn=partial(Block, attn_class=MemEffAttention), **conf),
        conf,
    )


def distill_factory(
    teacher_name: Literal[
        "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
    ],
    student_name: Literal["compvits14", "compvitb14", "compvitl14", "compvitg14"],
):
    config_path = CONFIG_PATH / "distill.yaml"
    distill_conf = OmegaConf.load(config_path)

    teacher, dino_conf = dinov2_factory(teacher_name)

    student, compvit_conf = compvit_factory(student_name)

    return (
        student,
        teacher,
        {**dino_conf, **compvit_conf, **distill_conf},
    )


if __name__ == "__main__":
    model, conf = compvit_factory("compvits14")