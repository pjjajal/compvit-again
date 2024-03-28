import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List
import random

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tvt
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from compvit.factory import distill_factory
from compvit.models.compvit import CompViT
from datasets import create_dataset
from datasets.imagenet_ffcv import create_train_pipeline, create_val_pipeline
from dinov2.models.vision_transformer import DinoVisionTransformer
from utils.schedulers import CosineAnnealingWithWarmup

CONFIG_PATH = Path("./configs")
DEFAULT_CHECKPOINTS_PATH = Path("./checkpoints")

torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "cifar100", "imagenet", "imagenet-21k"],
    )
    parser.add_argument("--downsize", required=True, type=int, default=224)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--checkpoints_path", type=Path, default=None)
    parser.add_argument(
        "--precision",
        choices=[
            "32-true",
            "32",
            "16-mixed",
            "bf16-mixed",
            "transformer-engine",
            "16-true",
            "bf16-true",
            "64-true",
        ],
        default="bf16-mixed",
    )
    parser.add_argument(
        "--overfit_batches",
        type=float,
        default=0,
        help="Overfit on a subset of the data for debugging purposes",
    )
    parser.add_argument(
        "--augmentations", default=False, action="store_true", help="Use augmentations"
    )
    parser.add_argument(
        "--symmetric", default=False, action="store_true", help="Use symmetric downsize"
    )
    parser.add_argument(
        "--use_mixup", default=False, action="store_true", help="Use mixup"
    )
    parser.add_argument(
        "--round_robin", default=False, action="store_true", help="Robin robin downsize"
    )
    parser.add_argument(
        "--ccr_loss", default=False, action="store_true", help="Use CCR loss"
    )
    parser.add_argument(
        "--cache_save_path", type=Path, default=None, help="Path where to cache data"
    )

    return parser.parse_args()


class LightningDistill(L.LightningModule):
    def __init__(
        self,
        student: CompViT,
        teacher: DinoVisionTransformer,
        args,
        hyperparameters,
        config,
    ):
        super().__init__()
        # Args, hyperparameters, and config.
        self.args = args
        self.hyperparameters = hyperparameters
        self.config = config
        self.distill_conf = config["distill"]

        # Student and teacher models.
        self.student = student
        self.teacher = teacher

        # Decoder.
        self.decoder = nn.Linear(student.embed_dim * 2, teacher.embed_dim * 2)

        # Transformations.
        self.downsize = tvt.Resize(args.downsize)
        if args.round_robin:
            self.downsize = tvt.RandomChoice(
                [
                    tvt.Resize(56),
                    tvt.Resize(112),
                    tvt.Resize(224),
                ]
            )

        if args.use_mixup:
            self.mixup = tvt.MixUp(
                alpha=hyperparameters["mixup_alpha"],
                num_classes=hyperparameters["mixup_classes"],
            )

        # Loss tracking.
        self.running_loss = 0
        self.lowest_batch_loss = float("inf")

    @torch.no_grad()
    def forward_teacher(self, x):
        x = self.teacher.forward(x, is_training=True)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"]
        x = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return x

    def forward_student(self, x):
        x = self.student.forward(x, is_training=True)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"]
        x = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return x

    def calculate_loss(self, x, x_teacher):
        if self.distill_conf["loss"] == "l2":
            return F.mse_loss(x, x_teacher, reduction="mean")
        elif self.distill_conf["loss"] == "smooth":
            return F.smooth_l1_loss(x, x_teacher, reduction="mean")
        else:
            raise NotImplementedError(
                f"Loss {self.distill_conf['loss']} not implemented"
            )

    def calculate_ccr_loss(self, x, x_teacher):
        x_norm = x.norm(dim=-1)
        x_teacher_norm = x_teacher.norm(dim=-1)

        cc_x = (x @ x.T) / x_norm.outer(x_norm)
        cc_x_teacher = (x_teacher @ x_teacher.T) / x_teacher_norm.outer(x_teacher_norm)

        return F.mse_loss(cc_x, cc_x_teacher, reduction="mean")

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.args.use_mixup:
            x, y = self.mixup(x, y)

        # Teacher forward.
        teacher_encodings = self.forward_teacher(self.downsize(x))

        # Student forward.
        student_encodings = self.forward_student(
            self.downsize(x) if self.args.symmetric else x
        )
        decoded_encodings = self.decoder(student_encodings)

        # Loss.
        loss = self.calculate_loss(decoded_encodings, teacher_encodings)

        if self.args.ccr_loss:
            # CCR Loss.
            ccr_loss = self.distill_conf["ccr_beta"] * self.calculate_ccr_loss(
                decoded_encodings, teacher_encodings
            )
            loss += ccr_loss
            self.log(
                "ccr loss",
                ccr_loss,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # Running loss.
        self.running_loss += loss.detach().item()
        # if self.global_rank == 0:
        self.log(
            "train loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        parameters = list(self.student.parameters()) + list(self.decoder.parameters())
        optimizer = optim.AdamW(
            parameters,
            lr=self.hyperparameters["lr"],
            weight_decay=5e-2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWithWarmup(
                    optimizer,
                    T_max=self.hyperparameters["epochs"],
                    eta_min=self.hyperparameters["min_lr"],
                    warmup_epochs=self.hyperparameters["warmup_epochs"],
                ),
                "interval": "epoch",
            },
        }

    def on_train_epoch_end(self) -> None:
        if self.running_loss < self.lowest_batch_loss:
            self.lowest_batch_loss = self.running_loss
            self.running_loss = 0
            # Save Model
            if self.global_rank == 0:
                save_path = self.args.save_loc / f"best_performing.pth"
                save_path_decoder = self.args.save_loc / f"best_decoder.pth"
                torch.save(self.student.state_dict(), save_path)
                torch.save(self.decoder.state_dict(), save_path_decoder)


def main(args):
    config_path = CONFIG_PATH / (args.dataset + "_pt_dino" + ".yaml")
    configs = OmegaConf.load(config_path)
    teacher_config = configs["teacher"]
    student_config = configs["student"]
    hyperparameters = configs["hyperparameters"]

    # Merging config with CLI args. CLI is prioritized over config.
    args = OmegaConf.merge(
        configs["args"],
        vars(args),
    )

    # Get checkpoint paths.
    teacher_checkpoint = teacher_config["checkpoint"]
    student_checkpoint = student_config["checkpoint"]
    decoder_checkpoint = student_config["decoder_checkpoint"]

    # Create MAE.
    student, teacher, config = distill_factory(
        teacher_name=teacher_config["name"], student_name=student_config["name"]
    )
    if teacher_checkpoint:
        teacher.load_state_dict(torch.load(teacher_checkpoint))
    if student_checkpoint:
        student.load_state_dict(torch.load(student_checkpoint), strict=False)
    if decoder_checkpoint:
        student.load_state_dict(torch.load(decoder_checkpoint))

    model = LightningDistill(student, teacher, args, hyperparameters, config)

    # # Setup W&B.
    wandb_logger = WandbLogger(project="compvit-again-rcac")

    # Create lr monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Create trainer.
    trainer = L.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=hyperparameters["accumulations"],
        max_epochs=hyperparameters["epochs"],
        logger=wandb_logger,
        benchmark=True,  # cudnn benchmarking, allows for faster training.
        enable_checkpointing=False,  # Disable automatic checkpointing (we do this manually).
        callbacks=[lr_monitor],
        overfit_batches=args.overfit_batches,
        log_every_n_steps=50,
        strategy="ddp_find_unused_parameters_true",
    )

    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(
            {
                "architecture": "mae",
                # "dataset": args.dataset,
                "teacher": teacher_config["name"],
                "student": student_config["name"],
                "teacher_checkpoint": teacher_config["checkpoint"],
                "student_checkpoint": student_config["checkpoint"],
                **config,
                **hyperparameters,
                **args,
            }
        )

    # Create dataset and train loader.
    train_dataset, test_dataset = create_dataset(args)

    # Cache data for imagnet-21k.
    if args.cache_save_path and args.dataset == "imagenet-21k":
        cache_data = train_dataset.cache_data()
        args.cache_save_path.mkdir(parents=True, exist_ok=True)
        save_path = args.cache_save_path / f"{args.dataset}_cache_data.json"
        with open(save_path, "w") as f:
            json.dump(cache_data, f)

    # Create train loader.
    loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=False if args.overfit_batches else True,
        num_workers=hyperparameters["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    # Trainer Fit.
    trainer.fit(model, loader)


if __name__ == "__main__":
    args = parse_args()

    now = "distill_" + datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = DEFAULT_CHECKPOINTS_PATH / now
    if args.checkpoints_path:
        save_loc = args.checkpoints_path / now

    if not save_loc.exists():
        save_loc.mkdir(parents=True, exist_ok=True)

    args.save_loc = save_loc
    args.pretraining = True
    main(args)
