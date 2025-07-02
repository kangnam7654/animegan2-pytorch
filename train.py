import argparse

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from datamodules.animegan_datamodule import AnimeDataSet
from models.criterion import AdversarialLoss, ColorLoss, ContentLoss, GrayscaleLoss
from models.discriminator import Discriminator
from models.generator import Generator
from models.vgg import Vgg19
from pipelines.animegan_pipeline import AnimeganPipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo_dir", type=str)
    parser.add_argument("--anime_dir", type=str)
    parser.add_argument("--smooth_dir", type=str)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--pretraining", action="store_true")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--g_lr", type=float, default=8e-5)
    parser.add_argument("--d_lr", type=float, default=1e-4)
    parser.add_argument("--w_adv", type=float, default=300)
    parser.add_argument("--w_con", type=float, default=1.5)
    parser.add_argument("--w_gray", type=float, default=3)
    parser.add_argument("--w_col", type=float, default=10)
    args = parser.parse_args()
    return args


def main(args):
    seed_everything(args.seed)
    # | Model Load |

    generator = Generator()
    discriminator = Discriminator()
    vgg = Vgg19()

    # | Weight Load |
    if args.weight is not None:
        state_dict = torch.load(args.weight, map_location=torch.device("cpu"))
        try:
            generator.load_state_dict(state_dict=state_dict["G"])
            discriminator.load_state_dict(state_dict=state_dict["D"])
        except Exception:
            generator.load_state_dict(state_dict=state_dict)

    # | Dataset Load |
    transform = Compose(
        [
            Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = AnimeDataSet(
        photo_dir=args.photo_dir,
        anime_dir=args.anime_dir,
        smooth_dir=args.smooth_dir,
        transform=transform,
    )
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=args.batch_size)

    adv_loss = AdversarialLoss()
    con_loss = ContentLoss()
    gray_loss = GrayscaleLoss()
    col_loss = ColorLoss()

    pipeline = AnimeganPipeline(
        generator=generator,
        discriminator=discriminator,
        vgg=vgg,
        adv_loss=adv_loss,  # AdversarialLoss(),
        con_loss=con_loss,  # ContentLoss(),
        gray_loss=gray_loss,  # GrayscaleLoss(),
        col_loss=col_loss,  # ColorLoss(),
        save_every=args.save_every,
        pretraining=args.pretraining,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        w_adv=args.w_adv,
        w_con=args.w_con,
        w_gray=args.w_gray,
        w_col=args.w_col,
    )
    logger_mlflow = MLFlowLogger(
        experiment_name="animegan2", run_name="animegan2_training"
    )
    logger_wandb = WandbLogger(project="animegan2", name="animegan2_training")
    loggers = [logger_mlflow, logger_wandb]
    trainer = Trainer(logger=loggers)
    trainer.fit(model=pipeline, train_dataloaders=loader)


if __name__ == "__main__":
    args = get_args()
    main(args)
