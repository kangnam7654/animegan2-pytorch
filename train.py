import argparse

from torch.utils.data import DataLoader
from torchvision.transforms import v2
import pytorch_lightning as pl

from datamodules.animegan_datamodule import AnimeDataSet
from pipelines.animegan_pipeline import AnimeganPipeline
from models.discriminator import Discriminator
from models.generator import Generator
from models.vgg import Vgg19


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo_dir", type=str)
    parser.add_argument("--anime_dir", type=str)
    parser.add_argument("--smooth_dir", type=str)
    args = parser.parse_args()
    return args


def main(args):
    # | Model Load |
    generator = Generator()
    discriminator = Discriminator()
    vgg = Vgg19()

    # | Dataset Load |
    transform = v2.Compose(
        [
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.LANCZOS),
            v2.ToTensor(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = AnimeDataSet(
        photo_dir=args.photo_dir,
        anime_dir=args.anime_dir,
        smooth_dir=args.smooth_dir,
        transform=transform,
    )
    loader = DataLoader(dataset=dataset)

    pipeline = AnimeganPipeline(
        generator=generator, discriminator=discriminator, vgg=vgg,
    )

    trainer = pl.Trainer()
    trainer.fit(model=pipeline, train_dataloaders=loader)


if __name__ == "__main__":
    args = get_args()
    main(args)
