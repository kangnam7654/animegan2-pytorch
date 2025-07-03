from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchvision.utils import make_grid

from animegan2.models import Discriminator, Generator, Vgg19
from animegan2.models.criterion import (
    AdversarialLoss,
    ColorLoss,
    ContentLoss,
    GrayscaleLoss,
)


class AnimeganPipeline(LightningModule):
    def __init__(
        self,
        *,
        generator: Generator,
        discriminator: Discriminator,
        vgg: Vgg19,
        adv_loss: AdversarialLoss,
        con_loss: ContentLoss,
        gray_loss: GrayscaleLoss,
        col_loss: ColorLoss,
        w_adv: float = 300,
        w_con: float = 1.5,
        w_gray: float = 3,
        w_col: float = 10,
        g_lr: float = 8e-5,
        d_lr: float = 1e-4,
        save_every: int = 5000,
        pretraining: bool = False,
    ):
        super().__init__()
        # ==========
        # | Models |
        # ==========
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg

        # ==============
        # | Optimizers |
        # ==============
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.automatic_optimization = False

        # ==================
        # | Loss Functions |
        # ==================
        self.adv_loss = adv_loss
        self.con_loss = con_loss
        self.gray_loss = gray_loss
        self.col_loss = col_loss

        # ====================
        # | Weights for loss |
        # ====================
        self.w_adv = w_adv
        self.w_con = w_con
        self.w_gray = w_gray
        self.w_col = w_col

        # ==========================
        # | Pre-training Parameters |
        # ==========================
        self.pretraining = pretraining
        self.training_step_counter = 0

        # =========================
        # | Save Checkpoint Every |
        # =========================
        self.save_every = save_every

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        # ==========================
        # | Pre-training if needed |
        # ==========================
        if self.pretraining and self.current_epoch == 0:
            self._pre_training(batch)

        else:
            # After Pre-training
            photo, anime, gray, smooth = batch  # p, a, x, y
            g_opt, d_opt = self.configure_optimizers()

            # =================
            # | Discriminator |
            # =================
            self.generator.requires_grad_(False)
            self.generator.eval()

            self.discriminator.requires_grad_(True)
            self.discriminator.train()

            fake = self.generator(photo)  # G(p)

            real_out = self.discriminator(anime)  # D(a)
            fake_out = self.discriminator(fake)  # D(G(p))
            gray_out = self.discriminator(gray)  # D(x)
            smooth_out = self.discriminator(smooth)  # D(y)

            # ======================
            # | Discirminator Loss |
            # ======================
            # E[(D(a) - 1)^2]
            # d_real_loss = torch.square(real_out - torch.ones_like(real_out)).mean()
            d_real_loss = self.adv_loss(real_out, torch.ones_like(real_out))

            # E[(D(G(p)))^2]
            # d_fake_loss = torch.square(fake_out).mean()
            d_fake_loss = self.adv_loss(fake_out, torch.zeros_like(fake_out))

            # E[(D(x))^2]
            # d_gray_loss = torch.square(gray_out).mean()
            d_gray_loss = self.adv_loss(gray_out, torch.zeros_like(gray_out))

            # E[(D(y))^2]
            # d_smooth_loss = torch.square(smooth_out).mean()
            d_smooth_loss = self.adv_loss(smooth_out, torch.zeros_like(smooth_out))

            d_loss = self.w_adv * (
                d_real_loss + d_fake_loss + d_gray_loss + 0.1 * d_smooth_loss
            )

            # ==============
            # | D Backward |
            # ==============
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()

            # ------------------------------------------------------------------------

            # =============
            # | Generator |
            # =============
            self.generator.requires_grad_(True)
            self.generator.train()

            self.discriminator.requires_grad_(False)
            self.discriminator.eval()

            fake = self.generator(photo)  # G(p)
            fake_out = self.discriminator(fake)  # D(G(p))

            # | For perceptual Loss |
            vgg_photo = self.vgg(photo)  # VGG(p)
            vgg_fake = self.vgg(fake)  # VGG(G(p))
            vgg_gray = self.vgg(gray)  # VGG(G(x))

            # | Generator Loss |
            # E[(G(p) - 1)^2]
            """
            24.01.04 by Kangnam Kim
            It should be E[(D(G(p)) - 1)^2]
            """
            # g_adv_loss = torch.square(fake_out - torch.ones_like(fake_out)).mean()
            g_adv_loss = self.adv_loss(fake_out, torch.ones_like(fake_out))

            # E[|| VGG(p) - VGG(G(p))||_1]
            g_con_loss = self.con_loss(vgg_photo, vgg_fake)

            # E[||Gram(VGG(G(p))) - Gram(VGG(x))||_1]
            g_gray_loss = self.gray_loss(vgg_gray, vgg_fake)

            # E[||Y(G(p)) - Y(p)||_1 + ||U(G(p)) - U(p)||_Huber + ||V(G(p)) - V(p))||_Huber]
            g_color_loss = self.col_loss(photo, fake)

            g_loss = (
                self.w_adv * g_adv_loss
                + self.w_con * g_con_loss
                + self.w_gray * g_gray_loss
                + self.w_col * g_color_loss
            )

            # ==============
            # | G Backward |
            # ==============
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()

            # | Logging |
            to_log = {"G_loss": g_loss, "D_loss": d_loss}
            self.log_dict(to_log, prog_bar=True)

            # | Image Logging |
            if self.training_step_counter % 100 == 0:
                save_dir = Path(__file__).parents[1].joinpath("./logs/images")
                save_dir.mkdir(parents=True, exist_ok=True)
                full_path = save_dir.joinpath(
                    f"{self.training_step_counter}".zfill(8) + ".jpg"
                )
                image = self.tensor_to_image(photo, fake, anime, gray, smooth, save_n=0)
                cv2.imwrite(str(full_path), image)

            # | Save Checkpoint |
            if self.training_step_counter % self.save_every == 0:
                save_dir = Path(__file__).parents[1].joinpath("./logs/checkpoints")
                save_dir.mkdir(parents=True, exist_ok=True)
                full_path = save_dir.joinpath(
                    f"{self.training_step_counter}".zfill(8) + ".pt"
                )
                to_save = {
                    "G": self.generator.state_dict(),
                    "D": self.discriminator.state_dict(),
                }
                torch.save(to_save, full_path)
            self.training_step_counter += 1

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr)
        return g_opt, d_opt

    def _pre_training(self, batch):
        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)
        g_opt, _ = self.configure_optimizers()

        photo, _, _, _ = batch
        fake = self.generator(photo)

        vgg_photo = self.vgg(photo)
        vgg_fake = self.vgg(fake)

        con_loss = self.w_con * (F.l1_loss(vgg_photo, vgg_fake))

        g_opt.zero_grad()
        self.manual_backward(con_loss)
        g_opt.step()

        self.log("Pre Loss", con_loss, prog_bar=True)

    def tensor_to_image(self, *tensor: torch.Tensor, save_n: int = 0):
        concatenated = torch.concat([*tensor], dim=-1)
        if save_n > concatenated.size(0):
            raise ValueError("save_n must smaller than batch size.")
        if save_n != 0:
            concatenated = concatenated.split(save_n)
        grid = make_grid(concatenated, nrow=1, normalize=True, value_range=(-1, 1))
        grid = grid * 255
        grid = grid.permute(1, 2, 0)
        image = grid.detach().cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
