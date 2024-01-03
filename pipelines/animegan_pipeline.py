from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnimeganPipeline(pl.LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        vgg: nn.Module,
        g_lr=2e-4,
        d_lr=8e-5,
        w_adv=300,
        w_con=1.5,
        w_gray=3,
        w_col=10,
    ):
        super().__init__()
        # Model
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg

        # optimizers
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.g_opt, self.d_opt = self.configure_optimizers()
        self.automatic_optimization = False

        # weights for loss
        self.w_adv = w_adv
        self.w_con = w_con
        self.w_gray = w_gray
        self.w_col = w_col

    def training_step(self, batch, batch_idx):
        # | Pre-training |
        if self.current_epoch == 0:
            self._pre_training(batch)

        else:
            # After Pre-training
            photo, anime, gray, smooth = batch  # p, a, x, y

            # | Discriminator |
            self.generator.requires_grad_(False)
            self.discriminator.requires_grad_(True)

            fake = self.generator(photo)  # G(p)

            real_out = self.discriminator(anime)  # D(a)
            fake_out = self.discriminator(fake)  # D(G(p))
            gray_out = self.discriminator(gray)  # D(x)
            smooth_out = self.discriminator(smooth)  # D(y)

            # | Discirminator Loss |
            # E[(D(a) - 1)^2]
            d_real_loss = torch.square(real_out - torch.ones_like(real_out)).mean()

            # E[(D(G(p)))^2]
            d_fake_loss = torch.square(fake_out).mean()

            # E[(D(x))^2]
            d_gray_loss = torch.square(gray_out).mean()

            # E[(D(y))^2]
            d_smooth_loss = torch.square(smooth_out).mean()

            d_loss = self.w_adv * (
                d_real_loss + d_fake_loss + d_gray_loss + 0.1 * d_smooth_loss
            )

            # | D Backward |
            self.d_opt.zero_grad()
            self.manual_backward(d_loss)
            self.d_opt.step()

            # ------------------------------------------------------------------------

            # | Generator |
            self.generator.requires_grad_(True)
            self.discriminator.requires_grad_(False)

            fake = self.generator(photo)  # G(p)
            fake_out = self.discriminator(fake)  # D(G(p))

            # | RGB -> YUV |
            yuv_photo = self.rgb_to_yuv(photo)
            yuv_fake = self.rgb_to_yuv(fake)

            # | For perceptual Loss |
            vgg_photo = self.vgg(photo)  # VGG(p)
            vgg_fake = self.vgg(fake)  # VGG(G(p))
            vgg_gray = self.vgg(gray)  # VGG(G(x))

            # | Gram |
            gram_fake = self.gram(vgg_fake)  # Gram(VGG(G(p)))
            gram_gray = self.gram(vgg_gray)  # Gram(VGG(x))

            # | Generator Loss |
            # E[(G(p) - 1)^2]
            g_adv_loss = torch.square(fake - torch.ones_like(fake)).mean()

            # E[|| VGG(p) - VGG(G(p))||_1]
            g_con_loss = F.l1_loss(vgg_photo, vgg_fake)

            # E[||Gram(VGG(G(p))) - Gram(VGG(x))||_1]
            g_gray_loss = F.l1_loss(gram_fake, gram_gray)

            # E[||Y(G(p)) - Y(p)||_1 + ||U(G(p)) - U(p)||_Huber + ||V(G(p)) - V(p))||_Huber]
            g_color_loss = (
                F.l1_loss(yuv_fake[:, 0, :, :], yuv_photo[:, 0, :, :])
                + F.huber_loss(yuv_fake[:, 1, :, :], yuv_photo[:, 1, :, :])
                + F.huber_loss(yuv_fake[:, 2, :, :], yuv_photo[:, 2, :, :])
            )
            g_loss = (
                self.w_adv * g_adv_loss
                + self.w_con * g_con_loss
                + self.w_gray * g_gray_loss
                + self.w_col * g_color_loss
            )

            # | G Backward |
            self.g_opt.zero_grad()
            self.manual_backward(g_loss)
            self.g_opt.step()

            # | Logging |
            to_log = {"G_loss": g_loss, "D_loss": d_loss}
            self.log_dict(to_log, prog_bar=True)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr)
        return g_opt, d_opt

    def gram(self, input):
        """
        Calculate Gram Matrix

        https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
        """
        b, c, w, h = input.size()

        x = input.view(b * c, w * h)

        G = torch.mm(x, x.T)

        # normalize by total elements
        return G.div(b * c * w * h)

    def rgb_to_yuv(self, image):
        """
        https://en.wikipedia.org/wiki/YUV

        output: Image of shape (H, W, C) (channel last)
        """
        rgb_to_yuv_kernel = torch.tensor(
            [
                [0.299, -0.14714119, 0.61497538],
                [0.587, -0.28886916, -0.51496512],
                [0.114, 0.43601035, -0.10001026],
            ]
        ).float()

        # -1 1 -> 0 1
        image = (image + 1.0) / 2.0

        yuv_img = torch.tensordot(
            image, rgb_to_yuv_kernel, dims=([image.ndim - 3], [0])
        )

        return yuv_img

    def _pre_training(self, batch):
        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)

        photo, _, _, _ = batch
        fake = self.generator(photo)

        vgg_photo = self.vgg(photo)
        vgg_fake = self.vgg(fake)

        con_loss = self.w_con * (F.l1_loss(vgg_photo, vgg_fake))

        self.g_opt.zero_grad()
        self.manual_backward(con_loss)
        self.g_opt.step()

        self.log("Pre Loss", con_loss, prog_bar=True)
