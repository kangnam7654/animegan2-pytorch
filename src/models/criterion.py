"""
# Data
S_(data)(p) = {p_i | i = 1, 2, ..., N} is the real photo dataset
S_(data)(a) = {a_i | i = 1, 2, ..., M} is the anime dataset
S_(data)(x) = {x_i | i = 1, 2, ..., M} is the anime dataset with grayscale
S_(data)(e) = {e_i | i = 1, 2, ..., M} is the anime dataset with edge smooth
S_(data)(y) = {y_i | i = 1, 2, ..., M} is the anime dataset with edge smooth and grayscale


# Loss
L(G, D) = w_adv * L_adv(G, D) + w_con * L_con(G, D) + w_gra * L_gra(G, D) + w_col * L_col(G, D)
L(G) = w_adv * E_(p_i~S_(data)(p))[(G(p_i) - 1)**2] + w_con * L_con(G, D) + w_gra * L_gra(G, D) + w_col * L_col(G, D)
L(D) = w_adv * E_(a_i~S_(data)(a))[(D(a_i) - 1)**2] + E_(p_i~S_(data)(p))[D(G(p_i))**2]]
       + E_(x_i~S_(data)(x))[D(x_i)**2] + 0.1 * E_(y_i~S_(data)(y))[D(y_i)**2]

## Loss Details
L_adv(G, D): the adversarial loss.
             L_adv(G) = E_(p_i~S_(data)(p))[(G(p_i) - 1)**2]
             L_adv(D) = E_(p_i~S_(data)(a))[(D(a_i) - 1)**2]

L_con(G, D): the content loss which helps to make the generated image retain the content of the input photo.
             L_con(G, D) = E_(p_i~S_(data)(p))[||VGG_l(p_i) - VGG_l(G(p_i))||_1]

L_gra(G, D): represents the grayscale style loss which makes the generated images have the clear anime style
             on the textures and lines.
             L_gra(G, D) = E_(p_i~S_(data))(p), E_(x_i~S_(data))(x)[||Gram(VGG_l(G(p_i))) - Gram(VGG_l(x_i))||_1]

L_col(G, D): is used as the color reconstruction loss to make the generated images have the color of
             the original photos. In order to make the image color reconstruction better, we convert the image
             color in RGB format to the YUV format to build the color reconstruction loss
             L_col(G, D). In L_col(G, D), L1 loss is used for the Y channel and Huber Loss is
             used for the U and V channels.
             L_col(G, D) = E_(p_i~S_(data)(p))[||Y(G(p_i)) - Y(p_i)||_1]
                           + E_(p_i~S_(data)(p))[||U(G(p_i)) - U(p_i)||_Huber]
                           + E_(p_i~S_(data)(p))[||V(G(p_i)) - V(p_i)||_Huber]

In the paper, weights are set as:
w_adv = 300
w_con = 1.5
w_gra = 3
w_col = 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    def __init__(self):
        """Adversarial loss for GANs."""
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(prediction, target)


class ContentLoss(nn.Module):
    def __init__(self):
        """Content loss based on VGG features."""
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(prediction, target)
        return loss


class GrayscaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_ = self.gram_matrix(prediction)
        target_ = self.gram_matrix(target)
        loss = self.criterion(prediction_, target_)
        return loss

    def gram_matrix(self, feature):
        """
        Computes the Gram matrix of a given feature.
        :param feature: torch.Tensor, feature matrix of shape (B, C, H, W)
        :return: torch.Tensor, Gram matrix of shape (B, C, C)
        """
        (b, c, h, w) = feature.size()
        feature = feature.view(b, c, h * w)
        feature_t = feature.transpose(1, 2)
        gram = torch.bmm(feature, feature_t)  # Batch matrix multiplication
        return gram / (c * h * w)  # Normalize by dividing by the number of elements


class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, photo: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        yuv_photo = self.rgb_to_yuv(photo)
        yuv_fake = self.rgb_to_yuv(fake)
        # print(yuv_fake.shape, yuv_photo.shape)

        photo_y, photo_u, photo_v = torch.split(yuv_photo, 1, 1)
        fake_y, fake_u, fake_v = torch.split(yuv_fake, 1, 1)

        g_color_loss = (
            F.l1_loss(fake_y, photo_y)
            + F.huber_loss(fake_u, photo_u)
            + F.huber_loss(fake_v, photo_v)
        )
        return g_color_loss

    def rgb_to_yuv(self, image):
        """
        image: (B, 3, H, W)
        return: (B, 3, H, W)
        """
        kernel = (
            torch.tensor(
                [
                    [0.299, 0.587, 0.114],  # Y
                    [-0.14713, -0.28886, 0.436],  # U
                    [0.615, -0.51499, -0.10001],  # V
                ]
            )
            .float()
            .to(image.device)
            .type_as(image)
        )  # (3, 3)

        image = (image + 1.0) / 2.0  # normalize to [0, 1]

        # einsum: (B, C, H, W), (YUV, RGB) → (B, YUV, H, W)
        yuv = torch.einsum("bchw,dc->bdhw", image, kernel)
        return yuv
