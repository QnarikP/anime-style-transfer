#!/usr/bin/env python3
"""
File: vgg_gan.py
Location: project/models/vgg_gan.py

Description:
    Defines a VGG-based GAN model for anime style transfer.
    The generator uses a pre-trained VGG19 network as an encoder (which can be fine-tuned)
    and a custom decoder to produce the stylized output. The discriminator is the same as
    the PatchGAN discriminator used in the previous models.

    A train_step method is added to facilitate training using LSGAN (MSELoss).

Author: [Your Name]
Date: [Date]
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Define the adversarial loss for LSGAN.
adversarial_loss = nn.MSELoss()


class VGGGenerator(nn.Module):
    """
    Generator that uses a pre-trained VGG19 as encoder and a custom decoder.
    """

    def __init__(self, output_nc=3, freeze_encoder=True):
        """
        Initialize the VGG-based generator.

        Args:
            output_nc (int): Number of channels in the output image.
            freeze_encoder (bool): Whether to freeze the encoder layers.
        """
        super(VGGGenerator, self).__init__()
        # Load a pre-trained VGG19 network.
        # We use layers up to relu4_1. Note: this produces features with 512 channels.
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Define a decoder that upsamples the encoded features back to image size.
        # Updated the first convolution to accept 512 channels (instead of 256) since encoder outputs 512.
        self.decoder = nn.Sequential(
            # Now input is 512 channels.
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, output_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass through the VGG generator.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, 3, H, W).

        Returns:
            Tensor: Generated image tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VGGGAN(nn.Module):
    """
    VGGGAN wraps the VGG-based generator and a PatchGAN discriminator.
    """

    def __init__(self, output_nc=3, freeze_encoder=True, input_nc=3):
        super(VGGGAN, self).__init__()
        from models.discriminator import Discriminator  # reuse existing discriminator
        self.generator = VGGGenerator(output_nc=output_nc, freeze_encoder=freeze_encoder)
        self.discriminator = Discriminator(input_nc=input_nc)

    def forward(self, x):
        """Forward pass using the generator."""
        return self.generator(x)

    def train_step(self, real_images, optimizer_G, optimizer_D, device):
        """
        Perform one training step for VGGGAN using LSGAN loss.

        Args:
            real_images (Tensor): Batch of real images.
            optimizer_G (Optimizer): Optimizer for the generator.
            optimizer_D (Optimizer): Optimizer for the discriminator.
            device (torch.device): Device for computations.

        Returns:
            dict: Losses for generator and discriminator.
        """
        batch_size = real_images.size(0)

        # Generate fake images
        fake_images = self.generator(real_images)

        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = self.discriminator(real_images)
        pred_fake = self.discriminator(fake_images.detach())

        valid = torch.ones_like(pred_real, device=device)
        fake = torch.zeros_like(pred_fake, device=device)

        loss_real = adversarial_loss(pred_real, valid)
        loss_fake = adversarial_loss(pred_fake, fake)
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        pred_fake = self.discriminator(fake_images)
        loss_G = adversarial_loss(pred_fake, valid)

        loss_G.backward()
        optimizer_G.step()

        return {"loss_G": loss_G.item(), "loss_D": loss_D.item()}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGGAN().to(device)
    print("VGGGAN Model:")
    print(model)

    # Test with dummy input.
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    output = model(dummy_input)
    print("Output shape:", output.shape)
