#!/usr/bin/env python3
"""
File: gan.py
Location: project/models/gan.py

Description:
    Combines the generator and discriminator classes into one GAN model wrapper.
    Provides functions for a training step and loss calculations for adversarial training.

    **LSGAN Update:**
    Instead of using Binary Cross-Entropy loss (BCEWithLogitsLoss), we now use Mean Squared Error (MSELoss)
    as used in LSGAN. In LSGAN, the discriminator's output is compared with target values of 1 for real images
    and 0 for fake images using an MSE loss. This tends to stabilize training in many cases.
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Import generator and discriminator from their respective files.
from models.generator import Generator
from models.discriminator import Discriminator

# LSGAN uses MSELoss for adversarial training.
adversarial_loss = nn.MSELoss()


class GANModel(nn.Module):
    """
    GANModel wraps the generator and discriminator, and provides a single training step.
    """
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        """
        Initialize the GAN model.

        Args:
            input_nc (int): Number of channels in the input images.
            output_nc (int): Number of channels in the output images.
            n_residual_blocks (int): Number of residual blocks in the generator.
        """
        super(GANModel, self).__init__()
        self.generator = Generator(input_nc, output_nc, n_residual_blocks)
        self.discriminator = Discriminator(input_nc)

    def forward(self, x):
        """
        Forward pass through the generator.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Generated fake image.
        """
        return self.generator(x)

    def train_step(self, real_images, optimizer_G, optimizer_D, device):
        """
        Performs one training step, including updating both the generator and discriminator.

        Args:
            real_images (Tensor): Batch of real images.
            optimizer_G (torch.optim.Optimizer): Optimizer for generator.
            optimizer_D (torch.optim.Optimizer): Optimizer for discriminator.
            device (torch.device): Device to run the computations on.

        Returns:
            dict: Dictionary containing losses for generator and discriminator.
        """
        batch_size = real_images.size(0)

        # Generate fake images from real images
        fake_images = self.generator(real_images)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Get discriminator outputs
        pred_real = self.discriminator(real_images)
        pred_fake = self.discriminator(fake_images.detach())

        # Create target tensors that match the discriminator output shape.
        valid = torch.ones_like(pred_real, device=device)
        fake = torch.zeros_like(pred_fake, device=device)

        # Compute losses using MSE (LSGAN loss)
        loss_real = adversarial_loss(pred_real, valid)
        loss_fake = adversarial_loss(pred_fake, fake)
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # Generator tries to fool the discriminator
        pred_fake = self.discriminator(fake_images)
        loss_G = adversarial_loss(pred_fake, valid)

        loss_G.backward()
        optimizer_G.step()

        return {"loss_G": loss_G.item(), "loss_D": loss_D.item()}


# Example test run for the GANModel training step.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = GANModel().to(device)
    print("GAN Model created.")

    # Setup dummy optimizers
    optimizer_G = torch.optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Create dummy input: batch of 4 images (3 channels, 256x256)
    dummy_real = torch.randn(4, 3, 256, 256, device=device)
    losses = gan.train_step(dummy_real, optimizer_G, optimizer_D, device)
    print("Training step losses:", losses)