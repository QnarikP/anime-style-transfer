"""
File: models/discriminator.py
Location: project/models/discriminator.py

Description:
    Defines a PatchGAN discriminator network that evaluates the realism of generated images.
    The discriminator processes the input image through several convolutional blocks to produce
    a feature map where each element corresponds to the "realness" score of a local patch.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    It distinguishes between real and generated images by processing patches of the input.
    """

    def __init__(self, input_nc=3):
        """
        Initialize the discriminator network.

        Args:
            input_nc (int): Number of channels in input images.
        """
        super(Discriminator, self).__init__()

        # Function to create a convolutional block for the discriminator.
        # The block consists of a convolution layer, optional instance normalization, and LeakyReLU activation.
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Build the discriminator model using sequential blocks.
        self.model = nn.Sequential(
            # First block: no normalization on the first layer.
            *discriminator_block(input_nc, 64, normalization=False),
            # Subsequent blocks.
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # Final layer: output one-channel prediction map.
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, input_nc, H, W).

        Returns:
            Tensor: Output feature map of shape (batch_size, 1, H_out, W_out) where each element is
                    a score indicating whether the corresponding patch in the input image is real or fake.
        """
        return self.model(x)


# Testing code to verify the discriminator architecture.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netD = Discriminator().to(device)
    print("Discriminator architecture:")
    print(netD)

    # Create a dummy input tensor with batch size 1 and 256x256 image.
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    dummy_output = netD(dummy_input)
    print("Output shape:", dummy_output.shape)
