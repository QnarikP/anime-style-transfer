"""
File: models/generator.py
Location: project/models/generator.py

Description:
    Defines the generator network for the GAN. This generator uses a ResNet-based architecture:
    - An initial convolution block (with reflection padding) to process the input.
    - Two downsampling layers to reduce spatial dimensions.
    - A series of residual blocks (default 9) to learn complex features.
    - Two upsampling layers (using transposed convolutions) to restore the image size.
    - A final convolution layer with Tanh activation to produce output images in [-1,1].
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block using two convolution layers with instance normalization and ReLU activation.
    The output of the block is added to its input (skip connection) to allow gradients to flow easily.
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            # First convolution: 3x3 kernel, stride=1, padding=1 for same size output
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            # Second convolution: another 3x3 layer to further process features
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        # Adding skip connection
        return x + self.block(x)


class Generator(nn.Module):
    """
    Generator network for the GAN.
    It transforms an input image to a stylized output image.
    """

    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        """
        Initialize the generator architecture.

        Args:
            input_nc (int): Number of channels in input images.
            output_nc (int): Number of channels in output images.
            n_residual_blocks (int): Number of residual blocks.
        """
        super(Generator, self).__init__()

        # Initial convolution block with reflection padding to avoid border artifacts.
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling layers: progressively reduce spatial size while increasing channel depth.
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Add residual blocks to learn complex features.
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling layers: restore the spatial size while decreasing the number of channels.
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer: reflection padding and convolution to produce final output,
        # followed by Tanh activation to scale the output image pixel values to [-1,1].
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
            nn.Tanh()
        ]

        # Combine all layers into a sequential model.
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Forward pass of the generator.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, input_nc, H, W).

        Returns:
            Tensor: Generated image tensor of shape (batch_size, output_nc, H, W).
        """
        return self.model(x)


# Testing code to verify the generator architecture.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    print("Generator architecture:")
    print(netG)

    # Create a dummy input tensor with batch size 1 and 256x256 image.
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    dummy_output = netG(dummy_input)
    print("Output shape:", dummy_output.shape)
