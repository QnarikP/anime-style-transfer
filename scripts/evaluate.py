"""
File: evaluate.py
Location: project/scripts/evaluate.py

Description:
    This script evaluates the pre-trained GAN model on a test dataset.
    It loads the configuration (configs/config.yaml), initializes the model and loads
    the specified checkpoint, then runs inference on images from the processed folder.
    Generated images are saved to the output directory.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint experiments/checkpoints/gan_epoch_28.pth --output_dir experiments/generated_images
"""

import os
import yaml
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import sys
import os

# Get the absolute path of the parent directory
module_path = os.path.abspath(r"C:\Users\User\Projects\NPUA\anime-style-transfer")

# Append the module path to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

from models.gan import GANModel
from utils.dataset import ImageFolderDataset
from utils.logger import setup_logger


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_test_loader(root_dir, batch_size=1, transform=None):
    """
    Build a DataLoader for the test dataset.

    Args:
        root_dir (str): Directory containing test images.
        batch_size (int): Batch size.
        transform (callable): Transformations to apply to each image.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = ImageFolderDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save generated images")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logger = setup_logger(log_dir=config["logging"]["log_dir"], log_file="evaluate.log")
    logger.info("Starting evaluation.")

    # Define evaluation transforms (resize and convert to tensor)
    eval_transforms = transforms.Compose([
        transforms.Resize(tuple(config["preprocessing"]["target_size"])),
        transforms.ToTensor()
    ])

    # Build test DataLoader from processed images
    test_loader = build_test_loader(config["dataset"]["processed_dir"],
                                    batch_size=1,
                                    transform=eval_transforms)
    logger.info(f"Loaded test dataset with {len(test_loader.dataset)} images.")

    # Initialize GAN model and load checkpoint
    gan = GANModel().to(device)
    gan.load_state_dict(torch.load(args.checkpoint, map_location=device))
    gan.eval()
    logger.info(f"Loaded model checkpoint from {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            real_images = batch["image"].to(device)
            filenames = batch["filename"]
            # Generate fake images using the generator
            fake_images = gan.generator(real_images)
            for i in range(fake_images.size(0)):
                # Denormalize if generator outputs in [-1, 1]
                img_tensor = fake_images[i].cpu()
                img_tensor = (img_tensor + 1) / 2  # Scale to [0,1]
                img = transforms.ToPILImage()(img_tensor)
                out_path = os.path.join(args.output_dir, f"generated_{filenames[i]}")
                img.save(out_path)
                logger.info(f"Saved generated image: {out_path}")

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
