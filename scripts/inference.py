"""
File: inference.py
Location: project/scripts/inference.py

Description:
    This script performs inference using the trained GAN model for anime style transfer.
    It loads a single input image, applies the required transformations, loads the trained
    model checkpoint, runs the generator to convert the image into an anime-styled output,
    and saves the generated image.

Usage:
    python inference.py --config configs/config.yaml --checkpoint experiments/checkpoints/gan_epoch_69.pth --input_image experiments/original_images/naruto.jpg --output_image experiments/eval_images/naruto.jpg
"""

import os
import sys
import argparse
import yaml
import torch
import torchvision.transforms as transforms
from PIL import Image

# Ensure that the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the GAN model (we only use the generator for inference)
from models.gan import GANModel
from utils.logger import setup_logger


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path, transform):
    """
    Loads an image from disk and applies the given transform.

    Args:
        image_path (str): Path to the input image.
        transform (callable): Transformation to apply to the image.

    Returns:
        Tensor: Transformed image tensor.
    """
    img = Image.open(image_path).convert("RGB")
    return transform(img)


def save_image(tensor, output_path):
    """
    Saves a tensor as an image to disk.

    Args:
        tensor (Tensor): Image tensor with values in [-1,1] or [0,1].
        output_path (str): Destination path to save the image.
    """
    # If tensor values are in [-1, 1], scale them to [0, 1]
    tensor = (tensor + 1) / 2 if tensor.min() < 0 else tensor
    # Convert tensor to PIL Image
    image = transforms.ToPILImage()(tensor.cpu())
    image.save(output_path)
    print(f"[INFO] Saved generated image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference script for GAN anime style transfer")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save the output (generated) image")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Setup logger (optional)
    logger = setup_logger(log_dir=config["logging"]["log_dir"], log_file="inference.log")
    logger.info("Starting inference process.")

    # Define transforms: same resize as during training, and convert image to tensor.
    transform = transforms.Compose([
        transforms.Resize(tuple(config["preprocessing"]["target_size"])),
        transforms.ToTensor()
    ])

    # Load input image
    input_tensor = load_image(args.input_image, transform).unsqueeze(0).to(device)
    logger.info(f"Loaded input image: {args.input_image}")

    # Initialize GAN model and load checkpoint (we only need the generator part for inference)
    gan = GANModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    gan.load_state_dict(checkpoint)
    gan.eval()
    logger.info(f"Loaded model checkpoint from {args.checkpoint}")

    # Run inference: generate anime-styled image using the generator
    with torch.no_grad():
        generated = gan.generator(input_tensor)
    logger.info("Inference completed.")

    # Save generated image
    save_image(generated.squeeze(0), args.output_image)


if __name__ == "__main__":
    main()
