"""
File: main.py
Location: project/main.py

Description:
    Main entry point for training the GAN model for anime style transfer.
    It loads configuration from a YAML file, sets up logging and TensorBoard,
    builds the dataset and DataLoader, initializes the GAN model along with its
    optimizers, and runs the training loop.

Usage:
    python main.py --config configs/config.yaml
"""

import os
import yaml
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

# Import our modules
from models.gan import GANModel
from utils.dataset import build_dataloader
from utils.logger import setup_logger, setup_tensorboard


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Main training script for GAN anime style transfer")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()

    # Load configuration.
    config = load_config(args.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Setup logger and TensorBoard writer.
    logger = setup_logger(log_dir=config["logging"]["log_dir"])
    writer = setup_tensorboard(log_dir=config["logging"]["tensorboard_dir"])
    logger.info("Starting main training loop.")

    # Define image transformations for training.
    train_transforms = transforms.Compose([
        transforms.Resize(tuple(config["preprocessing"]["target_size"])),
        transforms.ToTensor()
    ])

    # Build DataLoader for processed images.
    dataloader = build_dataloader(config["dataset"]["processed_dir"],
                                  batch_size=config["training"]["batch_size"],
                                  shuffle=True,
                                  num_workers=4,
                                  transform=train_transforms)
    logger.info(f"Loaded dataset with {len(dataloader.dataset)} images from '{config['dataset']['processed_dir']}'.")

    # Initialize the GAN model.
    gan = GANModel().to(device)
    logger.info("Initialized GAN model.")

    # Setup optimizers.
    optimizer_G = optim.Adam(gan.generator.parameters(),
                             lr=config["training"]["learning_rate"]["generator"],
                             betas=(config["training"]["beta1"], config["training"]["beta2"]))
    optimizer_D = optim.Adam(gan.discriminator.parameters(),
                             lr=config["training"]["learning_rate"]["discriminator"],
                             betas=(config["training"]["beta1"], config["training"]["beta2"]))

    # Create checkpoint directory.
    checkpoint_dir = os.path.join("experiments", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    num_epochs = config["training"]["epochs"]
    log_interval = config["training"]["log_interval"]

    # Training loop.
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs} starting.")
        for batch_idx, batch in enumerate(dataloader):
            real_images = batch["image"].to(device)
            # Perform one training step.
            losses = gan.train_step(real_images, optimizer_G, optimizer_D, device)
            global_step += 1

            # Log training metrics at defined intervals.
            if global_step % log_interval == 0:
                writer.add_scalar("Loss/Generator", losses["loss_G"], global_step)
                writer.add_scalar("Loss/Discriminator", losses["loss_D"], global_step)
                logger.info(f"Step {global_step}: Loss_G = {losses['loss_G']:.4f}, Loss_D = {losses['loss_D']:.4f}")

        # Save a model checkpoint at the end of each epoch.
        checkpoint_path = os.path.join(checkpoint_dir, f"gan_epoch_{epoch}.pth")
        torch.save(gan.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    writer.close()
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()