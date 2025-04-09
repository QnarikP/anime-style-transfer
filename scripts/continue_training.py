#!/usr/bin/env python3
"""
File: continue_training.py
Location: project/scripts/continue_training.py

Description:
    This script resumes training of the GAN model for anime style transfer from a specific checkpoint.
    It loads configuration from a YAML file (configs/config.yaml), loads the model weights from a provided
    checkpoint file, and then continues training from the next epoch. A learning rate scheduler (StepLR)
    is used to decay the learning rate after every epoch. Additionally, every specified log interval,
    a sample batch of generated images is saved to disk.

Usage:
    python scripts/continue_training.py --config configs/config.yaml --checkpoint experiments/checkpoints/gan_epoch_65.pth
"""

import os
import yaml
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import torchvision.utils as vutils
import sys
import re  # for extracting epoch number from checkpoint filename

# Append the project root to sys.path for module imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the GAN model and utilities.
from models.gan import GANModel
from utils.dataset import build_dataloader
from utils.logger import setup_logger, setup_tensorboard

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_epoch(checkpoint_path):
    """
    Extract the epoch number from the checkpoint filename.
    Expected filename format: "gan_epoch_{epoch}.pth"

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        int: Epoch number extracted from the filename.
    """
    match = re.search(r"gan_epoch_(\d+)\.pth", os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    else:
        return 0

def save_generated_images(fake_images, epoch, step, save_dir):
    """
    Save generated images to disk.

    Args:
        fake_images (Tensor): Batch of generated images (assumed in range [-1, 1]).
        epoch (int): Current epoch number.
        step (int): Global training step.
        save_dir (str): Directory where the images will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Normalize images to [0, 1]
    fake_images = (fake_images + 1) / 2
    image_path = os.path.join(save_dir, f"generated_epoch{epoch}_step{step}.png")
    vutils.save_image(fake_images, image_path, normalize=True, nrow=8)
    print(f"[INFO] Saved generated images to {image_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Continue training GAN model for anime style transfer from a checkpoint with LR scheduling and image saving")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.pth file) to resume from")
    args = parser.parse_args()

    # Load configuration and set device.
    config = load_config(args.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Setup logger and TensorBoard writer.
    logger = setup_logger(log_dir=config["logging"]["log_dir"], log_file="continue_training.log")
    writer = setup_tensorboard(log_dir=config["logging"]["tensorboard_dir"])
    logger.info("Resuming training process.")

    # Define transforms for training images.
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
    logger.info(f"Loaded dataset with {len(dataloader.dataset)} images.")

    # Initialize the GAN model.
    gan = GANModel().to(device)
    logger.info("Initialized GAN model.")

    # Load checkpoint weights.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    gan.load_state_dict(checkpoint)
    logger.info(f"Loaded model checkpoint from {args.checkpoint}")

    # Extract starting epoch from the checkpoint filename.
    start_epoch = extract_epoch(args.checkpoint)
    logger.info(f"Resuming training from epoch {start_epoch + 1}")

    # Setup optimizers.
    optimizer_G = optim.Adam(gan.generator.parameters(),
                             lr=config["training"]["learning_rate"]["generator"],
                             betas=(config["training"]["beta1"], config["training"]["beta2"]))
    optimizer_D = optim.Adam(gan.discriminator.parameters(),
                             lr=config["training"]["learning_rate"]["discriminator"],
                             betas=(config["training"]["beta1"], config["training"]["beta2"]))

    # Setup learning rate schedulers (StepLR).
    scheduler_G = StepLR(optimizer_G,
                         step_size=config["training"].get("lr_decay_step", 10),
                         gamma=config["training"].get("lr_decay_factor", 0.5))
    scheduler_D = StepLR(optimizer_D,
                         step_size=config["training"].get("lr_decay_step", 10),
                         gamma=config["training"].get("lr_decay_factor", 0.5))

    global_step = 0
    total_epochs = config["training"]["epochs"]

    # Continue training from start_epoch + 1 until total_epochs.
    for epoch in range(start_epoch + 1, total_epochs + 1):
        logger.info(f"Epoch {epoch}/{total_epochs} starting.")
        for batch_idx, batch in enumerate(dataloader):
            real_images = batch["image"].to(device)
            losses = gan.train_step(real_images, optimizer_G, optimizer_D, device)
            global_step += 1

            # Log scalar losses to TensorBoard at specified intervals.
            if global_step % config["training"]["log_interval"] == 0:
                writer.add_scalar("Loss/Generator", losses["loss_G"], global_step)
                writer.add_scalar("Loss/Discriminator", losses["loss_D"], global_step)
                logger.info(f"Step {global_step}: Loss_G = {losses['loss_G']:.4f}, Loss_D = {losses['loss_D']:.4f}")

                # Save a sample of generated images.
                with torch.no_grad():
                    # Use a small batch (e.g., first 8 images of the current batch)
                    sample_real = real_images[:8]
                    fake_images = gan.generator(sample_real)
                    save_generated_images(fake_images, epoch, global_step, config["logging"]["generated_images_dir"])

        # Step the learning rate schedulers after each epoch.
        scheduler_G.step()
        scheduler_D.step()
        logger.info(f"Updated learning rates: Generator LR = {optimizer_G.param_groups[0]['lr']}, "
                    f"Discriminator LR = {optimizer_D.param_groups[0]['lr']}")

        # Save a checkpoint after each epoch.
        checkpoint_dir = os.path.join("experiments", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"gan_epoch_{epoch}.pth")
        torch.save(gan.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    writer.close()
    logger.info("Training resumed and completed.")

if __name__ == "__main__":
    main()