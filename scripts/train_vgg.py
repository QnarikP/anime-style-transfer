#!/usr/bin/env python3
"""
File: train_vgg.py
Location: project/scripts/train_vgg.py

Description:
    This script trains the VGG-based GAN model for anime style transfer.
    It loads configuration from a YAML file (configs/config_vgg.yaml), sets up logging,
    builds the DataLoader for processed images, initializes the VGG-based GAN model,
    and runs the training loop while logging progress, saving generated images, and checkpoints.
    It can also resume training if a checkpoint is provided.

Usage:
    To start fresh:
        python train_vgg.py --config configs/config_vgg.yaml
    To resume training:
        python train_vgg.py --config configs/config_vgg.yaml --checkpoint experiments/checkpoints/vgggan_epoch_20.pth

Author: [Your Name]
Date: [Date]
"""

import os
import yaml
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
import sys
import re

# Append project root to sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vgg_gan import VGGGAN
from utils.dataset import build_dataloader
from utils.logger import setup_logger, setup_tensorboard

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_epoch(checkpoint_path):
    """
    Extract epoch number from checkpoint filename.
    Expected format: "vgggan_epoch_{epoch}.pth"
    """
    match = re.search(r"vgggan_epoch_(\d+)\.pth", os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    else:
        return 0

def save_generated_images(fake_images, epoch, step, save_dir):
    """
    Save generated images to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    fake_images = (fake_images + 1) / 2  # Normalize to [0,1]
    image_path = os.path.join(save_dir, f"vgggan_generated_epoch{epoch}_step{step}.png")
    vutils.save_image(fake_images, image_path, normalize=True, nrow=8)
    print(f"[INFO] Saved generated images to {image_path}")

def main():
    parser = argparse.ArgumentParser(description="Train VGG-based GAN model for anime style transfer")
    parser.add_argument("--config", type=str, required=True, help="Path to config_vgg.yaml file")
    parser.add_argument("--checkpoint", type=str, help="Optional: path to model checkpoint to resume training")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    logger = setup_logger(log_dir=config["logging"]["log_dir"], log_file="train_vgg.log")
    writer = setup_tensorboard(log_dir=config["logging"]["tensorboard_dir"])
    logger.info("Starting VGG-based GAN training.")

    train_transforms = transforms.Compose([
        transforms.Resize(tuple(config["preprocessing"]["target_size"])),
        transforms.ToTensor()
    ])

    dataloader = build_dataloader(config["dataset"]["processed_dir"],
                                  batch_size=config["training"]["batch_size"],
                                  shuffle=True,
                                  num_workers=4,
                                  transform=train_transforms)
    logger.info(f"Loaded dataset with {len(dataloader.dataset)} images.")

    # Initialize the VGG-based GAN model.
    gan = VGGGAN(output_nc=config["model"]["output_nc"], freeze_encoder=config["model"].get("freeze_encoder", True)).to(device)
    logger.info("Initialized VGG-based GAN model.")

    # Optionally resume training if a checkpoint is provided.
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        gan.load_state_dict(checkpoint)
        start_epoch = extract_epoch(args.checkpoint)
        logger.info(f"Resuming training from epoch {start_epoch + 1}")

    optimizer_G = optim.Adam(gan.generator.parameters(),
                               lr=config["training"]["learning_rate"]["generator"],
                               betas=(config["training"]["beta1"], config["training"]["beta2"]))
    optimizer_D = optim.Adam(gan.discriminator.parameters(),
                               lr=config["training"]["learning_rate"]["discriminator"],
                               betas=(config["training"]["beta1"], config["training"]["beta2"]))

    scheduler_G = StepLR(optimizer_G,
                         step_size=config["training"].get("lr_decay_step", 10),
                         gamma=config["training"].get("lr_decay_factor", 0.5))
    scheduler_D = StepLR(optimizer_D,
                         step_size=config["training"].get("lr_decay_step", 10),
                         gamma=config["training"].get("lr_decay_factor", 0.5))

    global_step = 0
    total_epochs = config["training"]["epochs"]

    for epoch in range(start_epoch + 1, total_epochs + 1):
        logger.info(f"Epoch {epoch}/{total_epochs} starting.")
        for batch_idx, batch in enumerate(dataloader):
            real_images = batch["image"].to(device)
            losses = gan.train_step(real_images, optimizer_G, optimizer_D, device)
            global_step += 1

            if global_step % config["training"]["log_interval"] == 0:
                writer.add_scalar("Loss/Generator", losses["loss_G"], global_step)
                writer.add_scalar("Loss/Discriminator", losses["loss_D"], global_step)
                logger.info(f"Step {global_step}: Loss_G = {losses['loss_G']:.4f}, Loss_D = {losses['loss_D']:.4f}")

                with torch.no_grad():
                    sample_real = real_images[:8]
                    fake_images = gan.generator(sample_real)
                    save_generated_images(fake_images, epoch, global_step, config["logging"]["generated_images_dir"])

        scheduler_G.step()
        scheduler_D.step()
        logger.info(f"Updated learning rates: Generator LR = {optimizer_G.param_groups[0]['lr']}, "
                    f"Discriminator LR = {optimizer_D.param_groups[0]['lr']}")

        checkpoint_dir = os.path.join("experiments", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"vgggan_epoch_{epoch}.pth")
        torch.save(gan.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    writer.close()
    logger.info("VGG-based GAN training completed.")

if __name__ == "__main__":
    main()
