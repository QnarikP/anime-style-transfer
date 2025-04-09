"""
File: train.py
Location: project/scripts/train.py

Description:
    This script trains the GAN model for anime style transfer. It loads configuration from
    a YAML file (configs/config.yaml), sets up the logger and TensorBoard writer, builds the
    DataLoader for processed images, initializes the GAN model (generator and discriminator),
    and runs the training loop while logging progress and saving checkpoints.

Usage:
    python train.py --config ../configs/config.yaml
"""

import os
import yaml
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the combined GAN model from models/gan.py
from models.gan import GANModel
# Import dataset builder from utils/dataset.py
from utils.dataset import build_dataloader
# Import logger and TensorBoard setup from utils/logger.py
from utils.logger import setup_logger, setup_tensorboard

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_generated_images(fake_images, epoch, step, save_dir):
    """Save generated images to disk."""
    os.makedirs(save_dir, exist_ok=True)
    # Normalize the images back to [0, 1] for saving
    fake_images = (fake_images + 1) / 2  # Assuming images are in the range [-1, 1]
    # Save the generated images
    image_path = os.path.join(save_dir, f"generated_{epoch}_{step}.png")
    vutils.save_image(fake_images, image_path, normalize=True, nrow=8)
    print(f"Saved generated images to {image_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GAN model for anime style transfer")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file")
    args = parser.parse_args()

    # Load configuration and set device
    config = load_config(args.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))
    # Setup logger and TensorBoard writer
    logger = setup_logger(log_dir=config["logging"]["log_dir"])
    writer = setup_tensorboard(log_dir=config["logging"]["tensorboard_dir"])
    logger.info("Starting training process.")

    # Define transforms for training images
    train_transforms = transforms.Compose([
        transforms.Resize(tuple(config["preprocessing"]["target_size"])),
        transforms.ToTensor()
    ])

    # Build DataLoader for images from the processed folder
    dataloader = build_dataloader(config["dataset"]["processed_dir"],
                                  batch_size=config["training"]["batch_size"],
                                  shuffle=True,
                                  num_workers=4,
                                  transform=train_transforms)
    logger.info(f"Loaded dataset with {len(dataloader.dataset)} images.")

    # Initialize the GAN model and optimizers
    gan = GANModel().to(device)
    logger.info("Initialized GAN model.")

    optimizer_G = optim.Adam(gan.generator.parameters(),
                             lr=config["training"]["learning_rate"]["generator"],
                             betas=(config["training"]["beta1"], config["training"]["beta2"]))
    optimizer_D = optim.Adam(gan.discriminator.parameters(),
                             lr=config["training"]["learning_rate"]["discriminator"],
                             betas=(config["training"]["beta1"], config["training"]["beta2"]))

    global_step = 0
    for epoch in range(1, config["training"]["epochs"] + 1):
        logger.info(f"Epoch {epoch}/{config['training']['epochs']} starting.")
        for batch_idx, batch in enumerate(dataloader):
            real_images = batch["image"].to(device)
            losses = gan.train_step(real_images, optimizer_G, optimizer_D, device)
            global_step += 1

            # Log scalar losses to TensorBoard at specified intervals
            if global_step % config["training"]["log_interval"] == 0:
                writer.add_scalar("Loss/Generator", losses["loss_G"], global_step)
                writer.add_scalar("Loss/Discriminator", losses["loss_D"], global_step)
                logger.info(f"Step {global_step}: Loss_G = {losses['loss_G']:.4f}, Loss_D = {losses['loss_D']:.4f}")

                # Generate fake images and save them
                # Generate fake images and save them
                # Generate fake images and save them
                with torch.no_grad():
                    # Use a batch of real images instead of random noise
                    # Take a small batch (e.g., 8 images) from your real images
                    sample_real = real_images[:8].to(device)  # Take the first 8 images from the batch

                    # Generate fake images by applying style transfer to these real images
                    fake_images = gan.generator(sample_real)

                    # Save the generated images
                    save_generated_images(fake_images, epoch, global_step, config["logging"]["generated_images_dir"])

        # Save a checkpoint after each epoch
        checkpoint_dir = os.path.join("experiments", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"gan_epoch_{epoch}.pth")
        torch.save(gan.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    writer.close()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
