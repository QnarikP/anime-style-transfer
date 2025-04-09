"""
File: transforms.py
Location: project/utils/transforms.py

Description:
    Provides custom data augmentation functions for image transformations.
    Includes functions for random horizontal flipping, random rotation, and random cropping.
    These functions can be used individually or composed into a pipeline.
"""

import random
from PIL import Image


def random_horizontal_flip(image, p=0.5):
    """
    Randomly flip the given PIL.Image horizontally with probability p.

    Args:
        image (PIL.Image.Image): Input image.
        p (float): Probability of flipping the image.

    Returns:
        PIL.Image.Image: Flipped image if condition met; otherwise, the original image.
    """
    if random.random() < p:
        # Flip image horizontally
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def random_rotate(image, max_angle=30):
    """
    Randomly rotate the image within a range of [-max_angle, max_angle] degrees.

    Args:
        image (PIL.Image.Image): Input image.
        max_angle (int): Maximum rotation angle in degrees.

    Returns:
        PIL.Image.Image: Rotated image.
    """
    # Random angle between -max_angle and +max_angle
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, resample=Image.BILINEAR)


def random_crop(image, crop_size=(224, 224)):
    """
    Randomly crop the image to the given crop_size.

    Args:
        image (PIL.Image.Image): Input image.
        crop_size (tuple): Desired output size (width, height).

    Returns:
        PIL.Image.Image: Cropped image.
    """
    width, height = image.size
    crop_width, crop_height = crop_size

    if width < crop_width or height < crop_height:
        # If the image is smaller than the crop size, resize it first.
        image = image.resize((max(width, crop_width), max(height, crop_height)), Image.BILINEAR)
        width, height = image.size

    # Randomly select the top-left corner of the crop
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height

    return image.crop((left, top, right, bottom))


def custom_transform(image):
    """
    Apply a series of custom augmentations to the image.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Augmented image.
    """
    # Apply random horizontal flip
    image = random_horizontal_flip(image, p=0.5)
    # Apply random rotation
    image = random_rotate(image, max_angle=20)
    # Apply random crop (if desired; adjust crop size as needed)
    image = random_crop(image, crop_size=(256, 256))
    return image


# Example test run for custom transforms.
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    # Open a sample image (update path to an existing image file)
    image_path = "data/raw/sample.jpg"
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open sample image: {e}")
        exit(1)

    # Apply custom transformation
    aug_img = custom_transform(img)

    # Visualize the original and augmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Augmented Image")
    plt.imshow(aug_img)
    plt.axis("off")
    plt.show()