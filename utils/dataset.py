"""
File: dataset.py
Location: project/utils/dataset.py

Description:
    Defines a custom PyTorch Dataset to load images from a given directory.
    Also includes a function to build a DataLoader for training and validation.

    The dataset expects images in either the raw or processed folders.
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageFolderDataset(Dataset):
    """
    Custom Dataset for loading images from a folder.
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        # List only image files based on extension
        self.image_files = [f for f in os.listdir(root_dir)
                            if os.path.isfile(os.path.join(root_dir, f))
                            and os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}]
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get the image and its filename at the specified index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            dict: Dictionary with keys 'image' (the transformed image tensor)
                  and 'filename' (the original filename).
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        # Open image using PIL
        image = Image.open(img_path).convert("RGB")
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        return {"image": image, "filename": img_name}


def build_dataloader(root_dir, batch_size=16, shuffle=True, num_workers=4, transform=None):
    """
    Build and return a DataLoader for the dataset.

    Args:
        root_dir (str): Directory with the images.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
        transform (callable, optional): Optional transform to be applied on each sample.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = ImageFolderDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


# Test run for the dataset module.
if __name__ == "__main__":
    # Define a simple transform: resize to 256x256 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Specify the directory (change path as needed)
    data_dir = "data/processed"
    # Build the DataLoader
    loader = build_dataloader(data_dir, batch_size=4, transform=transform)

    print(f"Total samples: {len(loader.dataset)}")
    # Iterate through one batch
    for batch in loader:
        print("Batch filenames:", batch["filename"])
        print("Batch image tensor shape:", batch["image"].shape)
        break
