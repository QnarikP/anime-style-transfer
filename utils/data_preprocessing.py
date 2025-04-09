"""
File: data_preprocessing.py
Location: project/utils/data_preprocessing.py

Description:
    Downloads a dataset from a specified URL (if not already downloaded),
    unzips it into a designated folder (if not already unzipped), copies image files
    from the unzipped folder into a "data/raw" folder, then preprocesses those images
    (by converting them to RGB and resizing them to a target size using Image.Resampling.LANCZOS)
    and saves them into a "data/processed" folder.

    Detailed progress is printed, including the number of files processed and remaining.

Usage:
    Run this file once to prepare your dataset. If the dataset zip exists and images are already unzipped,
    it will skip those steps.
"""

import os
import requests
import zipfile
from tqdm import tqdm
from PIL import Image
import shutil
import sys

dataset_url = "https://huggingface.co/datasets/BangumiBase/narutoshippuden/resolve/main/all.zip"
dataset_dir = "../data/"
zip_file_path = os.path.join(dataset_dir, "naruto_shippuden.zip")
raw_dir = "data/raw"
processed_dir = "data/processed"
target_size = (256, 256)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Created directory: {directory}")


def download_dataset(url, destination):
    if os.path.exists(destination):
        print(f"[INFO] Zip file already exists: {destination}")
        return

    print(f"[INFO] Downloading dataset from {url} to {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            bar.update(len(data))
            file.write(data)
    print(f"[INFO] Download completed: {destination}")


def unzip_dataset(zip_path, extract_to):
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"[INFO] Extraction folder '{extract_to}' already contains files. Skipping unzip.")
        return
    print(f"[INFO] Unzipping {zip_path} to {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("[INFO] Unzip completed.")
    except zipfile.BadZipFile as e:
        print(f"[ERROR] Bad zip file: {e}", file=sys.stderr)


def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize(target_size, resample=Image.Resampling.LANCZOS)
    return img


def preprocess_dataset(raw_folder, processed_folder, target_size=(256, 256)):
    create_dir(processed_folder)

    files = []
    for root, _, filenames in os.walk(raw_folder):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
                files.append(os.path.join(root, f))

    total_files = len(files)
    print(f"[INFO] Found {total_files} files in '{raw_folder}' to process.")

    for idx, file_path in enumerate(tqdm(files, desc="Processing images"), start=1):
        try:
            processed_img = preprocess_image(file_path, target_size=target_size)

            rel_path = os.path.relpath(file_path, raw_folder)
            out_path = os.path.join(processed_folder, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            processed_img.save(out_path)
        except Exception as e:
            print(f"[ERROR] Failed to process file {file_path}: {e}", file=sys.stderr)


def main():
    create_dir(dataset_dir)
    download_dataset(dataset_url, zip_file_path)
    unzip_dataset(zip_file_path, dataset_dir)
    create_dir(raw_dir)
    create_dir(processed_dir)

    if not os.listdir(raw_dir):
        print(f"[INFO] Copying images from {dataset_dir} to {raw_dir} ...")
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        count = 0
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(raw_dir, os.path.relpath(src_path, dataset_dir))
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    count += 1
        print(f"[INFO] Copied {count} image files to {raw_dir}")
    else:
        print(f"[INFO] Raw folder '{raw_dir}' already contains files. Skipping copy.")

    preprocess_dataset(raw_dir, processed_dir, target_size=target_size)
    print("[INFO] Data preprocessing completed.")


if __name__ == "__main__":
    main()
