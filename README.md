# Anime Style Transfer GAN

This project implements a Generative Adversarial Network (GAN) for transferring the style of anime to real images. The model is designed to work with an anime dataset (e.g., [BangumiBase/narutoshippuden](https://huggingface.co/datasets/BangumiBase/narutoshippuden)) and applies state-of-the-art training practices including residual network generators, PatchGAN discriminators, TensorBoard logging, and configurable training via YAML.

## Project Structure

```
project/
├── configs/
│   └── config.yaml         # Hyperparameters, dataset paths, and training options.
├── data/
│   ├── raw/                # Directory for original images.
│   └── processed/          # Directory for preprocessed images.
├── experiments/
│   └── checkpoints/        # Saved model checkpoints.
├── models/
│   ├── gan.py              # GAN wrapper combining generator and discriminator; defines training step.
│   ├── generator.py        # GAN generator (ResNet-based architecture).
│   └── discriminator.py    # PatchGAN discriminator.
├── scripts/
│   ├── train.py            # Command-line script to run training.
│   └── evaluate.py         # Command-line script to evaluate the model.
├── utils/
│   ├── data_preprocessing.py  # Downloads, unzips, and preprocesses dataset images.
│   ├── dataset.py          # PyTorch Dataset and DataLoader for images.
│   ├── logger.py           # Logging and TensorBoard setup.
│   └── transforms.py       # Custom data augmentation functions.
└── main.py                 # Main entry point for training.
```

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd project
   ```

2. **Install Dependencies:**

   Make sure you have Python 3 installed. Then install the required Python packages (e.g., using pip):

   ```bash
   pip install torch torchvision tqdm pyyaml pillow tensorboard
   ```

3. **Download and Preprocess the Dataset:**

   The dataset download and preprocessing script is located in `utils/data_preprocessing.py`. You can run it to:
   - Download the dataset from Hugging Face.
   - Unzip the dataset.
   - Copy images to `data/raw` and process them into `data/processed`.

   Example:
   ```bash
   python utils/data_preprocessing.py
   ```

4. **Configure the Project:**

   Edit the file `configs/config.yaml` to update hyperparameters, dataset paths, and training options as needed.

5. **Train the Model:**

   You can run training either using the training script in the `scripts` folder or by running the main file:

   ```bash
   # Using the training script
   python scripts/train.py --config configs/config.yaml

   # Or using the main training loop
   python main.py --config configs/config.yaml
   ```

6. **Evaluate the Model:**

   To evaluate the model and generate stylized images, use the evaluation script:

   ```bash
   python scripts/evaluate.py --config configs/config.yaml --checkpoint <path_to_checkpoint> --output_dir outputs
   ```

## Usage Guidelines

- **Logging:**  
  Training progress and evaluation metrics are logged to both the console and a log file located in `experiments/logs`. TensorBoard logs are stored in the `tensorboard` folder. To view them, run:
  ```bash
  tensorboard --logdir tensorboard
  ```

- **Checkpoints:**  
  Model checkpoints are saved in `experiments/checkpoints` after each epoch. Use these checkpoints for evaluation or to resume training.

- **Customization:**  
  - The generator and discriminator architectures are defined in `models/generator.py` and `models/discriminator.py`.
  - Custom data augmentation functions are available in `utils/transforms.py`.
  - All hyperparameters and file paths are stored in `configs/config.yaml` for easy modifications.

## Additional Notes

- Ensure your GPU (e.g., Nvidia RTX 4060) is correctly configured for CUDA if available.
- Adjust batch sizes and image resolutions according to your available GPU memory.
- The code includes detailed inline comments and docstrings for clarity.