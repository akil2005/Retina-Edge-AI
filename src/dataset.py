import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# --- Transformation Pipeline ---
# These parameters are derived from SOTA medical imaging papers (like TOVIT).
# We use 224x224 to balance the ViT's patch requirements with T4 GPU memory.
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Augmentation to improve generalization
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Mimics varying lighting in retinal cameras
    transforms.ToTensor(), # Converts PIL Image (0-255) to Tensor (0-1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

class RetinalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the train.csv file.
            img_dir (string): Directory with all the category subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # --- CRITICAL: Folder Mapping ---
        # These MUST match the folder names inside /content/data/colored_images/
        self.folder_map = {
            0: 'No_DR',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe',
            4: 'Proliferate_DR'  # Corrected from 'Proliferative' to match your ls output
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Extract Image ID and Diagnosis Label from CSV
        # Assuming Col 0 is ID and Col 1 is Diagnosis
        img_id = str(self.data.iloc[idx, 0])
        label = int(self.data.iloc[idx, 1])
        
        # 2. Resolve the Directory Path
        # We fetch the subfolder name based on the numerical label
        folder_name = self.folder_map[label]
        
        # Ensure the filename has the correct extension (.png for APTOS)
        filename = img_id if img_id.endswith('.png') else f"{img_id}.png"
        img_path = os.path.join(self.img_dir, folder_name, filename)
        
        # 3. Load and Process the Image
        try:
            # Convert to RGB to ensure 3-channel input for EfficientNet and ViT
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Safety fallback: In case the CSV has an ID that doesn't exist in that folder
            raise FileNotFoundError(f"Missing file: {img_path}. Check if ID matches folder.")

        if self.transform:
            image = self.transform(image)

        # Return as (Tensor, Tensor)
        return image, torch.tensor(label)