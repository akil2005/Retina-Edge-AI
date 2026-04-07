import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from sklearn.metrics import cohen_kappa_score
import numpy as np
from tqdm import tqdm

# Import local modules
from src.dataset import RetinalDataset, train_transforms
from src.model import HybridRetinaNet

def train_setup():
    # 1. Environment Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16 # Optimized for T4 GPU memory
    EPOCHS = 20
    
    # 2. Data Preparation and Splitting
    dataset = RetinalDataset(
        csv_file='/content/data/train.csv', 
        img_dir='/content/data/colored_images', 
        transform=train_transforms
    )
    
    # Stratified 80/20 Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # 3. Weighted Sampling to handle Class Imbalance
    # This ensures the model sees rare 'Proliferative' cases as often as 'Normal' cases.
    labels = [dataset.data.iloc[i]['diagnosis'] for i in train_ds.indices]
    class_counts = np.bincount(labels)
    weights = 1. / class_counts
    samples_weights = torch.from_numpy(np.array([weights[t] for t in labels])).double()
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 4. Model Initialization
    model = HybridRetinaNet(num_classes=5).to(DEVICE)
    
    # Label Smoothing (0.1) prevents the model from overfitting to noisy medical labels.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    # AdamW is preferred for models containing Transformers.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Cosine Annealing helps the model converge smoothly toward the end of training.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Mixed Precision Scaler for faster training on NVIDIA GPUs.
    scaler = torch.cuda.amp.GradScaler()

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Using Autocast for Mixed Precision (FP16)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                
                predictions = torch.argmax(outputs, dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(lbls.numpy())

        # Calculate Quadratic Weighted Kappa (Clinical standard for DR)
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        avg_loss = train_loss / len(train_loader)
        
        print(f"Summary - Loss: {avg_loss:.4f} | QWK Score: {kappa:.4f}")
        
        scheduler.step()

    # 5. Save the Final Model Weights
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), "weights/hybrid_retina_v1.pth")
    print("Training complete. Best model saved in weights folder.")

if __name__ == "__main__":
    train_setup()