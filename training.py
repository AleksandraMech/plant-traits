import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch.nn as nn

# ==== CONFIG ====
CSV_PATH = './planttraits2024/train_small.csv'
IMAGE_DIR = './planttraits2024/train_images'
MODEL_NAME = 'swin_small_patch4_window7_224'  # Try 'swin_base...' next
EPOCHS = 30
BATCH_SIZE = 32
NUM_WORKERS = 2
LR = 1e-4
PATIENCE = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = f'{MODEL_NAME}_best.pth'

# ==== Utility ====
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return 1 - ss_res / ss_tot

# ==== Dataset ====
class PlantDataset(Dataset):
    def __init__(self, df, root_dir, transform):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.targets = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.root_dir, row['id'] + '.jpeg')).convert("RGB")
        image = self.transform(image)

        labels = row[self.targets].values.astype(np.float32)
        return image, torch.tensor(labels)

# ==== Main Training Pipeline ====
def train():
    df = pd.read_csv(CSV_PATH, dtype={'id': str})

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = PlantDataset(train_df, IMAGE_DIR, transform)
    val_ds = PlantDataset(val_df, IMAGE_DIR, transform)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=6)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_r2 = -np.inf
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs = model(images).cpu().numpy()
                val_preds.append(outputs)
                val_labels.append(labels.numpy())
        
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        r2_norm = r2_score(val_labels, val_preds)
        mean_r2_norm = np.mean(r2_norm)

        scheduler.step(np.mean((val_preds - val_labels)**2))
        print(f"Epoch {epoch}: Train Loss: {train_loss / len(train_loader.dataset):.4f} | "
              f"Val R² (norm): {mean_r2_norm:.4f}")

        # Early stopping
        if mean_r2_norm > best_r2:
            best_r2 = mean_r2_norm
            torch.save(model.state_dict(), SAVE_PATH)
            print("✅ Model improved and saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping.")
                break

if __name__ == '__main__':
    train()
