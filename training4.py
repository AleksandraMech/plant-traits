import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import r2_score

# === CONFIGURATION ===
MODEL_NAME = 'swin_small_patch4_window7_224'
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 5
KFOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_CSV = './planttraits2024/train.csv'
IMG_DIR = './planttraits2024/train_images'
MODEL_DIR = './models_cv'

os.makedirs(MODEL_DIR, exist_ok=True)

mean_cols = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']

# === DATASET ===
class PlantDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['id'] + '.jpeg'
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mean_vals = row[mean_cols]
        mean_vals = mean_vals.astype(np.float32).values
        means = torch.tensor(mean_vals, dtype=torch.float32)
        return image, means


# === TRAINING FUNCTION ===
def train_one_fold(fold, train_idx, val_idx, full_df, target_means, target_stds):
    print(f"\nFold {fold+1}/{KFOLDS}")

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = PlantDataset(train_df, IMG_DIR, transform)
    val_dataset = PlantDataset(val_df, IMG_DIR, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)

        # 🔁 De-normalize for R²
        train_preds = train_preds * target_stds.values + target_means.values
        train_targets = train_targets * target_stds.values + target_means.values

        train_r2 = r2_score(train_targets, train_preds, multioutput='raw_values')
        train_r2_mean = np.mean(train_r2)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                val_preds.append(outputs.detach().cpu().numpy())
                val_targets.append(targets.detach().cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        # 🔁 De-normalize for R²
        val_preds = val_preds * target_stds.values + target_means.values
        val_targets = val_targets * target_stds.values + target_means.values

        val_r2 = r2_score(val_targets, val_preds, multioutput='raw_values')
        val_r2_mean = np.mean(val_r2)

        print(f"\n Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Train R² (mean): {train_r2_mean:.4f} | Val R² (mean): {val_r2_mean:.4f}")
        print(f"   Val R² (per trait): {np.round(val_r2, 4)}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(MODEL_DIR, f'swin_base_fold{fold+1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model for fold {fold+1}")


# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === RUN K-FOLD TRAINING ===
def run_kfold_training():
    df = pd.read_csv(DATA_CSV, dtype={'id': str})
    df = df.dropna(subset=mean_cols)

    # 🔁 Compute normalization stats
    target_means = df[mean_cols].mean()
    target_stds = df[mean_cols].std()

    df[mean_cols] = (df[mean_cols] - target_means) / target_stds

    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_one_fold(fold, train_idx, val_idx, df, target_means, target_stds)


if __name__ == '__main__':
    run_kfold_training()
