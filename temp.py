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
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# === CONFIGURATION ===
MODEL_NAME = 'swin_base_patch4_window7_224'
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_CSV = './planttraits2024/train.csv'
IMG_DIR = './planttraits2024/train_images'
MODEL_DIR = './models_no_kfold'

os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COLS = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
LOG_TRANSFORM_COLS = ['X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']

# === NORMALIZATION STATS (to be updated if needed) ===
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# === DATASET ===
class PlantDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, normalize_targets=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.normalize_targets = normalize_targets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['id'] + '.jpeg'
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        targets = row[TARGET_COLS].infer_objects().fillna(0).astype(np.float32).values

        if self.normalize_targets:
            targets = (targets - self.normalize_targets['mean']) / self.normalize_targets['std']

        targets = torch.tensor(targets, dtype=torch.float32)
        return image, targets

class SwinRegressor(nn.Module):
    def __init__(self, backbone_name, num_outputs):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='avg')
        self.head = nn.Linear(self.backbone.num_features, num_outputs)

    def forward(self, x):
        return self.head(self.backbone(x))


# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# === TRAIN FUNCTION ===
def train_model(train_df, val_df, normalize_targets=None):
    train_dataset = PlantDataset(train_df, IMG_DIR, transform, normalize_targets)
    val_dataset = PlantDataset(val_df, IMG_DIR, transform, normalize_targets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # model = SwinRegressor(MODEL_NAME, NUM_CLASSES).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_r2 = -np.inf
    best_val_loss = np.inf

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

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
        scheduler.step()

        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        mean = normalize_targets['mean'].reshape(1, -1)
        std = normalize_targets['std'].reshape(1, -1)
        train_preds_denorm = train_preds * std + mean
        train_targets_denorm = train_targets * std + mean
        train_r2 = r2_score(train_targets_denorm, train_preds_denorm, multioutput='raw_values')
        train_r2_mean = np.mean(train_r2)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

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
        mean = normalize_targets['mean'].reshape(1, -1)
        std = normalize_targets['std'].reshape(1, -1)
        val_preds_denorm = val_preds * std + mean
        val_targets_denorm = val_targets * std + mean
        # val_preds_denorm[:, 1:] = np.expm1(val_preds_denorm[:, 1:])
        # val_targets_denorm[:, 1:] = np.expm1(val_targets_denorm[:, 1:])
        val_r2 = r2_score(val_targets_denorm, val_preds_denorm, multioutput='raw_values')
        val_r2_mean = np.mean(val_r2)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train R² (mean): {train_r2_mean:.4f} | Val R² (mean): {val_r2_mean:.4f}")
        print(f"  Val R² (per trait): {np.round(val_r2, 4)}")

        if epoch == 0 or val_loss < best_val_loss:
            # best_val_r2 = val_r2_mean
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'best_model_v2_epoch{epoch+1}.pt'))
            print("✅ Saved best model.")


# === MAIN ===
if __name__ == '__main__':
    df = pd.read_csv(DATA_CSV, dtype={'id': str})
    df = df.dropna(subset=TARGET_COLS)
    # df[LOG_TRANSFORM_COLS] = np.log1p(df[LOG_TRANSFORM_COLS])

    target_values = df[TARGET_COLS].astype(np.float32).values    

    normalize_targets = {
        'mean': target_values.mean(axis=0),
        'std': target_values.std(axis=0) + 1e-8
    }

    # small_df = df.sample(32, random_state=42)
    # train_model(small_df, small_df, normalize_targets)
    train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
    train_model(train_df, val_df, normalize_targets)
