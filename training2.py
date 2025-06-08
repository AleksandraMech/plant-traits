import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch.nn as nn
import time

# === CONFIGURATION ===
DATA_DIR = './planttraits2024/train_images'
CSV_PATH = './planttraits2024/train_supersmall.csv'
MODEL_SAVE_PATH = './model_custom.pth'
MODEL_NAME = 'swin_small_patch4_window7_224'
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot
    return r2

class PlantDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.mean_cols = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filename = row['id'] + '.jpeg'
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        mean_vals = row[self.mean_cols].infer_objects().fillna(0).astype(float).values
        means = torch.tensor(mean_vals, dtype=torch.float32)
        return image, means

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(CSV_PATH, dtype={'id': str})
    dataset = PlantDataset(df, DATA_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_r2 = -np.inf
    patience_counter = 0
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        all_preds = []
        all_labels = []

        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(targets.cpu().numpy())

        all_preds_np = np.concatenate(all_preds, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        r2 = r2_score(all_labels_np, all_preds_np)
        mean_r2 = np.mean(r2)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(dataset):.4f}, R2 mean: {mean_r2:.4f}, R2 per trait: {r2}")

        # Early stopping
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Model improved and saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping.")
                break

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
