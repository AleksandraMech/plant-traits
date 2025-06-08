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
DATA_DIR = './planttraits2024/test_images'
CSV_PATH = './planttraits2024/test.csv'
MODEL_PATH = './model_custom.pth'
OUTPUT_CSV = './baseline_predictions.csv'
MODEL_NAME = 'swin_large_patch4_window12_384.ms_in22k_ft_in1k'
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot
    return r2

class PlantDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, is_train=False):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filename = row['id'] + '.jpeg'
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.is_train:
            mean_cols = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
            mean_vals = row[mean_cols]
            if mean_vals.isnull().any():
                print(f"Warning: Missing values at index {idx}")
            mean_vals = mean_vals.infer_objects()
            mean_vals = mean_vals.fillna(0).astype(float).values
            means = torch.tensor(mean_vals, dtype=torch.float32)
            return image, means, filename
        else:
            return image, filename


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Load train data for evaluation
    train_df = pd.read_csv('./planttraits2024/train_supersmall.csv', dtype={'id': str})

    train_dataset = PlantDataset(train_df, './planttraits2024/train_images', transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    true_labels_list = []
    predictions_list = []
    batch_counter = 0
    with torch.no_grad():
        for images, means, fnames in train_loader:
            images = images.to(DEVICE)
            labels = means.to(DEVICE)
            if batch_counter % 10 == 0:
                print(f"Processed {batch_counter * BATCH_SIZE} samples")
            batch_counter += 1

            outputs = model(images)
            predictions_list.append(outputs.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())

    all_preds = np.concatenate(predictions_list, axis=0)
    all_true = np.concatenate(true_labels_list, axis=0)
    r2_norm = r2_score(all_true, all_preds)
    print("Normalized R2 scores for each trait:", r2_norm)
    print("Mean Normalized R2 score:", np.mean(r2_norm))
    
    mean_cols = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    global_means = train_df[mean_cols].mean(axis=0).values
    global_stds = train_df[mean_cols].std(axis=0).values 

    all_preds_denorm = all_preds * global_stds + global_means
    all_true_denorm = all_true * global_stds + global_means
    r2_denorm = r2_score(all_true_denorm, all_preds_denorm)

    print("Denormalized R2 scores for each trait:", r2_denorm)
    print("Mean Denormalized R2 score:", np.mean(r2_denorm))

    mean_prediction = np.tile(global_means, (all_true.shape[0], 1))
    r2_baseline = r2_score(all_true_denorm, mean_prediction)
    print("Baseline R2 (predicting global mean):", r2_baseline)

