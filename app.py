import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import warnings

#  for Terminal
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 8 
EPOCHS = 50 
LEARNING_RATE = 5e-5 

# --- Transforms ---
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- Robust Dataset ---
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, file_list=None):
        self.transform = transform
        if file_list is None:
            self.pairs = []
            masks_dict = {}
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if "_mask" in f and f.lower().endswith((".png", ".jpg")):
                        key = f.split("_mask")[0].strip()
                        masks_dict[key] = os.path.join(root, f)
            
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if "_mask" not in f and f.lower().endswith((".png", ".jpg")):
                        key = os.path.splitext(f)[0].strip()
                        if key in masks_dict:
                            self.pairs.append((os.path.join(root, f), masks_dict[key]))
            self.file_list = self.pairs
        else:
            self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_list[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 127).astype(np.float32) 

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if mask.ndimension() == 2:
                mask = mask.unsqueeze(0)
        return image, mask

# --- Utility Functions ---
def calculate_dice(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def save_visual_results(dataset, model, num_samples=3):
    model.eval()
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        image, mask = dataset[idx]
        
        with torch.no_grad():
            input_tensor = image.unsqueeze(0).to(DEVICE)
            output = model(input_tensor)
            prediction = torch.sigmoid(output).cpu().squeeze()
            prediction = (prediction > 0.5).float()

        
        img_display = image.permute(1, 2, 0).cpu().numpy()
        img_display = (img_display * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img_display = np.clip(img_display, 0, 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_display)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(prediction.numpy(), cmap='gray')
        plt.title("Model Prediction")
        plt.axis('off')
        
        plt.savefig(f'segmentation_result_{i}.png', bbox_inches='tight')
        plt.close()
    print(f" Saved {num_samples} visual results to your folder.")

# --- Initialization ---
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
bce_loss_fn = nn.BCEWithLogitsLoss()

def hybrid_loss(pred, target):
    return (0.8 * dice_loss_fn(pred, target)) + (0.2 * bce_loss_fn(pred, target))

# --- Main Execution ---
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\HomePC\Downloads\archive(1)\Dataset_BUSI_with_GT"
    
    full_dataset = BreastCancerDataset(DATA_PATH)
    
    if len(full_dataset) == 0:
        print(" Error: No pairs found! Check your path and mask names.")
    else:
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_data, test_data = random_split(full_dataset, [train_size, test_size])
        
        train_ds = BreastCancerDataset(DATA_PATH, transform=train_transform, file_list=[full_dataset.file_list[i] for i in train_data.indices])
        test_ds = BreastCancerDataset(DATA_PATH, transform=val_transform, file_list=[full_dataset.file_list[i] for i in test_data.indices])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        print(f" Ready! Found {len(full_dataset)} total images.")
        print(f" Training on: {len(train_ds)} | Testing on: {len(test_ds)}\n")

        scaler = GradScaler('cuda')
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss, epoch_dice = 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for imgs, msks in pbar:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    out = model(imgs)
                    loss = hybrid_loss(out, msks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                d_score = calculate_dice(out, msks)
                epoch_loss += loss.item()
                epoch_dice += d_score.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d_score.item():.4f}")
            
            avg_dice = epoch_dice / len(train_loader)
            avg_loss = epoch_loss / len(train_loader)
            print(f" Epoch [{epoch+1}/{EPOCHS}] Summary: Loss = {avg_loss:.4f} | Dice Accuracy = {avg_dice:.4f}")
            print("-" * 40)

        torch.save(model.state_dict(), "breast_cancer_model.pth")
        print("\n Generating Sample Results...")
        save_visual_results(test_ds, model)
        print(" Process Complete.")
