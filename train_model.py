# ============================================================
# TRAINING CNN UNTUK KLASIFIKASI GESTUR ROCK-PAPER-SCISSORS
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================

CONFIG = {
    'data_path': 'Dataset/rps',
    'image_size': 128,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'num_classes': 3,
    'classes': ['rock', 'paper', 'scissors'],
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print("="*60)
print("🔧 KONFIGURASI TRAINING")
print("="*60)
print(f"✅ Device: {CONFIG['device']}")
print(f"✅ Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
print(f"✅ Epochs: {CONFIG['epochs']}")
print("="*60)

# ==================== DATASET CLASS ====================

class RPSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = CONFIG['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.exists(cls_path):
                print(f"⚠️ Folder {cls_path} tidak ditemukan!")
                continue
                
            for img_name in os.listdir(cls_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls])
        
        print(f"✅ Dataset loaded: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                image = Image.open(img_path).convert('RGB')
                label = self.labels[idx]
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"⚠️ Failed to load {img_path} after {max_retries} retries: {e}")
                    # Return dummy data instead of crashing
                    dummy_image = Image.new('RGB', (128, 128))
                    if self.transform:
                        dummy_image = self.transform(dummy_image)
                    return dummy_image, 0
                # Try random image
                idx = np.random.randint(0, len(self.images))
                img_path = self.images[idx]

# ==================== DATA TRANSFORM ====================

transform_train = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ==================== LOAD DATASET ====================

print("\n📂 Loading dataset...")

full_dataset = RPSDataset(CONFIG['data_path'], transform=transform_train)

# Validasi dataset tidak kosong
if len(full_dataset) == 0:
    print("❌ ERROR: Dataset kosong! Periksa path dataset.")
    print(f"Path yang dicari: {CONFIG['data_path']}")
    exit(1)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Pastikan minimal ada satu sample di train dan val
if train_size < 1:
    train_size = 1
if val_size < 1:
    val_size = 1

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

val_dataset.dataset.transform = transform_val

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"✅ Train: {len(train_dataset)} gambar")
print(f"✅ Validation: {len(val_dataset)} gambar")

# ==================== ARSITEKTUR CNN ====================

class GestureCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(GestureCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1: 128x128x3 → 64x64x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 64x64x32 → 32x32x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 32x32x64 → 16x16x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 16x16x128 → 8x8x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 8x8x256 → 4x4x512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Inisialisasi model
model = GestureCNN(num_classes=CONFIG['num_classes'])
model = model.to(CONFIG['device'])

print(f"\n🧠 Total parameter: {sum(p.numel() for p in model.parameters()):,}")

# ==================== TRAINING ====================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

train_losses = []
val_accs = []
best_val_acc = 0

print("\n🚀 Memulai Training...")
print("="*60)

# Buat folder Model
os.makedirs('Model', exist_ok=True)

for epoch in range(CONFIG['epochs']):
    # Training
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = correct / total
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'Model/best_gesture_cnn.pth')
        print(f"⭐ Model saved! Best Val Acc: {best_val_acc:.4f}")

print(f"\n✅ Selesai! Best Validation Accuracy: {best_val_acc:.4f}")

# ==================== SAVE CONFIG ====================

config = {
    'num_classes': CONFIG['num_classes'],
    'classes': CONFIG['classes'],
    'image_size': CONFIG['image_size'],
    'best_accuracy': float(best_val_acc)
}

with open('Model/gesture_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print("✅ Config saved: Model/gesture_config.json")

# ==================== PLOT HASIL ====================

if len(train_losses) > 0 and len(val_accs) > 0:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("✅ Graph saved: training_results.png")
    try:
        plt.show()
    except:
        print("⚠️ Display tidak tersedia, tapi graph sudah disimpan.")
else:
    print("⚠️ Tidak ada data untuk diplot.")

print("\n🎉 Training selesai!")
print(f"📁 Model tersimpan di: Model/best_gesture_cnn.pth")
print(f"📁 Config tersimpan di: Model/gesture_config.json")