import torch
import json
from PIL import Image
from torchvision import transforms
import os
import torch.nn as nn

# ==================== LOAD CONFIG ====================
with open('Model/gesture_config.json', 'r') as f:
    config = json.load(f)

print("="*50)
print("🧪 TESTING MODEL GESTURE RECOGNITION")
print("="*50)
print(f"✅ Classes: {config['classes']}")
print(f"✅ Image size: {config['image_size']}x{config['image_size']}")
print(f"✅ Best accuracy: {config['best_accuracy']:.2%}")

# ==================== DEFINE MODEL ====================
class GestureCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(GestureCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==================== LOAD MODEL ====================
model = GestureCNN(num_classes=config['num_classes'])
model.load_state_dict(torch.load('Model/best_gesture_cnn.pth', map_location='cpu'))
model.eval()
print("✅ Model loaded successfully!")

# ==================== TRANSFORM ====================
transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ==================== PREDICT FUNCTION ====================
def predict_image(image_path):
    """Prediksi gestur dari gambar"""
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
        
        label = config['classes'][pred.item()]
        confidence = confidence.item()
        all_probs = {config['classes'][i]: probs[0][i].item() 
                    for i in range(len(config['classes']))}
        
        return label, confidence, all_probs
    except Exception as e:
        print(f"Error: {e}")
        return None, 0, {}

# ==================== TEST WITH DATASET ====================
print("\n" + "="*50)
print("📸 TESTING DENGAN DATASET")
print("="*50)

results = []

for cls in config['classes']:
    cls_path = os.path.join('Dataset/rps', cls)
    if os.path.exists(cls_path):
        files = [f for f in os.listdir(cls_path) if f.endswith('.png')]
        if files:
            # Ambil 3 gambar pertama
            test_files = files[:3]
            correct = 0
            
            for f in test_files:
                img_path = os.path.join(cls_path, f)
                label, conf, probs = predict_image(img_path)
                
                is_correct = (label == cls)
                if is_correct:
                    correct += 1
                
                # Tampilkan detail
                status = "✅" if is_correct else "❌"
                print(f"{status} {f} → Prediksi: {label.upper()} ({conf:.1%})")
            
            print(f"   Akurasi untuk {cls}: {correct}/{len(test_files)} ({correct/len(test_files):.0%})")
            print()

# ==================== TEST WITH CUSTOM IMAGE ====================
print("\n" + "="*50)
print("📸 TESTING DENGAN GAMBAR KUSTOM")
print("="*50)

# Option untuk user jika ingin run interaktif: uncomment baris di bawah
# import sys
# if not sys.stdin.isatty():
#     print("Skipping interactive mode - run script dengan -i flag untuk input")
# else:
#     print("Masukkan path gambar untuk diuji, atau tekan Enter untuk keluar")
#     while True:
#         img_path = input("\n🔍 Path gambar (atau 'q' untuk keluar): ").strip()
#         
#         if img_path.lower() == 'q' or img_path == '':
#             break
#         
#         if os.path.exists(img_path):
#             label, conf, probs = predict_image(img_path)
#             
#             if label:
#                 print(f"\n📷 Gambar: {os.path.basename(img_path)}")
#                 print(f"✅ Prediksi: {label.upper()} ({conf:.1%})")
#                 print("📊 Probabilitas:")
#                 for k, v in probs.items():
#                     bar = "█" * int(v * 20)
#                     print(f"   {k}: {bar} {v:.1%}")
#         else:
#             print("❌ File tidak ditemukan!")

print("✅ Untuk test gambar custom, buka test_model.py dan uncomment bagian interaktif")
print("\n🎉 Testing selesai!")