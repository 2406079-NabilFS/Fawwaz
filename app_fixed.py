# ============================================================
# APLIKASI DETEKSI GESTUR TANGAN DENGAN STREAMLIT (FIXED)
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="Gesture Control Dino",
    page_icon="🦖",
    layout="centered"
)

st.title("🦖 Gesture Control Dino Chrome")
st.markdown("""
    Kontrol game Dino Chrome menggunakan gestur tangan!
    
    | **Gestur** | **Aksi Game** |
    |-----------|---------------|
    | 👊 **Rock** | Diam (Idle) |
    | ✋ **Paper** | Lompat (Jump) |
    | ✌️ **Scissors** | Jongkok (Duck) |
""")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load model dan konfigurasi"""
    
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
    
    try:
        with open('Model/gesture_config.json', 'r') as f:
            config = json.load(f)
        
        model = GestureCNN(num_classes=config['num_classes'])
        model.load_state_dict(torch.load('Model/best_gesture_cnn.pth', map_location='cpu'))
        model.eval()
        
        return model, config
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

try:
    model, config = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ==================== TRANSFORM ====================
transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ==================== FUNGSI PREDIKSI ====================
def predict_image(image):
    """Prediksi gestur dari gambar"""
    try:
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
        st.error(f"❌ Error during prediction: {e}")
        return None, 0, {}

# ==================== MAIN UI ====================

st.markdown("---")
st.subheader("📸 Upload Gambar Tangan")

uploaded_file = st.file_uploader(
    "Pilih gambar tangan (jpg, jpeg, png):",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Prediksi
        label, confidence, probs = predict_image(image)
        
        if label:
            # Tampilkan hasil
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption='Gambar Tangan', use_container_width=True)
            
            with col2:
                emoji = {'rock': '👊', 'paper': '✋', 'scissors': '✌️'}
                action = {
                    'rock': '⏸️ Diam (Idle)',
                    'paper': '⬆️ Lompat (Jump)',
                    'scissors': '⬇️ Jongkok (Duck)'
                }
                color = {'rock': '#e74c3c', 'paper': '#2ecc71', 'scissors': '#3498db'}
                
                st.markdown(f"""
                <div style="text-align:center;padding:20px;border-radius:15px;
                            background:{color[label]}22;border:3px solid {color[label]}">
                    <span style="font-size:64px">{emoji[label]}</span>
                    <h2 style="color:{color[label]}">{label.upper()}</h2>
                    <p style="font-size:18px">Confidence: {confidence:.1%}</p>
                    <p style="font-size:16px">🎮 {action[label]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probabilitas per kelas
            st.subheader("📊 Probabilitas per Kelas")
            for cls in config['classes']:
                prob = probs[cls]
                emoji_map = {'rock': '👊', 'paper': '✋', 'scissors': '✌️'}
                st.progress(prob, text=f"{emoji_map[cls]} {cls.upper()}: {prob:.1%}")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

# ==================== INFO ====================
st.markdown("---")
st.markdown("**ℹ️ Informasi Model:**")
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Dataset", "2,520 images")
with col_info2:
    st.metric("Accuracy", "100%")
with col_info3:
    st.metric("Classes", "3")

st.caption("Dibangun dengan PyTorch + CNN | Dataset: Rock-Paper-Scissors")
st.caption("📚 Tugas Besar Prak. Kecerdasan Buatan - Institut Teknologi Garut")
