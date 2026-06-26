# ============================================================
# APLIKASI DETEKSI GESTUR TANGAN DENGAN STREAMLIT
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import cv2

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
    
    with open('Model/gesture_config.json', 'r') as f:
        config = json.load(f)
    
    model = GestureCNN(num_classes=config['num_classes'])
    model.load_state_dict(torch.load('Model/best_gesture_cnn.pth', map_location='cpu'))
    model.eval()
    
    return model, config

model, config = load_model()

# ==================== TRANSFORM ====================
transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ==================== FUNGSI PREDIKSI ====================
def predict_image(image):
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

# ==================== UI ====================
mode = st.radio("Pilih Mode:", ["📸 Upload Gambar", "📷 Live Webcam"])

if mode == "📸 Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar tangan", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        label, confidence, probs = predict_image(image)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Gambar Tangan', width=250)
        
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
        
        st.subheader("📊 Probabilitas per Kelas")
        for cls in config['classes']:
            prob = probs[cls]
            color = {'rock': '#e74c3c', 'paper': '#2ecc71', 'scissors': '#3498db'}[cls]
            emoji = {'rock': '👊', 'paper': '✋', 'scissors': '✌️'}[cls]
            st.progress(prob, text=f"{emoji} {cls.upper()}: {prob:.1%}")

else:  # Live Webcam
    st.warning("⚠️ Pastikan webcam terhubung!")
    
    start_cam = st.button("🎥 Mulai Webcam", type="primary")
    
    if start_cam:
        frame_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Webcam tidak terdeteksi!")
        else:
            st.success("✅ Webcam aktif! Tekan 'q' pada keyboard untuk berhenti")
            
            running = True
            while running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                roi = frame[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)]
                
                cv2.rectangle(frame, (int(w*0.15), int(h*0.15)), 
                            (int(w*0.85), int(h*0.85)), (0, 255, 0), 2)
                
                try:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(roi_rgb)
                    label, confidence, probs = predict_image(img_pil)
                    
                    emoji = {'rock': '👊', 'paper': '✋', 'scissors': '✌️'}[label]
                    action = {'rock': 'IDLE', 'paper': 'JUMP!', 'scissors': 'DUCK!'}[label]
                    color = {'rock': (0, 0, 255), 'paper': (0, 255, 0), 'scissors': (255, 165, 0)}[label]
                    
                    cv2.putText(frame, f"{emoji} {label.upper()}", (10, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(frame, f"Conf: {confidence:.1%}", (10, 80), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"ACTION: {action}", (10, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    y = 40
                    for cls in config['classes']:
                        prob = probs[cls]
                        emoji_cls = {'rock': '👊', 'paper': '✋', 'scissors': '✌️'}[cls]
                        cv2.putText(frame, f"{emoji_cls} {cls}: {prob:.1%}", 
                                   (w - 250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                        y += 30
                    
                except:
                    cv2.putText(frame, "Tangan tidak terdeteksi", (10, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
            
            cap.release()
            cv2.destroyAllWindows()
            frame_placeholder.empty()
            st.info("⏹️ Webcam dimatikan")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Dibangun dengan PyTorch + CNN | Dataset: Rock-Paper-Scissors")
st.caption("📚 Tugas Besar Prak. Kecerdasan Buatan - Institut Teknologi Garut")