import cv2

# Cek kamera
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Webcam terdeteksi!")
    cap.release()
else:
    print("❌ Webcam tidak terdeteksi!")