#!/usr/bin/env python
# ============================================================
# Script untuk menjalankan Streamlit App dengan benar
# ============================================================

import subprocess
import sys
import os

print("=" * 60)
print("🦖 Gesture Control Dino - Streamlit App Launcher")
print("=" * 60)

# Pastikan di folder yang benar
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("\n📍 Lokasi app:", os.path.abspath("app.py"))
print("🐍 Python:", sys.executable)

# Jalankan streamlit
print("\n🚀 Meluncurkan Streamlit app...")
print("-" * 60)

try:
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--logger.level=error"
    ], check=True)
except KeyboardInterrupt:
    print("\n\n⛔ App dihentikan oleh user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
