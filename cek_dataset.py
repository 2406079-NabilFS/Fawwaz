import os

base_path = 'Dataset/rps'
classes = ['rock', 'paper', 'scissors']

print("📊 Cek Dataset")
print("="*40)

total = 0
for cls in classes:
    path = os.path.join(base_path, cls)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if f.endswith('.png')])
        print(f"✅ {cls}: {count} gambar")
        total += count
    else:
        print(f"❌ {cls}: folder tidak ditemukan!")

print("="*40)
print(f"📊 Total: {total} gambar")