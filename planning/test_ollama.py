import requests
import json
import base64
import os
from PIL import Image
import io

def test_ollama_vision():
    # --- KONFIGURASI ---
    url = "http://10.7.101.217:11434/v1/chat/completions"
    model_name = "qwen3.5:27b"
    
    # GANTI INI dengan path gambar hasil jepretan D455 kamu
    image_path = "/home/b401/Documents/pick_place_occlusion_noetic/data/d455_capture/post_scene_rgb.jpg" 
    # -------------------

    if not os.path.exists(image_path):
        print(f"❌ File gambar tidak ditemukan: {image_path}")
        return

    print(f"--- 1. Memproses Gambar: {image_path} ---")
    
    try:
        # Resize gambar untuk mencegah Out of Memory (OOM) di Ollama
        img = Image.open(image_path)
        print(f"   Ukuran asli: {img.size}")
        
        img.thumbnail((800, 800)) # Batasi resolusi maksimal 800x800
        print(f"   Ukuran setelah resize: {img.size}")
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"   Ukuran Payload Base64: {len(base64_image) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ Gagal memproses gambar: {e}")
        return

    print("\n--- 2. Menyiapkan Payload & Mengirim ke Server ---")
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the main object in this image briefly."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }

    try:
        # Menggunakan timeout yang panjang karena model VLM butuh waktu untuk encode gambar
        response = requests.post(url, json=payload, timeout=180)
        
        if response.status_code == 200:
            data = response.json()
            answer = data['choices'][0]['message']['content']
            print("\n✅ BERHASIL! Jawaban Model:")
            print(f"   '{answer.strip()}'")
        else:
            print(f"\n❌ GAGAL! Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("\n❌ Timeout! Server butuh waktu terlalu lama (> 180 detik) untuk memproses gambar.")
    except Exception as e:
        print(f"\n❌ Error koneksi: {e}")

if __name__ == "__main__":
    test_ollama_vision()