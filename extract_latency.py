import re
import os
import csv

# Path ke folder tempat log Anda berada
LOG_FOLDER = './logs' 
OUTPUT_FILE = 'rekap_latensi.csv'

def hitung_latensi(file_path):
    # Pola regex untuk mencari timestamp [123.456]
    pattern = r"\[\s*(\d+\.\d+)\s*\]"
    
    timestamps = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Mengambil angka di dalam kurung siku pertama/kedua yang formatnya timestamp
                matches = re.findall(pattern, line)
                for match in matches:
                    timestamps.append(float(match))
        
        if len(timestamps) >= 2:
            # Latensi = Waktu terakhir - Waktu pertama
            return round(timestamps[-1] - timestamps[0], 2)
    except Exception as e:
        print(f"Error membaca {file_path}: {e}")
    return "N/A"

# Proses semua file di folder
data_hasil = []
for filename in os.listdir(LOG_FOLDER):
    if filename.endswith(".log"):
        path = os.path.join(LOG_FOLDER, filename)
        latensi = hitung_latensi(path)
        data_hasil.append([filename, latensi])

# Simpan ke CSV
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Latency_sec"])
    writer.writerows(data_hasil)

print(f"Selesai! Data tersimpan di {OUTPUT_FILE}")