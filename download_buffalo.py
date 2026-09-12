import os
import sys
import urllib.request
import zipfile
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ZIP_PATH = os.path.join(MODELS_DIR, "buffalo_l.zip")

def download_and_extract():
    print("=" * 60, flush=True)
    print("  BUFFALO_L MODELLERİ İNDİRİLİYOR (Yüz Tespiti + Tanıma)", flush=True)
    print("=" * 60, flush=True)

    if not os.path.exists(os.path.join(MODELS_DIR, "w600k_r50.onnx")):
        print(f"[Indirme] {BUFFALO_URL}...", flush=True)
        start_time = time.time()

        def reporthook(count, block_size, total_size):
            if total_size <= 0:
                return
            downloaded = count * block_size
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            elapsed = max(time.time() - start_time, 0.001)
            speed = (downloaded / (1024 * 1024)) / elapsed
            sys.stdout.write(f"\r  -> %{percent:.1f} | {mb_down:.1f}/{mb_total:.1f} MB | {speed:.1f} MB/s ")
            sys.stdout.flush()

        urllib.request.urlretrieve(BUFFALO_URL, ZIP_PATH, reporthook=reporthook)
        print(f"\n[Ayiklama] {ZIP_PATH} cikariliyor...", flush=True)

        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)

        # Temizlik
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)

        print("[Basarili] Yüz modelleri ayiklandi:", flush=True)
        for f in os.listdir(MODELS_DIR):
            print(f"  - {f}", flush=True)
    else:
        print("[Mevcut] buffalo_l modelleri zaten var.", flush=True)

if __name__ == "__main__":
    download_and_extract()
