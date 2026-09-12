import os
import sys
import urllib.request
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Güvenilir HuggingFace model bağlantıları
MODELS = {
    "scrfd_2.5g_bnkps.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/scrfd_2.5g_bnkps.onnx",
    "w600k_r50.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/w600k_r50.onnx",
    "inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
}

def download_with_progress(url: str, dest_path: str):
    print(f"\n[Indirme] Hedef dosya: {os.path.basename(dest_path)}", flush=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
        print(f"[Mevcut] {os.path.basename(dest_path)} zaten indirilmis ({os.path.getsize(dest_path) / (1024*1024):.1f} MB). Atlaniyor.", flush=True)
        return

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

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print(f"\n[Basarili] {os.path.basename(dest_path)} tamamlandi.", flush=True)

def main():
    print("=" * 60, flush=True)
    print("  AI NÖRAL MODELLER İNDİRİLİYOR (HuggingFace)", flush=True)
    print("=" * 60, flush=True)

    for name, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        try:
            download_with_progress(url, dest)
        except Exception as e:
            print(f"[Hata] {name} indirilemedi: {e}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Tum modeller hazir!", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
