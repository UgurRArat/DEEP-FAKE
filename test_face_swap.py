import cv2
import time
import os
import sys
import numpy as np

# Proje ana dizinini Python path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

from src.camera import CameraManager
from src.spout_sender import SpoutSenderManager
from src.face_engine import FaceEngine
from src.config import DEFAULT_CAMERA_INDEX, DEFAULT_WIDTH, DEFAULT_HEIGHT, TARGET_FPS, SPOUT_SENDER_NAME, MODELS_DIR, ASSETS_DIR

def main():
    print("=" * 60, flush=True)
    print("  FAZ 2: GPU AI Canlı Yüz Değiştirici (InSwapper + Spout2 + OBS)", flush=True)
    print("=" * 60, flush=True)

    # 1. AI Motorunu Başlat
    engine = FaceEngine(models_dir=MODELS_DIR)
    if not engine.load_models():
        print("[Hata] Modeller yüklenemedi. Lütfen önce download_models.py'nin bitmesini bekleyin.", flush=True)
        return

    # 2. Varsayılan Hedef Yüzü Seç
    target_img_path = os.path.join(ASSETS_DIR, "target.png")
    if not os.path.exists(target_img_path):
        target_img_path = os.path.join(ASSETS_DIR, "target.jpg")

    if os.path.exists(target_img_path):
        engine.set_source_image(target_img_path)
    else:
        print(f"[Bilgi] Hedef yüz seçilmedi. Fotoğraf seçmek için 't' tuşuna basabilirsiniz.", flush=True)

    # 3. Kamerayı Başlat
    cam = CameraManager(
        camera_index=DEFAULT_CAMERA_INDEX,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        target_fps=TARGET_FPS
    )
    if not cam.start():
        print("[Hata] Kamera açılamadı!", flush=True)
        return

    # 4. Spout2 GPU Vericisini Başlat
    spout = SpoutSenderManager(
        sender_name=SPOUT_SENDER_NAME,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT
    )

    print("-" * 60, flush=True)
    print("KONTROLLER:", flush=True)
    print("  [s] : Yüz değiştirmeyi AÇ / KAPAT (Orijinal vs AI karşılaştırması)", flush=True)
    print("  [t] : Bilgisayardan yeni hedef fotoğraf seç (Dosya Seçici)", flush=True)
    print("  [q] : Çıkış\n", flush=True)

    swap_enabled = True
    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1

            # Yüz Değiştirme İşlemi (GPU)
            t0 = time.perf_counter()
            if swap_enabled and engine.source_face is not None:
                output_frame = engine.swap_face(frame)
            else:
                output_frame = frame.copy()
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # FPS Hesapla
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 0.001))
            prev_time = curr_time

            # Bilgi yazıları
            status_text = "AI YUZ AKTIF" if swap_enabled and engine.source_face is not None else "ORIJINAL KAMERA"
            color = (0, 255, 0) if "AKTIF" in status_text else (0, 165, 255)
            
            cv2.putText(output_frame, f"FPS: {fps:.1f} | AI: {infer_ms:.1f}ms | {status_text}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(output_frame, "[s] Ac/Kapat | [t] Foto Sec | [q] Cikis", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Spout2 GPU kanalına gönder (OBS)
            spout.send_frame(output_frame)

            # Durum çıktısı
            if frame_count % 30 == 0:
                print(f"[Yayin] FPS: {fps:.1f} | GPU AI Süresi: {infer_ms:.1f} ms | Durum: {status_text}", flush=True)

            # Yerel Önizleme
            cv2.imshow("AI Canli Yuz Degistirici - OBS Aktif", output_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                swap_enabled = not swap_enabled
                print(f"\n[Ayar] Yüz değiştirme: {'AÇIK' if swap_enabled else 'KAPALI'}", flush=True)
            elif key == ord('t'):
                # Dosya seçici aç
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                selected_file = filedialog.askopenfilename(
                    title="Hedef Yüz Fotoğrafı Seç",
                    filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.webp")]
                )
                root.destroy()
                if selected_file:
                    print(f"\n[Secim] Yeni fotoğraf seçildi: {selected_file}", flush=True)
                    engine.set_source_image(selected_file)

    except KeyboardInterrupt:
        print("\n[Bilgi] Kullanıcı tarafından durduruldu.")
    finally:
        cam.stop()
        spout.release()
        cv2.destroyAllWindows()
        print("[Bilgi] Faz 2 testi tamamlandı, kaynaklar serbest bırakıldı.")

if __name__ == "__main__":
    main()
