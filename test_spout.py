import cv2
import time
import os
import sys

# Proje ana dizinini Python path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

from src.camera import CameraManager
from src.spout_sender import SpoutSenderManager
from src.config import DEFAULT_CAMERA_INDEX, DEFAULT_WIDTH, DEFAULT_HEIGHT, TARGET_FPS, SPOUT_SENDER_NAME

def main():
    print("=" * 60, flush=True)
    print("  FAZ 1 TESTI: Donanim Kamerasi -> Spout2 -> OBS Baglantisi", flush=True)
    print("=" * 60, flush=True)
    print(f"Kamera: {DEFAULT_CAMERA_INDEX} ({DEFAULT_WIDTH}x{DEFAULT_HEIGHT})", flush=True)
    print(f"Spout2 Kanal Adi: '{SPOUT_SENDER_NAME}'", flush=True)
    print("-" * 60, flush=True)
    print("Kapatmak icin acilan pencerede 'q' tusuna basin veya terminali durdurun.\n", flush=True)

    # 1. Kamerayi Baslat
    cam = CameraManager(
        camera_index=DEFAULT_CAMERA_INDEX,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        target_fps=TARGET_FPS
    )
    if not cam.start():
        print("[Hata] Kamera baslatilamadi. Cikiliyor.", flush=True)
        return

    # 2. Spout2 Vericisini Baslat
    spout = SpoutSenderManager(
        sender_name=SPOUT_SENDER_NAME,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT
    )

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

            # FPS Hesapla
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 0.001))
            prev_time = curr_time

            # Bilgi yazisi ekle
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f} | Spout: {SPOUT_SENDER_NAME}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Spout2 GPU kanalina gonder
            spout.send_frame(display_frame)

            # Her 30 karede bir konsola durum yaz
            if frame_count % 30 == 0:
                print(f"[Canli Yayin] Kare: {frame_count} | Anlik FPS: {fps:.1f} | Spout: {SPOUT_SENDER_NAME} aktif", flush=True)

            # Yerel onizleme penceresi
            cv2.imshow("Canli Kamera (Spout2 Testi - Cikis icin 'q')", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[Bilgi] Kullanıcı tarafından durduruldu.")
    finally:
        cam.stop()
        spout.release()
        cv2.destroyAllWindows()
        print("[Bilgi] Test tamamlandı, tüm kaynaklar serbest bırakıldı.")

if __name__ == "__main__":
    main()
