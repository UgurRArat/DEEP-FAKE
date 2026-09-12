import cv2
import threading
import time
from typing import Optional, Tuple
import numpy as np

class CameraManager:
    """
    Sıfır Gecikmeli (Zero-Lag LIFO) Donanım Kamera Yöneticisi.
    Kameradan gelen kareleri ayrı bir iş parçacığında (thread) okur ve
    gecikme birikmesini önlemek için her zaman sadece en son tamamlanan kareyi saklar.
    """
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, target_fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Kamerayı başlatır ve arka plan okuma thread'ini açar."""
        # Windows DirectShow backend öncelikli
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not self.cap or not self.cap.isOpened():
            # DirectShow başarısız olursa standart backend'e dön
            print(f"[Camera] DirectShow ile kamera {self.camera_index} açılamadı, varsayılan deneniyor...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            print(f"[Camera Hata] Kamera açılamadı! Başka bir uygulama (Discord/WhatsApp) kamerayı kullanıyor olabilir.")
            return False

        # Kamera donanım çözünürlük ve FPS ayarları
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Gerçek ayarlanan değerleri kontrol et
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"[Camera] Kamera açıldı: {actual_w}x{actual_h} @ ~{actual_fps} FPS")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        """Kamerayı arka planda kesintisiz okuyan döngü (Eski kareleri atar)."""
        while self._running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """En taze kareyi döndürür (Gecikme süresi: 0 ms)."""
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
            return False, None

    def stop(self):
        """Kamerayı ve thread'i güvenli şekilde serbest bırakır."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[Camera] Kamera bağlantısı güvenli şekilde kapatıldı.")
