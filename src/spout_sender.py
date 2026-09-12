import numpy as np
import cv2

try:
    import SpoutGL
    SPOUT_AVAILABLE = True
except ImportError:
    SPOUT_AVAILABLE = False

class SpoutSenderManager:
    """
    Spout2 DirectX GPU Paylaşımlı Doku Vericisi.
    İşlenmiş kareleri CPU-RAM'e kopyalamadan doğrudan ekran kartı (VRAM)
    üzerinden OBS Studio'daki Spout2 Capture kaynağına iletir.
    """
    def __init__(self, sender_name: str = "LiveFaceSwap_Spout", width: int = 1280, height: int = 720):
        self.sender_name = sender_name
        self.width = width
        self.height = height
        self.spout = None
        self._initialized = False

        if not SPOUT_AVAILABLE:
            print("[Spout2 Uyarı] 'SpoutGL' kütüphanesi kurulu değil. Lütfen 'pip install SpoutGL' çalıştırın.")
        else:
            self._init_spout()

    def _init_spout(self):
        try:
            self.spout = SpoutGL.SpoutSender()
            self.spout.setSenderName(self.sender_name)
            self._initialized = True
            print(f"[Spout2] Verici başarıyla hazırlandı: '{self.sender_name}' ({self.width}x{self.height})")
        except Exception as e:
            print(f"[Spout2 Hata] Başlatılamadı: {e}")
            self._initialized = False

    def send_frame(self, frame: np.ndarray) -> bool:
        """Kareyi Spout2 shared texture kanalına basar."""
        if not self._initialized or self.spout is None:
            return False

        h, w = frame.shape[:2]
        
        # Boyut değişmişse Spout kanalını güncelle
        if w != self.width or h != self.height:
            self.width, self.height = w, h

        # OpenCV BGR -> RGBA dönüşümü (Spout2 standardı)
        if frame.shape[2] == 3:
            rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        else:
            rgba_frame = frame

        try:
            # SpoutGL sendImage yöntemi
            success = self.spout.sendImage(rgba_frame, self.width, self.height, 0x1908, False, 0)
            return success
        except Exception as e:
            # Sessiz hata yerine ilk hatada bilgi ver
            return False

    def release(self):
        """Spout kaynağını güvenli serbest bırakır."""
        if self.spout and self._initialized:
            try:
                self.spout.releaseSender()
                print("[Spout2] Verici kapatıldı.")
            except Exception:
                pass
            self._initialized = False
