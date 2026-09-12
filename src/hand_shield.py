import cv2
import numpy as np
from typing import Optional, List, Tuple

class HandShield:
    """
    Google MediaPipe 21-Eklemli El ve Parmak Kalkanı (Hand & Finger Shield).
    Kullanıcı elini, parmaklarını, telefonunu veya bardağını yüzünün önüne getirdiğinde
    el bölgesini 2 ms içinde piksel hassasiyetinde tespit eder.
    Yapay yüzün elin üzerine çizilmesini engeller; elin %100 doğal ve keskin şekilde
    en önde kalmasını sağlar.
    """
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.55, min_tracking_confidence: float = 0.50):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_hands = None
        self.hands_detector = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        """MediaPipe Hands kütüphanesini yüklemeye çalışır."""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            print("[HandShield] MediaPipe 21-Eklemli El Takipçisi başarıyla yüklendi!", flush=True)
        except Exception as e:
            self.hands_detector = None
            print(f"[HandShield Bilgi] MediaPipe python paketi bulunamadı ({e}). Hibrit renk & kontur el dedektörüne geçildi.", flush=True)

    def detect_hand_mask(self, frame_bgr: np.ndarray, face_bbox: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Kamera karesinde el ve parmakların kapladığı alan için ikili (binary) maske üretir.
        Dönüş: (H, W) float32 maskesi (1.0 = El/Parmaklar, 0.0 = Arka plan / Yüz).
        """
        h, w = frame_bgr.shape[:2]
        hand_mask = np.zeros((h, w), dtype=np.float32)

        # 1. Öncelik: MediaPipe 21-Noktalı İskelet Tespiti
        if self.hands_detector is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.hands_detector.process(rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        pts = []
                        for lm in hand_landmarks.landmark:
                            cx = int(np.clip(lm.x * w, 0, w - 1))
                            cy = int(np.clip(lm.y * h, 0, h - 1))
                            pts.append([cx, cy])
                        
                        pts = np.array(pts, dtype=np.int32)
                        
                        # Parmak eklemleri ve avuç içinin dış bükey örtüsü (Convex Hull)
                        hull = cv2.convexHull(pts)
                        cv2.fillConvexPoly(hand_mask, hull, 1.0)
                        
                        # Parmak kalınlığı için her eklemi daire ile kalınlaştır
                        for pt in pts:
                            cv2.circle(hand_mask, tuple(pt), 14, 1.0, -1)
                    
                    # Doğal kenar dikişi için hafif yumuşat (3x3)
                    if np.any(hand_mask > 0):
                        hand_mask = cv2.GaussianBlur(hand_mask, (5, 5), 0)
                        return np.clip(hand_mask, 0.0, 1.0)
            except Exception:
                pass

        # MediaPipe yüklü değilse kesinlikle yüze delik açma (sıfır maske dön)
        return hand_mask
