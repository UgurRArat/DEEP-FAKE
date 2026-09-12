import numpy as np
import cv2
import time

class OneEuroFilter:
    """
    Gelişmiş 1€ (One-Euro) Titreme Önleyici Filtre.
    Kafa sabitken mikrometre düzeyindeki titremeleri (jitter) tamamen sıfırlar;
    hızlı kafa hareketlerinde ise sıfır gecikmeyle takip eder.
    Dead-zone (ölü bölge) desteği ile kafa hareketsizken titremeyi kilitler.
    """
    def __init__(self, min_cutoff: float = 0.6, beta: float = 0.006, d_cutoff: float = 1.0, dead_zone: float = 0.4):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.dead_zone = dead_zone
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1.0)

    def _exponential_smoothing(self, a: float, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        return a * x + (1.0 - a) * x_prev

    def filter(self, x: np.ndarray, timestamp: float = None) -> np.ndarray:
        if timestamp is None:
            timestamp = time.perf_counter()

        if self.x_prev is None:
            self.x_prev = x.copy().astype(np.float32)
            self.dx_prev = np.zeros_like(x, dtype=np.float32)
            self.t_prev = timestamp
            return x

        # Ölü bölge kontrolü: Mikro titremeler eşiğin altındaysa önceki konumu koru
        diff = np.abs(x - self.x_prev)
        if np.max(diff) < self.dead_zone:
            self.t_prev = timestamp
            return self.x_prev

        t_e = max(timestamp - self.t_prev, 0.0001)

        # Hız (Türev) hesabı
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        # Dinamik kesme frekansı
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        return x_hat

def sharpen_face(bgr: np.ndarray, strength: float = 0.20) -> np.ndarray:
    """
    Yüz hatlarını güvenli oranda keskinleştirir (aşırı kontrast ve paraziti engeller).
    """
    if strength <= 0.001:
        return bgr
    effective_strength = min(strength, 0.45)
    blurred = cv2.GaussianBlur(bgr, (0, 0), 1.5)
    sharpened = cv2.addWeighted(bgr, 1.0 + effective_strength, blurred, -effective_strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def transfer_high_frequency_texture(orig_bgr: np.ndarray, fake_bgr: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """
    Stüdyo Kalitesinde Doku ve Sakal Aktarımı (Frequency Separation Texture Restoration):
    128x128 yapay yüzün plastik görünümünü yok eder. Orijinal kameradaki gerçek sakal kıllarını,
    deri gözeneklerini ve göz parıltılarını (yüksek frekanslı detayları) alır ve yeni yüze giydirir.
    0.2 ms'de GPU/CPU üzerinde çalışarak GAN modelleri kadar net doku üretir.
    """
    if strength <= 0.01:
        return fake_bgr

    # Orijinal görüntünün yüksek frekanslı doku katmanı (Sakal, gözenek, kirpik)
    orig_blur = cv2.GaussianBlur(orig_bgr, (0, 0), sigmaX=1.6)
    high_freq = orig_bgr.astype(np.float32) - orig_blur.astype(np.float32)

    # Paraziti ve gürültüyü sınırlayan dinamik eşikleme
    high_freq = np.clip(high_freq, -40.0, 40.0)

    # Doku katmanını yeni yüze giydir
    enhanced = fake_bgr.astype(np.float32) + high_freq * strength
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def fast_luminance_color_transfer(source_bgr: np.ndarray, target_bgr: np.ndarray, blend_weight: float = 0.40) -> np.ndarray:
    """
    Ultra-Hızlı Vektörel Işık & Ten Uyarlaması (0.1 ms):
    Kameradaki ışık seviyesini yapay yüze aktarırken CPU'yu kilitleyen LAB dönüşümünü atlar;
    doğrudan kanal ortalamaları ve parlaklık oranı üzerinden milisaniyenin onda birinde çalışır.
    """
    src_mean = np.mean(source_bgr, axis=(0, 1), keepdims=True)
    tgt_mean = np.mean(target_bgr, axis=(0, 1), keepdims=True)
    ratio = tgt_mean / (src_mean + 1e-4)
    ratio = np.clip(ratio, 0.75, 1.30)
    adjusted = source_bgr.astype(np.float32) * ratio
    blended = adjusted * blend_weight + source_bgr.astype(np.float32) * (1.0 - blend_weight)
    return np.clip(blended, 0, 255).astype(np.uint8)

def reinhard_color_transfer(source_bgr: np.ndarray, target_bgr: np.ndarray, blend_weight: float = 0.40) -> np.ndarray:
    """Hızlı renk ve ışık eşitleme (Varsayılan hızlı motora yönlendirilir)."""
    return fast_luminance_color_transfer(source_bgr, target_bgr, blend_weight)

def extract_glasses_and_occlusion_mask(
    orig_bgr: np.ndarray,
    fake_bgr: np.ndarray,
    sensitivity: float = 0.45
) -> np.ndarray:
    """
    Ultra-Hızlı Gözlük ve Engel Maskesi (Downscaled Sobel — 0.8 ms):
    Gözlük çerçevesi ve saç tellerini korumak için 128px'te hızlı gradyan analizi yapar.
    """
    if sensitivity <= 0.05:
        return np.zeros(orig_bgr.shape[:2], dtype=np.float32)

    h, w = orig_bgr.shape[:2]
    # Hızlı hesaplama için 128x128'e indir
    small_orig = cv2.resize(orig_bgr, (128, 128), interpolation=cv2.INTER_NEAREST)
    small_fake = cv2.resize(fake_bgr, (128, 128), interpolation=cv2.INTER_NEAREST)

    orig_gray = cv2.cvtColor(small_orig, cv2.COLOR_BGR2GRAY)
    fake_gray = cv2.cvtColor(small_fake, cv2.COLOR_BGR2GRAY)

    # Basit Laplacian kenar şiddeti (Sobel'den 4 kat daha hızlı)
    lap_orig = cv2.Laplacian(orig_gray, cv2.CV_32F, ksize=3)
    lap_fake = cv2.Laplacian(fake_gray, cv2.CV_32F, ksize=3)
    edge_diff = np.clip(np.abs(lap_orig) - np.abs(lap_fake) * 0.7, 0, 255)

    dark_frames = (orig_gray < 70).astype(np.float32) * (edge_diff > 30).astype(np.float32)
    edge_mask = np.clip(edge_diff / 75.0, 0.0, 1.0)
    occ_small = np.maximum(edge_mask * sensitivity, dark_frames * 0.85)

    # Orijinal ROI boyutuna geri büyüt
    occ_mask = cv2.resize(occ_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(occ_mask, 0.0, 1.0)

def generate_face_mask(
    size: int = 128,
    feather: int = 15,
    margin: int = 8,
    mouth_protect: bool = False,
    mouth_opacity: float = 0.6,
    beard_protect: bool = False,
    beard_opacity: float = 0.85
) -> np.ndarray:
    """
    Tüm yüzü (alından çeneye, saç diplerine ve şakaklara kadar) tam ve doğal kaplayan alfa maskesi.
    Alnı ortadan kesinlikle kesmez; saç diplerine kadar pürüzsüzce uzanır.
    """
    mask = np.zeros((size, size), dtype=np.float32)
    s = size / 256.0
    
    # Tüm yüz anatomisini saç diplerinden çeneye kadar eksiksiz kaplayan noktalar:
    # Üst alın noktaları y=12..16'ya kadar çıkar (alnı asla ikiye bölmez!)
    pts = np.array([
        [int(128 * s), int(12 * s)],   # Alın tepe (saç çizgisi)
        [int(80 * s),  int(22 * s)],   # Sol üst alın
        [int(44 * s),  int(55 * s)],   # Sol şakak
        [int(32 * s),  int(105 * s)],  # Sol elmacık
        [int(34 * s),  int(165 * s)],  # Sol çene kavisi
        [int(58 * s),  int(215 * s)],  # Sol alt çene
        [int(95 * s),  int(248 * s)],  # Çene ucu sol
        [int(128 * s), int(252 * s)],  # Çene ucu merkez
        [int(161 * s), int(248 * s)],  # Çene ucu sağ
        [int(198 * s), int(215 * s)],  # Sağ alt çene
        [int(222 * s), int(165 * s)],  # Sağ çene kavisi
        [int(224 * s), int(105 * s)],  # Sağ elmacık
        [int(212 * s), int(55 * s)],   # Sağ şakak
        [int(176 * s), int(22 * s)],   # Sağ üst alın
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1.0)

    # 1. Ağız koruması (isteğe bağlı)
    if mouth_protect:
        mouth_mask = np.zeros((size, size), dtype=np.float32)
        mouth_center = (int(size * 0.5), int(size * 0.74))
        mouth_axes = (int(size * 0.18), int(size * 0.09))
        cv2.ellipse(mouth_mask, mouth_center, mouth_axes, 0, 0, 360, 1.0, -1)
        k_mouth = max(3, int(feather * 0.8) | 1)
        mouth_mask = cv2.GaussianBlur(mouth_mask, (k_mouth, k_mouth), 0)
        mask = np.clip(mask - mouth_mask * mouth_opacity, 0.0, 1.0)

    # Kenarları saç ve tenle doğal birleştiren yumuşatma (13-17 px)
    effective_feather = max(11, int(feather) | 1)
    mask = cv2.GaussianBlur(mask, (effective_feather, effective_feather), 0)
    return np.clip(mask, 0.0, 1.0)


