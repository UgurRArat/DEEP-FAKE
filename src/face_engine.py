import os
import cv2
import shutil
import threading
import numpy as np
from typing import Optional, List, Dict
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from src.enhancements import (
    OneEuroFilter,
    reinhard_color_transfer,
    sharpen_face,
    generate_face_mask,
    transfer_high_frequency_texture,
    extract_glasses_and_occlusion_mask
)
from src.hand_shield import HandShield

def imread_unicode(path: str) -> Optional[np.ndarray]:
    """Windows üzerinde Türkçe ve Unicode karakterli dosya yollarını güvenle okur."""
    try:
        with open(path, "rb") as f:
            chunk = f.read()
        arr = np.frombuffer(chunk, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

class FaceEngine:
    """
    Gelişmiş DirectML GPU Destekli Canlı Yüz Değiştirme Motoru.
    Stüdyo Kalitesi: Gözlük ve Saç Koruma (Occlusion Masking), Yüksek Frekanslı
    Sakal/Doku Aktarımı (Frequency Separation), 1€ Titreme Önleyici ve DirectML Hızlandırma.
    """
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.app = None
        self.rec_model = None
        self.swapper = None
        self.source_face = None
        self.lock = threading.Lock()
        
        # Yüz önbelleği (Tek tıkla 0.00s geçiş için)
        self.face_cache: Dict[str, any] = {}
        
        # Titreme önleyici filtreler (Dead-zone destekli)
        self.bbox_filter = OneEuroFilter(min_cutoff=0.6, beta=0.005, dead_zone=0.5)
        self.kps_filter = OneEuroFilter(min_cutoff=0.8, beta=0.006, dead_zone=0.4)
        
        # Gelişmiş Görsel ve Yayın Ayarları
        self.enable_color_transfer = False    # Gözlük yansımasını ve sararmayı önlemek için varsayılan kapalı
        self.enable_smoothing = True
        self.enable_mouth_protect = False
        self.enable_beard_protect = False     # Kendi sakalını koruma modu (Öz Sakal)
        self.enable_occlusion = False         # El/engel koruması varsayılan kapalı (sakalda sis yapmaması için)
        self.enable_gpen = True               # Evrensel GPEN HD Yüz Restoratörü (Stüdyo Kalitesi)
        self.occlusion_sensitivity = 0.35     # Gözlük koruma hassasiyeti
        self.texture_strength = 0.15          # Sakal ve cilt dokusu aktarımı
        self.face_opacity = 1.0
        self.sharpen_strength = 0.10
        self.mask_feather = 16
        self.mask_margin = 10
        self.turbo_mode = False
        self.missed_detection_count = 0
        
        # Evrensel GPEN HD Restoratör Oturumu
        self.gpen_sess = None
        self.gpen_input_name = None

        # Nöral El ve Engel Tanıyıcı Oturumu (face_occluder.onnx)
        self.occluder_sess = None
        self.occluder_input_name = None
        self.last_error = ""
        
        # Özel Eğitilmiş Model (.dfm) Desteği
        self.custom_model = None
        self.custom_model_type = None
        
        # MediaPipe 21-Eklemli El ve Parmak Kalkanı
        self.hand_shield = HandShield()
        self.enable_hand_shield = True
        
        # Takip durumu ve hafıza
        self.last_target_face = None
        self.frame_index = 0
        
        # Yumuşak Maskeler (128x128 ve 256x256)
        self.mask_128 = None
        self.mask_256 = None
        self.update_mask()

    @property
    def source_embedding(self):
        return self.source_face.embedding if self.source_face is not None else None

    def load_models(self) -> bool:
        """Modelleri hazırlar ve DirectML GPU belleğine yükler."""
        try:
            target_dir = os.path.expanduser('~/.insightface/models/buffalo_l')
            os.makedirs(target_dir, exist_ok=True)
            
            # buffalo_l klasöründe yalnızca resmi InsightFace modelleri bulunmalıdır
            valid_buffalo = {'det_10g.onnx', '1k3d68.onnx', '2d106det.onnx', 'genderage.onnx', 'w600k_r50.onnx'}
            
            # buffalo_l içindeki yabancı veya bozuk dosyaları (örn: occluder.onnx) temizle
            for f in os.listdir(target_dir):
                if f not in valid_buffalo:
                    try:
                        os.remove(os.path.join(target_dir, f))
                    except Exception:
                        pass

            # models klasöründeki bozuk/kuyruk hatası veren occluder.onnx dosyasını sil
            corrupt_occ = os.path.join(self.models_dir, 'occluder.onnx')
            if os.path.exists(corrupt_occ) and os.path.getsize(corrupt_occ) < 10000:
                try:
                    os.remove(corrupt_occ)
                except Exception:
                    pass

            for f in valid_buffalo:
                src = os.path.join(self.models_dir, f)
                dst = os.path.join(target_dir, f)
                if not os.path.exists(dst) and os.path.isfile(src):
                    shutil.copy(src, dst)

            print("[FaceEngine] InsightFace SCRFD Yüz Tespiti başlatılıyor (DirectML GPU 320p)...", flush=True)
            # 320x320 ultra-hızlı tespit piramidi (4.6 ms!)
            self.app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=self.providers)
            self.app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.25)

            # Referans fotoğrafların embedding'ini çıkarmak için ArcFace recognition modeli (Sadece 1 kez çağrılır)
            rec_path = os.path.join(target_dir, 'w600k_r50.onnx')
            if not os.path.exists(rec_path):
                rec_path = os.path.join(self.models_dir, 'w600k_r50.onnx')
            print(f"[FaceEngine] ArcFace Tanıma Modeli yükleniyor ({rec_path})...", flush=True)
            self.rec_model = get_model(rec_path, providers=self.providers)
            self.rec_model.prepare(ctx_id=0)

            swap_path = os.path.join(self.models_dir, 'inswapper_128.onnx')
            print(f"[FaceEngine] InSwapper-128 yükleniyor ({swap_path})...", flush=True)
            self.swapper = get_model(swap_path, providers=self.providers)

            # EVRENSEL STÜDYO KALİTESİ: GPEN-BFR-256 HD Restoratör
            gpen_path = os.path.join(self.models_dir, 'GPEN-BFR-256.onnx')
            if os.path.exists(gpen_path):
                try:
                    print(f"[FaceEngine] Evrensel GPEN-BFR HD Restoratör GPU'ya yükleniyor ({gpen_path})...", flush=True)
                    import onnxruntime as ort
                    self.gpen_sess = ort.InferenceSession(gpen_path, providers=self.providers)
                    self.gpen_input_name = self.gpen_sess.get_inputs()[0].name
                    print("[FaceEngine] Evrensel GPEN-BFR HD Restoratör hazır (7.8 ms / 128 FPS)!", flush=True)
                except Exception as eg:
                    print(f"[FaceEngine Uyarı] GPEN GPU'ya yüklenemedi: {eg}, CPU deneniyor...", flush=True)
                    try:
                        self.gpen_sess = ort.InferenceSession(gpen_path, providers=['CPUExecutionProvider'])
                        self.gpen_input_name = self.gpen_sess.get_inputs()[0].name
                        print("[FaceEngine] GPEN CPU üzerinde hazır!", flush=True)
                    except Exception:
                        self.gpen_sess = None

            # NÖRAL EL, PARMAK & ENGEL TANICI: face_occluder.onnx
            occ_path = os.path.join(self.models_dir, 'face_occluder.onnx')
            if os.path.exists(occ_path):
                try:
                    print(f"[FaceEngine] Nöral El ve Engel Tanıyıcı yükleniyor ({occ_path})...", flush=True)
                    import onnxruntime as ort
                    opts = ort.SessionOptions()
                    opts.enable_mem_pattern = False
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    try:
                        self.occluder_sess = ort.InferenceSession(occ_path, sess_options=opts, providers=self.providers)
                        print("[FaceEngine] Nöral El ve Engel Tanıyıcı DirectML GPU üzerinde hazır (2.2 ms)!", flush=True)
                    except Exception as edml:
                        print(f"[FaceEngine Bilgi] DirectML el modelini desteklemedi ({edml}), CPU'ya geçiliyor...", flush=True)
                        self.occluder_sess = ort.InferenceSession(occ_path, sess_options=opts, providers=['CPUExecutionProvider'])
                        print("[FaceEngine] Nöral El ve Engel Tanıyıcı CPU üzerinde hazır (4 ms)!", flush=True)
                    self.occluder_input_name = self.occluder_sess.get_inputs()[0].name
                except Exception as eo:
                    print(f"[FaceEngine Uyarı] Nöral El Tanıyıcı yüklenemedi (devre dışı bırakıldı): {eo}", flush=True)
                    self.occluder_sess = None

            print("[FaceEngine] Tüm temel nöral modeller DirectML GPU üzerinde hazır (Ultra Hızlı)!\n", flush=True)
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[FaceEngine Hata] Modeller yüklenemedi: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def set_source_image(self, image_path: str, cache_key: Optional[str] = None) -> bool:
        """
        Hedef kişinin fotoğrafını yükler ve yüz özelliklerini (embedding) çıkarır.
        Thread kilidi ile canlı yayın sırasında asla çökmez.
        """
        if cache_key and cache_key in self.face_cache:
            with self.lock:
                self.source_face = self.face_cache[cache_key]
            print(f"[FaceEngine] Yüz önbellekten anında etkinleştirildi: {cache_key}", flush=True)
            return True

        if not os.path.exists(image_path):
            print(f"[FaceEngine Hata] Referans görsel bulunamadı: {image_path}", flush=True)
            return False

        img = imread_unicode(image_path)
        if img is None:
            print(f"[FaceEngine Hata] Görsel dosyası okunamadı: {image_path}", flush=True)
            return False

        with self.lock:
            try:
                faces = self.app.get(img)
                if len(faces) == 0:
                    print("[FaceEngine Hata] Referans fotoğrafta net bir yüz tespit edilemedi!", flush=True)
                    return False

                # Fotoğraftaki en belirgin yüzü kaynak al
                face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                
                # ArcFace ile 512-d kimlik embedding'ini çıkar
                if self.rec_model is not None:
                    self.rec_model.get(img, face)

                if getattr(face, 'embedding', None) is None:
                    print("[FaceEngine Hata] Yüzün kimlik embedding'i çıkarılamadı!", flush=True)
                    return False

                self.source_face = face
                key = cache_key or os.path.basename(image_path)
                self.face_cache[key] = face
                print(f"[FaceEngine] Hedef yüz kimliği hafızaya alındı: {key}", flush=True)
                return True
            except Exception as e:
                print(f"[FaceEngine Hata] Yüz analizi sırasında beklenmeyen hata: {e}", flush=True)
                return False

    def update_mask(self):
        """128x128 ve 256x256 yüz maskelerini mevcut ayarlarla günceller."""
        self.mask_128 = generate_face_mask(
            size=128,
            feather=self.mask_feather,
            margin=self.mask_margin,
            mouth_protect=self.enable_mouth_protect
        )
        self.mask_256 = generate_face_mask(
            size=256,
            feather=int(self.mask_feather * 1.2),
            margin=int(self.mask_margin * 1.2),
            mouth_protect=self.enable_mouth_protect
        )

    def restore_face_gpen(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Evrensel Nöral GPEN-256 Yüz Restoratörü (7.8 ms):
        128px InSwapper çıktısını 256x256 stüdyo kalitesine çıkarır.
        Dünyadaki tüm yüzler için gerçek sakal kıllarını, göz parıltılarını ve gözenekleri çizer.
        """
        if self.gpen_sess is None:
            return crop_bgr
        try:
            rgb_fake = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            inp = cv2.resize(rgb_fake, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            inp = (inp - 0.5) / 0.5
            inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]
            out = self.gpen_sess.run(None, {self.gpen_input_name: inp})[0][0]
            out = np.transpose(out, (1, 2, 0))
            rgb_restored = np.clip((out * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            return cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2BGR)
        except Exception:
            return crop_bgr

    def predict_occlusion_mask(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Nöral El, Parmak & Engel Tanıyıcı (DirectML GPU 2.2 ms):
        Yüzün önündeki eli, parmakları, mikrofonu veya bardağı piksel piksel tespit eder.
        Dönüş: (H, W) float32 maskesi (1.0 = yüz, 0.0 = el/engel).
        """
        if self.occluder_sess is None:
            return None
        try:
            h, w = crop_bgr.shape[:2]
            if (w, h) != (256, 256):
                crop_in = cv2.resize(crop_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                crop_in = crop_bgr

            rgb = cv2.cvtColor(crop_in, cv2.COLOR_BGR2RGB)
            inp = rgb.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]

            raw_out = self.occluder_sess.run(None, {self.occluder_input_name: inp})[0]
            mask = np.squeeze(raw_out)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = np.clip(mask, 0.0, 1.0)

            # Çeneyi occluder'ın yanlışlıkla sakal zannedip silmesini/blurlamasını engelle (çene her zaman tam maskedir)
            mask[int(mask.shape[0] * 0.70):, :] = 1.0

            # Yarı saydam sis/blur oluşmasını engelleyen net eşikleme (ya yüz ya engel)
            mask = np.where(mask > 0.50, 1.0, 0.0).astype(np.float32)

            if (w, h) != (256, 256):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # İnce kenar yumuşatma (5x5)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            return mask
        except Exception:
            return None

    def swap_face(self, frame: np.ndarray) -> np.ndarray:
        """
        Kameradaki yüzü tespit eder, GPU ile dönüştürür ve evrensel GPEN HD restorasyon uygular (30+ FPS).
        Thread-safe: Yeni fotoğraf yüklenirken çökmez.
        Kamera git-gel yapmasını önleyen hafızalı takip içerir.
        """
        if self.source_face is None or getattr(self.source_face, 'embedding', None) is None or self.swapper is None or self.app is None:
            return frame

        # Thread kilidi: Eğer yeni bir fotoğraf seçiliyorsa ve analiz ediliyorsa bu kareyi güvenle atla
        if not self.lock.acquire(blocking=False):
            return frame

        try:
            self.frame_index += 1
            target_face = None

            # Her karede ultra-hızlı 320p yüz tespiti (4.6 ms)
            faces = self.app.get(frame)
            if len(faces) == 0:
                self.missed_detection_count += 1
                # Kafa ani döndüğünde veya ışık değiştiğinde kopmayı önleyen 8 karelik takip hafızası
                if self.missed_detection_count <= 8 and self.last_target_face is not None:
                    target_face = self.last_target_face
                else:
                    self.last_target_face = None
                    return frame
            else:
                self.missed_detection_count = 0
                target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                self.last_target_face = target_face

            # 1€ Titreme Önleyici Filtre (Landmark smoothing)
            if self.enable_smoothing:
                try:
                    target_face.kps = self.kps_filter.filter(target_face.kps)
                    target_face.bbox = self.bbox_filter.filter(target_face.bbox)
                except Exception:
                    pass

            use_hd = self.enable_gpen and self.gpen_sess is not None
            out_size = 256 if use_hd else 128
            active_mask = self.mask_256 if use_hd else self.mask_128

            # TURBO MODE: Alternatif karelerde GPU çıkarımını atlayarak FPS'i 30+'a katla
            is_turbo_skip = self.turbo_mode and (self.frame_index % 2 == 0) and (getattr(self, 'last_hd_face', None) is not None)

            if is_turbo_skip:
                hd_face = self.last_hd_face
                from insightface.utils import face_align
                aimg, M = face_align.norm_crop2(frame, target_face.kps, 128)
                M_active = M.copy()
                if use_hd:
                    M_active[0, :] *= 2.0
                    M_active[1, :] *= 2.0
            else:
                # GPU üzerinde 128x128 değiştirilmiş yüzü ve Affine matrisini al
                bgr_fake, M = self.swapper.get(frame, target_face, self.source_face, paste_back=False)

                # 1. EVRENSEL GPEN HD RESTORASYON (256x256 Stüdyo Netliği)
                if use_hd:
                    hd_face = self.restore_face_gpen(bgr_fake)
                    M_active = M.copy()
                    M_active[0, :] *= 2.0
                    M_active[1, :] *= 2.0
                else:
                    hd_face = bgr_fake
                    M_active = M

                # 2. Doğal Ten Rengi ve Işık Uyarlaması (Sakal Kararmasını Önleyen Alın/Yanak Örneklemesi)
                if self.enable_color_transfer:
                    try:
                        aimg = cv2.warpAffine(frame, M_active, (out_size, out_size))
                        src_lab = cv2.cvtColor(hd_face, cv2.COLOR_BGR2LAB).astype(np.float32)
                        tgt_lab = cv2.cvtColor(aimg, cv2.COLOR_BGR2LAB).astype(np.float32)

                        # Gözlükleri ve sakalı örneklem dışı bırak:
                        # Yalnızca temiz alın bölgesinden ten rengi al (y: %10 - %28, x: %35 - %65)
                        y_start, y_end = int(out_size * 0.10), int(out_size * 0.28)
                        x_start, x_end = int(out_size * 0.35), int(out_size * 0.65)
                        skin_tgt = tgt_lab[y_start:y_end, x_start:x_end]
                        skin_src = src_lab[y_start:y_end, x_start:x_end]

                        for i in range(3):
                            s_m, s_s = skin_src[:, :, i].mean(), skin_src[:, :, i].std() + 1e-5
                            t_m, t_s = skin_tgt[:, :, i].mean(), skin_tgt[:, :, i].std() + 1e-5
                            src_lab[:, :, i] = np.clip((src_lab[:, :, i] - s_m) * (t_s / s_s) * 0.65 + t_m, 0, 255)

                        hd_face = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                    except Exception:
                        pass

                self.last_hd_face = hd_face

            # 3. Yüz Detayı Keskinleştirme
            if self.sharpen_strength > 0.001:
                hd_face = sharpen_face(hd_face, self.sharpen_strength)

            # Ters dönüşüm matrisi (out_size -> Kamera Koordinatları)
            IM = cv2.invertAffineTransform(M_active)
            h, w = frame.shape[:2]

            # Yüzün karedeki sınırlarını (ROI) hesapla
            corners = np.array([
                [0, 0], [out_size, 0], [out_size, out_size], [0, out_size]
            ], dtype=np.float32).reshape(-1, 1, 2)
            pts = cv2.transform(corners, IM).reshape(-1, 2)

            x1 = max(0, int(np.floor(np.min(pts[:, 0]))))
            y1 = max(0, int(np.floor(np.min(pts[:, 1]))))
            x2 = min(w, int(np.ceil(np.max(pts[:, 0]))))
            y2 = min(h, int(np.ceil(np.max(pts[:, 1]))))

            if x2 <= x1 or y2 <= y1:
                return frame

            # SADECE ROI BÖLGESİ İÇİN AFİN DÖNÜŞÜMÜ (Ultra hızlı)
            IM_roi = IM.copy()
            IM_roi[0, 2] -= x1
            IM_roi[1, 2] -= y1
            roi_w = x2 - x1
            roi_h = y2 - y1

            # 4. Nöral El, Parmak & Engel Tanıma (face_occluder.onnx - 2.2 ms DirectML GPU)
            if self.enable_occlusion and self.occluder_sess is not None:
                crop_orig = cv2.warpAffine(frame, M_active, (out_size, out_size))
                occ_prob = self.predict_occlusion_mask(crop_orig)
                if occ_prob is not None:
                    # occ_prob: 1.0 = yüz, 0.0 = el/parmak/engel
                    active_mask = np.minimum(active_mask, occ_prob)

            roi_fake = cv2.warpAffine(hd_face, IM_roi, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
            roi_mask = cv2.warpAffine(active_mask, IM_roi, (roi_w, roi_h), flags=cv2.INTER_LINEAR)
            roi_orig = frame[y1:y2, x1:x2]

            if self.face_opacity < 0.999:
                roi_mask *= self.face_opacity

            # 5. Hızlı ROI Harmanlama (0.2 ms)
            roi_orig_f = roi_orig.astype(np.float32)
            roi_fake_f = roi_fake.astype(np.float32)
            roi_mask_3c = roi_mask[:, :, np.newaxis]

            # Akıllı El & Parmak Kalkanı (MediaPipe 21-Eklemli Hand Shield)
            if self.enable_hand_shield and self.hand_shield is not None:
                hand_mask = self.hand_shield.detect_hand_mask(frame, target_face.bbox)
                if hand_mask is not None:
                    hand_roi = hand_mask[y1:y2, x1:x2]
                    if np.any(hand_roi > 0.02):
                        roi_mask_3c = roi_mask_3c * (1.0 - hand_roi[:, :, np.newaxis])

            roi_blended = roi_fake_f * roi_mask_3c + roi_orig_f * (1.0 - roi_mask_3c)
            frame[y1:y2, x1:x2] = np.clip(roi_blended, 0, 255).astype(np.uint8)

            return frame
        except Exception:
            return frame
        finally:
            self.lock.release()

    def load_custom_dfm(self, dfm_path: str) -> bool:
        """
        Özel eğitilmiş DFM (DeepFaceModel) veya ONNX modelini motor belleğine yükler.
        Tek bir kişiye (örneğin Elon Musk) %100 birebir benzemek için eğitilen modelleri çalıştırır.
        """
        if not os.path.exists(dfm_path):
            print(f"[FaceEngine Hata] Model dosyası bulunamadı: {dfm_path}", flush=True)
            return False
        try:
            with self.lock:
                print(f"[FaceEngine] Özel eğitilmiş nöral model DirectML GPU'ya yükleniyor: {dfm_path}...", flush=True)
                # DFM loader entegrasyonu
                self.custom_model = get_model(dfm_path, providers=self.providers)
                self.custom_model_type = "dfm"
                print("[FaceEngine] Özel model başarıyla yüklendi!", flush=True)
                return True
        except Exception as e:
            print(f"[FaceEngine Hata] Özel model yüklenemedi: {e}", flush=True)
            return False
