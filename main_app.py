import os
import sys
import time
import threading
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Proje ana dizinini Python path'e ekle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

from src.camera import CameraManager
from src.spout_sender import SpoutSenderManager
from src.face_engine import FaceEngine
from src.search_engine import FaceSearchEngine
from src.config import (
    DEFAULT_CAMERA_INDEX, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    TARGET_FPS, SPOUT_SENDER_NAME, MODELS_DIR, ASSETS_DIR
)

# CustomTkinter Tema Ayarları
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class LiveFaceSwapApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Canlı Yüz Değiştirici — OBS & Spout2 Stüdyo Paneli (30+ FPS)")
        self.geometry("1240x840")
        self.minsize(1020, 720)

        # Temel Motorlar
        self.engine = FaceEngine(models_dir=MODELS_DIR)
        self.search_engine = FaceSearchEngine()
        self.cam = None
        self.spout = None

        # Durum Değişkenleri
        self.is_running = False
        self.swap_enabled = False  # İlk açılışta kullanıcının kendi doğal yüzüyle başlar
        self.mirror_mode = True  # Canlı yayıncılar için ayna modu varsayılan açık
        self.active_face_name = "Kendi Yüzün (Doğal)"
        self.fps = 0.0
        self.infer_ms = 0.0
        
        # Galeri Kayıtları: {name: image_path}
        self.gallery = {}
        self.latest_display_frame = None

        # Arayüz Bileşenlerini İnşa Et
        self._build_ui()

        # GUI Yenileme Döngüsünü Başlat (Bağımsız 30 FPS)
        self._render_gui_loop()

        # Motorları Başlat
        self._init_backend()

    def _build_ui(self):
        # Grid Yapısı: Sol Kontrol Paneli (370px) | Sağ Önizleme & Galeri
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # SOL PANEL: KAYDIRILABİLİR KONTROL ALANI
        # ==========================================
        self.sidebar = ctk.CTkScrollableFrame(self, width=370, corner_radius=12)
        self.sidebar.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        # Başlık & Logo
        self.title_label = ctk.CTkLabel(
            self.sidebar, text="🎭 AI FACE SWAPPER",
            font=ctk.CTkFont(size=21, weight="bold")
        )
        self.title_label.pack(padx=16, pady=(12, 2))

        self.sub_label = ctk.CTkLabel(
            self.sidebar, text="DirectML GPU (RTX 4060) & OBS Spout2 30 FPS",
            font=ctk.CTkFont(size=12), text_color="#3498db"
        )
        self.sub_label.pack(padx=16, pady=(0, 14))

        # --- 1. BÖLÜM: İNTERNETTEN YÜZ ARA ---
        self.search_group = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.search_group.pack(padx=8, pady=6, fill="x")

        self.search_title = ctk.CTkLabel(
            self.search_group, text="🔍 İnternetten Yüz Ara",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.search_title.pack(padx=12, pady=(8, 2), anchor="w")

        self.search_entry = ctk.CTkEntry(
            self.search_group, placeholder_text="Kişi veya ünlü adı (örn: Brad Pitt)...",
            height=34
        )
        self.search_entry.pack(padx=12, pady=4, fill="x")
        self.search_entry.bind("<Return>", lambda event: self._on_search_clicked())

        self.search_btn = ctk.CTkButton(
            self.search_group, text="Bul ve Canlı Uygula ⚡",
            height=34, fg_color="#1f538d", hover_color="#14375e",
            command=self._on_search_clicked
        )
        self.search_btn.pack(padx=12, pady=(4, 6), fill="x")

        self.search_status = ctk.CTkLabel(
            self.search_group, text="", font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.search_status.pack(padx=12, pady=(0, 6))

        # --- 2. BÖLÜM: BİLGİSAYARDAN FOTOĞRAF / MODEL YÜKLE ---
        self.upload_group = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.upload_group.pack(padx=8, pady=6, fill="x")

        self.upload_title = ctk.CTkLabel(
            self.upload_group, text="📁 Fotoğraf veya Özel Model",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.upload_title.pack(padx=12, pady=(8, 2), anchor="w")

        self.upload_btn = ctk.CTkButton(
            self.upload_group, text="Arkadaşının / Özel Fotoğraf Seç 👤",
            height=34, fg_color="#2b2b2b", hover_color="#3d3d3d",
            command=self._on_upload_clicked
        )
        self.upload_btn.pack(padx=12, pady=4, fill="x")

        self.upload_dfm_btn = ctk.CTkButton(
            self.upload_group, text="Özel Eğitilmiş Model (.dfm / .onnx) 🧠",
            height=34, fg_color="#2c3e50", hover_color="#34495e",
            command=self._on_upload_dfm_clicked
        )
        self.upload_dfm_btn.pack(padx=12, pady=(0, 8), fill="x")

        # --- 3. BÖLÜM: POPÜLER HAZIR PROFİLLER (1-TIK GEÇİŞ) ---
        self.presets_group = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.presets_group.pack(padx=8, pady=6, fill="x")

        self.presets_title = ctk.CTkLabel(
            self.presets_group, text="⭐ Popüler Hazır Yüzler",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.presets_title.pack(padx=12, pady=(8, 4), anchor="w")

        self.presets_grid = ctk.CTkFrame(self.presets_group, fg_color="transparent")
        self.presets_grid.pack(padx=10, pady=(0, 8), fill="x")
        self.presets_grid.grid_columnconfigure((0, 1), weight=1)

        presets = [
            ("⚡ Elraenn", "elraenn"),
            ("🚀 Elon Musk", "Elon Musk"),
            ("🎬 Brad Pitt", "Brad Pitt"),
            ("🕶️ Keanu Reeves", "Keanu Reeves"),
            ("🏆 C. Ronaldo", "Cristiano Ronaldo"),
            ("💪 Gigachad", "Gigachad"),
            ("🎥 DiCaprio", "Leonardo DiCaprio"),
            ("✨ Scarlett J.", "Scarlett Johansson")
        ]

        for idx, (label, query) in enumerate(presets):
            row = idx // 2
            col = idx % 2
            btn = ctk.CTkButton(
                self.presets_grid, text=label, height=30,
                fg_color="#182c44", hover_color="#2980b9",
                font=ctk.CTkFont(size=11),
                command=lambda q=query: self._apply_preset(q)
            )
            btn.grid(row=row, column=col, padx=3, pady=3, sticky="ew")

        # --- 4. BÖLÜM: CANLI YAYIN & PERFORMANS AYARLARI ---
        self.settings_group = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.settings_group.pack(padx=8, pady=6, fill="x")

        self.settings_title = ctk.CTkLabel(
            self.settings_group, text="🎛️ Yayın & Görsel İnce Ayarları",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.settings_title.pack(padx=12, pady=(8, 4), anchor="w")

        # Switch'ler
        self.swap_switch = ctk.CTkSwitch(self.settings_group, text="AI Yüz Değiştirme", command=self._on_toggle_swap)
        self.swap_switch.deselect()  # İlk açılışta kapalı (kullanıcının kendi yüzü)
        self.swap_switch.pack(padx=12, pady=4, anchor="w")

        self.gpen_switch = ctk.CTkSwitch(
            self.settings_group, text="✨ Evrensel GPEN HD Netlik (Stüdyo)",
            command=self._on_toggle_gpen
        )
        self.gpen_switch.select()
        self.gpen_switch.pack(padx=12, pady=4, anchor="w")

        self.hand_shield_switch = ctk.CTkSwitch(
            self.settings_group, text="🖐️ Akıllı El Kalkanı (MediaPipe 21-Nokta)",
            command=self._on_toggle_hand_shield
        )
        self.hand_shield_switch.select()
        self.hand_shield_switch.pack(padx=12, pady=4, anchor="w")

        self.mirror_switch = ctk.CTkSwitch(self.settings_group, text="Yatay Ayna Modu (Flip)", command=self._on_toggle_mirror)
        self.mirror_switch.select()
        self.mirror_switch.pack(padx=12, pady=4, anchor="w")

        self.turbo_switch = ctk.CTkSwitch(self.settings_group, text="Turbo Kare Atlama", command=self._on_toggle_turbo)
        self.turbo_switch.pack(padx=12, pady=4, anchor="w")

        self.color_switch = ctk.CTkSwitch(self.settings_group, text="Akıllı Işık & Ten Eşitleme", command=self._on_toggle_color)
        self.color_switch.deselect()
        self.color_switch.pack(padx=12, pady=4, anchor="w")

        self.mouth_switch = ctk.CTkSwitch(self.settings_group, text="Ağız Koruması (Doğal Konuşma)", command=self._on_toggle_mouth)
        self.mouth_switch.pack(padx=12, pady=4, anchor="w")

        self.smooth_switch = ctk.CTkSwitch(self.settings_group, text="1€ Titreme Önleyici Filtre", command=self._on_toggle_smooth)
        self.smooth_switch.select()
        self.smooth_switch.pack(padx=12, pady=4, anchor="w")

        # --- SLIDERLAR ---

        # 2. HD Keskinleştirici (Güvenli kalibre: %10)
        self.sharpen_label = ctk.CTkLabel(
            self.settings_group, text="HD Netleştirici (Sharpen): 10%",
            font=ctk.CTkFont(size=11)
        )
        self.sharpen_label.pack(padx=12, pady=(4, 0), anchor="w")

        self.sharpen_slider = ctk.CTkSlider(
            self.settings_group, from_=0.0, to=0.45, number_of_steps=20,
            command=self._on_slider_sharpen
        )
        self.sharpen_slider.set(0.10)
        self.sharpen_slider.pack(padx=12, pady=(0, 6), fill="x")

        # 3. Yüz Karışım Oranı (Opaklık)
        self.opacity_label = ctk.CTkLabel(
            self.settings_group, text="Yüz Karışım Gücü: 100%",
            font=ctk.CTkFont(size=11)
        )
        self.opacity_label.pack(padx=12, pady=(4, 0), anchor="w")

        self.opacity_slider = ctk.CTkSlider(
            self.settings_group, from_=0.5, to=1.0, number_of_steps=20,
            command=self._on_slider_opacity
        )
        self.opacity_slider.set(1.0)
        self.opacity_slider.pack(padx=12, pady=(0, 6), fill="x")

        # 4. Kenar Yumuşatma (Feather)
        self.feather_label = ctk.CTkLabel(
            self.settings_group, text="Kenar Yumuşatma: 16 px",
            font=ctk.CTkFont(size=11)
        )
        self.feather_label.pack(padx=12, pady=(4, 0), anchor="w")

        self.feather_slider = ctk.CTkSlider(
            self.settings_group, from_=5, to=30, number_of_steps=25,
            command=self._on_slider_feather
        )
        self.feather_slider.set(16)
        self.feather_slider.pack(padx=12, pady=(0, 10), fill="x")

        # --- 5. BÖLÜM: DURUM & METRİKLER ---
        self.status_box = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#121820")
        self.status_box.pack(padx=8, pady=8, fill="x")

        self.spout_indicator = ctk.CTkLabel(
            self.status_box, text="🟢 OBS Spout: LiveFaceSwap_Spout",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#2ecc71"
        )
        self.spout_indicator.pack(padx=10, pady=(8, 2), anchor="w")

        self.gpu_badge = ctk.CTkLabel(
            self.status_box, text="⚡ Donanım: NVIDIA RTX 4060 (DirectML)",
            font=ctk.CTkFont(size=11), text_color="#95a5a6"
        )
        self.gpu_badge.pack(padx=10, pady=2, anchor="w")

        self.fps_label = ctk.CTkLabel(
            self.status_box, text="🚀 FPS: --  |  AI: -- ms",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#2ecc71"
        )
        self.fps_label.pack(padx=10, pady=(4, 1), anchor="w")

        self.active_face_label = ctk.CTkLabel(
            self.status_box, text="👤 Yüz: Kendi Yüzün",
            font=ctk.CTkFont(size=11), text_color="#3498db"
        )
        self.active_face_label.pack(padx=10, pady=(1, 8), anchor="w")

        # ==========================================
        # SAĞ PANEL: CANLI ÖNİZLEME & HIZLI GALERİ
        # ==========================================
        self.main_area = ctk.CTkFrame(self, corner_radius=12)
        self.main_area.grid(row=0, column=1, padx=12, pady=12, sticky="nsew")
        self.main_area.grid_rowconfigure(0, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        # Canlı Kamera Görüntü Ekranı (Label)
        self.video_display = ctk.CTkLabel(
            self.main_area, text="Kamera ve modeller hazırlanıyor...",
            corner_radius=8, fg_color="#0e0e0e"
        )
        self.video_display.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        # Alt Hızlı Geçiş Galerisi (Presets)
        self.gallery_frame = ctk.CTkFrame(self.main_area, height=100, corner_radius=10)
        self.gallery_frame.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")
        self.gallery_frame.grid_columnconfigure(0, weight=1)

        self.gallery_header = ctk.CTkLabel(
            self.gallery_frame, text="⚡ AKTİF YÜZ HAVUZU (Tek Tıkla Anında 0.00s Geçiş)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#3498db"
        )
        self.gallery_header.pack(padx=10, pady=(6, 4), anchor="w")

        self.gallery_scroll = ctk.CTkScrollableFrame(self.gallery_frame, height=55, orientation="horizontal")
        self.gallery_scroll.pack(padx=8, pady=(0, 6), fill="x", expand=True)

        # Kalıcı "Kendi Yüzüm" Butonu (Tek tıkla anında kendi yüzüne dönüş)
        self.my_face_btn = ctk.CTkButton(
            self.gallery_scroll,
            text="👤 Kendi Yüzüm (Doğal)",
            width=145, height=38,
            fg_color="#1b4332", hover_color="#2d6a4f",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_select_my_face
        )
        self.my_face_btn.pack(side="left", padx=4, pady=2)

    def _init_backend(self):
        """Yapay Zeka, Kamera ve Spout2 motorlarını arka planda başlatır."""
        threading.Thread(target=self._start_engines_async, daemon=True).start()

    def _start_engines_async(self):
        print("[App] Modeller GPU belleğine yükleniyor...", flush=True)
        if not self.engine.load_models():
            err_msg = getattr(self.engine, 'last_error', '')
            detail = f"\n\nHata Detayı:\n{err_msg}" if err_msg else ""
            messagebox.showerror("Hata", f"Nöral modeller yüklenemedi!{detail}")
            return

        # Önbellekteki varsayılanları galeriye ekle
        cache_dir = os.path.join(ASSETS_DIR, "cache")
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    name = os.path.splitext(file)[0].replace('_', ' ').capitalize()
                    full_p = os.path.join(cache_dir, file)
                    self._add_to_gallery(name, full_p)

        # Yüzleri önbelleğe (cache) al ki tıklandığında anında (0.00s) geçebilsin
        elraenn_p = os.path.join(cache_dir, "elraenn.jpg")
        if os.path.exists(elraenn_p):
            self.engine.set_source_image(elraenn_p, cache_key="elraenn")

        target_path = os.path.join(ASSETS_DIR, "target.png")
        if not os.path.exists(target_path):
            target_path = os.path.join(ASSETS_DIR, "target.jpg")
        if os.path.exists(target_path):
            self.engine.set_source_image(target_path, cache_key="Hedef")
            self._add_to_gallery("Hedef", target_path)

        # KULLANICI İSTEĞİ: İlk açılışta kendi doğal yüzüyle başlar
        self.swap_enabled = False
        self.active_face_name = "Kendi Yüzün (Doğal)"
        if not self.is_running:
            return
        try:
            self.search_status.configure(
                text="👤 Kendi doğal yüzündesin. Yüz değiştirmek için galeriden bir yüz seçebilirsin.",
                text_color="#2ecc71"
            )
        except Exception:
            return

        # Kamerayı Başlat
        self.cam = CameraManager(
            camera_index=DEFAULT_CAMERA_INDEX,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            target_fps=TARGET_FPS
        )
        if not self.cam.start():
            messagebox.showerror("Hata", "Kamera açılamadı!")
            return

        # Spout2 Başlat
        self.spout = SpoutSenderManager(
            sender_name=SPOUT_SENDER_NAME,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT
        )

        self.is_running = True
        print("[App] Sistem hazır, canlı yayın döngüsü başlatılıyor (30 FPS)...", flush=True)
        self._video_loop()

    def _video_loop(self):
        """Kareleri işleyen ve hem ekrana hem OBS'e basan ana döngü."""
        prev_time = time.time()
        frame_counter = 0

        while self.is_running:
            ret, frame = self.cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_counter += 1

            # Yayıncılar için yatay ayna (mirror) modu
            if self.mirror_mode:
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()

            # Yapay Zeka Yüz Değiştirme
            if self.swap_enabled and self.engine.source_face is not None:
                out_frame = self.engine.swap_face(frame)
            else:
                out_frame = frame.copy()

            self.infer_ms = (time.perf_counter() - t0) * 1000.0

            # FPS Ölçümü
            curr_time = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(curr_time - prev_time, 0.001))
            prev_time = curr_time

            # Spout2 ile OBS'e Gönder (DirectX GPU VRAM - 0.2 ms)
            if self.spout:
                self.spout.send_frame(out_frame)

            # GUI Thread'inin bağımsız çizmesi için en güncel kareyi paylaş (Thread kilidini kaldırır -> 30+ FPS)
            self.latest_display_frame = out_frame

    def _render_gui_loop(self):
        """Tkinter ana iş parçacığında çalışan bağımsız önizleme yenileyici."""
        if self.latest_display_frame is not None:
            self._update_preview(self.latest_display_frame)
        self.after(33, self._render_gui_loop)

    def _update_preview(self, bgr_frame):
        try:
            # Önizleme boyutunu orantılı küçült (GUI donmasını engellemek için)
            h, w = bgr_frame.shape[:2]
            scale = min(740 / w, 480 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            preview_img = cv2.resize(bgr_frame, (new_w, new_h))

            # Canlı Önizleme Üzerine Hafif HUD Rozeti (OBS Spout akışına basılmaz, sadece GUI ekranında görünür)
            hud_color = (46, 204, 113) if self.fps >= 24 else (241, 196, 15) if self.fps >= 15 else (231, 76, 60)
            cv2.rectangle(preview_img, (10, 10), (168, 36), (20, 20, 20), -1)
            cv2.rectangle(preview_img, (10, 10), (168, 36), (60, 60, 60), 1)
            cv2.putText(preview_img, f"FPS: {self.fps:.1f} | {self.infer_ms:.1f}ms", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 1, cv2.LINE_AA)

            # BGR -> RGB -> CTkImage
            rgb_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))

            # GUI Thread üzerinde güncelle
            self.video_display.configure(image=ctk_img, text="")
            self.fps_label.configure(
                text=f"🚀 FPS: {self.fps:.1f}  |  AI: {self.infer_ms:.1f} ms"
            )
            face_disp = self.active_face_name
            if len(face_disp) > 22:
                face_disp = face_disp[:20] + "..."
            self.active_face_label.configure(text=f"👤 Yüz: {face_disp}")
        except Exception:
            pass

    def _apply_preset(self, query: str):
        """Popüler hazır yüz butonuna basıldığında çağrılır."""
        self.search_entry.delete(0, 'end')
        self.search_entry.insert(0, query)
        self._on_search_clicked()

    def _on_search_clicked(self):
        query = self.search_entry.get().strip()
        if not query:
            return

        self.search_status.configure(text=f"'{query}' aranıyor...", text_color="#f39c12")
        self.search_btn.configure(state="disabled")

        def _async_search():
            img_path = self.search_engine.search_and_download(query)
            if img_path and os.path.exists(img_path):
                ok = self.engine.set_source_image(img_path, cache_key=query)
                if ok:
                    self.active_face_name = query
                    self.swap_enabled = True
                    self.swap_switch.select()
                    self._add_to_gallery(query, img_path)
                    self.search_status.configure(text=f"'{query}' uygulandı! ✓", text_color="#2ecc71")
                else:
                    self.search_status.configure(text="Yüz bulunamadı, başka isim deneyin.", text_color="#e74c3c")
            else:
                self.search_status.configure(text="Görsel bulunamadı.", text_color="#e74c3c")
            self.search_btn.configure(state="normal")

        threading.Thread(target=_async_search, daemon=True).start()

    def _on_upload_clicked(self):
        file_path = filedialog.askopenfilename(
            title="Hedef Yüz Fotoğrafı Seç",
            filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.webp")]
        )
        if not file_path:
            return

        name = os.path.splitext(os.path.basename(file_path))[0].capitalize()
        self.search_status.configure(text=f"'{name}' taranıyor...", text_color="#f39c12")

        def _async_upload():
            ok = self.engine.set_source_image(file_path, cache_key=name)
            if ok:
                self.active_face_name = name
                self.swap_enabled = True
                self.swap_switch.select()
                self._add_to_gallery(name, file_path)
                self.search_status.configure(text=f"'{name}' yüklendi! ✓", text_color="#2ecc71")
            else:
                self.search_status.configure(text=f"'{name}' içinde net yüz bulunamadı!", text_color="#e74c3c")

        threading.Thread(target=_async_upload, daemon=True).start()

    def _on_upload_dfm_clicked(self):
        file_path = filedialog.askopenfilename(
            title="Özel Eğitilmiş Model Seç (.dfm veya .onnx)",
            filetypes=[("Nöral Model Dosyaları", "*.dfm *.onnx *.pb")]
        )
        if not file_path:
            return

        name = os.path.splitext(os.path.basename(file_path))[0].capitalize()
        self.search_status.configure(text=f"Model yükleniyor: '{name}'...", text_color="#f39c12")

        def _async_load_dfm():
            ok = self.engine.load_custom_dfm(file_path)
            if ok:
                self.active_face_name = f"🧠 {name}"
                self.swap_enabled = True
                self.swap_switch.select()
                self.search_status.configure(text=f"Özel model aktif: '{name}'! ✓", text_color="#2ecc71")
            else:
                self.search_status.configure(text=f"'{name}' modeli yüklenemedi!", text_color="#e74c3c")

        threading.Thread(target=_async_load_dfm, daemon=True).start()

    def _add_to_gallery(self, name: str, img_path: str):
        """Galeriye küçük önizleme kartı ekler (0.00s hızlı geçiş)."""
        if name in self.gallery:
            return

        self.gallery[name] = img_path
        btn = ctk.CTkButton(
            self.gallery_scroll,
            text=f"🎭 {name}",
            width=110, height=38,
            fg_color="#1e272e", hover_color="#0984e3",
            command=lambda n=name, p=img_path: self._switch_face(n, p)
        )
        btn.pack(side="left", padx=4, pady=2)

    def _on_select_my_face(self):
        """Kalıcı butona basıldığında anında kendi doğal yüzüne döner."""
        self.swap_enabled = False
        self.swap_switch.deselect()
        self.active_face_name = "Kendi Yüzün (Doğal)"
        self.search_status.configure(text="👤 Kendi doğal yüzündesin.", text_color="#2ecc71")

    def _switch_face(self, name: str, img_path: str):
        self.search_status.configure(text=f"'{name}' geçiliyor...", text_color="#f39c12")
        def _async_switch():
            ok = self.engine.set_source_image(img_path, cache_key=name)
            if ok:
                self.active_face_name = name
                self.swap_enabled = True
                self.swap_switch.select()
                self.search_status.configure(text=f"'{name}' aktif! ⚡", text_color="#3498db")
            else:
                self.search_status.configure(text=f"'{name}' yüklenemedi.", text_color="#e74c3c")
        threading.Thread(target=_async_switch, daemon=True).start()

    def _on_toggle_swap(self):
        self.swap_enabled = self.swap_switch.get()
        if not self.swap_enabled:
            self.active_face_name = "Kendi Yüzün (Doğal)"
            self.search_status.configure(text="👤 Kendi doğal yüzündesin.", text_color="#2ecc71")
        else:
            if self.engine.source_face is not None:
                self.search_status.configure(text=f"🎭 AI Aktif: {self.active_face_name}", text_color="#3498db")
            else:
                self.search_status.configure(text="Lütfen bir yüz seçin veya arayın.", text_color="#f39c12")

    def _on_toggle_gpen(self):
        self.engine.enable_gpen = self.gpen_switch.get()

    def _on_toggle_occlusion(self):
        self.engine.enable_occlusion = self.occlusion_switch.get()

    def _on_toggle_mirror(self):
        self.mirror_mode = self.mirror_switch.get()

    def _on_toggle_turbo(self):
        self.engine.turbo_mode = self.turbo_switch.get()

    def _on_toggle_color(self):
        self.engine.enable_color_transfer = self.color_switch.get()

    def _on_toggle_mouth(self):
        self.engine.enable_mouth_protect = self.mouth_switch.get()
        self.engine.update_mask()

    def _on_toggle_hand_shield(self):
        self.engine.enable_hand_shield = self.hand_shield_switch.get()

    def _on_toggle_smooth(self):
        self.engine.enable_smoothing = self.smooth_switch.get()

    def _on_slider_sharpen(self, val):
        self.engine.sharpen_strength = float(val)
        self.sharpen_label.configure(text=f"HD Netleştirici (Sharpen): {int(float(val)*100)}%")

    def _on_slider_opacity(self, val):
        self.engine.face_opacity = float(val)
        self.opacity_label.configure(text=f"Yüz Karışım Gücü: {int(float(val)*100)}%")

    def _on_slider_feather(self, val):
        self.engine.mask_feather = int(float(val))
        self.engine.update_mask()
        self.feather_label.configure(text=f"Kenar Yumuşatma: {int(float(val))} px")

    def on_closing(self):
        self.is_running = False
        time.sleep(0.1)
        if self.cam:
            self.cam.stop()
        if self.spout:
            self.spout.release()
        self.destroy()

if __name__ == "__main__":
    app = LiveFaceSwapApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
