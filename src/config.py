import os

# --- Kamera Ayarları ---
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
TARGET_FPS = 30

# --- Spout2 Ayarları (OBS Entegrasyonu) ---
SPOUT_SENDER_NAME = "LiveFaceSwap_Spout"

# --- Dizin Yolları ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Klasörlerin var olduğundan emin ol
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
