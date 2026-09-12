# AI Canlı Yüz Değiştirici (Live Face Swapper)

OBS Studio ve Spout2 DirectX paylaşımlı doku mimarisine dayalı, sıfır gecikmeli gerçek zamanlı canlı yüz değiştirme sistemi.

## 📁 Proje Yapısı
```
live_face_swap/
├── models/                  # AI modelleri (SCRFD, InSwapper, BiSeNet)
├── assets/                  # Örnek referans fotoğraflar
├── src/
│   ├── config.py            # Çözünürlük, FPS ve kanal konfigürasyonu
│   ├── camera.py            # Sıfır gecikmeli kamera okuyucu (DirectShow)
│   └── spout_sender.py      # GPU VRAM Spout2 vericisi
├── requirements.txt         # Gerekli kütüphaneler
└── test_spout.py            # Faz 1: Kamera ➔ Spout2 ➔ OBS test betiği
```

## 🔌 OBS Studio Spout2 Eklentisi Kurulumu (1 Seferlik)
OBS'in görüntüyü GPU'dan doğrudan alabilmesi için resmi açık kaynaklı Spout2 eklentisinin kurulu olması gerekir:
1. GitHub'dan `obs-spout2-plugin` Windows Installer dosyasını indirin (veya zip olarak OBS eklenti klasörüne atın).
2. OBS'i açtığınızda **Kaynaklar (+) -> Spout2 Capture** seçeneği görünmelidir.

## 🚀 Faz 1 Testini Çalıştırma
```powershell
pip install -r requirements.txt
python test_spout.py
```
* Açılan kamera penceresinde FPS değerini göreceksiniz.
* OBS'i açıp `Kaynak Ekle (+) -> Spout2 Capture` dediğinizde kanal listesinde `LiveFaceSwap_Spout` görünecek ve kamera anında OBS sahnesine yansıyacaktır!
