# Özel Yapay Zeka Modeli Eğitimi & DFM Entegrasyon Rehberi

Bu rehber; hedeflediğin stüdyo kalitesindeki (Elon Musk, yayıncı veya herhangi biri) %100 kusursuz mimik ve cilt benzerliğini sağlayan **Özel Nöral Yüz Modelini (SAEHD / DFM)** kendi RTX 4060 GPU'nda nasıl eğitebileceğini ve projemize nasıl bağlayacağını anlatır.

---

## 1. Neden Özel Model (DFM)?
* **Zero-Shot (InSwapper):** Tek fotoğrafla anında çalışır; ancak 128px olduğu için bizim eklediğimiz HD Frequency Texture & Occlusion katmanıyla desteklenir.
* **Trained Model (DFM / SAEHD):** Tek bir kişi için eğitilir. Hedef kişinin binlerce gülümsemesini, kaş çatmasını, dişlerini ve sakal yönünü ezberlemiştir. 320x320 - 512x512 stüdyo netliğinde 35-60 FPS canlı çalışır.

---

## 2. Kendi RTX 4060 Ekran Kartında Model Eğitme Adımları

### Adım 1: Hedef Kişinin Videosunu Hazırlama
* Hedef kişinin (örn. Elon Musk, arkadaşın veya kendin) 2-3 dakikalık net, 1080p röportaj veya konuşma videosunu indir.
* Farklı açılardan (önden, profilden, gülerken, konuşurken) yaklaşık 1500 - 3000 kare yüz kırpılır.

### Adım 2: Model Eğitimi (SAEHD / DFL Trainer)
* Mimari: **SAEHD (Resolution: 224x224 veya 320x320, Architecture: LIAE veya DF)**
* RTX 4060 Laptop GPU (8GB VRAM) ile batch_size=8 veya 16 seçilerek eğitim başlatılır.
* Yaklaşık 100.000 - 250.000 iterasyonda (yaklaşık 2-4 saat) model fotogerçekçi seviyeye ulaşır.

### Adım 3: DFM (.dfm) Olarak Dışa Aktarma
* Eğitim tamamlandığında: `Export DFM` seçeneğiyle tek bir `.dfm` (veya `.onnx`) dosyası üretilir.
* Örnek dosya adı: `elon_musk_320_rtx4060.dfm`

---

## 3. Modeli Kendi Programımıza Bağlama

1. Kendi programımızı (`main_app.py`) aç.
2. Sol menüdeki **"Özel Eğitilmiş Model (.dfm / .onnx) 🧠"** butonuna bas.
3. Ürettiğin `.dfm` dosyasını seç.
4. Program modeli DirectML üzerinden RTX 4060'ın VRAM'ine yükler ve canlı yayında sıfır gecikmeyle o kişinin birebir nöral ikizini ekrana yansıtır!

---

> [!TIP]
> **Hazır DFM Modelleri:**
> İnternette açık kaynak topluluklar (DFL / DeepFaceLive) tarafından önceden eğitilmiş binlerce ünlü yüzü (Elon Musk, Keanu Reeves, Jim Carrey, Cristiano Ronaldo vb.) `.dfm` formatında ücretsiz olarak mevcuttur. Dilersen saatlerce eğitim beklemeden doğrudan hazır bir `.dfm` dosyasını da programımıza yükleyebilirsin.
