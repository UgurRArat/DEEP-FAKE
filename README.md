# 🎭 DEEP FAKE - Gerçek Zamanlı AI Canlı Yüz Değiştirici (Live Face Swap)

DirectML GPU hızlandırması, **InsightFace**, **GPEN-BFR HD Restoratör** ve **Spout2 DirectX VRAM paylaşım mimarisi** üzerine kurulu, sıfır gecikmeli (zero-latency) profesyonel canlı yayın ve kamera yüz değiştirme sistemi.

Sanal kamera (Virtual Camera) gecikmeleri ve kalite kayıpları olmadan doğrudan GPU belleği (DirectX dokuları) üzerinden **OBS Studio** içerisine saniyede 30-60 FPS ile canlı aktarım sağlar.

---

## ✨ Öne Çıkan Özellikler

* ⚡ **DirectML GPU Desteği:** NVIDIA, AMD Radeon ve Intel Arc ekran kartlarında tam donanım hızlandırması.
* 🛡️ **HandShield & Nöral Engel Koruması:** Yüzünüzün önüne eliniz, telefonunuz, gözlüğünüz veya mikrofon geçtiğinde yapay zekanın bozulmasını engelleyen koruma kalkanı.
* 💎 **Evrensel GPEN-BFR-256 HD Restorasyon:** Yüz değiştirme sonrası oluşan bulanıklığı gideren, stüdyo kalitesinde göz, ten ve diş detayları oluşturan nöral iyileştirici.
* 🧔 **Anatomik Çene ve Sakal Koruyucu:** Kullanıcının doğal sakalını, bıyığını ve çene hattını koruyan gelişmiş maskeleme sistemi.
* 🖼️ **Anında Yüz Değiştirme Galerisi:** Tek tıkla gecikmesiz hazır ünlüler veya hedef yüzler arasında geçiş.
* 🔍 **Entegre Görsel Arama Motoru:** İnternetten aradığınız herhangi bir kişinin yüzünü anında programa indirip tek tıkla canlı yayına aktarma.
* 🚀 **Spout2 Sıfır Gecikmeli OBS Köprüsü:** CPU'yu yormadan doğrudan ekran kartı belleğinden OBS'e görüntü aktarımı.

---

## 🖥️ Sistem Gereksinimleri

* **İşletim Sistemi:** Windows 10 / 11 (64-bit)
* **Python:** Python 3.10 veya üzeri
* **Ekran Kartı:** DirectX 12 destekli modern GPU (NVIDIA GTX/RTX serisi veya AMD RX serisi önerilir)
* **Yayın Yazılımı:** OBS Studio (v28 veya üzeri)
* **Webcam:** 720p veya 1080p standart USB / Dahili Kamera

---

## 📦 Adım Adım Kurulum Kılavuzu

### 1. Projeyi Bilgisayarınıza İndirin
```powershell
git clone https://github.com/UgurRArat/DEEP-FAKE.git
cd DEEP-FAKE
```

### 2. Gerekli Python Kütüphanelerini Yükleyin
```powershell
pip install -r requirements.txt
```

### 3. Nöral AI Modellerini İndirin
GitHub dosya boyutu sınırları nedeniyle büyük ONNX yapay zeka modelleri repoya dahil edilmemiştir. Otomatik indirme betiklerini çalıştırarak gerekli tüm modelleri tek seferde indirin:

```powershell
# 1. Ana InSwapper, SCRFD ve Yüz Modellerini İndir
python download_models.py

# 2. Buffalo_l Tanıma Modellerini İndir
python download_buffalo.py
```

---

## 🔌 OBS Studio Spout2 Eklentisi Kurulumu (1 Seferlik)

Uygulamanın görüntüyü doğrudan ekran kartından OBS'e sıfır gecikmeyle aktarabilmesi için OBS'te resmi açık kaynaklı **Spout2** eklentisi bulunmalıdır.

### Kolay Kurulum:
1. [OBS Spout2 Plugin GitHub Releases](https://github.com/Off-World-Live/obs-spout2-plugin/releases) sayfasına gidin.
2. `obs-spout2-plugin-windows-installer.exe` (veya en güncel sürümün installer dosyasını) indirin.
3. OBS Studio kapalıyken kurulumu tamamlayın.
4. *(Alternatif)* Proje klasöründeki PowerShell betiğini de kullanabilirsiniz:
   ```powershell
   powershell -ExecutionPolicy Bypass -File install_plugin.ps1
   ```

---

## 🚀 Uygulamanın Kullanımı

### 1. Programı Başlatın
```powershell
python main_app.py
```

### 2. Arayüz Rehberi
1. **İlk Açılış (Doğal Yüz Modu):**  
   Program ilk açıldığında doğrudan kameranızı açar ve sizi doğal yüzünüzle gösterir. Canlı yayındayken yanlışlıkla farklı bir yüzle başlamanızı önler.
2. **Yüz Seçimi:**  
   * Sol taraftaki **Galeri** üzerinden önbellekteki hazır yüzlere (Elraenn, Beren Saat, Cristiano Ronaldo vb.) tıklayarak **0.0 saniye** gecikmeyle o yüze geçiş yapabilirsiniz.
   * Kendi fotoğrafınızı kullanmak için **"➕ Bilgisayardan Fotoğraf Yükle"** butonuna basarak net bir vesikalık/portre fotoğrafı seçebilirsiniz.
3. **İnternetten Yüz Arama:**  
   * Arama kutusuna istediğiniz kişinin adını yazıp **"Ara"** dediğinizde internetten uygun portre fotoğrafları bulunur. Beğendiğiniz sonuca tıkladığınızda yüz otomatik olarak algılanıp sisteme entegre edilir.
4. **İnce Ayar Sürgüleri:**
   * **Yüz Karışım Oranı (%):** Hedef yüz ile kendi yüzünüz arasındaki geçiş yüzdesi (Varsayılan: %100).
   * **GPEN HD Gücü (%):** Nöral netleştirme ve çözünürlük iyileştirme gücü (Varsayılan: %75).
   * **Kenar Yumuşatma (px):** Yüz sınırlarının doğal teninizle kaynaşma yumuşaklığı.
   * **HandShield Hassasiyeti (%):** El, gözlük veya mikrofon yüzünüzü kapattığında devreye giren korumanın duyarlılığı.

---

## 🎥 OBS Studio Entegrasyonu (Adım Adım)

Uygulamayı açtıktan sonra görüntüyü OBS Studio'ya aktarmak için aşağıdaki adımları uygulayın:

### 1. OBS Studio'yu Açın
OBS Studio'yu başlatın ve yayın/kayıt yapacağınız sahneyi (Scene) seçin.

### 2. Spout2 Kaynağı Ekleyin
1. **Kaynaklar (Sources)** panelinin altındaki **`+` (Ekle)** butonuna tıklayın.
2. Açılan menüden **`Spout2 Capture`** seçeneğini seçin.  
   *(Eğer bu seçenek görünmüyorsa OBS eklentisi henüz yüklenmemiştir veya OBS yeniden başlatılmalıdır).*
3. Kaynağa bir isim verin (Örn: `AI Canlı Yüz`) ve **Tamam**'a basın.

```
[Kaynaklar (+)] ➔ [Spout2 Capture] ➔ [İsim: AI Canlı Yüz]
```

### 3. Kanalı Bağlayın
Açılan özellikler penceresinde:
* **Spout Send name:** Açılır menüden **`LiveFaceSwap_Spout`** seçeneğini işaretleyin.
* **Composite Mode:** `Default` (Varsayılan) bırakın.
* **Tamam** butonuna basın.

🎉 **Tebrikler!** Canlı yüz değiştirici çıktınız doğrudan ekran kartı VRAM'i üzerinden sıfır milisaniye gecikmeyle OBS sahnenize aktarılmıştır!

---

## 🛠️ Sık Karşılaşılan Sorunlar ve Çözümleri (FAQ)

### 1. "Kamera açılamadı!" uyarısı alıyorum
* Başka bir programın (Zoom, Discord, Teams veya başka bir tarayıcı sekmesi) kamerayı meşgul etmediğinden emin olun.
* Birden fazla kameranız varsa `src/config.py` dosyasını açıp `DEFAULT_CAMERA_INDEX = 0` değerini `1` veya `2` olarak değiştirin.

### 2. OBS'te `Spout2 Capture` seçeneği görünmüyor
* OBS Studio'nun 64-bit sürümünü kullandığınızdan emin olun.
* Spout2 eklentisini yükledikten sonra OBS'i tamamen kapatıp yeniden başlatın.

### 3. Yüzüm tam oturmuyor veya kenarlar belirgin görünüyor
* Arayüzdeki **"Kenar Yumuşatma"** sürgüsünü `15-25 px` aralığına getirin.
* Aydınlatmanızın yeterli olduğundan emin olun; yüzünüze doğrudan gelen yumuşak bir ışık yapay zekanın kusursuz çalışmasını sağlar.

---

## ⚖️ Yasal Uyarı ve Kullanım Şartları

Bu proje; eğitim, araştırma, yayıncılık, dijital avatar ve yaratıcı içerik üretimi amacıyla geliştirilmiştir. 
* Kişilerin rızası olmadan kimlik taklidi yapmak veya yanıltıcı/kötü niyetli faaliyetlerde bulunmak etik ve yasal sorumluluk doğurur.
* Projenin kullanımından doğan tüm hukuki sorumluluk kullanıcıya aittir.
