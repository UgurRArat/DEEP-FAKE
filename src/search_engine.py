import os
import sys
import re
import urllib.request
import urllib.parse
import json
import time
from typing import Optional, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "assets", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Güvenilir, yüksek çözünürlüklü, cepheden net ünlü portreleri (Bozuk çizim ve yan pozları engeller)
PRESET_URLS = {
    "elon_musk": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Elon_Musk_-_54820081119_%28cropped%29.jpg",
    "scarlett_johansson": "https://upload.wikimedia.org/wikipedia/commons/a/ad/Scarlett_Johansson-8588.jpg",
    "cristiano_ronaldo": "https://upload.wikimedia.org/wikipedia/commons/8/8c/Cristiano_Ronaldo_2018.jpg",
    "brad_pitt": "https://upload.wikimedia.org/wikipedia/commons/4/4c/Brad_Pitt_2019_by_Glenn_Francis.jpg",
    "keanu_reeves": "https://upload.wikimedia.org/wikipedia/commons/3/33/ReevesWebCrop2012.jpg",
    "leonardo_dicaprio": "https://upload.wikimedia.org/wikipedia/commons/2/25/Leonardo_DiCaprio_2014.jpg",
    "gigachad": "https://upload.wikimedia.org/wikipedia/en/thumb/9/96/Meme_Gigachad.jpg/440px-Meme_Gigachad.jpg"
}

class FaceSearchEngine:
    """
    Kişi/Ünlü İsimlerinden Otomatik Yüksek Kaliteli Gerçek Yüz Portresi Bulan Motor.
    Wikipedia Resmi Biyografi Portreleri ve Filtreli Web Arama kullanır.
    """
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        self._clean_bad_cache()

    def _clean_bad_cache(self):
        """Daha önceden hatalı veya çizim olarak inmiş önbellek dosyalarını temizler."""
        bad_files = ["elon_musk.jpg", "scarlett_johansson.jpg", "elraenn.jpg"]
        for f in bad_files:
            p = os.path.join(CACHE_DIR, f)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def search_and_download(self, query: str) -> Optional[str]:
        """
        Verilen isim için en kaliteli portre fotoğrafını bulup indirir.
        Dönen değer: İndirilen görselin yerel dosya yolu.
        """
        clean_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', query).strip().replace(' ', '_').lower()
        if not clean_name:
            return None

        cached_file = os.path.join(CACHE_DIR, f"{clean_name}.jpg")
        if os.path.exists(cached_file) and os.path.getsize(cached_file) > 20000:
            print(f"[Arama] '{query}' önbellekten yüklendi.", flush=True)
            return cached_file

        print(f"[Arama] '{query}' için internette gerçek insan portresi aranıyor...", flush=True)

        # 1. Yöntem: Hazır onaylı URL'ler
        img_url = PRESET_URLS.get(clean_name)

        # 2. Yöntem: Wikipedia Resmi Biyografi Portresi
        if not img_url:
            img_url = self._search_wikipedia_summary(query)

        # 3. Yöntem: DuckDuckGo Gerçek Fotoğraf Arama
        if not img_url:
            img_url = self._search_duckduckgo(f'"{query}" real photo portrait face hd')

        if not img_url:
            print(f"[Arama Hata] '{query}' için uygun bir görsel bulunamadı.", flush=True)
            return None

        # Görseli İndir
        try:
            req = urllib.request.Request(img_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read()
                if len(content) > 10000:
                    with open(cached_file, "wb") as f:
                        f.write(content)
                    print(f"[Arama Başarılı] '{query}' net portresi indirildi: {cached_file}", flush=True)
                    return cached_file
        except Exception as e:
            print(f"[Arama Hata] İndirme başarısız ({img_url}): {e}", flush=True)

        return None

    def _search_wikipedia_summary(self, query: str) -> Optional[str]:
        """Wikipedia resmi sayfasından teyitli portre çeker."""
        formatted_title = urllib.parse.quote(query.replace(" ", "_"))
        for lang in ["en", "tr"]:
            try:
                url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{formatted_title}"
                req = urllib.request.Request(url, headers={"User-Agent": "LiveFaceSwap/1.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    if "originalimage" in data and "source" in data["originalimage"]:
                        return data["originalimage"]["source"]
                    if "thumbnail" in data and "source" in data["thumbnail"]:
                        thumb = data["thumbnail"]["source"]
                        return re.sub(r'/\d+px-', '/640px-', thumb)
            except Exception:
                continue
        return None

    def _search_duckduckgo(self, query: str) -> Optional[str]:
        """DuckDuckGo üzerinden filtreli gerçek fotoğraf arar."""
        try:
            try:
                from ddgs import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.images(query, max_results=8))
                    for res in results:
                        url = res.get("image", "")
                        # Çizim, karikatür ve vektör sitelerini filtrele
                        bad_keywords = ["behance", "deviantart", "vector", "cartoon", "illustration", "drawing", "clipart"]
                        if not any(k in url.lower() for k in bad_keywords):
                            return url
            except ImportError:
                pass
        except Exception:
            pass
        return None
