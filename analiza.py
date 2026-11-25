import json
import re
from datetime import datetime, timezone, timedelta
from io import BytesIO
import numpy as np
import requests
from PIL import Image
import cv2

URL = "https://meteo.hr/prognoze.php?section=prognoze_model&param=ala_k&el=web_uv10_SENJ_P"
HEADERS = {"User-Agent": "Mozilla/5.0"}
NTFY_KANAL = "preluk-vjetar-andrej-99"
DANI_HR = ["ponedjeljak", "utorak", "srijeda", "četvrtak", "petak", "subota", "nedjelja"]

def pronadji_bazu_slika():
    # 1. Pokušaj naći točan puni URL slike u HTML-u stranice
    r = requests.get(URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    puni = re.findall(r'https?://[^\s"\'<>\\]+web_uv10_SENJ[^\s"\'<>\\]*\.(?:gif|png|jpg)', r.text, re.IGNORECASE)
    if puni:
        return list(dict.fromkeys(puni))

    # 2. Ako su putanje skrivene u JavaScriptu, isprobaj poznate DHMZ putanje za ALADIN
    kandidati_baze = [
        "https://prognoza.hr/aladinHR/web_uv10_SENJ_",
        "https://meteo.hr/aladinHR/web_uv10_SENJ_",
        "https://meteo.hr/data/aladinHR/web_uv10_SENJ_",
    ]
    for baza in kandidati_baze:
        for ext in ("gif", "png"):
            test_url = f"{baza}03.{ext}"
            resp = requests.head(test_url, headers=HEADERS, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                return [f"{baza}{h:02d}.{ext}" for h in range(3, 75, 3)]

    return []

def analiziraj_zonu_opa(img_bytes, spremi_kao=None):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h = img.size

    # Uski izrez oko natpisa "Opa" / Preluk
    crop_box = (int(w * 0.48), int(h * 0.28), int(w * 0.60), int(h * 0.40))
    zona = img.crop(crop_box)
    if spremi_kao:
        zona.save(spremi_kao)

    arr = np.array(zona)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    maska_boje = (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 75)
    udio_vjetra = float(np.mean(maska_boje) * 100)

    sivo = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(sivo, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(sivo, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    kut = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    jaki_rubovi = mag > 60
    if np.sum(jaki_rubovi) > 0:
        sjever_jug = ((kut < 30) | (kut > 150)) & jaki_rubovi
        udio_smjera = float(np.sum(sjever_jug) / np.sum(jaki_rubovi) * 100)
    else:
        udio_smjera = 0.0

    povoljno = (udio_vjetra >= 12.0) and (udio_smjera >= 35.0)
    return povoljno, round(udio_vjetra, 1), round(udio_smjera, 1)

def posalji_obavijest(naslov, tekst):
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_KANAL}",
            data=tekst.encode("utf-8"),
            headers={"Title": naslov.encode("utf-8"), "Priority": "default"},
            timeout=15
        )
    except Exception as e:
        print("Greška slanja obavijesti:", e)

def main():
    sada = datetime.now(timezone(timedelta(hours=1)))
    start_sat = 1 if sada.hour < 14 else 13
    start_modela = sada.replace(hour=start_sat, minute=0, second=0, microsecond=0)

    ciljani_dani = [
        (sada + timedelta(days=1)).date(),
        (sada + timedelta(days=2)).date(),
        (sada + timedelta(days=3)).date(),
    ]

    rezultati_dana = []

    try:
        slike = pronadji_bazu_slika()
        if not slike:
            raise ValueError("Nijedna poznata DHMZ adresa za ALADIN slike nije vratila 200 OK.")

        for idx_dan, datum in enumerate(ciljani_dani, start=1):
            ime_dana = DANI_HR[datum.weekday()]
            jutarnje_slike = []

            for i, url_slike in enumerate(slike, start=1):
                termin = start_modela + timedelta(hours=i * 3)
                if termin.date() == datum and 6 <= termin.hour <= 10:
                    jutarnje_slike.append((termin, url_slike))

            dan_povoljan = False
            max_vjetar = 0.0
            max_smjer = 0.0
            slika_datoteka = f"dan{idx_dan}.png"
            spremljena_slika = ""

            for j, (termin, url_slike) in enumerate(jutarnje_slike):
                img_r = requests.get(url_slike, headers=HEADERS, timeout=20)
                if img_r.status_code != 200:
                    continue
                spremi = slika_datoteka if not spremljena_slika else None
                pov, vj, sm = analiziraj_zonu_opa(img_r.content, spremi_kao=spremi)
                if spremi:
                    spremljena_slika = slika_datoteka
                if pov:
                    dan_povoljan = True
                max_vjetar = max(max_vjetar, vj)
                max_smjer = max(max_smjer, sm)

            status_tekst = "povoljno" if dan_povoljan else "nepovoljno"
            rezultati_dana.append({
                "dan": ime_dana,
                "datum": datum.strftime("%d.%m.%Y."),
                "povoljno": dan_povoljan,
                "status_tekst": status_tekst,
                "vjetar_posto": max_vjetar,
                "smjer_posto": max_smjer,
                "slika": spremljena_slika
            })

        sazetak = ", ".join([f"Preluk {d['dan']} {d['status_tekst']}" for d in rezultati_dana])
        ima_povoljnog = any(d["povoljno"] for d in rezultati_dana)
        naslov = "Moguće povoljno vrijeme za surfanje" if ima_povoljnog else "Preluk prognoza (3 dana)"

        posalji_obavijest(naslov, sazetak)

        izlaz = {
            "azurirano": sada.strftime("%d.%m.%Y. %H:%M"),
            "pronadjeno_slika": len(slike),
            "primjer_url": slike[0],
            "naslov": naslov,
            "sazetak": sazetak,
            "dani": rezultati_dana
        }

    except Exception as e:
        izlaz = {
            "azurirano": sada.strftime("%d.%m.%Y. %H:%M"),
            "pronadjeno_slika": 0,
            "naslov": "Greška pri dohvatu",
            "sazetak": str(e),
            "dani": []
        }

    with open("status.json", "w", encoding="utf-8") as f:
        json.dump(izlaz, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
