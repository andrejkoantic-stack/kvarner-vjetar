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

# OVDJE UPIŠI SVOJE TAJNO IME KANALA ZA OBAVIJESTI (isto upišeš u aplikaciju ntfy na iPhoneu)
NTFY_KANAL = "preluk-vjetar-andrej-99"

DANI_HR = ["ponedjeljak", "utorak", "srijeda", "četvrtak", "petak", "subota", "nedjelja"]

def dohvati_slike():
    r = requests.get(URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    uzorci = re.findall(r'https?://[^\s"\'<>]+web_uv10_SENJ[^\s"\'<>]+\.(?:gif|png|jpg)', r.text)
    if not uzorci:
        rel = re.findall(r'[^\s"\'<>]*aladin[^\s"\'<>]*web_uv10_SENJ[^\s"\'<>]+\.(?:gif|png|jpg)', r.text)
        uzorci = ["https://meteo.hr/" + u.lstrip("/") if not u.startswith("http") else u for u in rel]
    return list(dict.fromkeys(uzorci))

def analiziraj_zonu_opa(img_bytes, spremi_kao=None):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h = img.size

    # Uski izrez točno oko natpisa "Opa" / Preluk (cca 7-8 najbližih strelica)
    # Koordinate u postocima slike: (lijevo, gore, desno, dolje)
    crop_box = (int(w * 0.48), int(h * 0.28), int(w * 0.60), int(h * 0.40))
    zona = img.crop(crop_box)
    if spremi_kao:
        zona.save(spremi_kao)

    arr = np.array(zona)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    # 1. Ima li vjetra (obojeni pikseli sa skale brzine)
    maska_boje = (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 75)
    udio_vjetra = float(np.mean(maska_boje) * 100)

    # 2. Orijentacija strelica (generalni smjer sjevera -> okomite linije S-J)
    sivo = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(sivo, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(sivo, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    kut = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    jaki_rubovi = mag > 60
    if np.sum(jaki_rubovi) > 0:
        # Raspon za generalni sjeverni smjer (N, NNE, NNW)
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
    sada = datetime.now(timezone(timedelta(hours=1)))  # CET (Hrvatska)
    # ALADIN-HR20 starta u 01:00 ili 13:00 po zimskom vremenu, korak je 3 sata
    start_sat = 1 if sada.hour < 14 else 13
    start_modela = sada.replace(hour=start_sat, minute=0, second=0, microsecond=0)

    ciljani_dani = [
        (sada + timedelta(days=1)).date(),
        (sada + timedelta(days=2)).date(),
        (sada + timedelta(days=3)).date(),
    ]

    rezultati_dana = []

    try:
        slike = dohvati_slike()
        for idx_dan, datum in enumerate(ciljani_dani, start=1):
            ime_dana = DANI_HR[datum.weekday()]
            jutarnje_slike = []

            for i, url_slike in enumerate(slike):
                termin = start_modela + timedelta(hours=i * 3)
                if termin.date() == datum and 6 <= termin.hour <= 10:
                    jutarnje_slike.append((termin, url_slike))

            # Ako model još nema 3. dan do kraja, uzmi najbližu dostupnu sliku
            dan_povoljan = False
            max_vjetar = 0.0
            max_smjer = 0.0
            slika_datoteka = f"dan{idx_dan}.png"

            for j, (termin, url_slike) in enumerate(jutarnje_slike):
                img_r = requests.get(url_slike, headers=HEADERS, timeout=20)
                img_r.raise_for_status()
                spremi = slika_datoteka if j == 0 else None
                pov, vj, sm = analiziraj_zonu_opa(img_r.content, spremi_kao=spremi)
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
                "slika": slika_datoteka if jutarnje_slike else ""
            })

        sazetak = ", ".join([f"Preluk {d['dan']} {d['status_tekst']}" for d in rezultati_dana])
        ima_povoljnog = any(d["povoljno"] for d in rezultati_dana)
        naslov = "Moguće povoljno vrijeme za surfanje" if ima_povoljnog else "Preluk prognoza (3 dana)"

        posalji_obavijest(naslov, sazetak)

        izlaz = {
            "azurirano": sada.strftime("%d.%m.%Y. %H:%M"),
            "naslov": naslov,
            "sazetak": sazetak,
            "dani": rezultati_dana
        }

    except Exception as e:
        izlaz = {
            "azurirano": sada.strftime("%d.%m.%Y. %H:%M"),
            "naslov": "Greška pri dohvatu",
            "sazetak": str(e),
            "dani": []
        }

    with open("status.json", "w", encoding="utf-8") as f:
        json.dump(izlaz, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
