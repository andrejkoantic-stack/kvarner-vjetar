import json
import math
import re
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from io import BytesIO
from zoneinfo import ZoneInfo
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
import cv2

URL = "https://meteo.hr/prognoze.php?section=prognoze_model&param=ala_k&el=web_uv10_SENJ_P"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DANI_HR = ["ponedjeljak", "utorak", "srijeda", "četvrtak", "petak", "subota", "nedjelja"]
ZAGREB_TZ = ZoneInfo("Europe/Zagreb")

# Ispod ovog postotka zasjenjenosti oko Opatije (a >= 12%) piše "povoljno ali slabašan vjetar"
PRAG_JAKOG_VJETRA = 60.0

def pronadji_bazu_slika():
    r = requests.get(URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    puni = re.findall(r'https?://[^\s"\'<>\\]+web_uv10_SENJ[^\s"\'<>\\]*\.(?:gif|png|jpg)', r.text, re.IGNORECASE)

    kandidati = []
    if puni:
        m = re.match(r'^(https?://.+?_)\d{2}\.(gif|png|jpg)$', puni[0], re.IGNORECASE)
        if m:
            kandidati.append((m.group(1), m.group(2)))

    for baza in ("https://prognoza.hr/aladinHR/web_uv10_SENJ_", "https://meteo.hr/aladinHR/web_uv10_SENJ_"):
        for ext in ("png", "gif"):
            if (baza, ext) not in kandidati:
                kandidati.append((baza, ext))

    for baza, ext in kandidati:
        # Prvo provjeri postoje li satne slike (01, 02...)
        r1 = requests.head(f"{baza}01.{ext}", headers=HEADERS, timeout=10, allow_redirects=True)
        r2 = requests.head(f"{baza}02.{ext}", headers=HEADERS, timeout=10, allow_redirects=True)
        if r1.status_code == 200 and r2.status_code == 200:
            return [(h, f"{baza}{h:02d}.{ext}") for h in range(1, 73)], 1

        # Rezervno: 3-satne slike (03, 06...)
        r3 = requests.head(f"{baza}03.{ext}", headers=HEADERS, timeout=10, allow_redirects=True)
        if r3.status_code == 200:
            return [(h, f"{baza}{h:02d}.{ext}") for h in range(3, 75, 3)], 3

    return [], 3

def odredi_start_modela(prvi_url, sada_utc):
    ref_utc = sada_utc
    try:
        resp = requests.head(prvi_url, headers=HEADERS, timeout=10, allow_redirects=True)
        lm = resp.headers.get("Last-Modified")
        if lm:
            ref_utc = parsedate_to_datetime(lm).astimezone(timezone.utc)
    except Exception:
        pass

    # ALADIN starta svakih 6 sati (00, 06, 12, 18 UTC), a karte izlaze ~2-4h kasnije
    pomaknuto = ref_utc - timedelta(hours=2)
    run_sat = (pomaknuto.hour // 6) * 6
    run_utc = pomaknuto.replace(hour=run_sat, minute=0, second=0, microsecond=0)

    return run_utc.astimezone(ZAGREB_TZ)

def spoji_slike_mreza(stavke, izlazna_datoteka, sirina_kartice=145):
    if not stavke:
        return ""

    prva = stavke[0][1]
    visina_kartice = max(1, int(sirina_kartice * prva.height / prva.width))
    visina_naslova = 18
    razmak = 5

    stupaca = 2
    redaka = math.ceil(len(stavke) / stupaca)

    ukupna_sirina = stupaca * sirina_kartice + (stupaca - 1) * razmak
    ukupna_visina = redaka * (visina_naslova + visina_kartice) + (redaka - 1) * razmak

    platno = Image.new("RGB", (ukupna_sirina, ukupna_visina), (15, 23, 42))
    crtanje = ImageDraw.Draw(platno)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()

    for idx, (oznaka_sata, slika) in enumerate(stavke):
        red = idx // stupaca
        stupac = idx % stupaca

        if len(stavke) == 1:
            x = (ukupna_sirina - sirina_kartice) // 2
        else:
            x = stupac * (sirina_kartice + razmak)
        y = red * (visina_naslova + visina_kartice + razmak)

        smanjena = slika.resize((sirina_kartice, visina_kartice), Image.Resampling.LANCZOS)
        platno.paste(smanjena, (x, y + visina_naslova))

        bbox = crtanje.textbbox((0, 0), oznaka_sata, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        crtanje.text(
            (x + (sirina_kartice - tw) // 2, y + (visina_naslova - th) // 2 - 1),
            oznaka_sata,
            fill=(226, 232, 240),
            font=font,
        )

    platno.save(izlazna_datoteka)
    return izlazna_datoteka

def izracunaj_zonu(img, crop_rel, min_strelica):
    w, h = img.size
    x1, y1, x2, y2 = crop_rel
    zona = img.crop((int(w * x1), int(h * y1), int(w * x2), int(h * y2)))

    arr = np.array(zona)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    maska_boje = (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 75)
    udio_vjetra = float(np.mean(maska_boje) * 100)

    sivo = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    crno = (sivo < 85).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crno, connectivity=8)

    strelice_sjever = 0
    strelice_jug = 0

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 12 or area > 220 or bh < 6:
            continue

        objekt = (labels[y:y+bh, x:x+bw] == i)
        polovica = bh // 2
        gornja_masa = int(np.sum(objekt[:polovica, :]))
        donja_masa = int(np.sum(objekt[bh - polovica:, :]))

        if donja_masa > gornja_masa * 1.15:
            strelice_sjever += 1
        elif gornja_masa > donja_masa * 1.15:
            strelice_jug += 1

    ukupno = strelice_sjever + strelice_jug
    udio_sjever = (strelice_sjever / ukupno * 100) if ukupno > 0 else 0.0
    smjer_ok = (strelice_sjever >= min_strelica) and (udio_sjever >= 60.0)

    return smjer_ok, round(udio_vjetra, 1), round(udio_sjever, 1), zona

def analiziraj_obje_zone(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # 1. Mala lokalna zona (Opa / Preluk) — koristi se za izračun
    ok_lok, vj_lok, sm_lok, _ = izracunaj_zonu(
        img, (0.48, 0.33, 0.60, 0.45), min_strelica=3
    )

    # 2. Velika šira zona (Kvarner: Učka, Opa, RI, Labin, Cres, Krk) — izračun + prikaz
    ok_sir, vj_sir, sm_sir, zona_sir = izracunaj_zonu(
        img, (0.38, 0.28, 0.68, 0.68), min_strelica=5
    )

    # Povoljno samo ako ima vjetra i OBJE zone pokazuju sjeverni smjer
    povoljno = (vj_lok >= 12.0) and ok_lok and ok_sir
    return povoljno, vj_lok, sm_lok, vj_sir, sm_sir, zona_sir

def main():
    sada_utc = datetime.now(timezone.utc)
    sada = sada_utc.astimezone(ZAGREB_TZ)

    # Prije 10:00 današnje jutro još nije prošlo -> danas, sutra, prekosutra
    # Od 10:00 nadalje -> sljedeća 3 jutra
    pomak = 0 if sada.hour < 10 else 1
    ciljani_dani = [
        (sada + timedelta(days=pomak)).date(),
        (sada + timedelta(days=pomak + 1)).date(),
        (sada + timedelta(days=pomak + 2)).date(),
    ]

    rezultati_dana = []

    try:
        slike, korak = pronadji_bazu_slika()
        if not slike:
            raise ValueError("Nisu pronađene ALADIN slike.")

        start_modela = odredi_start_modela(slike[0][1], sada_utc)
        krajnji_sat = 9 if korak == 1 else 10

        for idx_dan, datum in enumerate(ciljani_dani, start=1):
            ime_dana = DANI_HR[datum.weekday()]
            jutarnje_slike = []

            for sat_pomak, url_slike in slike:
                termin = start_modela + timedelta(hours=sat_pomak)
                if termin.date() == datum and 6 <= termin.hour <= krajnji_sat:
                    jutarnje_slike.append((termin, url_slike))

            dan_povoljan = False
            dan_jak_vjetar = False
            obradjeno_slika = 0
            max_vj_lok = max_sm_lok = 0.0
            max_vj_sir = max_sm_sir = 0.0
            velika_dat = f"dan{idx_dan}_siroko.png"
            velike_za_spajanje = []

            for termin, url_slike in jutarnje_slike:
                img_r = requests.get(url_slike, headers=HEADERS, timeout=20)
                if img_r.status_code != 200:
                    continue
                obradjeno_slika += 1
                pov, vl, sl, vs, ss, z_sir = analiziraj_obje_zone(img_r.content)

                oznaka = termin.strftime("%H:%M")
                velike_za_spajanje.append((oznaka, z_sir))

                if pov:
                    dan_povoljan = True
                    if vl >= PRAG_JAKOG_VJETRA:
                        dan_jak_vjetar = True
                max_vj_lok = max(max_vj_lok, vl)
                max_sm_lok = max(max_sm_lok, sl)
                max_vj_sir = max(max_vj_sir, vs)
                max_sm_sir = max(max_sm_sir, ss)

            spremljena_velika = spoji_slike_mreza(velike_za_spajanje, velika_dat)

            if obradjeno_slika == 0:
                status_tekst = "nema još karata"
            elif dan_povoljan and dan_jak_vjetar:
                status_tekst = "povoljno"
            elif dan_povoljan:
                status_tekst = "povoljno ali slabašan vjetar"
            else:
                status_tekst = "nepovoljno"

            rezultati_dana.append({
                "dan": ime_dana,
                "datum": datum.strftime("%d.%m.%Y."),
                "povoljno": dan_povoljan,
                "status_tekst": status_tekst,
                "vjetar_posto": max_vj_lok,
                "smjer_posto": max_sm_lok,
                "vjetar_siroko": max_vj_sir,
                "smjer_siroko": max_sm_sir,
                "slika": "",
                "slika_siroko": spremljena_velika
            })

        sazetak = ", ".join([f"Preluk {d['dan']} {d['status_tekst']}" for d in rezultati_dana])
        ima_povoljnog = any(d["povoljno"] for d in rezultati_dana)
        naslov = "Moguće povoljno vrijeme za surfanje" if ima_povoljnog else "Preluk prognoza (3 dana)"

        izlaz = {
            "azurirano": sada.strftime("%d.%m.%Y. %H:%M"),
            "start_modela": start_modela.strftime("%d.%m.%Y. %H:%M"),
            "korak_sati": korak,
            "pronadjeno_slika": len(slike),
            "primjer_url": slike[0][1],
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
