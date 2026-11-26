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

# Udio žute/narančaste/crvene u zoni RI–RI_A (Bakarski zaljev) iznad kojeg je "nepovoljno (jaka bura)"
PRAG_JAKE_BURE = 5.0

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

        bbox = crtanje.textbbox((0, 0), oznaka_sata, font
