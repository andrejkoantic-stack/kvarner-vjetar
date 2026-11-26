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

# Udio žute/narančaste/crvene u zoni RI–RI_A (Bakarski zaljev) iznad kojeg je "jaka bura"
PRAG_JAKE_BURE = 5.0

# Udio zelene/žute/narančaste/crvene u uskoj zoni Punta (Krk) za detekciju bure
PRAG_BURE_PUNAT = 8.0

# Koordinate za Punat (Krk) — isključuje Bašku i ostatak Krka
ZONA_PUNAT_BOJA = (0.61, 0.55, 0.675, 0.63)
ZONA_PUNAT_SMJER = (0.59, 0.53, 0.69, 0.65)
ZONA_PUNAT_PRIKAZ = (0.52, 0.44, 0.74, 0.70)

def je_dan_za_surfanje(termin):
    # Ljetno računanje vremena (kraj 3. do kraja 10. mjeseca): 06:00–22:00
    # Zimsko računanje vremena (kraj 10. do kraja 3. mjeseca): 06:00–18:00
    je_ljeto = bool(termin.dst())
    krajnji_sat = 22 if je_ljeto else 18
    return 6 <= termin.hour <= krajnji_sat

def pronadji_bazu_slika(sesija):
    r = sesija.get(URL, headers=HEADERS, timeout=20)
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
        r1 = sesija.head(f"{baza}01.{ext}", headers=HEADERS, timeout=10, allow_redirects=True)
        r2 = sesija.head(f"{baza}02.{ext}", headers=HEADERS, timeout=10, allow_redirects=True)
        if r1.status_code == 200 and r2.status_code == 200:
            return [(h, f"{baza}{h:02d}.{ext}") for h in range(1, 73)], 1

        # Rezervno: 3-satne slike (03, 06...)
        r3 = sesija.head(f"{baza}03.{ext}", headers=HEADERS, timeout=10, allow_redirects=True)
        if r3.status_code == 200:
            return [(h, f"{baza}{h:02d}.{ext}") for h in range(3, 75, 3)], 3

    return [], 3

def odredi_start_modela(sesija, prvi_url, sada_utc):
    ref_utc = sada_utc
    try:
        resp = sesija.head(prvi_url, headers=HEADERS, timeout=10, allow_redirects=True)
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

def spoji_slike_mreza(stavke, izlazna_datoteka, sirina_kartice=145, stupaca=2):
    if not stavke:
        return ""

    if len(stavke) > 4 and stupaca == 2:
        stupaca = 3
        sirina_kartice = 115

    prva = stavke[0][1]
    visina_kartice = max(1, int(sirina_kartice * prva.height / prva.width))
    visina_naslova = 18
    razmak = 5

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

def provjeri_boju_bure(img, crop_rel, prag, ukljuci_zelenu=False):
    w, h = img.size
    x1, y1, x2, y2 = crop_rel
    zona = img.crop((int(w * x1), int(h * y1), int(w * x2), int(h * y2)))

    arr = np.array(zona)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    if ukljuci_zelenu:
        # Bilo kakva zelena, žuta, narančasta i crvena boja na DHMZ skali vjetra
        maska_bura = ((hsv[:, :, 0] <= 88) | (hsv[:, :, 0] >= 160)) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 75)
    else:
        # Samo žuta, narančasta i crvena boja na DHMZ skali vjetra
        maska_bura = ((hsv[:, :, 0] <= 35) | (hsv[:, :, 0] >= 160)) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 100)

    udio_bure = float(np.mean(maska_bura) * 100)
    return udio_bure >= prag, round(udio_bure, 1)

def analiziraj_obje_zone(img):
    # 1. Mala lokalna zona (Opa / Preluk) — koristi se za izračun
    ok_lok, vj_lok, sm_lok, _ = izracunaj_zonu(
        img, (0.48, 0.33, 0.60, 0.45), min_strelica=3
    )

    # 2. Velika šira zona (Kvarner: Učka, Opa, RI, Labin, Cres, Krk) — izračun + prikaz
    ok_sir, vj_sir, sm_sir, zona_sir = izracunaj_zonu(
        img, (0.38, 0.28, 0.68, 0.68), min_strelica=5
    )

    # 3. Zona jake bure (Bakarski zaljev / između RI i RI_A)
    jaka_bura, udio_bure = provjeri_boju_bure(img, (0.58, 0.36, 0.67, 0.48), PRAG_JAKE_BURE)

    # Povoljno samo ako ima vjetra, OBJE zone pokazuju sjeverni smjer i nema jake bure
    povoljno = (vj_lok >= 12.0) and ok_lok and ok_sir and (not jaka_bura)
    return povoljno, vj_lok, sm_lok, vj_sir, sm_sir, jaka_bura, udio_bure, zona_sir

def analiziraj_punat(img):
    # Boja (bilo kakva zelena, žuta, narančasta ili crvena) strogo u uskoj zoni Punta
    ima_boju, udio_bure = provjeri_boju_bure(img, ZONA_PUNAT_BOJA, PRAG_BURE_PUNAT, ukljuci_zelenu=True)
    if not ima_boju:
        return False, udio_bure, 0.0, None

    # Smjer: sjeverni kvadrant (SZ, S, SI) oko Punta
    smjer_ok, _, udio_sjever, _ = izracunaj_zonu(img, ZONA_PUNAT_SMJER, min_strelica=3)
    if not smjer_ok:
        return False, udio_bure, udio_sjever, None

    w, h = img.size
    x1, y1, x2, y2 = ZONA_PUNAT_PRIKAZ
    prikaz = img.crop((int(w * x1), int(h * y1), int(w * x2), int(h * y2)))
    return True, udio_bure, udio_sjever, prikaz

def formatiraj_sate(sati):
    if not sati:
        return ""
    rasponi = []
    poc = kraj = sati[0]
    for s in sati[1:]:
        if s == kraj + 1:
            kraj = s
        else:
            rasponi.append(f"{poc}h" if poc == kraj else f"{poc}-{kraj}h")
            poc = kraj = s
    rasponi.append(f"{poc}h" if poc == kraj else f"{poc}-{kraj}h")
    return ", ".join(rasponi)

def kratki_status(status):
    if status == "povoljno ali slabašan vjetar":
        return "slabašan"
    if status == "nema još karata":
        return "nema karata"
    return status

def main():
    sada_utc = datetime.now(timezone.utc)
    sada = sada_utc.astimezone(ZAGREB_TZ)
    pocetak_buducnosti = sada.replace(minute=0, second=0, microsecond=0)

    # Prije 10:00 današnje jutro još nije prošlo -> danas, sutra, prekosutra
    # Od 10:00 nadalje -> sljedeća 3 jutra
    pomak = 0 if sada.hour < 10 else 1
    ciljani_dani = [
        (sada + timedelta(days=pomak)).date(),
        (sada + timedelta(days=pomak + 1)).date(),
        (sada + timedelta(days=pomak + 2)).date(),
    ]

    rezultati_dana = []
    punat_rezultati = []

    try:
        sesija = requests.Session()
        slike, korak = pronadji_bazu_slika(sesija)
        if not slike:
            raise ValueError("Nisu pronađene ALADIN slike.")

        start_modela = odredi_start_modela(sesija, slike[0][1], sada_utc)
        krajnji_sat = 9 if korak == 1 else 10

        # Strukture za prikupljanje podataka u jednom prolazu kroz slike
        preluk_podaci = {
            d: {
                "povoljan": False,
                "jak_vjetar": False,
                "jaka_bura": False,
                "obradjeno": 0,
                "max_vj_lok": 0.0,
                "max_sm_lok": 0.0,
                "max_vj_sir": 0.0,
                "max_sm_sir": 0.0,
                "max_bura": 0.0,
                "slike": [],
            }
            for d in ciljani_dani
        }
        punat_po_danu = {}

        for sat_pomak, url_slike in slike:
            termin = start_modela + timedelta(hours=sat_pomak)
            treba_preluk = (termin.date() in preluk_podaci) and (6 <= termin.hour <= krajnji_sat)
            treba_punat = (termin >= pocetak_buducnosti) and je_dan_za_surfanje(termin)

            if not treba_preluk and not treba_punat:
                continue

            img_r = sesija.get(url_slike, headers=HEADERS, timeout=20)
            if img_r.status_code != 200:
                continue

            img = Image.open(BytesIO(img_r.content)).convert("RGB")
            oznaka = termin.strftime("%H:%M")

            # 1. Analiza za Preluk (jutarnji termini za 3 ciljana dana)
            if treba_preluk:
                p = preluk_podaci[termin.date()]
                p["obradjeno"] += 1
                pov, vl, sl, vs, ss, jaka_bura, udio_bure, z_sir = analiziraj_obje_zone(img)
                p["slike"].append((oznaka, z_sir))

                if jaka_bura:
                    p["jaka_bura"] = True
                if pov:
                    p["povoljan"] = True
                    if vl >= PRAG_JAKOG_VJETRA:
                        p["jak_vjetar"] = True
                p["max_vj_lok"] = max(p["max_vj_lok"], vl)
                p["max_sm_lok"] = max(p["max_sm_lok"], sl)
                p["max_vj_sir"] = max(p["max_vj_sir"], vs)
                p["max_sm_sir"] = max(p["max_sm_sir"], ss)
                p["max_bura"] = max(p["max_bura"], udio_bure)

            # 2. Analiza za Punat (samo dnevni sati: zima 06–18h, ljeto 06–22h)
            if treba_punat:
                bura_punat, udio_p, smjer_p, prikaz_p = analiziraj_punat(img)
                if bura_punat:
                    d_kljuc = termin.date()
                    if d_kljuc not in punat_po_danu:
                        punat_po_danu[d_kljuc] = {
                            "sati": [],
                            "slike": [],
                            "max_bura": 0.0,
                            "max_smjer": 0.0,
                        }
                    pd = punat_po_danu[d_kljuc]
                    pd["sati"].append(termin.hour)
                    pd["slike"].append((oznaka, prikaz_p))
                    pd["max_bura"] = max(pd["max_bura"], udio_p)
                    pd["max_smjer"] = max(pd["max_smjer"], smjer_p)

        # Sastavi rezultate za Preluk
        for idx_dan, datum in enumerate(ciljani_dani, start=1):
            ime_dana = DANI_HR[datum.weekday()]
            p = preluk_podaci[datum]
            if p["jaka_bura"]:
                p["povoljan"] = False

            velika_dat = f"dan{idx_dan}_siroko.png"
            spremljena_velika = spoji_slike_mreza(p["slike"], velika_dat)

            if p["obradjeno"] == 0:
                status_tekst = "nema još karata"
            elif p["jaka_bura"]:
                status_tekst = "jaka bura"
            elif p["povoljan"] and p["jak_vjetar"]:
                status_tekst = "povoljno"
            elif p["povoljan"]:
                status_tekst = "povoljno ali slabašan vjetar"
            else:
                status_tekst = "nepovoljno"

            rezultati_dana.append({
                "dan": ime_dana,
                "datum": datum.strftime("%d.%m.%Y."),
                "povoljno": p["povoljan"],
                "status_tekst": status_tekst,
                "vjetar_posto": p["max_vj_lok"],
                "smjer_posto": p["max_sm_lok"],
                "vjetar_siroko": p["max_vj_sir"],
                "smjer_siroko": p["max_sm_sir"],
                "jaka_bura_posto": p["max_bura"],
                "slika": "",
                "slika_siroko": spremljena_velika
            })

        # Sastavi rezultate za Punat (samo dani kad je bura najavljena po danu)
        for idx_p, datum in enumerate(sorted(punat_po_danu.keys()), start=1):
            ime_dana = DANI_HR[datum.weekday()]
            pd = punat_po_danu[datum]
            punat_dat = f"dan_punat_{idx_p}.png"
            spremljena_punat = spoji_slike_mreza(pd["slike"], punat_dat)
            sati_tekst = formatiraj_sate(pd["sati"])

            punat_rezultati.append({
                "dan": ime_dana,
                "datum": datum.strftime("%d.%m.%Y."),
                "sati": pd["sati"],
                "sati_tekst": sati_tekst,
                "bura_posto": pd["max_bura"],
                "smjer_posto": pd["max_smjer"],
                "slika": spremljena_punat
            })

        # Kratki sažetak za push obavijest (svaki dan u novom redu)
        preluk_kratko = ",\n".join(
            [f"{d['dan'][:3].capitalize()} {kratki_status(d['status_tekst'])}" for d in rezultati_dana]
        )
        sazetak = f"Preluk:\n{preluk_kratko}"

        ima_povoljnog = any(d["povoljno"] for d in rezultati_dana)
        ima_punat_bure = len(punat_rezultati) > 0

        if ima_punat_bure:
            punat_kratko = ",\n".join(
                [f"{p['dan'][:3].capitalize()} bura {p['sati_tekst']}" for p in punat_rezultati]
            )
            sazetak += f"\n\nPunat:\n{punat_kratko}"

        if ima_povoljnog and ima_punat_bure:
            naslov = "Preluk povoljno + Bura u Puntu"
        elif ima_punat_bure:
            naslov = "Bura u Puntu (Krk)"
        elif ima_povoljnog:
            naslov = "Moguće povoljno vrijeme za surfanje"
        else:
            naslov = "Preluk prognoza (3 dana)"

        izlaz = {
            "azurirano": sada.strftime("%d.%m.%Y. %H:%M"),
            "start_modela": start_modela.strftime("%d.%m.%Y. %H:%M"),
            "korak_sati": korak,
            "pronadjeno_slika": len(slike),
            "primjer_url": slike[0][1],
            "naslov": naslov,
            "sazetak": sazetak,
            "dani": rezultati_dana,
            "punat_dani": punat_rezultati
        }

    except Exception as e:
        izlaz = {
            "azurirano": sada.strftime("%d.%m.%Y. %H:%M"),
            "pronadjeno_slika": 0,
            "naslov": "Greška pri dohvatu",
            "sazetak": str(e),
            "dani": [],
            "punat_dani": []
        }

    with open("status.json", "w", encoding="utf-8") as f:
        json.dump(izlaz, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
