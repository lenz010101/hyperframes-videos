#!/usr/bin/env python3
"""Builds the reference-edit schedule for a base video.

Reads scenes.json (holds the reference video's cut times: scene_times_refe),
pipes the reference video through OCR (-L spa) at every reference cut to find
the big overlay word that shows there, maps every beat onto the base video's
own duration, and writes the result back into scenes.json as:
  overlay_words    [[t_ref, word], ...]   (deduped consecutive equal words)
  dur_whatsapp     base video duration in seconds
"""
import json, re, subprocess, sys, tempfile, os

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)

def probe_dur(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path]).stdout
    return float(out.strip().splitlines()[0])

def ocr_word(video, t, keep):
    # frame OCR: keep only plausible lines, last meaningful line = the big word
    try:
        run(["ffmpeg", "-y", "-ss", f"{t+0.15:.3f}", "-i", video,
             "-frames:v", "1", keep],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        txt = run(["tesseract", keep, "stdout", "-l", "spa", "--psm", "6"]).stdout
    except subprocess.CalledProcessError:
        return None
    clean = []
    for line in txt.splitlines():
        line = line.strip()
        if "@" in line:            # watermark handles
            continue
        line = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ¿?¡!0-9' ]+", " ", line).strip()
        if len(line) >= 2:
            clean.append(line)
    if not clean:
        return None
    return clean[-1]

base, ref, scenes_path = sys.argv[1], sys.argv[2], sys.argv[3]
sc = json.load(open(scenes_path, encoding="utf-8"))
dur_wa = probe_dur(base)
dur_refe = float(sc["dur_refe"])

keep = os.path.join(tempfile.gettempdir(), "ref_frame.png")
overlay = []
for t in sc["scene_times_refe"]:
    if t > dur_refe - 1.2:
        continue
    word = ocr_word(ref, t, keep)
    if word and (not overlay or overlay[-1][1] != word):
        overlay.append([t, word])
        print(f"beat {t:6.2f}s -> {word!r}")

sc["overlay_words"] = overlay
sc["dur_whatsapp"] = dur_wa
json.dump(sc, open(scenes_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"schedule: {len(overlay)} beats, base {dur_wa:.2f}s, ref {dur_refe:.2f}s")
