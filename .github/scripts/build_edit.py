#!/usr/bin/env python3
"""Reverse-engineers the reference edit onto a base video.

Inputs:  base video, reference video, scenes.json (holds scene_times_refe)
Writes (back into scenes.json):
  overlay_words       [[t_ref, word], ...]  - big kinetic word per reference cut (OCR)
  captions_whatsapp   [[s, e, text], ...]   - real auto-captions (faster-whisper)
  dur_whatsapp        base duration in seconds
"""
import json, os, re, subprocess, sys, tempfile

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

def probe_dur(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path]).stdout
    return float(out.strip().splitlines()[0])

def ocr_word(video, t, keep):
    try:
        run(["ffmpeg", "-y", "-ss", f"{t+0.15:.3f}", "-i", video, "-frames:v", "1", keep])
        txt = run(["tesseract", keep, "stdout", "-l", "spa", "--psm", "6"]).stdout
    except (subprocess.CalledProcessError, EnvironmentError) as e:
        print(f"  ocr fail @ {t:.2f}: {e.__class__.__name__}")
        return None
    clean = []
    for line in txt.splitlines():
        line = line.strip()
        if "@" in line:
            continue
        line = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ¿?¡!0-9' ]+", " ", line).strip()
        if len(line) >= 2:
            clean.append(line)
    return clean[-1] if clean else None

def split_long(s, e, txt):
    if len(txt) <= 84:
        return [(s, e, txt)]
    words = txt.split()
    mid = max(1, len(words) // 2)
    a, b = " ".join(words[:mid]), " ".join(words[mid:])
    m = s + (e - s) * (len(a) / max(1, len(a) + len(b)))
    return split_long(s, m, a) + split_long(m, e, b)

def transcribe(path):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print("faster-whisper unavailable:", e.__class__.__name__)
        return []
    try:
        model = WhisperModel("tiny")
        segments, _ = model.transcribe(path, vad_filter=True)
        caps = []
        for seg in segments:
            txt = (seg.text or "").strip()
            if not txt:
                continue
            end = min(seg.end, seg.start + 6.5)
            for a, b, t in split_long(seg.start, end, txt):
                caps.append([round(a, 2), round(b, 2), t])
        return caps
    except Exception as e:
        print("whisper failed:", e)
        return []

base, ref, scenes_path = sys.argv[1], sys.argv[2], sys.argv[3]
sc = json.load(open(scenes_path, encoding="utf-8"))
dur_wa = probe_dur(base)
dur_refe = float(sc["dur_refe"])

import tempfile
keep = os.path.join(tempfile.gettempdir(), "ref_frame.png")
overlay = []
for t in sc["scene_times_refe"]:
    if t > dur_refe - 1.2:
        continue
    word = ocr_word(ref, t, keep)
    if word and (not overlay or overlay[-1][1] != word):
        overlay.append([t, word])
        print(f"beat {t:6.2f}s -> {word!r}")

caps = transcribe(base)
print(f"captions: {len(caps)} segments")

sc["overlay_words"] = overlay
sc["captions_whatsapp"] = caps
sc["dur_whatsapp"] = dur_wa
json.dump(sc, open(scenes_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"schedule: {len(overlay)} beats, captions {len(caps)}, base {dur_wa:.2f}s, ref {dur_refe:.2f}s")
