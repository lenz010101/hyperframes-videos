#!/usr/bin/env python3
"""Reverse-engineers the reference edit onto a base video.

Reads scenes.json (holds scene_times_refe), OCRs the big overlay word per
reference cut, transcribes the base video's speech with faster-whisper, then
GENERATES a static contract-compliant wa-edit.html (all timed elements written
in the markup, tweens literal, no runtime fetch).
Writes back into scenes.json: overlay_words, captions_whatsapp, dur_whatsapp.
"""
import json, os, re, subprocess, sys, tempfile, wave

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

def probe_dur(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path]).stdout
    return float(out.strip().splitlines()[0])

def ocr_word(video, t, keep):
    try:
        run(["ffmpeg", "-y", "-ss", f"{t+0.15:.3f}", "-i", video,
             "-vf", "crop=iw*0.92:ih*0.5:(iw-iw*0.92)/2:ih*0.28,scale=iw*1.5:ih*1.5:flags=lanczos",
             "-frames:v", "1", keep])
        txt = run(["tesseract", keep, "stdout", "-l", "spa", "--psm", "6"]).stdout
    except (subprocess.CalledProcessError, EnvironmentError) as e:
        print(f"  ocr fail @ {t:.2f}: {e.__class__.__name__}")
        return None
    cands = []
    for line in txt.splitlines():
        line = line.strip()
        if "@" in line or "?" in line:
            continue
        line = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ¡!]+", " ", line).strip()
        letters = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", "", line)
        if len(letters) >= 4 and len(line) <= 18:
            # kinetic strips are short; prefer all-caps single words
            cands.append(((1 if letters.isupper() else 0, -len(line)), line))
    return max(cands, key=lambda x: x[0])[1] if cands else None

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
        import numpy as np
        from faster_whisper import WhisperModel
    except Exception as e:
        print("faster-whisper unavailable:", e.__class__.__name__)
        return []
    try:
        # decode the audio to raw PCM ourselves; faster-whisper's internal
        # av.decode path breaks on this host (metadata_errors kwarg)
        wav = os.path.join(tempfile.gettempdir(), "base_audio.wav")
        run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-v", "error", wav])
        with wave.open(wav, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        model = WhisperModel("tiny")
        segments, _ = model.transcribe(audio, language="es", vad_filter=True)
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
        import traceback
        print("whisper failed:", e)
        traceback.print_exc()
        return []

def esc(x):
    return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def gen_words_html(overlay):
    return "\n".join(f'      <div class="word" id="w{i}">{esc(w)}</div>'
                     for i, (t, w) in enumerate(overlay))

def gen_caps_html(caps):
    return "\n".join(f'      <div class="cap" id="cap{i}">{esc(t)}</div>'
                     for i, (s, e, t) in enumerate(caps))

def gen_sfx_html(overlay, caps, dur_wa, dur_refe):
    out = []
    for i, (t, _) in enumerate(overlay):
        tw = t * dur_wa / dur_refe
        if tw >= dur_wa - 0.6:
            continue
        out.append(f'      <audio id="sfwho{i}" class="clip" data-start="{tw:.3f}" data-duration="0.65" data-volume="0.5" src="assets/sfx/whoosh.wav"></audio>')
        if i % 8 == 0:
            out.append(f'      <audio id="sfding{i}" class="clip" data-start="{tw:.3f}" data-duration="0.35" data-volume="0.35" src="assets/sfx/ding.wav"></audio>')
    for i, (s, e, _) in enumerate(caps):
        if s >= dur_wa - 0.2:
            continue
        out.append(f'      <audio id="sftick{i}" class="clip" data-start="{s:.3f}" data-duration="0.12" data-volume="0.3" src="assets/sfx/tick.wav"></audio>')
    return "\n".join(out)

def gen_words(overlay, dur_wa, dur_refe):
    lines = []
    for i, (t, _) in enumerate(overlay):
        tw = t * dur_wa / dur_refe
        if tw >= dur_wa - 0.6:
            continue
        nxt = dur_wa
        if i + 1 < len(overlay):
            nxt = overlay[i + 1][0] * dur_wa / dur_refe
        show = min(1.2, max(0.4, nxt - tw - 0.05))
        lines.append(f"tl.fromTo('#w{i}',{{opacity:0,y:34,scale:0.92}},{{opacity:1,y:0,scale:1,duration:0.25,ease:'power4.out'}},{tw:.3f});")
        lines.append(f"tl.to('#w{i}',{{opacity:0,y:-22,scale:0.95,duration:0.25,ease:'power4.in'}},{tw+show:.3f});")
        lines.append(f"tl.fromTo('#vwrap',{{scale:1}},{{scale:1.12,duration:0.12,ease:'power2.out'}},{tw:.3f});")
        lines.append(f"tl.to('#vwrap',{{scale:1,duration:0.5,ease:'power1.inOut'}},{tw+0.3:.3f});")
        if i % 4 == 0:
            lines.append(f"tl.fromTo('#flash',{{opacity:0}},{{opacity:0.75,duration:0.1,ease:'power2.out'}},{tw:.3f});")
            lines.append(f"tl.to('#flash',{{opacity:0,duration:0.3,ease:'power1.in'}},{tw+0.1:.3f});")
            lines.append(f"tl.fromTo('#vwrap',{{x:18}},{{x:0,duration:0.42,ease:'power3.out'}},{tw+0.05:.3f});")
    return "\n".join(lines)

def gen_caps(caps, dur_wa):
    lines = []
    for i, (s, e, _) in enumerate(caps):
        if e <= s + 0.05 or s >= dur_wa - 0.2:
            continue
        e = min(e, dur_wa - 0.05)
        lines.append(f"tl.fromTo('#cap{i}',{{opacity:0,y:26}},{{opacity:1,y:0,duration:0.28,ease:'power4.out'}},{s:.3f});")
        lines.append(f"tl.to('#cap{i}',{{opacity:0,y:-16,duration:0.24,ease:'power4.in'}},{e-0.24:.3f});")
    return "\n".join(lines)

COMPO_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>WhatsApp Edit Match Refe - Motion Graphics</title>
<style>
html,body{margin:0;height:100%;background:#000;overflow:hidden}
#main{position:absolute;inset:0;background:#000;overflow:hidden}
#vwrap{position:absolute;inset:0;transform-origin:50% 50%;z-index:1}
video{width:100%;height:100%;object-fit:cover;transform:translate3d(0,0,0);filter:contrast(1.08) saturate(1.15) brightness(1.02)}
.words{position:absolute;inset:0;display:grid;place-items:center;pointer-events:none;z-index:3}
.word{font-family:'Archivo Black',Arial,Helvetica,sans-serif;font-size:clamp(42px,12vw,118px);color:#fff;text-transform:uppercase;letter-spacing:1px;text-shadow:0 3px 0 rgba(0,0,0,.6),0 0 34px rgba(0,0,0,.85);opacity:0;text-align:center;max-width:88%}
.caps{position:absolute;left:0;right:0;bottom:9%;display:grid;place-items:center;pointer-events:none;z-index:4}
.cap{font-family:'Archivo Black',Arial,Helvetica,sans-serif;font-size:clamp(24px,5.6vw,62px);color:#fff;opacity:0;text-align:center;max-width:86%;letter-spacing:.4px;line-height:1.18;
text-shadow:1px 1px 0 #8b3dff,2px 2px 0 #8b3dff,3px 3px 0 #6d24e0,4px 4px 0 #5618bd,5px 5px 0 #3f0d99,6px 6px 0 #2c0876,7px 7px 14px rgba(0,0,0,.65),0 0 30px rgba(150,90,255,.4)}
.flash{position:absolute;inset:0;z-index:2;pointer-events:none;opacity:0;background:linear-gradient(115deg,#ff3b5c 0%,#b23bff 45%,#3bd2ff 100%)}
.bar{position:absolute;left:0;right:0;bottom:0;height:6px;background:linear-gradient(90deg,#ff3b5c,#ff9a3c);transform:scaleX(0);transform-origin:left;z-index:5}
</style>
</head>
<body>
<div id="main" data-composition-id="main" data-start="0" data-duration="§§DUR_WA§§" data-width="1080" data-height="1920">
  <div id="vwrap">
    <video id="baseVideo" class="clip" data-start="0" data-duration="§§DUR_WA§§" src="whatsapp.mp4" playsinline preload="auto" data-has-audio="true"></video>
  </div>
  <div class="flash" id="flash"></div>
  <div class="words" id="words">
§§WORDS_HTML§§
  </div>
  <div class="caps" id="caps">
§§CAPS_HTML§§
  </div>
§§SFX_HTML§§
  <div class="bar" id="bar"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script>
const tl=gsap.timeline({paused:true});
tl.to('#bar',{scaleX:1,duration:§§DUR_WA§§,ease:'none'},0);
// baseVideo is a timed class="clip" element: the runtime per-frame seeks it
§§WORDS_TWEENS§§
§§CAPS_TWEENS§§
window.__timelines={main:tl};
</script>
</body>
</html>
"""

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
    if not word:
        continue
    if re.fullmatch(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ¿?¡!]*", word):
        continue
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

COMPO = (COMPO_TEMPLATE
         .replace("§§DUR_WA§§", f"{dur_wa:.3f}")
         .replace("§§DUR_REFE§§", f"{dur_refe:.3f}")
         .replace("§§WORDS_TWEENS§§", gen_words(overlay, dur_wa, dur_refe))
         .replace("§§CAPS_TWEENS§§", gen_caps(caps, dur_wa))
         .replace("§§WORDS_HTML§§", gen_words_html(overlay))
         .replace("§§CAPS_HTML§§", gen_caps_html(caps))
         .replace("§§SFX_HTML§§", gen_sfx_html(overlay, caps, dur_wa, dur_refe)))
open("wa-edit.html", "w", encoding="utf-8").write(COMPO)
print(f"wa-edit.html written: {len(COMPO)} bytes")
