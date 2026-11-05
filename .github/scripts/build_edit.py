#!/usr/bin/env python3
"""Reverse-engineered speech-cut editor.

faster-whisper transcribes the base video (raw PCM decoded by ffmpeg, no
PyAV) with word-level timestamps. From the transcript this GENERATES a
static, contract-compliant wa-edit.html implementing real jump-cut editing:

  - continuous footage split into sequential <video> clips data-media-start
    trims snapped to speech beats (the picture actually CUTS)
  - camera punch (scale 1.14 -> 1) on every cut + alternating side jerks
  - one kinetic word per segment: the speaker's own longest word
  - 3D purple extruded captions with the segment phrase
  - subtle 2-frame white flash every 3rd cut, tick SFX per cut, whoosh 1/4

Writes back into scenes.json: captions_whatsapp, words_flat, cut_times,
kinetic_words, dur_whatsapp.
"""
import json, os, re, subprocess, sys, tempfile, wave

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

def probe_dur(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path]).stdout
    return float(out.strip().splitlines()[0])

def split_long(s, e, txt):
    if len(txt) <= 84:
        return [(s, e, txt)]
    words = txt.split()
    mid = max(1, len(words) // 2)
    a, b = " ".join(words[:mid]), " ".join(words[mid:])
    m = s + (e - s) * (len(a) / max(1, len(a) + len(b)))
    return split_long(s, m, a) + split_long(m, e, b)

def transcribe(path):
    """Returns (phrase_caps, flat_words) = [[s,e,txt]...], [[s,e,word]...]."""
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except Exception as e:
        print("faster-whisper unavailable:", e.__class__.__name__)
        return [], []
    try:
        # decode the audio to raw PCM ourselves; faster-whisper's internal
        # av.decode path breaks on this host (metadata_errors kwarg)
        wav = os.path.join(tempfile.gettempdir(), "base_audio.wav")
        run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-v", "error", wav])
        with wave.open(wav, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        model = WhisperModel("tiny")
        try:
            segments, _ = model.transcribe(audio, language="es", vad_filter=True,
                                           word_timestamps=True)
            got_word_ts = True
        except TypeError:
            segments, _ = model.transcribe(audio, language="es", vad_filter=True)
            got_word_ts = False
        caps, flat = [], []
        for seg in segments:
            txt = (seg.text or "").strip()
            if txt:
                end = min(seg.end, seg.start + 6.5)
                for a, b, t in split_long(seg.start, end, txt):
                    caps.append([round(a, 2), round(b, 2), t])
            if got_word_ts and seg.words:
                for wd in seg.words:
                    word = (wd.word or "").strip()
                    if word and wd.end > wd.start:
                        flat.append([wd.start, wd.end, word])
            elif txt:
                # approximate word positions spread across the segment
                end = min(seg.end, seg.start + 6.5)
                ws = txt.split()
                span = end - seg.start
                per = span / max(1, len(ws))
                for k, wword in enumerate(ws):
                    flat.append([round(seg.start + k * per, 2),
                                 round(seg.start + (k + 1) * per, 2), wword])
        return caps, flat
    except Exception as e:
        import traceback
        print("whisper failed:", e)
        traceback.print_exc()
        return [], []

def build_cuts(flat, dur):
    """Partition [0, dur] into jump-cut segments snapped to speech beats."""
    targets = [1.35, 1.9, 1.15, 1.65, 2.15]
    cuts = [0.0]
    ti = 0
    if not flat:
        t = 0.0
        while dur - t > 1.9:
            t += 1.7
            cuts.append(round(t, 3))
        cuts.append(round(dur, 3))
        return [[cuts[k], cuts[k + 1]] for k in range(len(cuts) - 1)]
    while True:
        prev = cuts[-1]
        if dur - prev <= 2.2:
            break
        target = targets[ti % 5]
        ti += 1
        nxt = None
        for s, e, wd in flat:
            if s >= prev + target:
                nxt = s
                break
        if nxt is None or nxt > prev + 3.2:
            nxt = prev + 3.2          # long silence: hard cut anyway
        nxt = min(nxt, dur - 0.4)
        if nxt <= prev + 0.5:
            nxt = prev + 0.5
        cuts.append(round(nxt, 3))
    cuts.append(round(dur, 3))
    return [[cuts[k], cuts[k + 1]] for k in range(len(cuts) - 1)]

def clean_word(wd):
    lw = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ¿¡!]", "", wd)
    return lw

def seg_content(flat, a, b):
    """(kinetic_word, phrase) inside segment [a, b)."""
    inseg = [w for w in flat if w[0] >= a - 0.02 and w[0] < b - 0.02]
    best = None
    for s, e, wd in inseg:
        lw = clean_word(wd)
        if len(lw) >= 4 and (best is None or len(lw) > len(best)):
            best = lw
    phrase = " ".join(clean_word(w) for s, e, w in inseg
                      if clean_word(w)).strip(" ¡!?¿")
    return best, phrase

def esc(x):
    return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def gen_clips_html(cuts):
    out = []
    for k, (a, b) in enumerate(cuts):
        dur = max(0.2, b - a - 0.02)
        out.append(f'    <video id="seg{k}" class="clip" data-start="{a:.3f}" '
                   f'data-duration="{dur:.3f}" data-media-start="{a:.3f}" '
                   f'src="whatsapp.mp4" playsinline preload="auto" data-has-audio="true"></video>')
    return "\n".join(out)

def gen_words_html(segs_words):
    return "\n".join(f'      <div class="word" id="w{i}">{esc(w)}</div>'
                     for i, (w, _) in enumerate([(w, p) for w, p in segs_words]) if w)

def gen_caps_html(segs_words, cuts):
    out = []
    idx = 0
    for i, ((w, phrase), (a, b)) in enumerate(zip(segs_words, cuts)):
        if not phrase or b - a < 0.45:
            continue
        out.append(f'      <div class="cap" id="cap{idx}">{esc(phrase)}</div>')
        idx += 1
    return "\n".join(out)

def gen_sfx_html(cuts):
    out = []
    for k, (a, b) in enumerate(cuts[:-1]):
        out.append(f'      <audio id="tick{k}" class="clip" data-start="{a:.3f}" '
                   f'data-duration="0.12" data-volume="0.28" src="assets/sfx/tick.wav"></audio>')
        if k % 4 == 0:
            out.append(f'      <audio id="whoosh{k}" class="clip" data-start="{a:.3f}" '
                       f'data-duration="0.65" data-volume="0.5" src="assets/sfx/whoosh.wav"></audio>')
    return "\n".join(out)

def gen_tweens(cuts, segs_words, dur):
    lines = ["tl.to('#bar',{scaleX:1,duration:%.3f,ease:'none'},0);" % dur]
    wi = 0
    ci = 0
    for i, ((w, phrase), (a, b)) in enumerate(zip(segs_words, cuts)):
        nxt = cuts[i + 1][0] if i + 1 < len(cuts) else dur
        # camera punch on every cut
        lines.append(f"tl.fromTo('#vwrap',{{scale:1.14}},{{scale:1,duration:0.24,ease:'power3.out'}},{a:.3f});")
        if i % 3 == 0:
            lines.append(f"tl.fromTo('#vwrap',{{x:12}},{{x:0,duration:0.34,ease:'power2.out'}},{a:.3f});")
        elif i % 5 == 2:
            lines.append(f"tl.fromTo('#vwrap',{{x:-12}},{{x:0,duration:0.34,ease:'power2.out'}},{a:.3f});")
        # subtle white flash every 3rd cut
        if i % 3 == 0:
            lines.append(f"tl.fromTo('#flash',{{opacity:0}},{{opacity:0.55,duration:0.055,ease:'none'}},{a:.3f});")
            lines.append(f"tl.to('#flash',{{opacity:0,duration:0.14,ease:'power1.out'}},{a+0.055:.3f});")
        # kinetic word: one at a time until the next cut
        if w:
            show = min(1.05, max(0.35, nxt - a - 0.08))
            lines.append(f"tl.fromTo('#w{wi}',{{opacity:0,y:30,scale:0.94}},{{opacity:1,y:0,scale:1,duration:0.18,ease:'power4.out'}},{a:.3f});")
            lines.append(f"tl.to('#w{wi}',{{opacity:0,y:-18,scale:0.97,duration:0.18,ease:'power4.in'}},{a+show:.3f});")
            wi += 1
        # 3D caption for the phrase
        if phrase and b - a >= 0.45:
            lines.append(f"tl.fromTo('#cap{ci}',{{opacity:0,y:24}},{{opacity:1,y:0,duration:0.2,ease:'power4.out'}},{a:.3f});")
            lines.append(f"tl.to('#cap{ci}',{{opacity:0,y:-14,duration:0.2,ease:'power4.in'}},{max(a+0.3, nxt-0.22):.3f});")
            ci += 1
    return "\n".join(lines)

COMPO_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>WhatsApp Edit - Speech Jump Cuts</title>
<style>
html,body{margin:0;height:100%;background:#000;overflow:hidden}
#main{position:absolute;inset:0;background:#000;overflow:hidden}
#vwrap{position:absolute;inset:0;transform-origin:50% 40%;z-index:1}
#vwrap video{position:absolute;left:0;top:0;width:100%;height:100%;object-fit:cover;filter:contrast(1.06) saturate(1.12)}
.words{position:absolute;inset:0;display:grid;place-items:center;pointer-events:none;z-index:3}
.word{font-family:'Archivo Black',Arial,Helvetica,sans-serif;font-size:clamp(40px,11vw,110px);color:#fff;text-transform:uppercase;letter-spacing:1px;text-shadow:0 3px 0 rgba(0,0,0,.65),0 0 30px rgba(0,0,0,.9);opacity:0;text-align:center;max-width:90%}
.caps{position:absolute;left:0;right:0;bottom:11%;display:grid;place-items:center;pointer-events:none;z-index:4}
.cap{font-family:'Archivo Black',Arial,Helvetica,sans-serif;font-size:clamp(22px,5.2vw,58px);color:#fff;opacity:0;text-align:center;max-width:86%;letter-spacing:.4px;line-height:1.18;
text-shadow:1px 1px 0 #8b3dff,2px 2px 0 #8b3dff,3px 3px 0 #6d24e0,4px 4px 0 #5618bd,5px 5px 0 #3f0d99,6px 6px 0 #2c0876,7px 7px 14px rgba(0,0,0,.65),0 0 30px rgba(150,90,255,.4)}
.flash{position:absolute;inset:0;z-index:2;pointer-events:none;opacity:0;background:#fff}
.bar{position:absolute;left:0;right:0;bottom:0;height:6px;background:linear-gradient(90deg,#ff3b5c,#ff9a3c);transform:scaleX(0);transform-origin:left;z-index:5}
</style>
</head>
<body>
<div id="main" data-composition-id="main" data-start="0" data-duration="§§DUR§§" data-width="1080" data-height="1920">
  <div id="vwrap">
§§CLIPS_HTML§§
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
§§TWEENS§§
// the runtime drives each timed <video class="clip"> segment itself
window.__timelines={main:tl};
</script>
</body>
</html>
"""

base, ref, scenes_path = sys.argv[1], sys.argv[2], sys.argv[3]
sc = json.load(open(scenes_path, encoding="utf-8"))
dur_wa = probe_dur(base)

print("transcribing with word timestamps ...")
caps, flat = transcribe(base)
print(f"captions: {len(caps)} phrases, {len(flat)} words")

cuts = build_cuts(flat, dur_wa)
print(f"cut plan: {len(cuts)} segments")

segs_words = [seg_content(flat, a, b) for a, b in cuts]
kinetic = [[a, w] for (a, b), (w, _) in zip(cuts, segs_words) if w]
for a, w in kinetic:
    print(f"  cut {a:6.2f}s -> {w}")

COMPO = (COMPO_TEMPLATE
         .replace("§§DUR§§", f"{dur_wa:.3f}")
         .replace("§§CLIPS_HTML§§", gen_clips_html(cuts))
         .replace("§§WORDS_HTML§§", gen_words_html(segs_words))
         .replace("§§CAPS_HTML§§", gen_caps_html(segs_words, cuts))
         .replace("§§SFX_HTML§§", gen_sfx_html(cuts))
         .replace("§§TWEENS§§", gen_tweens(cuts, segs_words, dur_wa)))
open("wa-edit.html", "w", encoding="utf-8").write(COMPO)
print(f"wa-edit.html written: {len(COMPO)} bytes")

sc["captions_whatsapp"] = caps
sc["words_flat"] = [[round(s, 2), round(e, 2), w] for s, e, w in flat]
sc["cut_times"] = [round(a, 2) for a, b in cuts]
sc["kinetic_words"] = kinetic
sc["dur_whatsapp"] = dur_wa
json.dump(sc, open(scenes_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"schedule: {len(cuts)} cuts, {len(kinetic)} kinetic words, base {dur_wa:.2f}s")
