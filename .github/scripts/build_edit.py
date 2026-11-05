#!/usr/bin/env python3
"""Reverse-engineered reel editor (styled after a montage contact sheet).

faster-whisper transcribes the base video (raw PCM decoded by ffmpeg, no
PyAV) with word-level timestamps. From the transcript this GENERATES a
static, contract-compliant wa-edit.html that reproduces the montage grammar
of the reference edit, using the ONE available talking-head source:

  - continuous footage split into sequential <video> clips (data-media-start
    trims) snapped to speech beats: the picture genuinely cut-cuts every ~2s
  - each segment gets a different LOOK (zoom/crop/offset/mirror/color grade)
    so consecutive cuts read as distinct scenes, like the reference sheet
  - captions: sentence-case white bold with black stroke, centered band,
    the segment's key word rendered BIG inline above the phrase
  - graphic "plates" (colored cards with a big word) cut in for 0.55s,
    standing in for the reference's neon/logo cutaways
  - camera punch (scale 1.14 -> 1) every cut, subtle 2-frame white flashes,
    tick SFX per cut, whoosh every 4th, ding on plates

Writes back into scenes.json: captions_whatsapp, words_flat, cut_times,
kinetic_words, plates, dur_whatsapp.
"""
import json, glob, os, re, subprocess, sys, tempfile, unicodedata, wave

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
        wav = os.path.join(tempfile.gettempdir(), "base_audio.wav")
        run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-v", "error", wav])
        with wave.open(wav, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        model = WhisperModel("small", compute_type="int8")
        prompt = ("Transcripción en español de una joven hablando de participación "
                  "ciudadana, juventud y política en México.")
        try:
            segments, _ = model.transcribe(audio, language="es", vad_filter=True,
                                           word_timestamps=True, initial_prompt=prompt)
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
    targets = [2.0, 2.6, 1.7, 2.3, 2.9]   # reference cadence ~2.2s per scene
    cuts = [0.0]
    ti = 0
    if not flat:
        t = 0.0
        while dur - t > 2.4:
            t += 2.2
            cuts.append(round(t, 3))
        cuts.append(round(dur, 3))
        return [[cuts[k], cuts[k + 1]] for k in range(len(cuts) - 1)]
    while True:
        prev = cuts[-1]
        if dur - prev <= 2.6:
            break
        target = targets[ti % 5]
        ti += 1
        nxt = None
        for s, e, wd in flat:
            if s >= prev + target:
                nxt = s
                break
        if nxt is None or nxt > prev + 3.6:
            nxt = prev + 3.6
        nxt = min(nxt, dur - 0.4)
        if nxt <= prev + 0.6:
            nxt = prev + 0.6
        cuts.append(round(nxt, 3))
    cuts.append(round(dur, 3))
    return [[cuts[k], cuts[k + 1]] for k in range(len(cuts) - 1)]

def clean_word(wd):
    return re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ¿¡!]", "", wd)

def seg_content(flat, a, b):
    """(kinetic_word, phrase) inside segment [a, b)."""
    inseg = [w for w in flat if w[0] >= a - 0.02 and w[0] < b - 0.02]
    best = None
    for s, e, wd in inseg:
        lw = clean_word(wd)
        if len(lw) >= 4 and (best is None or len(lw) > len(best)):
            best = lw
    phrase = " ".join(clean_word(w) for s, e, w in inseg
                      if clean_word(w)).strip(" ¡!?¿.,")
    return best, phrase

def esc(x):
    return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ------------------------------------------------------------- scene looks
# f = CSS filter, s = scale (punch-in), x/y = framing offset,
# fx = mirror flag, t = per-scene color wash (color identity, like the ref)
LOOKS = [
    {"f": "none", "s": 1.03, "x": "0%", "y": "0%", "fx": 1, "t": "rgba(28,64,168,0.14)"},
    {"f": "sepia(0.22) saturate(1.4) contrast(1.12) brightness(1.03)", "s": 1.32, "x": "-6%", "y": "-12%", "fx": 1, "t": "rgba(255,146,40,0.16)"},
    {"f": "saturate(0.95) hue-rotate(-14deg) contrast(1.08) brightness(0.97)", "s": 1.16, "x": "5%", "y": "-4%", "fx": 1, "t": "rgba(0,178,168,0.14)"},
    {"f": "saturate(1.2) contrast(1.22) brightness(0.94)", "s": 1.12, "x": "0%", "y": "-7%", "fx": -1, "t": "rgba(198,30,84,0.16)"},
    {"f": "sepia(0.3) saturate(1.7) contrast(1.28) brightness(0.85) hue-rotate(-18deg)", "s": 1.45, "x": "-4%", "y": "-9%", "fx": 1, "t": "rgba(96,48,210,0.15)"},
    {"f": "grayscale(1) contrast(1.38) brightness(1.04)", "s": 1.52, "x": "0%", "y": "-16%", "fx": 1, "t": None},
    {"f": "none", "s": 1.85, "x": "0%", "y": "-22%", "fx": 1, "t": "rgba(255,110,150,0.12)"},
    {"f": "grayscale(0.5) contrast(1.32) saturate(1.35)", "s": 1.62, "x": "8%", "y": "-10%", "fx": 1, "t": "rgba(22,88,255,0.2)"},
]
SEQ = [0, 1, 2, 6, 3, 0, 4, 1, 5, 2, 0, 7, 3, 1, 4, 2, 5, 0, 3, 2, 1, 4, 0, 5, 2, 3, 1]

def build_seq(n, seq):
    """Look index per segment, never repeating the previous one."""
    out = []
    for k in range(n):
        li = seq[k % len(seq)]
        if k and li == out[-1]:
            for off in range(1, len(seq)):
                cand = seq[(k + off) % len(seq)]
                if cand != out[-1]:
                    li = cand
                    break
        out.append(li)
    return out

def gen_segments_html(cuts, seq):
    seq = build_seq(len(cuts), seq)
    out = []
    for k, (a, b) in enumerate(cuts):
        lk = LOOKS[seq[k]]
        d = max(0.2, b - a - 0.02)
        style = (f"filter:{lk['f']};transform:scale({lk['s']}) "
                 f"translate({lk['x']},{lk['y']}) scaleX({lk['fx']})")
        tint = lk.get("t") or ""
        tint_html = f'<div class="tint" style="background:{tint}"></div>' if tint else ""
        out.append(f'    <div class="lseg" style="{style}">'
                   f'<video id="seg{k}" class="clip" data-start="{a:.3f}" '
                   f'data-duration="{d:.3f}" data-media-start="{a:.3f}" '
                   f'src="whatsapp.mp4" playsinline preload="auto" data-has-audio="true"></video>{tint_html}</div>')
    return "\n".join(out)

def gen_caps_html(segs_words, cuts):
    out, idx = [], 0
    for (w, phrase), (a, b) in zip(segs_words, cuts):
        if not (w or phrase):
            continue
        if b - a < 0.45:
            continue
        body = phrase
        kb = ""
        if w and phrase:
            tokens = phrase.split()
            target = w.lower()
            tokens2 = [t for t in tokens
                       if re.sub(r"[^a-záéíóúüñ]", "", t.lower()) != target]
            body = " ".join(tokens2)
            kb = f'<span class="kb">{esc(w)}</span>'
        out.append(f'      <div class="cap" id="cap{idx}">{kb}{esc(body)}</div>')
        idx += 1
    return "\n".join(out)

def gen_plates(cuts, segs_words, dur):
    """Massive-word graphic cards, in for 0.55s, like the reference cutaways."""
    grads_css = """
.p0{background:linear-gradient(115deg,#ff3b5c 0%,#b23bff 55%,#3bd2ff 100%)}
.p1{background:radial-gradient(circle at 32% 28%,#2532a8,#0a0f3d 72%)}
.p2{background:linear-gradient(160deg,#101010,#3d0a0a 65%,#12010c)}
"""
    html, events = [], []
    idx = 0
    for k, (a, b) in enumerate(cuts[:-1]):
        if k % 5 != 2 or b - a < 1.0:
            continue
        w, _ = segs_words[k]
        word = (w or "").upper()
        if not word:
            continue
        t0 = a + 0.25
        t1 = min(a + 0.8, b - 0.1)
        if t1 <= t0 + 0.3:
            continue
        n = len(word)
        if n <= 8:
            fs = ""
        elif n <= 12:
            fs = " style=\"font-size:clamp(36px,9.5vw,96px)\""
        else:
            fs = " style=\"font-size:clamp(30px,7vw,72px);letter-spacing:.5px\""
        html.append(f'      <div class="plate p{idx % 3}" id="plate{idx}"{fs}>{esc(word)}</div>')
        events.append((round(t0, 3), round(t1, 3), idx))
        idx += 1
    return "\n".join(html), events, grads_css

def gen_sfx_html(cuts, plate_events):
    out = []
    for k, (a, b) in enumerate(cuts[:-1]):
        out.append(f'      <audio id="tick{k}" class="clip" data-start="{a:.3f}" '
                   f'data-duration="0.12" data-volume="0.28" src="assets/sfx/tick.wav"></audio>')
        if k % 4 == 0:
            out.append(f'      <audio id="whoosh{k}" class="clip" data-start="{a:.3f}" '
                       f'data-duration="0.65" data-volume="0.5" src="assets/sfx/whoosh.wav"></audio>')
    for (t0, t1, idx) in plate_events:
        out.append(f'      <audio id="ding{idx}" class="clip" data-start="{t0:.3f}" '
                   f'data-duration="0.35" data-volume="0.4" src="assets/sfx/ding.wav"></audio>')
    return "\n".join(out)

def gen_tweens(cuts, segs_words, dur, plate_events):
    lines = []
    ci = 0
    for i, ((w, phrase), (a, b)) in enumerate(zip(segs_words, cuts)):
        nxt = cuts[i + 1][0] if i + 1 < len(cuts) else dur
        # camera punch on every cut
        lines.append(f"tl.fromTo('#vwrap',{{scale:1.14}},{{scale:1,duration:0.24,ease:'power3.out'}},{a:.3f});")
        if i % 3 == 0:
            lines.append(f"tl.fromTo('#vwrap',{{x:12}},{{x:0,duration:0.34,ease:'power2.out'}},{a:.3f});")
        elif i % 5 == 2:
            lines.append(f"tl.fromTo('#vwrap',{{x:-12}},{{x:0,duration:0.34,ease:'power2.out'}},{a:.3f});")
        # subtle white flash every other cut
        if i % 2 == 0:
            lines.append(f"tl.fromTo('#flash',{{opacity:0}},{{opacity:0.35,duration:0.045,ease:'none'}},{a:.3f});")
            lines.append(f"tl.to('#flash',{{opacity:0,duration:0.12,ease:'power1.out'}},{a+0.045:.3f});")
        # caption for the phrase
        if (w or phrase) and b - a >= 0.45:
            lines.append(f"tl.fromTo('#cap{ci}',{{opacity:0,y:22}},{{opacity:1,y:0,duration:0.18,ease:'power4.out'}},{a:.3f});")
            lines.append(f"tl.to('#cap{ci}',{{opacity:0,y:-12,duration:0.18,ease:'power4.in'}},{max(a+0.3, nxt-0.2):.3f});")
            ci += 1
    for (t0, t1, idx) in plate_events:
        lines.append(f"tl.fromTo('#plate{idx}',{{opacity:0,scale:1.3}},{{opacity:1,scale:1,duration:0.09,ease:'power3.out'}},{t0:.3f});")
        lines.append(f"tl.to('#plate{idx}',{{opacity:0,scale:1.06,duration:0.12,ease:'power2.in'}},{t1-0.12:.3f});")
    return "\n".join(lines)

# ----------------------------------------------------------- motion layer
ACCENTS = ["#ffd643", "#ff5f6d", "#3bd2ff", "#b27bff"]

BURST_SVG = (
    '<svg class="burst" viewBox="-50 -50 100 100">'
    + "".join(
        '<line x1="0" y1="0" x2="{r:.0f}" y2="0" transform="rotate({d:.1f})" />'.format(
            r=44 * (0.55 + 0.45 * (i % 2)), d=i * 22.5)
        for i in range(16)
    )
    + "</svg>"
)

def norm_txt(s):
    t = unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", t)

TOPIC_WORDS = {
    "hands": ["mano", "levantar", "participar", "votar", "propo", "crear", "organizar"],
    "students": ["educac", "escuela", "estudi", "profesor", "clase", "preparac"],
    "phones": ["joven", "red", "internet", "movil", "telefon", "app", "social"],
    "street": ["calle", "infraes", "barrio", "construir", "mejorar", "obra", "pueblo"],
    "participation": ["ciudadan", "polit", "voto", "comunidad", "organiza", "cambios"],
    "talking": ["miedo", "hablar", "expresar", "diran", "decir", "escucha"],
    "community": ["vecin", "conjunto", "todos", "nosotro", "reunir", "junt"],
    "mexico": ["mexico", "pais", "republica", "nacion"],
}

def load_broll():
    """Canonical themed b-roll stills, if present in assets/broll."""
    order = ["hands", "students", "phones", "street", "participation",
             "talking", "community", "mexico"]
    found = {}
    for f in glob.glob("assets/broll/*"):
        key = os.path.splitext(os.path.basename(f))[0].lower()
        if key in order:
            found[key] = f.replace("\\", "/")
    return [(k, found[k]) for k in order if k in found]

def gen_broll(cuts, segs_words):
    """Themed stills: full-frame beats on longer scenes, polaroid cards on shorter."""
    broll = load_broll()
    html, tw, beats = [], [], []
    if not broll:
        return "\n".join(html), "\n".join(tw), beats
    slots = [("6%", "9%", "46%", -6), ("50%", "12%", "44%", 5),
             ("8%", "27%", "44%", 4), ("52%", "25%", "42%", -4)]
    bi = 0
    for k, (a, b) in enumerate(cuts):
        if k % 4 != 2 or b - a < 0.9:
            continue
        p = norm_txt(segs_words[k][1] or "")
        pick = None
        for name, path in broll:
            if any(w in p for w in TOPIC_WORDS.get(name, [])):
                pick = (name, path)
                break
        if pick is None:
            pick = broll[bi % len(broll)]
        name, path = pick
        t0, t1 = round(a + 0.12, 3), round(min(b - 0.12, a + 2.3), 3)
        if t1 - t0 < 0.75:
            continue
        beats.append([round(t0, 2), round(t1, 2), name])
        if b - a >= 1.5:
            html.append(f'      <div class="bfull" id="bf{bi}"><img src="{path}" alt=""></div>')
            tw.append(f"tl.fromTo('#bf{bi}',{{opacity:0}},{{opacity:1,duration:0.08,ease:'none'}},{t0:.3f});")
            tw.append(f"tl.fromTo('#bf{bi} img',{{scale:1.14}},{{scale:1.02,duration:{t1-t0:.3f},ease:'none'}},{t0:.3f});")
            tw.append(f"tl.to('#bf{bi}',{{opacity:0,duration:0.07,ease:'none'}},{t1:.3f});")
        else:
            s = slots[bi % len(slots)]
            r = s[3]
            html.append(f'      <div class="broll" id="br{bi}" style="left:{s[0]};top:{s[1]};width:{s[2]}"><img src="{path}" alt=""></div>')
            tw.append(f"tl.fromTo('#br{bi}',{{opacity:0,scale:0.55,rotate:{r+7},y:26}},{{opacity:1,scale:1,rotate:{r},y:0,duration:0.24,ease:'back.out(1.6)'}},{t0:.3f});")
            tw.append(f"tl.to('#br{bi}',{{y:8,duration:{max(0.4, t1-t0-0.32):.3f},ease:'sine.inOut'}},{t0+0.26:.3f});")
            tw.append(f"tl.to('#br{bi}',{{opacity:0,scale:0.92,duration:0.16,ease:'power2.in'}},{max(t0+0.3, t1-0.16):.3f});")
        bi += 1
    return "\n".join(html), "\n".join(tw), beats

def gen_slams(cuts, segs_words, flat):
    """Full-screen 3D kinetic word at its real spoken beat (every 2nd segment)."""
    html, tw, slams = [], [], []
    idx = 0
    for k, (w, phrase) in enumerate(segs_words):
        if not w or k % 2 != 0:
            continue
        a, b = cuts[k]
        target = norm_txt(w)
        st = en = None
        for s, e, wd in flat:
            if a - 0.05 <= s < b and norm_txt(wd) == target:
                st, en = round(s, 3), round(e, 3)
                break
        if st is None:
            st = round(a + 0.2, 3)
            en = round(min(b, st + 0.8), 3)
        show = min(max(0.55, en - st), b - a - 0.28)
        endt = round(st + show, 3)
        if endt - st < 0.4:
            continue
        acc = ACCENTS[idx % len(ACCENTS)]
        n = len(w)
        if n <= 7:
            fs = "clamp(46px,12vw,128px)"
        elif n <= 11:
            fs = "clamp(38px,9.5vw,104px)"
        elif n <= 15:
            fs = "clamp(30px,7.5vw,84px)"
        else:
            fs = "clamp(25px,6vw,68px)"
        html.append(f'      <div class="slam" id="slam{idx}">{BURST_SVG}'
                    f'<span style="font-size:{fs};--xa:{acc}">{esc(w.upper())}</span></div>')
        mode = idx % 4
        if mode == 0:
            tw.append(f"tl.fromTo('#slam{idx}',{{y:-150,opacity:0}},{{y:0,opacity:1,duration:0.2,ease:'back.out(1.7)'}},{st:.3f});")
        elif mode == 1:
            tw.append(f"tl.fromTo('#slam{idx}',{{x:-320,opacity:0}},{{x:0,opacity:1,duration:0.2,ease:'expo.out'}},{st:.3f});")
        elif mode == 2:
            tw.append(f"tl.fromTo('#slam{idx}',{{x:320,opacity:0}},{{x:0,opacity:1,duration:0.2,ease:'expo.out'}},{st:.3f});")
        else:
            tw.append(f"tl.fromTo('#slam{idx}',{{scale:0.42,opacity:0}},{{scale:1,opacity:1,duration:0.24,ease:'back.out(2.1)'}},{st:.3f});")
        tw.append(f"tl.fromTo('#slam{idx} .burst',{{scale:0.55,opacity:0,rotation:-24}},{{scale:1,opacity:0.9,rotation:0,duration:0.2,ease:'power3.out'}},{st:.3f});")
        tw.append(f"tl.to('#slam{idx} .burst',{{rotation:30,duration:{endt-st:.3f},ease:'none'}},{st+0.2:.3f});")
        tw.append(f"tl.to('#slam{idx}',{{opacity:0,scale:0.93,duration:0.14,ease:'power2.in'}},{endt:.3f});")
        slams.append([round(st, 2), w])
        idx += 1
    return "\n".join(html), "\n".join(tw), slams

COMPO_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>WhatsApp Edit - Reel Montage</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Anton&family=Archivo+Black&display=swap">
<style>
html,body{margin:0;height:100%;background:#000;overflow:hidden}
#main{position:absolute;inset:0;background:#000;overflow:hidden}
#vwrap{position:absolute;inset:0;transform-origin:50% 40%;z-index:1;overflow:hidden}
#vwrap video{position:absolute;left:0;top:0;width:100%;height:100%;object-fit:cover}
.lseg{position:absolute;inset:0;transform-origin:50% 30%}
.tint{position:absolute;inset:0;mix-blend-mode:overlay;pointer-events:none}
.broll{position:absolute;z-index:3;opacity:0;box-shadow:0 16px 46px rgba(0,0,0,.42);border:5px solid #fff;border-radius:14px;overflow:hidden;pointer-events:none;aspect-ratio:4/5}
.broll img{width:100%;height:100%;object-fit:cover;display:block}
.bfull{position:absolute;inset:0;z-index:2;opacity:0;pointer-events:none;overflow:hidden}
.bfull img{width:100%;height:100%;object-fit:cover;transform-origin:50% 40%}
.slam{position:absolute;left:0;right:0;top:4%;bottom:34%;z-index:5;display:grid;place-items:center;pointer-events:none;opacity:0}
.slam .burst{position:absolute;width:min(78vw,600px);height:min(78vw,600px);z-index:0}
.slam .burst line{stroke:rgba(255,255,255,.85);stroke-width:2.4;stroke-linecap:round}
.slam span{position:relative;z-index:1;font-family:'Anton','Archivo Black',Arial,Helvetica,sans-serif;color:#fff;text-transform:uppercase;line-height:.95;letter-spacing:1px;-webkit-text-stroke:2.4px #000;text-align:center;max-width:94%;text-shadow:0 6px 0 var(--xa),0 12px 0 rgba(0,0,0,.35),0 0 44px rgba(0,0,0,.35)}
.flash{position:absolute;inset:0;z-index:8;pointer-events:none;opacity:0;background:#fff}
.caps{position:absolute;left:0;right:0;top:52%;display:grid;place-items:center;pointer-events:none;z-index:6}
.cap{font-family:'Archivo Black',Arial,Helvetica,sans-serif;font-size:clamp(20px,4.4vw,50px);color:#fff;opacity:0;text-align:center;max-width:82%;letter-spacing:.3px;line-height:1.15;
-webkit-text-stroke:1.4px #000;text-shadow:0 3px 0 rgba(0,0,0,.55),0 6px 16px rgba(0,0,0,.5)}
.cap .kb{display:block;font-size:1.95em;letter-spacing:.5px;line-height:1.05;margin-bottom:2px}
.plate{position:absolute;inset:0;z-index:9;display:grid;place-items:center;pointer-events:none;opacity:0;
font-family:'Archivo Black',Arial,Helvetica,sans-serif;font-size:clamp(40px,12vw,120px);color:#fff;text-transform:uppercase;text-align:center;max-width:90%;letter-spacing:2px;
text-shadow:0 4px 0 rgba(0,0,0,.45),0 0 40px rgba(0,0,0,.5)}
§§PLATE_GRADS§§
</style>
</head>
<body>
<div id="main" data-composition-id="main" data-start="0" data-duration="§§DUR§§" data-width="1080" data-height="1920">
  <div id="vwrap">
§§CLIPS_HTML§§
  </div>
  <div class="flash" id="flash"></div>
  <div class="caps" id="caps">
§§CAPS_HTML§§
  </div>
§§BROLL_HTML§§
§§SLAM_HTML§§
§§PLATES_HTML§§
§§SFX_HTML§§
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

print("transcribing with word timestamps (small model) ...")
caps, flat = transcribe(base)
print(f"captions: {len(caps)} phrases, {len(flat)} words")

cuts = build_cuts(flat, dur_wa)
print(f"cut plan: {len(cuts)} segments")

segs_words = [seg_content(flat, a, b) for a, b in cuts]
kinetic = [[round(a, 2), w] for (a, b), (w, _) in zip(cuts, segs_words) if w]
for a, w in kinetic:
    print(f"  kinetic {a:6.2f}s -> {w}")

plates_html, plate_events, plate_grads = gen_plates(cuts, segs_words, dur_wa)
print(f"plates: {len(plate_events)}")

broll_html, broll_tweens, broll_beats = gen_broll(cuts, segs_words)
print(f"b-roll: {len(broll_beats)} beats ({len(load_broll())} images)")

slam_html, slam_tweens, slams = gen_slams(cuts, segs_words, flat)
print(f"slams: {len(slams)}")

COMPO = (COMPO_TEMPLATE
         .replace("§§DUR§§", f"{dur_wa:.3f}")
         .replace("§§PLATE_GRADS§§", plate_grads)
         .replace("§§CLIPS_HTML§§", gen_segments_html(cuts, SEQ))
         .replace("§§CAPS_HTML§§", gen_caps_html(segs_words, cuts))
         .replace("§§PLATES_HTML§§", plates_html)
         .replace("§§BROLL_HTML§§", broll_html)
         .replace("§§SLAM_HTML§§", slam_html)
         .replace("§§SFX_HTML§§", gen_sfx_html(cuts, plate_events))
         .replace("§§TWEENS§§", "\n".join([
             gen_tweens(cuts, segs_words, dur_wa, plate_events),
             slam_tweens,
             broll_tweens])))
open("wa-edit.html", "w", encoding="utf-8").write(COMPO)
print(f"wa-edit.html written: {len(COMPO)} bytes")

sc["captions_whatsapp"] = caps
sc["words_flat"] = [[round(s, 2), round(e, 2), w] for s, e, w in flat]
sc["cut_times"] = [round(a, 2) for a, b in cuts]
sc["kinetic_words"] = kinetic
sc["plates"] = [[t0, t1, idx] for (t0, t1, idx) in plate_events]
sc["slams"] = slams
sc["broll"] = broll_beats
sc["dur_whatsapp"] = dur_wa
json.dump(sc, open(scenes_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"schedule: {len(cuts)} cuts, {len(kinetic)} kinetic words, {len(plate_events)} plates, base {dur_wa:.2f}s")
