#!/usr/bin/env bash
# Builds a 6x4 contact sheet replica (24 frames, dark bg, labeled cells) from any video
# Usage: contact_sheet.sh <video-file>
set -euo pipefail
V="$1"
test -f "$V" || { echo "video not found: $V" >&2; exit 1; }
BG=0x0b0b0b
FONTDIR=/usr/share/fonts/truetype/dejavu
MONO="$FONTDIR/DejaVuSansMono.ttf"
BOLD="$FONTDIR/DejaVuSans-Bold.ttf"

W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$V")
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$V")
RATE=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$V")
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$V")
FPS=$(printf '%s' "$RATE" | awk -F/ '{ if ($2=="" || $2==0) $2=1; printf "%g", $1/$2 }')

mkdir -p cells out
rm -f cells/* || true

for i in $(seq 0 23); do
  T=$(python3 -c "print(($DUR*(0.5+$i))/24)")
  IDX=$i DUR=$DUR python3 - <<'PY' > "cells/label.txt"
import os
i = int(os.environ['IDX']); d = float(os.environ['DUR'])
t = d * (0.5 + i) / 24
m = int(t // 60); s = t - 60 * m
print(f"{i+1:02d} · {m:02d}:{s:04.1f}", end='')
PY
  ffmpeg -hide_banner -loglevel error -y -ss "$T" -i "$V" -frames:v 1 \
    -vf "scale=-2:270,setsar=1,drawtext=fontfile=$MONO:fontsize=17:fontcolor=0xC8C8C8:x=10:y=h-27:textfile=cells/label.txt" \
    "cells/cell_$i.png"
done

NAME=$(basename "$V")
printf 'CONTACT SHEET · %s' "$NAME" > out/title.txt
python3 -c "
w=$W; h=$H; fps='$FPS'; d=$DUR
print(f'{w}×{h} · {fps}fps · {d:.1f}s · 24 frames', end='')" > out/meta.txt

ffmpeg -hide_banner -loglevel error -y -framerate 1 -start_number 0 -i "cells/cell_%d.png" \
  -vf "tile=6x4:margin=10:padding=10:color=$BG,pad=iw:ih+56:0:0:color=$BG,drawtext=fontfile=$BOLD:fontsize=21:fontcolor=0xF2F2F2:x=14:y=18:textfile=out/title.txt,drawtext=fontfile=$MONO:fontsize=15:fontcolor=0x9E9E9E:x=w-tw-14:y=20:textfile=out/meta.txt" \
  -frames:v 1 "out/contact_sheet.png"
rm -rf cells

STEM=$(basename "$V"); STEM="${STEM%.*}"; STEM="${STEM// /_}"
mkdir -p sheets
cp out/contact_sheet.png "sheets/${STEM}_contact_sheet.png"
echo "built sheets/${STEM}_contact_sheet.png"
