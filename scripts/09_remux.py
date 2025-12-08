# scripts/09_remux.py
import subprocess
from pathlib import Path

VID = Path("data/episodes_raw")
DUB = Path("data/dub_audio")
OUT = Path("data/episodes_dubbed")
OUT.mkdir(parents=True, exist_ok=True)

def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

for mix in DUB.glob("*_fr_full.wav"):
    stem = mix.stem.replace("_fr_full", "")
    src = VID / f"{stem}.mkv"
    out = OUT / f"{stem}_FR.mkv"

    run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-i", str(mix),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-metadata:s:a:0", "language=fra",
        str(out)
    ])
