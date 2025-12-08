# scripts/08_mix_audio.py
import subprocess
from pathlib import Path

RAW = Path("data/audio_raw")
DUB = Path("data/dub_audio")

def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

for voices in DUB.glob("*_fr_voices.wav"):
    stem = voices.stem.replace("_fr_voices", "")
    original = RAW / f"{stem}_full.wav"
    out_mix = DUB / f"{stem}_fr_full.wav"

    # Ex : original volume 0.5, voix FR 1.2
    run([
        "ffmpeg", "-y",
        "-i", str(original),
        "-i", str(voices),
        "-filter_complex",
        "[0:a]volume=0.5[a0];[1:a]volume=1.2[a1];[a0][a1]amix=inputs=2:dropout_transition=0",
        "-c:a", "aac",
        str(out_mix)
    ])
