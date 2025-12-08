# scripts/01_extract_audio.py
import subprocess
from pathlib import Path

RAW = Path("data/episodes_raw")
OUT = Path("data/audio_raw")
OUT.mkdir(parents=True, exist_ok=True)

def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

for video in RAW.glob("*.mkv"):
    stem = video.stem
    full_wav = OUT / f"{stem}_full.wav"
    mono16 = OUT / f"{stem}_mono16k.wav"

    # Audio full qualité
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-map", "0:a:0", "-c:a", "pcm_s16le",
        str(full_wav)
    ])

    # Version mono 16k pour Whisper
    run([
        "ffmpeg", "-y", "-i", str(full_wav),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(mono16)
    ])
