# scripts/01_extract_audio.py
import argparse
import subprocess
from typing import Iterable

from utils_config import ensure_directories, get_data_path


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def iter_sources(stems_filter: set[str] | None) -> Iterable[str]:
    episodes_raw = get_data_path("episodes_raw_dir")
    for video in sorted(episodes_raw.glob("*.mkv")):
        stem = video.stem
        if stems_filter and stem not in stems_filter:
            continue
        yield stem


def extract_audio_for_all_sources(stems: set[str] | None = None) -> None:
    paths = ensure_directories(["audio_raw_dir"])
    audio_raw = paths["audio_raw_dir"]

    episodes_raw = get_data_path("episodes_raw_dir")

    for stem in iter_sources(stems):
        video = episodes_raw / f"{stem}.mkv"
        full_wav = audio_raw / f"{stem}_full.wav"
        mono16 = audio_raw / f"{stem}_mono16k.wav"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction audio par lot ou par épisode")
    parser.add_argument("--stem", action="append", help="Nom(s) de fichier (sans extension) à traiter uniquement")
    args = parser.parse_args()

    stems_filter = set(args.stem) if args.stem else None
    extract_audio_for_all_sources(stems_filter)
