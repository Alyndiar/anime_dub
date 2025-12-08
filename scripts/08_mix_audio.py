# scripts/08_mix_audio.py
import argparse
import subprocess
from typing import Iterable

from utils_config import ensure_directories, get_data_path


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def iter_targets(stems_filter: set[str] | None) -> Iterable[str]:
    dub_dir = get_data_path("dub_audio_dir")
    for voices in sorted(dub_dir.glob("*_fr_voices.wav")):
        stem = voices.stem.replace("_fr_voices", "")
        if stems_filter and stem not in stems_filter:
            continue
        yield stem


def mix_all(stems: set[str] | None = None):
    paths = ensure_directories(["dub_audio_dir"])
    raw_dir = get_data_path("audio_raw_dir")
    dub_dir = paths["dub_audio_dir"]

    for stem in iter_targets(stems):
        voices = dub_dir / f"{stem}_fr_voices.wav"
        original = raw_dir / f"{stem}_full.wav"
        out_mix = dub_dir / f"{stem}_fr_full.wav"

        run([
            "ffmpeg", "-y",
            "-i", str(original),
            "-i", str(voices),
            "-filter_complex",
            "[0:a]volume=0.5[a0];[1:a]volume=1.2[a1];[a0][a1]amix=inputs=2:dropout_transition=0",
            "-c:a", "aac",
            str(out_mix)
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixage voix FR + BGM d'origine")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode à mixer")
    args = parser.parse_args()

    stems_filter = set(args.stem) if args.stem else None
    mix_all(stems_filter)
