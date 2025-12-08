# scripts/09_remux.py
import argparse
import subprocess
from typing import Iterable

from utils_config import ensure_directories, get_data_path


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def iter_targets(stems_filter: set[str] | None) -> Iterable[str]:
    dub_dir = get_data_path("dub_audio_dir")
    for mix in sorted(dub_dir.glob("*_fr_full.wav")):
        stem = mix.stem.replace("_fr_full", "")
        if stems_filter and stem not in stems_filter:
            continue
        yield stem


def remux_all(stems: set[str] | None = None):
    paths = ensure_directories(["episodes_dubbed_dir"])
    vid_dir = get_data_path("episodes_raw_dir")
    dub_dir = get_data_path("dub_audio_dir")
    out_dir = paths["episodes_dubbed_dir"]

    for stem in iter_targets(stems):
        mix = dub_dir / f"{stem}_fr_full.wav"
        src = vid_dir / f"{stem}.mkv"
        out = out_dir / f"{stem}_FR.mkv"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remux vidéo + piste FR mixée")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode à remuxer")
    args = parser.parse_args()

    stems_filter = set(args.stem) if args.stem else None
    remux_all(stems_filter)
