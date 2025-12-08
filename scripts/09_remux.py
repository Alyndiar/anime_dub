# scripts/09_remux.py
import subprocess

from utils_config import ensure_directories, get_data_path


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def remux_all():
    paths = ensure_directories(["episodes_dubbed_dir"])
    vid_dir = get_data_path("episodes_raw_dir")
    dub_dir = get_data_path("dub_audio_dir")
    out_dir = paths["episodes_dubbed_dir"]

    for mix in dub_dir.glob("*_fr_full.wav"):
        stem = mix.stem.replace("_fr_full", "")
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
    remux_all()
