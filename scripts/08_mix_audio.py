# scripts/08_mix_audio.py
import subprocess

from utils_config import ensure_directories, get_data_path


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def mix_all():
    paths = ensure_directories(["dub_audio_dir"])
    raw_dir = get_data_path("audio_raw_dir")
    dub_dir = paths["dub_audio_dir"]

    for voices in dub_dir.glob("*_fr_voices.wav"):
        stem = voices.stem.replace("_fr_voices", "")
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
    mix_all()
