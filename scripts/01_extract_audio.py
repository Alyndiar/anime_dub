# scripts/01_extract_audio.py
import subprocess

from utils_config import ensure_directories, get_data_path


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_audio_for_all_sources() -> None:
    paths = ensure_directories(["audio_raw_dir"])
    episodes_raw = get_data_path("episodes_raw_dir")
    audio_raw = paths["audio_raw_dir"]

    for video in episodes_raw.glob("*.mkv"):
        stem = video.stem
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
    extract_audio_for_all_sources()
