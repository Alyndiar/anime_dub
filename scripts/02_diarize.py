# scripts/02_diarize.py
import argparse
import os
from typing import Iterable

from pyannote.audio import Pipeline

from utils_config import ensure_directories, get_data_path


def iter_targets(stems_filter: set[str] | None) -> Iterable[str]:
    audio_raw = get_data_path("audio_raw_dir")
    for wav in sorted(audio_raw.glob("*_mono16k.wav")):
        stem = wav.stem.replace("_mono16k", "")
        if stems_filter and stem not in stems_filter:
            continue
        yield stem


def diarize_all(stems: set[str] | None = None) -> None:
    paths = ensure_directories(["diarization_dir"])
    audio_raw = get_data_path("audio_raw_dir")
    diar_dir = paths["diarization_dir"]

    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "Définis HF_TOKEN=ton_token_HF"

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    for stem in iter_targets(stems):
        wav = audio_raw / f"{stem}_mono16k.wav"
        rttm_path = diar_dir / f"{stem}.rttm"

        diarization = pipeline(str(wav))

        with rttm_path.open("w", encoding="utf-8") as f:
            diarization.write_rttm(f)

        print("Diarisation écrite :", rttm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarisation pyannote pour un ou plusieurs épisodes")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode sans suffixe à traiter")
    args = parser.parse_args()

    stems_filter = set(args.stem) if args.stem else None
    diarize_all(stems_filter)
