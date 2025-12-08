# scripts/02_diarize.py
import os

from pyannote.audio import Pipeline

from utils_config import ensure_directories, get_data_path


def diarize_all() -> None:
    paths = ensure_directories(["diarization_dir"])
    audio_raw = get_data_path("audio_raw_dir")
    diar_dir = paths["diarization_dir"]

    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "Définis HF_TOKEN=ton_token_HF"

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    for wav in audio_raw.glob("*_mono16k.wav"):
        stem = wav.stem.replace("_mono16k", "")
        rttm_path = diar_dir / f"{stem}.rttm"

        diarization = pipeline(str(wav))

        with rttm_path.open("w", encoding="utf-8") as f:
            diarization.write_rttm(f)

        print("Diarisation écrite :", rttm_path)


if __name__ == "__main__":
    diarize_all()
