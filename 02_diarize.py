# scripts/02_diarize.py
import os
from pathlib import Path
import torch
from pyannote.audio import Pipeline

AUDIO_RAW = Path("data/audio_raw")
DIAR = Path("data/diarization")
DIAR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Définis HF_TOKEN=ton_token_HF"

# Pipeline speaker diarization communautaire (ex : 3.1)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN,
)

for wav in AUDIO_RAW.glob("*_mono16k.wav"):
    stem = wav.stem.replace("_mono16k", "")
    rttm_path = DIAR / f"{stem}.rttm"

    diarization = pipeline(str(wav))

    # Sauvegarde format RTTM
    with rttm_path.open("w", encoding="utf-8") as f:
        diarization.write_rttm(f)

    print("Diarisation écrite :", rttm_path)
