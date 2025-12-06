# scripts/05_build_speaker_bank.py
from pathlib import Path
import numpy as np
import soundfile as sf
from pyannote.audio import Model
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.pipelines.utils import SpeakerDiarizationMixin

AUDIO_RAW = Path("data/audio_raw")
DIAR = Path("data/diarization")
BANK = Path("data/speaker_bank")
BANK.mkdir(parents=True, exist_ok=True)

# Modèle d'embedding de locuteur pyannote (exemple)
embed_model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token=True  # ou HF_TOKEN
)

SAMPLE_RATE = 16000  # à adapter selon ton audio

def get_embedding(wav_path, start, end):
    audio, sr = sf.read(wav_path)
    if sr != SAMPLE_RATE:
        raise RuntimeError("adapter le resampling : sr != 16000")
    s = int(start * sr)
    e = int(end * sr)
    chunk = audio[s:e]
    emb = embed_model({"waveform": np.expand_dims(chunk, 0), "sample_rate": sr})
    return emb.detach().cpu().numpy().squeeze()

# Ensuite tu peux :
# - sélectionner quelques épisodes (manuellement dans le script)
# - pour chaque segment (RTTM), calculer un embedding
# - les stocker dans un .npz par épisode pour ensuite les clusteriser avec sklearn

