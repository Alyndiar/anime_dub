# scripts/05_build_speaker_bank.py
import numpy as np
import soundfile as sf
from pyannote.audio import Model

from utils_config import ensure_directories

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


def prepare_bank_directories() -> None:
    ensure_directories(["speaker_bank_dir"])


# Ensuite tu peux :
# - sélectionner quelques épisodes (manuellement dans le script)
# - pour chaque segment (RTTM), calculer un embedding
# - les stocker dans un .npz par épisode pour ensuite les clusteriser avec sklearn


if __name__ == "__main__":
    # Prépare simplement la structure pour la prochaine étape GUI / clustering.
    prepare_bank_directories()

