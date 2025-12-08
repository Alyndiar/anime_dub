# scripts/06_build_speaker_bank.py
import argparse
import logging
import os
import numpy as np
import soundfile as sf
from pyannote.audio import Model

from utils_config import ensure_directories
from utils_logging import init_logger, should_verbose

# Modèle d'embedding de locuteur pyannote (exemple)
embed_model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token=os.environ.get("HF_TOKEN", True),  # ou HF_TOKEN
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


def prepare_bank_directories(verbose: bool = False, logger: logging.Logger | None = None) -> None:
    logger = init_logger("build_speaker_bank", verbose, logger)
    paths = ensure_directories(["speaker_bank_dir"])
    logger.info("Répertoire banque de voix prêt : %s", paths["speaker_bank_dir"])


# Ensuite tu peux :
# - sélectionner quelques épisodes (manuellement dans le script)
# - pour chaque segment (RTTM), calculer un embedding
# - les stocker dans un .npz par épisode pour ensuite les clusteriser avec sklearn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialisation ou calcul de la banque de speakers")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    args = parser.parse_args()

    verbose = args.verbose or should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    logger = init_logger("build_speaker_bank", verbose)

    prepare_bank_directories(verbose=verbose, logger=logger)

