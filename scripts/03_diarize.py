# scripts/03_diarize.py
import argparse
import logging
import os
from typing import Iterable

import torch
from pyannote.audio import Pipeline

from utils_config import ensure_directories, get_data_path
from utils_logging import init_logger, parse_stems, should_verbose


def iter_targets(stems_filter: set[str] | None, logger: logging.Logger) -> Iterable[str]:
    audio_raw = get_data_path("audio_raw_dir")
    logger.debug("Recherche des audios dans %s", audio_raw)
    for wav in sorted(audio_raw.glob("*_mono16k.wav")):
        stem = wav.stem.replace("_mono16k", "")
        if stems_filter and stem not in stems_filter:
            logger.debug("Ignore %s car non sélectionné", stem)
            continue
        yield stem


def diarize_all(
    stems: set[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    logger = init_logger("diarize", verbose, logger)

    paths = ensure_directories(["diarization_dir"])
    audio_raw = get_data_path("audio_raw_dir")
    diar_dir = paths["diarization_dir"]

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error(
            "Token Hugging Face manquant (HF_TOKEN).\n"
            "Renseigne HF_TOKEN=ton_token_HF dans l'environnement avant de lancer la diarisation."
        )
        raise SystemExit(1)
    logger.info("Initialisation du pipeline pyannote (HF_TOKEN présent)")

    # PyTorch >= 2.6 charge les checkpoints en mode "weights_only=True" par défaut.
    # Certains checkpoints pyannote utilisent le type torch.torch_version.TorchVersion
    # dans leur state dict. On l'ajoute donc à la liste des globals autorisés pour
    # éviter l'erreur "Unsupported global ... TorchVersion" lors du chargement.
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    processed_any = False
    for stem in iter_targets(stems, logger):
        wav = audio_raw / f"{stem}_mono16k.wav"
        rttm_path = diar_dir / f"{stem}.rttm"

        logger.info("Diarisation en cours : %s", wav)
        diarization = pipeline(str(wav))

        with rttm_path.open("w", encoding="utf-8") as f:
            diarization.write_rttm(f)

        logger.info("Diarisation écrite : %s", rttm_path)
        processed_any = True

    if not processed_any:
        logger.warning("Aucun wav *_mono16k.wav trouvé pour diarisation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarisation pyannote pour un ou plusieurs épisodes")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode sans suffixe à traiter")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    args = parser.parse_args()

    verbose = args.verbose or should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    logger = init_logger("diarize", verbose)

    stems_filter = parse_stems(args.stem, logger)
    logger.info("Stems ciblés : %s", sorted(stems_filter) if stems_filter else "tous")

    diarize_all(stems_filter, verbose=verbose, logger=logger)
