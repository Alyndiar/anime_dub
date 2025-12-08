# scripts/01_extract_audio.py
"""Extraction audio avec instrumentation compatible CLI et GUI.

Ce module peut être importé par le GUI afin de réutiliser les fonctions
et centraliser les logs, ou exécuté directement depuis la ligne de
commande comme auparavant.
"""

import argparse
import logging
import os
import subprocess
from typing import Iterable

from utils_config import ensure_directories, get_data_path


def should_verbose(env_value: str | None) -> bool:
    """Retourne True si la valeur d'environnement active le mode verbose."""

    if not env_value:
        return False
    return env_value.lower() in {"1", "true", "yes", "on"}


def setup_logger(verbose: bool, external_logger: logging.Logger | None = None) -> logging.Logger:
    """Prépare un logger interne ou réutilise un logger fourni par le GUI."""

    if external_logger:
        return external_logger

    logger = logging.getLogger("extract_audio")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log(message: str, logger: logging.Logger, level: int = logging.DEBUG) -> None:
    """Envoie un message au logger central."""

    logger.log(level, message)


def run(cmd: list[str], logger: logging.Logger, verbose: bool) -> None:
    """Exécute une commande ffmpeg en journalisant la ligne complète."""

    command_str = " ".join(cmd)
    if verbose:
        log(f"Exécution : {command_str}", logger)
    else:
        log(command_str, logger, level=logging.INFO)
    subprocess.run(cmd, check=True)


def iter_sources(stems_filter: set[str] | None, logger: logging.Logger) -> Iterable[str]:
    """Itère sur les vidéos sources, en filtrant si nécessaire."""

    episodes_raw = get_data_path("episodes_raw_dir")
    log(f"Recherche des sources dans {episodes_raw}", logger)
    for video in sorted(episodes_raw.glob("*.mkv")):
        stem = video.stem
        if stems_filter and stem not in stems_filter:
            log(f"Ignore {stem} car non sélectionné", logger)
            continue
        log(f"Source détectée : {video}", logger)
        yield stem


def extract_audio_for_all_sources(
    stems: set[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Extrait les pistes audio principales et mono 16 kHz pour chaque source."""

    logger = setup_logger(verbose, logger)

    paths = ensure_directories(["audio_raw_dir"])
    audio_raw = paths["audio_raw_dir"]

    episodes_raw = get_data_path("episodes_raw_dir")
    log(f"Sorties audio dans {audio_raw}", logger)

    for stem in iter_sources(stems, logger):
        video = episodes_raw / f"{stem}.mkv"
        full_wav = audio_raw / f"{stem}_full.wav"
        mono16 = audio_raw / f"{stem}_mono16k.wav"

        log(f"Traitement de {stem}", logger)
        log(f"Fichier vidéo : {video}", logger)
        log(f"Fichier wav complet : {full_wav}", logger)
        log(f"Fichier wav mono16k : {mono16}", logger)

        # Audio full qualité
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_s16le",
                str(full_wav),
            ],
            logger,
            verbose,
        )

        # Version mono 16k pour Whisper
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(full_wav),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(mono16),
            ],
            logger,
            verbose,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extraction audio par lot ou par épisode")
    parser.add_argument("--stem", action="append", help="Nom(s) de fichier (sans extension) à traiter uniquement")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    return parser.parse_args()


def main():
    args = parse_args()

    env_verbose = should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    verbose = args.verbose or env_verbose

    logger = setup_logger(verbose)

    if verbose:
        log("Mode verbeux activé", logger)
        log(f"Stems filtrés : {args.stem if args.stem else 'tous'}", logger)

    stems_filter = set(args.stem) if args.stem else None
    extract_audio_for_all_sources(stems_filter, verbose=verbose, logger=logger)


if __name__ == "__main__":
    main()
