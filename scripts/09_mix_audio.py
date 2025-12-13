# scripts/09_mix_audio.py
import argparse
import logging
import os
import subprocess
from typing import Iterable

from utils_config import ensure_directories, get_data_path
from utils_logging import init_logger, parse_stems, should_verbose
from utils_paths import normalized_filter, stem_matches_filter


def run(cmd: list[str], logger: logging.Logger, verbose: bool) -> None:
    command_str = " ".join(cmd)
    if verbose:
        logger.debug("Exécution : %s", command_str)
    else:
        logger.info(command_str)
    subprocess.run(cmd, check=True)


def iter_targets(stems_filter: set[str] | None, logger: logging.Logger) -> Iterable[str]:
    dub_dir = get_data_path("dub_audio_dir")
    logger.debug("Recherche des voix FR dans %s", dub_dir)
    stems_filter_norm = normalized_filter(stems_filter)
    for voices in sorted(dub_dir.glob("*_fr_voices.wav")):
        stem = voices.stem.replace("_fr_voices", "")
        if not stem_matches_filter(stem, stems_filter_norm):
            logger.debug("Ignore %s car non sélectionné", stem)
            continue
        yield stem


def mix_all(
    stems: set[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    logger = init_logger("mix_audio", verbose, logger)

    paths = ensure_directories(["dub_audio_dir"])
    raw_dir = get_data_path("audio_raw_dir")
    dub_dir = paths["dub_audio_dir"]

    processed_any = False
    for stem in iter_targets(stems, logger):
        voices = dub_dir / f"{stem}_fr_voices.wav"
        original = raw_dir / f"{stem}_full.wav"
        out_mix = dub_dir / f"{stem}_fr_full.wav"

        logger.info("Mixage : %s", stem)
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(original),
                "-i",
                str(voices),
                "-filter_complex",
                "[0:a]volume=0.5[a0];[1:a]volume=1.2[a1];[a0][a1]amix=inputs=2:dropout_transition=0",
                "-c:a",
                "aac",
                str(out_mix),
            ],
            logger,
            verbose,
        )
        processed_any = True

    if not processed_any:
        logger.warning("Aucun fichier de voix FR trouvé pour le mixage.")


def main():
    parser = argparse.ArgumentParser(description="Mixage voix FR + BGM d'origine")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode à mixer")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    args = parser.parse_args()

    verbose = args.verbose or should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    logger = init_logger("mix_audio", verbose)

    stems_filter = parse_stems(args.stem, logger)
    stems_display = sorted(normalized_filter(stems_filter)) if stems_filter else "tous"
    logger.info("Stems ciblés (normalisés) : %s", stems_display)

    mix_all(stems_filter, verbose=verbose, logger=logger)


if __name__ == "__main__":
    main()
