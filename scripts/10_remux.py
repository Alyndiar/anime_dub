# scripts/10_remux.py
import argparse
import logging
import os
import subprocess
from typing import Iterable

from utils_config import ensure_directories, get_data_path
from utils_logging import init_logger, parse_stems, should_verbose


def run(cmd: list[str], logger: logging.Logger, verbose: bool) -> None:
    command_str = " ".join(cmd)
    if verbose:
        logger.debug("Exécution : %s", command_str)
    else:
        logger.info(command_str)
    subprocess.run(cmd, check=True)


def iter_targets(stems_filter: set[str] | None, logger: logging.Logger) -> Iterable[str]:
    dub_dir = get_data_path("dub_audio_dir")
    logger.debug("Recherche des mix FR dans %s", dub_dir)
    for mix in sorted(dub_dir.glob("*_fr_full.wav")):
        stem = mix.stem.replace("_fr_full", "")
        if stems_filter and stem not in stems_filter:
            logger.debug("Ignore %s car non sélectionné", stem)
            continue
        yield stem


def remux_all(
    stems: set[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    logger = init_logger("remux", verbose, logger)

    paths = ensure_directories(["episodes_dubbed_dir"])
    vid_dir = get_data_path("episodes_raw_dir")
    dub_dir = get_data_path("dub_audio_dir")
    out_dir = paths["episodes_dubbed_dir"]

    processed_any = False
    for stem in iter_targets(stems, logger):
        mix = dub_dir / f"{stem}_fr_full.wav"
        src = vid_dir / f"{stem}.mkv"
        out = out_dir / f"{stem}_FR.mkv"

        logger.info("Remux : %s", stem)
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-i",
                str(mix),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-metadata:s:a:0",
                "language=fra",
                str(out),
            ],
            logger,
            verbose,
        )
        processed_any = True

    if not processed_any:
        logger.warning("Aucun mix FR trouvé pour le remux.")


def main():
    parser = argparse.ArgumentParser(description="Remux vidéo + piste FR mixée")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode à remuxer")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    args = parser.parse_args()

    verbose = args.verbose or should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    logger = init_logger("remux", verbose)

    stems_filter = parse_stems(args.stem, logger)
    logger.info("Stems ciblés : %s", sorted(stems_filter) if stems_filter else "tous")

    remux_all(stems_filter, verbose=verbose, logger=logger)


if __name__ == "__main__":
    main()
