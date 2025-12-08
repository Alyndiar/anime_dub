# scripts/01_extract_audio.py
import argparse
import os
import subprocess
from typing import Iterable

from utils_config import ensure_directories, get_data_path


def should_verbose(env_value: str | None) -> bool:
    if not env_value:
        return False
    return env_value.lower() in {"1", "true", "yes", "on"}


def log(message: str, verbose: bool):
    if verbose:
        print(message)


def run(cmd, verbose: bool):
    command_str = " ".join(cmd)
    if verbose:
        log(f"[verbose] Exécution : {command_str}", verbose)
    else:
        print(command_str)
    subprocess.run(cmd, check=True)


def iter_sources(stems_filter: set[str] | None, verbose: bool) -> Iterable[str]:
    episodes_raw = get_data_path("episodes_raw_dir")
    log(f"[verbose] Recherche des sources dans {episodes_raw}", verbose)
    for video in sorted(episodes_raw.glob("*.mkv")):
        stem = video.stem
        if stems_filter and stem not in stems_filter:
            log(f"[verbose] Ignore {stem} car non sélectionné", verbose)
            continue
        log(f"[verbose] Source détectée : {video}", verbose)
        yield stem


def extract_audio_for_all_sources(stems: set[str] | None = None, verbose: bool = False) -> None:
    paths = ensure_directories(["audio_raw_dir"])
    audio_raw = paths["audio_raw_dir"]

    episodes_raw = get_data_path("episodes_raw_dir")
    log(f"[verbose] Sorties audio dans {audio_raw}", verbose)

    for stem in iter_sources(stems, verbose):
        video = episodes_raw / f"{stem}.mkv"
        full_wav = audio_raw / f"{stem}_full.wav"
        mono16 = audio_raw / f"{stem}_mono16k.wav"

        log(f"[verbose] Traitement de {stem}", verbose)
        log(f"[verbose] Fichier vidéo : {video}", verbose)
        log(f"[verbose] Fichier wav complet : {full_wav}", verbose)
        log(f"[verbose] Fichier wav mono16k : {mono16}", verbose)

        # Audio full qualité
        run([
            "ffmpeg", "-y", "-i", str(video),
            "-map", "0:a:0", "-c:a", "pcm_s16le",
            str(full_wav)
        ], verbose)

        # Version mono 16k pour Whisper
        run([
            "ffmpeg", "-y", "-i", str(full_wav),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(mono16)
        ], verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction audio par lot ou par épisode")
    parser.add_argument("--stem", action="append", help="Nom(s) de fichier (sans extension) à traiter uniquement")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    args = parser.parse_args()

    env_verbose = should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    verbose = args.verbose or env_verbose

    if verbose:
        log("[verbose] Mode verbeux activé", True)
        log(f"[verbose] Stems filtrés : {args.stem if args.stem else 'tous'}", True)

    stems_filter = set(args.stem) if args.stem else None
    extract_audio_for_all_sources(stems_filter, verbose=verbose)
