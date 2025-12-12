# scripts/02_separate_stems.py
"""Séparation des stems (voix / instrumental) avec Demucs ou une commande UVR externe.

Ce script complète le pipeline en proposant une étape dédiée à la
séparation des pistes audio extraites. Il s'appuie par défaut sur
Demucs (``python -m demucs.separate``) mais peut également appeler une
commande UVR personnalisée fournie par l'utilisateur.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

import torch

from utils_config import ensure_directories, get_data_path
from utils_logging import init_logger, parse_stems, should_verbose
from utils_paths import normalize_stem, normalized_filter, stem_matches_filter


def log(message: str, logger: logging.Logger, level: int = logging.DEBUG) -> None:
    """Envoie un message au logger central."""

    logger.log(level, message)


def resolve_demucs_device(requested: str, logger: logging.Logger, verbose: bool, require_cuda: bool) -> str:
    """Détermine l'appareil Demucs en expliquant les raisons du choix.

    Si ``require_cuda`` est actif et que CUDA n'est pas utilisable, une
    ``RuntimeError`` est levée afin d'éviter un fallback silencieux.
    """

    cuda_build = torch.version.cuda
    cuda_available = torch.cuda.is_available()
    cuda_built = torch.backends.cuda.is_built()
    torch_version = torch.__version__

    if verbose:
        log(
            (
                "PyTorch "
                f"{torch_version} (build CUDA: {cuda_build or 'aucun'}, torch.backends.cuda.is_built={cuda_built}) "
                f"– torch.cuda.is_available={cuda_available}"
            ),
            logger,
        )

    if requested == "auto":
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            except Exception:  # noqa: BLE001
                gpu_name = "GPU CUDA détecté"
            log(f"CUDA détecté ({gpu_name}), utilisation du GPU pour Demucs.", logger, level=logging.INFO)
            return "cuda"

        if cuda_build is None:
            log(
                "PyTorch installé sans support CUDA (torch.version.cuda=None) : installation GPU requise pour utiliser la RTX 4080.",
                logger,
                level=logging.WARNING,
            )
        else:
            log(
                "CUDA détecté dans la build mais indisponible (drivers / runtime manquants ?), bascule sur CPU.",
                logger,
                level=logging.WARNING,
            )
            log(
                "Vérifiez le driver NVIDIA, le toolkit CUDA et que demucs s'exécute dans l'environnement où torch==torchvision"
                " ont été installés avec le suffixe +cuXXX.",
                logger,
                level=logging.WARNING,
            )

        if require_cuda:
            raise RuntimeError("CUDA requis (--demucs-require-cuda) mais indisponible")
        return "cpu"

    if requested.startswith("cuda") and not cuda_available:
        fallback_reason = (
            "PyTorch sans CUDA" if cuda_build is None else "CUDA indisponible pour l'instant"
        )
        log(f"{requested} demandé mais {fallback_reason}, bascule sur CPU.", logger, level=logging.WARNING)
        if require_cuda:
            raise RuntimeError("CUDA requis (--demucs-require-cuda) mais indisponible")
        return "cpu"

    return requested


def run(cmd: list[str], logger: logging.Logger, verbose: bool) -> None:
    """Exécute une commande système en journalisant la ligne complète."""

    command_str = " ".join(cmd)
    if verbose:
        log(f"Exécution : {command_str}", logger)
    else:
        log(command_str, logger, level=logging.INFO)
    subprocess.run(cmd, check=True)


def iter_audio_sources(
    stems_filter: set[str] | None, logger: logging.Logger
) -> Iterable[tuple[str, str, Path]]:
    """Itère sur les wav complets issus de l'extraction.

    Retourne le stem tel que trouvé et sa version normalisée (sans espaces).
    """

    audio_raw = get_data_path("audio_raw_dir")
    log(f"Recherche des WAV complets dans {audio_raw}", logger)

    stems_filter_norm = normalized_filter(stems_filter)

    candidates = sorted(audio_raw.glob("*_full.wav"))
    if not candidates:
        log("Aucun fichier *_full.wav trouvé ; avez-vous exécuté 01_extract_audio ?", logger, level=logging.WARNING)
        return

    for wav_path in candidates:
        stem = wav_path.stem.replace("_full", "")
        if not stem_matches_filter(stem, stems_filter_norm):
            log(f"Ignoré (non sélectionné) : {stem}", logger)
            continue
        yield stem, normalize_stem(stem), wav_path


def separate_with_demucs(
    audio_path: Path,
    workspace: Path,
    model: str,
    device: str,
    require_cuda: bool,
    logger: logging.Logger,
    verbose: bool,
) -> Tuple[Path, Path]:
    """Lance Demucs en mode two-stems et retourne les chemins produits."""

    demucs_out = workspace / "demucs"
    demucs_out.mkdir(parents=True, exist_ok=True)

    chosen_device = resolve_demucs_device(device, logger, verbose, require_cuda)
    log(f"Appareil Demucs sélectionné : {chosen_device}", logger, level=logging.INFO if not verbose else logging.DEBUG)

    cmd = [
        "python",
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-n",
        model,
        "--out",
        str(demucs_out),
        "-d",
        chosen_device,
        str(audio_path),
    ]
    run(cmd, logger, verbose)

    model_dir = demucs_out / model
    stem_dir = model_dir / audio_path.stem
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"

    if not vocals.exists() or not instrumental.exists():
        raise FileNotFoundError(
            f"Sorties Demucs manquantes dans {stem_dir} (attendu vocals.wav et no_vocals.wav)"
        )
    return vocals, instrumental


def separate_with_uvr_command(
    audio_path: Path,
    workspace: Path,
    cmd_template: str,
    model_path: str | None,
    vocals_name: str,
    instrumental_name: str,
    logger: logging.Logger,
    verbose: bool,
) -> Tuple[Path, Path]:
    """Exécute une commande UVR externe fournie par l'utilisateur.

    Le template est évalué avec les variables ``{input}``, ``{output_dir}``
    et ``{model}`` pour s'adapter à différents CLIs UVR (portable ou pip).
    """

    uvr_out = workspace / "uvr"
    uvr_out.mkdir(parents=True, exist_ok=True)

    formatted = cmd_template.format(input=str(audio_path), output_dir=str(uvr_out), model=str(model_path or ""))
    cmd = shlex.split(formatted)
    run(cmd, logger, verbose)

    vocals = uvr_out / vocals_name
    instrumental = uvr_out / instrumental_name
    if not vocals.exists() or not instrumental.exists():
        raise FileNotFoundError(
            f"Sorties UVR non trouvées : {vocals_name} / {instrumental_name} dans {uvr_out}."
        )
    return vocals, instrumental


def separate_all_stems(
    stems_filter: set[str] | None,
    tool: str,
    demucs_model: str,
    demucs_device: str,
    demucs_require_cuda: bool,
    uvr_command: str | None,
    uvr_model: str | None,
    uvr_vocals_name: str,
    uvr_instrumental_name: str,
    overwrite: bool,
    keep_temp: bool,
    verbose: bool,
    logger: logging.Logger | None = None,
) -> None:
    """Sépare les stems pour tous les fichiers sélectionnés."""

    logger = init_logger("separate_stems", verbose, logger)
    paths = ensure_directories(["audio_stems_dir"])
    stems_dir = paths["audio_stems_dir"]
    workspace = stems_dir / "_work"
    workspace.mkdir(parents=True, exist_ok=True)

    processed_any = False

    errors: list[str] = []

    for stem_raw, stem_norm, audio_path in iter_audio_sources(stems_filter, logger):
        target_vocals = stems_dir / f"{stem_norm}_vocals.wav"
        target_instr = stems_dir / f"{stem_norm}_instrumental.wav"

        if target_vocals.exists() and target_instr.exists() and not overwrite:
            log(f"Déjà présent, saut : {stem_raw}", logger, level=logging.INFO)
            continue

        log(
            f"Séparation de {stem_raw} (sorties sans espaces : {stem_norm}) via {tool}",
            logger,
            level=logging.INFO,
        )
        try:
            if tool == "demucs":
                vocals_src, instr_src = separate_with_demucs(
                    audio_path,
                    workspace,
                    demucs_model,
                    demucs_device,
                    demucs_require_cuda,
                    logger,
                    verbose,
                )
            else:
                vocals_src, instr_src = separate_with_uvr_command(
                    audio_path,
                    workspace,
                    uvr_command or "",
                    uvr_model,
                    uvr_vocals_name,
                    uvr_instrumental_name,
                    logger,
                    verbose,
                )
        except Exception as exc:  # noqa: BLE001
            log(
                f"Échec de la séparation pour {stem_raw}: {exc}",
                logger,
                level=logging.ERROR,
            )
            errors.append(stem_raw)
            continue

        shutil.copyfile(vocals_src, target_vocals)
        shutil.copyfile(instr_src, target_instr)
        log(f"Stems écrits dans {stems_dir}", logger, level=logging.INFO)
        processed_any = True

        if not keep_temp:
            for folder in workspace.iterdir():
                if folder.is_dir():
                    shutil.rmtree(folder, ignore_errors=True)

    if not processed_any:
        log("Aucun stem traité (fichiers absents ou déjà présents).", logger, level=logging.WARNING)

    if errors:
        log(
            "Certaines séparations ont échoué : " + ", ".join(sorted(errors)),
            logger,
            level=logging.ERROR,
        )
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Séparation voix/BGM des WAV extraits")
    parser.add_argument("--stem", action="append", help="Nom(s) de fichier (sans extension) à traiter uniquement")
    parser.add_argument(
        "--tool",
        choices=["demucs", "uvr"],
        default="demucs",
        help="Outil utilisé pour la séparation (demucs recommandé)",
    )
    parser.add_argument("--demucs-model", default="htdemucs", help="Modèle Demucs à utiliser (ex: htdemucs, htdemucs_ft)")
    parser.add_argument(
        "--demucs-device",
        default="auto",
        help="Appareil Demucs (auto, cuda ou cpu). 'auto' choisit cuda si disponible, sinon cpu.",
    )
    parser.add_argument(
        "--demucs-require-cuda",
        action="store_true",
        help="Échoue si CUDA n'est pas utilisable (utile pour diagnostiquer l'environnement GPU).",
    )
    parser.add_argument(
        "--uvr-command",
        help="Commande UVR complète à exécuter (template avec {input}, {output_dir}, {model})",
    )
    parser.add_argument("--uvr-model", help="Chemin vers le modèle UVR si requis par la commande")
    parser.add_argument("--uvr-vocals-name", default="vocals.wav", help="Nom du fichier voix produit par UVR")
    parser.add_argument(
        "--uvr-instrumental-name", default="instrumental.wav", help="Nom du fichier instrumental produit par UVR"
    )
    parser.add_argument("--overwrite", action="store_true", help="Écrase les stems existants si présents")
    parser.add_argument("--keep-temp", action="store_true", help="Conserve les dossiers temporaires Demucs/UVR")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_verbose = should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    verbose = args.verbose or env_verbose

    if args.tool == "uvr" and not args.uvr_command:
        raise SystemExit("--uvr-command est requis quand --tool uvr est sélectionné")

    logger = init_logger("separate_stems", verbose)
    stems_filter = parse_stems(args.stem, logger)
    log(
        f"Stems ciblés : {sorted(stems_filter) if stems_filter else 'tous les fichiers *_full.wav'}",
        logger,
        level=logging.INFO if not verbose else logging.DEBUG,
    )

    separate_all_stems(
        stems_filter=stems_filter,
        tool=args.tool,
        demucs_model=args.demucs_model,
        demucs_device=args.demucs_device,
        demucs_require_cuda=args.demucs_require_cuda,
        uvr_command=args.uvr_command,
        uvr_model=args.uvr_model,
        uvr_vocals_name=args.uvr_vocals_name,
        uvr_instrumental_name=args.uvr_instrumental_name,
        overwrite=args.overwrite,
        keep_temp=args.keep_temp,
        verbose=verbose,
        logger=logger,
    )


if __name__ == "__main__":
    main()
