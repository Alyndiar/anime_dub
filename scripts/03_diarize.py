# scripts/03_diarize.py
import argparse
import logging
import os
from typing import Iterable

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

    try:
        import torch
    except OSError as exc:
        logger.error(
            "PyTorch ne peut pas se charger : %s",
            exc,
        )
        logger.info(
            "Cette erreur (fbgemm.dll ou dépendance manquante) survient souvent quand"
            " les runtimes Visual C++ 2015-2022 sont absents ou quand la version"
            " PyTorch/CUDA ne correspond pas au GPU/driver."
        )
        logger.info(
            "Actions suggérées :\n"
            "  1) Installer le runtime Microsoft Visual C++ 2015-2022 (x64).\n"
            "  2) Vérifier que le driver NVIDIA est à jour et que PyTorch correspond"
            " à la version CUDA de l'environnement (ex. pytorch-cuda=12.1).\n"
            "  3) Réinstaller PyTorch dans l'env dédié : conda install pytorch"
            " pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge."
        )
        raise SystemExit(1) from exc

    from pyannote.audio import Pipeline
    from pyannote.audio.core.task import Specifications

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

    # Vérifie si on tourne dans un environnement dédié pour la diarisation.
    # Cela permet de rester sur une pile PyTorch/pyannote/ffmpeg connue comme compatible
    # (voir config/diarization_env.yml). torchcodec est optionnel grâce au fallback.
    diar_env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
    if diar_env and "diar" not in diar_env:
        logger.warning(
            "Environnement actuel (%s) différent de l'environnement dédié diarisation."
            " Utilise de préférence 'anime_dub_diar' défini dans config/diarization_env.yml",
            diar_env,
        )
    elif not diar_env:
        logger.info(
            "Aucun environnement virtuel détecté. Pour éviter les conflits de dépendances,"
            " crée et active l'environnement conda 'anime_dub_diar' (config/diarization_env.yml)."
        )

    # PyTorch >= 2.6 charge les checkpoints en mode "weights_only=True" par défaut.
    # Certains checkpoints pyannote utilisent des classes personnalisées (TorchVersion,
    # Specifications) dans leur state dict. On les ajoute donc aux globals autorisés
    # pour éviter les erreurs "Unsupported global ..." lors du chargement.
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Specifications])

    # Le pipeline pyannote s'appuie sur torchcodec pour le décodage audio. En cas
    # d'installation manquante ou cassée (DLL introuvable), on avertit et on bascule
    # sur un préchargement manuel de l'audio pour éviter le blocage.
    has_torchcodec = True
    try:
        import torchcodec  # noqa: F401
    except Exception as exc:  # pragma: no cover - avertissement runtime seulement
        has_torchcodec = False
        logger.warning(
            "torchcodec n'est pas disponible ou mal installé : %s", exc,
        )
        logger.info(
            "Fallback : l'audio sera préchargé via torchaudio/soundfile pour contourner"
            " l'absence de torchcodec.",
        )
        logger.info(
            "Si une roue Windows est publiée ultérieurement, installe-la ainsi dans l'environnement"
            " courant (PyTorch %s) :\n  pip install --upgrade --no-deps torchcodec",
            torch.__version__,
        )
        logger.info(
            "Consulte la table de compatibilité torchcodec/ffmpeg si besoin : "
            "https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec",
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    processed_any = False
    for stem in iter_targets(stems, logger):
        wav = audio_raw / f"{stem}_mono16k.wav"
        rttm_path = diar_dir / f"{stem}.rttm"

        logger.info("Diarisation en cours : %s", wav)

        diar_input: str | dict
        if has_torchcodec:
            diar_input = str(wav)
        else:  # précharge pour contourner torchcodec manquant
            try:
                import torchaudio

                waveform, sample_rate = torchaudio.load(wav)
                diar_input = {"waveform": waveform, "sample_rate": sample_rate}
            except Exception as ta_exc:  # pragma: no cover - fallback de secours
                import soundfile as sf

                audio, sample_rate = sf.read(wav, dtype="float32")
                if audio.ndim == 1:
                    waveform = torch.from_numpy(audio).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(audio).T
                diar_input = {"waveform": waveform, "sample_rate": sample_rate}
                logger.warning(
                    "Décodage via torchaudio impossible (%s). Fallback soundfile utilisé.", ta_exc,
                )

        diarization = pipeline(diar_input)

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
