# scripts/03_diarize.py
import argparse
import logging
import os
from typing import Iterable

from utils_config import ensure_directories, get_data_path
from utils_logging import init_logger, parse_stems, should_verbose
from utils_paths import normalize_stem, normalized_filter, stem_matches_filter


def iter_targets(
    stems_filter: set[str] | None, logger: logging.Logger
) -> Iterable[tuple[str, str, os.PathLike[str], str]]:
    """Prépare la liste des stems à diariser en privilégiant les stems vocaux (étape 02).

    Retourne le stem tel que trouvé, sa version normalisée (sans espaces) et la
    source choisie.
    """

    audio_stems = get_data_path("audio_stems_dir")
    audio_raw = get_data_path("audio_raw_dir")
    logger.debug(
        "Recherche des audios (priorité stems) dans %s puis %s", audio_stems, audio_raw
    )

    stems_filter_norm = normalized_filter(stems_filter)

    chosen: dict[str, tuple[os.PathLike[str], str]] = {}

    for wav in sorted(audio_stems.glob("*_vocals.wav")):
        stem = wav.stem.removesuffix("_vocals")
        if not stem_matches_filter(stem, stems_filter_norm):
            logger.debug("Ignore %s car non sélectionné (stems)", stem)
            continue
        chosen[stem] = (wav, "stems (vocals)")

    for pattern, suffix, desc in (
        ("*_mono16k.wav", "_mono16k", "audio complet mono16k"),
        ("*_full.wav", "_full", "audio complet _full (resample 16 kHz)"),
    ):
        for wav in sorted(audio_raw.glob(pattern)):
            stem = wav.stem.replace(suffix, "")
            if not stem_matches_filter(stem, stems_filter_norm):
                logger.debug("Ignore %s car non sélectionné (audio brut)", stem)
                continue
            chosen.setdefault(stem, (wav, desc))

    for stem in sorted(chosen):
        yield stem, normalize_stem(stem), chosen[stem][0], chosen[stem][1]


def load_waveform(
    path: os.PathLike[str] | str,
    logger: logging.Logger,
    target_sample_rate: int = 16000,
) -> dict:
    """Charge l'audio en mono et, si possible, le rééchantillonne en 16 kHz."""

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            logger.info("Downmix stéréo→mono pour %s", path)
        if sample_rate != target_sample_rate:
            logger.info(
                "Rééchantillonnage de %s (%s Hz → %s Hz)",
                path,
                sample_rate,
                target_sample_rate,
            )
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
            sample_rate = target_sample_rate
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as ta_exc:  # pragma: no cover - fallback de secours
        logger.warning(
            "Décodage via torchaudio impossible (%s). Fallback soundfile utilisé.", ta_exc,
        )
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
            logger.info("Downmix stéréo→mono pour %s (fallback soundfile)", path)
        waveform = torch.from_numpy(audio).unsqueeze(0)
        if sample_rate != target_sample_rate:
            logger.warning(
                "Soundfile ne rééchantillonne pas : audio %s resté à %s Hz (pyannote rééchantillonnera si nécessaire)",
                path,
                sample_rate,
            )
        return {"waveform": waveform, "sample_rate": sample_rate}


def diarize_all(
    stems: set[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    logger = init_logger("diarize", verbose, logger)

    try:
        import torch
        globals()["torch"] = torch  # pour les fallbacks (load_waveform)
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
    # torchcodec reste optionnel grâce au fallback de préchargement.
    diar_env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
    if diar_env and "diar" not in diar_env:
        logger.warning(
            "Environnement actuel (%s) différent de l'environnement dédié diarisation."
            " Utilise de préférence l'environnement 'anime_dub_diar' décrit dans le README",
            diar_env,
        )
    elif not diar_env:
        logger.info(
            "Aucun environnement virtuel détecté. Pour éviter les conflits de dépendances,"
            " crée et active l'environnement conda 'anime_dub_diar' (voir README)."
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipeline.to(device)
        if device.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            except Exception:  # pragma: no cover - info only
                gpu_name = "GPU CUDA détecté"
            logger.info("Pipeline pyannote placé sur le GPU (%s)", gpu_name)
        else:
            logger.info("Pipeline pyannote exécuté sur CPU (CUDA indisponible)")
    except Exception as exc:  # pragma: no cover - garde-fou CUDA
        logger.warning(
            "Impossible de déplacer le pipeline sur CUDA (%s), fallback CPU.", exc
        )
        cpu_device = torch.device("cpu")
        pipeline.to(cpu_device)
        device = cpu_device

    def _write_rttm(result, file_obj, uri: str):
        """Compatibilité pyannote 3.x/4.x pour écrire un RTTM."""

        def _extract_annotation(obj):
            if hasattr(obj, "write_rttm"):
                return obj

            for attr in (
                "diarization",
                "speaker_diarization",
                "exclusive_speaker_diarization",
            ):
                nested = getattr(obj, attr, None)
                if nested is not None:
                    return nested
            return obj

        annotation = _extract_annotation(result)

        if hasattr(annotation, "uri"):
            try:
                annotation.uri = uri
            except Exception:  # pragma: no cover - setter absent/readonly
                pass

        if hasattr(annotation, "write_rttm"):
            annotation.write_rttm(file_obj)
            return

        for module_path in (
            "pyannote.core.io.rttm",
            "pyannote.core.utils.rttm",  # ancien chemin (pyannote 3.x)
        ):
            try:
                module = __import__(module_path, fromlist=["dump_rttm"])
                dump_rttm = getattr(module, "dump_rttm")
                dump_rttm(annotation, file_obj)
                return
            except ModuleNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - garde-fou
                logger.error(
                    "Impossible d'écrire le RTTM via %s : %s (résultat type %s)",
                    module_path,
                    exc,
                    type(annotation),
                )
                raise

        logger.error(
            "Impossible d'écrire le RTTM : aucun writer compatible trouvé (résultat type %s)",
            type(annotation),
        )
        raise RuntimeError("RTTM non écrit : writer introuvable")

    processed_any = False
    for stem_raw, stem_norm, wav, source_desc in iter_targets(stems, logger):
        rttm_path = diar_dir / f"{stem_norm}.rttm"

        logger.info("Diarisation en cours : %s (%s)", wav, source_desc)

        diar_input: str | dict
        if has_torchcodec:
            diar_input = str(wav)
        else:  # précharge pour contourner torchcodec manquant
            diar_input = load_waveform(wav, logger)

        diarization = pipeline(diar_input)

        with rttm_path.open("w", encoding="utf-8") as f:
            _write_rttm(diarization, f, uri=stem_norm)

        logger.info("Diarisation écrite : %s", rttm_path)
        processed_any = True

    if not processed_any:
        logger.warning(
            "Aucun stem vocal ou WAV brut (_mono16k/_full) trouvé pour la diarisation."
        )


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
