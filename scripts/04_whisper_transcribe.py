# scripts/04_whisper_transcribe.py
import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Iterable

import soundfile as sf
from faster_whisper import WhisperModel

from utils_config import ensure_directories, get_data_path
from utils_logging import init_logger, parse_stems, should_verbose
from utils_paths import normalized_filter, stem_matches_filter


def write_srt(segments, srt_path):
    def srt_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def iter_sources(
    stems_filter: set[str] | None, logger: logging.Logger
) -> Iterable[tuple[str, Path, str]]:
    """Détermine la source audio à transcrire (priorité aux stems vocaux)."""

    audio_stems = get_data_path("audio_stems_dir")
    audio_raw = get_data_path("audio_raw_dir")
    logger.debug(
        "Recherche des audios (priorité stems vocaux) dans %s puis %s", audio_stems, audio_raw
    )

    stems_filter_norm = normalized_filter(stems_filter)
    chosen: dict[str, tuple[Path, str]] = {}

    for wav in sorted(audio_stems.glob("*_vocals.wav")):
        stem = wav.stem.removesuffix("_vocals")
        if not stem_matches_filter(stem, stems_filter_norm):
            logger.debug("Ignore %s car non sélectionné (stems)", stem)
            continue
        chosen[stem] = (wav, "stems (vocals)")

    for pattern, suffix, desc in (
        ("*_mono16k.wav", "_mono16k", "audio complet mono16k"),
        ("*_full.wav", "_full", "audio complet brut"),
    ):
        for wav in sorted(audio_raw.glob(pattern)):
            stem = wav.stem.replace(suffix, "")
            if not stem_matches_filter(stem, stems_filter_norm):
                logger.debug("Ignore %s car non sélectionné (audio brut)", stem)
                continue
            chosen.setdefault(stem, (wav, desc))

    for stem in sorted(chosen):
        yield stem, chosen[stem][0], chosen[stem][1]


def ensure_mono16k(audio_path: Path, workdir: Path, logger: logging.Logger) -> Path:
    """Garantit une source mono 16 kHz (conversion ffmpeg si nécessaire)."""

    try:
        info = sf.info(str(audio_path))
        if info.channels == 1 and int(info.samplerate) == 16000:
            logger.debug("Audio déjà mono 16 kHz : %s", audio_path)
            return audio_path
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Lecture des métadonnées impossible pour %s (%s) ; conversion forcée en 16 kHz.",
            audio_path,
            exc,
        )

    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / f"{audio_path.stem}_mono16k.wav"

    if target.exists():
        try:
            t_info = sf.info(str(target))
            if t_info.channels == 1 and int(t_info.samplerate) == 16000:
                logger.debug("Réutilisation du cache 16 kHz : %s", target)
                return target
        except Exception:  # noqa: BLE001
            logger.debug("Cache 16 kHz invalide, nouvelle conversion : %s", target)

    logger.info("Conversion 16 kHz (mono) pour Whisper : %s", audio_path)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
    )
    return target


def transcribe_all(
    stems: set[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    logger = init_logger("whisper_transcribe", verbose, logger)

    paths = ensure_directories(["whisper_json_dir", "zh_srt_dir", "transcripts_root"])
    out_json = paths["whisper_json_dir"]
    out_srt = paths["zh_srt_dir"]
    whisper_work = paths["transcripts_root"] / "whisper_wav16k"

    model_size = "large-v3"
    logger.info("Chargement du modèle Whisper %s", model_size)
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    processed_any = False
    for stem, source_path, source_desc in iter_sources(stems, logger):
        wav = ensure_mono16k(source_path, whisper_work, logger)
        json_path = out_json / f"{stem}.json"
        srt_path = out_srt / f"{stem}_zh.srt"

        logger.info("Transcription (%s) : %s", source_desc, wav)

        segments_out = []
        segments, info = model.transcribe(str(wav), language="zh", beam_size=5)

        for seg in segments:
            segments_out.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })

        json_path.write_text(json.dumps({"segments": segments_out}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_srt(segments_out, srt_path)
        logger.info("Whisper OK : %s", stem)
        processed_any = True

    if not processed_any:
        logger.warning("Aucun fichier wav trouvé pour la transcription.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcription Whisper d'un ou plusieurs épisodes")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode sans suffixe à traiter")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    args = parser.parse_args()

    verbose = args.verbose or should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    logger = init_logger("whisper_transcribe", verbose)

    stems_filter = parse_stems(args.stem, logger)
    stems_display = sorted(normalized_filter(stems_filter)) if stems_filter else "tous"
    logger.info("Stems ciblés (normalisés) : %s", stems_display)

    transcribe_all(stems_filter, verbose=verbose, logger=logger)
