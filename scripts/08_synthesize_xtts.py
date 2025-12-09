# scripts/08_synthesize_xtts.py
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from TTS.api import TTS

from utils_config import PROJECT_ROOT, ensure_directories, get_data_path, load_xtts_config
from utils_logging import init_logger, parse_stems, should_verbose


def build_char_voices(reference_cfg, logger: logging.Logger):
    voices = {}
    for char, ref_path in reference_cfg.items():
        if not ref_path:
            logger.debug("Voix ignorée pour %s (aucun chemin)", char)
            continue
        path = Path(ref_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            voices[char] = path
            logger.debug("Voix chargée pour %s : %s", char, path)
        else:
            logger.warning("Fichier de référence introuvable pour %s : %s", char, path)
    return voices


def synthesize_episode(
    seg_file,
    tts,
    char_voices,
    sample_rate,
    out_dir,
    logger: logging.Logger,
):
    stem = seg_file.stem.replace("_segments", "")
    data = json.loads(seg_file.read_text(encoding="utf-8"))
    segs = data["segments"]

    total_dur = max(s["end"] for s in segs) + 1.0
    total_samples = int(total_dur * sample_rate)
    mix = np.zeros((total_samples,), dtype=np.float32)

    for seg in segs:
        text = seg["text_fr"]
        char = seg["character"]
        if char == "unknown" or char not in char_voices:
            logger.debug("Seg %s ignoré (personnage %s sans voix)", seg.get("id"), char)
            continue

        ref_wav = str(char_voices[char])
        logger.debug("Synthèse segment %s avec %s", seg.get("id"), ref_wav)
        audio = tts.tts(
            text=text,
            speaker_wav=ref_wav,
            language="fr",
        )

        start_sample = int(seg["start"] * sample_rate)
        end_sample = start_sample + len(audio)
        if end_sample > len(mix):
            end_sample = len(mix)
            audio = audio[: end_sample - start_sample]

        mix[start_sample:end_sample] += audio

    out_wav = out_dir / f"{stem}_fr_voices.wav"
    sf.write(out_wav, mix, sample_rate)
    logger.info("Piste voix FR écrite : %s", out_wav)


def iter_segments(stems_filter: set[str] | None, logger: logging.Logger) -> Iterable[str]:
    seg_in = get_data_path("segments_dir")
    logger.debug("Recherche des segments dans %s", seg_in)
    for seg_file in sorted(seg_in.glob("*_segments.json")):
        stem = seg_file.stem.replace("_segments", "")
        if stems_filter and stem not in stems_filter:
            logger.debug("Ignore %s car non sélectionné", stem)
            continue
        yield stem


def synthesize_all(
    stems: set[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
):
    logger = init_logger("synthesize_xtts", verbose, logger)

    paths = ensure_directories(["dub_audio_dir"])
    seg_in = get_data_path("segments_dir")
    out_dir = paths["dub_audio_dir"]

    xtts_cfg = load_xtts_config()
    model_name = xtts_cfg.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
    device = xtts_cfg.get("device", "cuda")
    sample_rate = int(xtts_cfg.get("sample_rate", 24000))
    char_voices = build_char_voices(xtts_cfg.get("reference_voices", {}), logger)

    logger.info("Initialisation du modèle XTTS : %s", model_name)
    tts = TTS(model_name).to(device)

    processed_any = False
    for stem in iter_segments(stems, logger):
        seg_file = seg_in / f"{stem}_segments.json"
        logger.info("Synthèse XTTS pour %s", seg_file)
        synthesize_episode(seg_file, tts, char_voices, sample_rate, out_dir, logger)
        processed_any = True

    if not processed_any:
        logger.warning("Aucun fichier de segments trouvé pour la synthèse.")


def main():
    parser = argparse.ArgumentParser(description="Synthèse XTTS pour un ou plusieurs épisodes")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode à traiter")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    args = parser.parse_args()

    verbose = args.verbose or should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    logger = init_logger("synthesize_xtts", verbose)

    stems_filter = parse_stems(args.stem, logger)
    logger.info("Stems ciblés : %s", sorted(stems_filter) if stems_filter else "tous")

    synthesize_all(stems_filter, verbose=verbose, logger=logger)


if __name__ == "__main__":
    main()
