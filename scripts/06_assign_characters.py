# scripts/06_assign_characters.py
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from pyannote.audio import Model

from utils_config import (
    PROJECT_ROOT,
    ensure_directories,
    get_data_path,
    load_characters_config,
)
from utils_logging import init_logger, parse_stems, should_verbose

SAMPLE_RATE = 16000


def parse_rttm(rttm_path: Path):
    segments = []
    for line in rttm_path.read_text().splitlines():
        if line.startswith("SPEAKER"):
            parts = line.split()
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            segments.append({"speaker": spk, "start": start, "end": start + dur})
    return segments


def load_audio(wav_path: Path, logger: logging.Logger):
    audio, sr = sf.read(wav_path)
    if sr != SAMPLE_RATE:
        logger.warning(
            "Taux d'échantillonnage inattendu (%s Hz) pour %s, attendu %s Hz", sr, wav_path, SAMPLE_RATE
        )
    return audio


def get_segment_embedding(embed_model, audio, start, end):
    s = int(start * SAMPLE_RATE)
    e = int(end * SAMPLE_RATE)
    chunk = audio[s:e]
    emb = embed_model({"waveform": np.expand_dims(chunk, 0), "sample_rate": SAMPLE_RATE})
    return emb.detach().cpu().numpy().squeeze()


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_character_bank():
    cfg = load_characters_config()
    bank = {}
    for char_key, char_cfg in cfg.get("characters", {}).items():
        emb_path = char_cfg.get("embedding_path")
        if not emb_path:
            continue
        path = Path(emb_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            bank[char_key] = np.load(path)
    matching_cfg = cfg.get("matching", {})
    threshold = float(matching_cfg.get("similarity_threshold", 0.7))
    fallback = matching_cfg.get("fallback_character", "unknown")
    return bank, threshold, fallback


def match_character(emb, bank, threshold, fallback_char):
    best_char, best_sim = fallback_char, -1.0
    for char, ref in bank.items():
        sim = cosine(emb, ref)
        if sim > best_sim:
            best_sim = sim
            best_char = char
    if best_sim < threshold:
        return fallback_char, best_sim
    return best_char, best_sim


def assign_for_episode(
    embed_model,
    bank,
    threshold,
    fallback_char,
    fr_json: Path,
    diar_dir: Path,
    audio_raw: Path,
    seg_out_dir: Path,
    logger: logging.Logger,
):
    stem = fr_json.stem.replace("_fr", "")
    diar_rttm = diar_dir / f"{stem}.rttm"
    wav_path = audio_raw / f"{stem}_mono16k.wav"

    if not diar_rttm.exists():
        logger.warning("Pas de diarisation pour %s", stem)
        return

    diar_segments = parse_rttm(diar_rttm)
    diar_segments.sort(key=lambda x: x["start"])

    data = json.loads(fr_json.read_text(encoding="utf-8"))
    text_segments = data["segments"]

    audio = load_audio(wav_path, logger)

    merged = []
    for seg in text_segments:
        mid = (seg["start"] + seg["end"]) / 2.0
        speaker = None
        for d in diar_segments:
            if d["start"] <= mid <= d["end"]:
                speaker = d["speaker"]
                seg_start, seg_end = d["start"], d["end"]
                break
        if speaker is None:
            speaker = "unknown"
            seg_start, seg_end = seg["start"], seg["end"]

        emb = get_segment_embedding(embed_model, audio, seg_start, seg_end)
        char, score = match_character(emb, bank, threshold, fallback_char)

        merged.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "character": char,
            "score": float(score),
            "text_fr": seg["text_fr"],
        })

    out_path = seg_out_dir / f"{stem}_segments.json"
    out_path.write_text(json.dumps({"segments": merged}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Segments fusionnés écrits : %s", out_path)


def iter_sources(stems_filter: set[str] | None, logger: logging.Logger) -> Iterable[str]:
    transcripts_fr = get_data_path("whisper_json_fr_dir")
    logger.debug("Recherche des transcriptions FR dans %s", transcripts_fr)
    for fr_json in sorted(transcripts_fr.glob("*_fr.json")):
        stem = fr_json.stem.replace("_fr", "")
        if stems_filter and stem not in stems_filter:
            logger.debug("Ignore %s car non sélectionné", stem)
            continue
        yield stem


def assign_all(
    stems: set[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
):
    logger = init_logger("assign_characters", verbose, logger)

    paths = ensure_directories(["segments_dir"])
    diar_dir = get_data_path("diarization_dir")
    audio_raw = get_data_path("audio_raw_dir")
    transcripts_fr = get_data_path("whisper_json_fr_dir")
    seg_out_dir = paths["segments_dir"]

    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "Définis HF_TOKEN=ton_token_HF"
    logger.info("Chargement du modèle d'embedding pyannote")
    embed_model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=hf_token,
    )

    bank, threshold, fallback_char = load_character_bank()
    if not bank:
        logger.warning(
            "Banque de personnages vide : ajoute des embeddings dans data/speaker_bank."
        )

    processed_any = False
    for stem in iter_sources(stems, logger):
        fr_json = transcripts_fr / f"{stem}_fr.json"
        logger.info("Appariement des personnages pour %s", fr_json)
        assign_for_episode(
            embed_model,
            bank,
            threshold,
            fallback_char,
            fr_json,
            diar_dir,
            audio_raw,
            seg_out_dir,
            logger,
        )
        processed_any = True

    if not processed_any:
        logger.warning("Aucun fichier de transcription FR trouvé pour l'appariement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Association segments/locuteurs pour un ou plusieurs épisodes")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode sans suffixe")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    args = parser.parse_args()

    verbose = args.verbose or should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    logger = init_logger("assign_characters", verbose)

    stems_filter = parse_stems(args.stem, logger)
    logger.info("Stems ciblés : %s", sorted(stems_filter) if stems_filter else "tous")

    assign_all(stems_filter, verbose=verbose, logger=logger)
