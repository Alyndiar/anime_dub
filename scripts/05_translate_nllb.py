# scripts/05_translate_nllb.py
import argparse
import json
import logging
import os
from typing import Iterable
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils_config import ensure_directories, get_data_path
from utils_logging import init_logger, parse_stems, should_verbose
from utils_paths import normalized_filter, stem_matches_filter


# Codes FLORES pour chinois simplifié / français
SRC_LANG = "zho_Hans"
TGT_LANG = "fra_Latn"


def translate_batch(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
            max_length=256,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


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
        lines.append(seg["text_fr"])
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def iter_sources(stems_filter: set[str] | None, logger: logging.Logger) -> Iterable[str]:
    src_json_dir = get_data_path("whisper_json_dir")
    logger.debug("Recherche des transcriptions JSON dans %s", src_json_dir)
    stems_filter_norm = normalized_filter(stems_filter)
    for jpath in sorted(src_json_dir.glob("*.json")):
        stem = jpath.stem
        if not stem_matches_filter(stem, stems_filter_norm):
            logger.debug("Ignore %s car non sélectionné", stem)
            continue
        yield stem


def translate_all(
    stems: set[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    logger = init_logger("translate_nllb", verbose, logger)

    paths = ensure_directories(["whisper_json_fr_dir", "fr_srt_dir"])
    src_json_dir = get_data_path("whisper_json_dir")
    out_json_dir = paths["whisper_json_fr_dir"]
    out_srt_dir = paths["fr_srt_dir"]

    model_name = "facebook/nllb-200-1.3B"
    logger.info("Chargement du modèle NLLB %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

    processed_any = False
    for stem in iter_sources(stems, logger):
        jpath = src_json_dir / f"{stem}.json"
        logger.info("Traduction : %s", jpath)
        data = json.loads(jpath.read_text(encoding="utf-8"))
        segs = data["segments"]

        texts = [s["text"] for s in segs]
        batch_size = 16
        translations = []
        for i in tqdm(range(0, len(texts), batch_size), desc=stem):
            batch = texts[i:i+batch_size]
            translations.extend(translate_batch(batch, tokenizer, model))

        for s, fr in zip(segs, translations):
            s["text_fr"] = fr.strip()

        out_json = out_json_dir / f"{stem}_fr.json"
        out_json.write_text(json.dumps({"segments": segs}, ensure_ascii=False, indent=2), encoding="utf-8")

        out_srt = out_srt_dir / f"{stem}_fr.srt"
        write_srt(segs, out_srt)

        logger.info("Traduction OK : %s", stem)
        processed_any = True

    if not processed_any:
        logger.warning("Aucun fichier json Whisper trouvé pour la traduction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traduction NLLB-200 d'un ou plusieurs épisodes")
    parser.add_argument("--stem", action="append", help="Nom(s) de fichier json (sans suffixe) à traduire")
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    args = parser.parse_args()

    verbose = args.verbose or should_verbose(os.environ.get("ANIME_DUB_VERBOSE"))
    logger = init_logger("translate_nllb", verbose)

    stems_filter = parse_stems(args.stem, logger)
    stems_display = sorted(normalized_filter(stems_filter)) if stems_filter else "tous"
    logger.info("Stems ciblés (normalisés) : %s", stems_display)

    translate_all(stems_filter, verbose=verbose, logger=logger)
