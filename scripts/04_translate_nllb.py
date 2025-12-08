# scripts/04_translate_nllb.py
import argparse
import json
from typing import Iterable
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils_config import ensure_directories, get_data_path


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


def iter_sources(stems_filter: set[str] | None) -> Iterable[str]:
    src_json_dir = get_data_path("whisper_json_dir")
    for jpath in sorted(src_json_dir.glob("*.json")):
        stem = jpath.stem
        if stems_filter and stem not in stems_filter:
            continue
        yield stem


def translate_all(stems: set[str] | None = None) -> None:
    paths = ensure_directories(["whisper_json_fr_dir", "fr_srt_dir"])
    src_json_dir = get_data_path("whisper_json_dir")
    out_json_dir = paths["whisper_json_fr_dir"]
    out_srt_dir = paths["fr_srt_dir"]

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

    for stem in iter_sources(stems):
        jpath = src_json_dir / f"{stem}.json"
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

        print("Traduction OK :", stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traduction NLLB-200 d'un ou plusieurs épisodes")
    parser.add_argument("--stem", action="append", help="Nom(s) de fichier json (sans suffixe) à traduire")
    args = parser.parse_args()

    stems_filter = set(args.stem) if args.stem else None
    translate_all(stems_filter)
