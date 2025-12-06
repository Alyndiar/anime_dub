# scripts/04_translate_nllb.py
from pathlib import Path
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

IN_JSON = Path("data/transcripts/whisper_json")
OUT_JSON = Path("data/transcripts/whisper_json_fr")
OUT_SRT = Path("data/transcripts/fr_srt")
OUT_JSON.mkdir(parents=True, exist_ok=True)
OUT_SRT.mkdir(parents=True, exist_ok=True)

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

# Codes FLORES pour chinois simplifié / français
SRC_LANG = "zho_Hans"
TGT_LANG = "fra_Latn"

def translate_batch(texts):
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

for jpath in IN_JSON.glob("*.json"):
    stem = jpath.stem
    data = json.loads(jpath.read_text(encoding="utf-8"))
    segs = data["segments"]

    texts = [s["text"] for s in segs]
    batch_size = 16
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc=stem):
        batch = texts[i:i+batch_size]
        translations.extend(translate_batch(batch))

    for s, fr in zip(segs, translations):
        s["text_fr"] = fr.strip()

    out_json = OUT_JSON / f"{stem}_fr.json"
    out_json.write_text(json.dumps({"segments": segs}, ensure_ascii=False, indent=2), encoding="utf-8")

    out_srt = OUT_SRT / f"{stem}_fr.srt"
    write_srt(segs, out_srt)

    print("Traduction OK :", stem)
