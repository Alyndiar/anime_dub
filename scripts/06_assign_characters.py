# scripts/06_assign_characters.py
from pathlib import Path
import json
import numpy as np
import soundfile as sf

DIAR = Path("data/diarization")
TRANS_FR = Path("data/transcripts/whisper_json_fr")
AUDIO_RAW = Path("data/audio_raw")
BANK = Path("data/speaker_bank")
SEG_OUT = Path("data/segments")
SEG_OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

# Chargement de la banque (simple : 1 profil par perso)
CHAR_BANK = {
    # "char_ning": np.load("data/speaker_bank/char_ning.npy"),
    # ...
}

def parse_rttm(rttm_path):
    # Format RTTM: on parse juste ce qu'il faut
    segments = []
    for line in rttm_path.read_text().splitlines():
        if line.startswith("SPEAKER"):
            parts = line.split()
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            segments.append({"speaker": spk, "start": start, "end": start + dur})
    return segments

def load_audio(wav_path):
    audio, sr = sf.read(wav_path)
    assert sr == SAMPLE_RATE
    return audio

def get_segment_embedding(embed_model, audio, start, end):
    s = int(start * SAMPLE_RATE)
    e = int(end * SAMPLE_RATE)
    chunk = audio[s:e]
    emb = embed_model({"waveform": np.expand_dims(chunk, 0), "sample_rate": SAMPLE_RATE})
    return emb.detach().cpu().numpy().squeeze()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_character(emb):
    best_char, best_sim = None, -1.0
    for char, ref in CHAR_BANK.items():
        sim = cosine(emb, ref)
        if sim > best_sim:
            best_sim = sim
            best_char = char
    if best_sim < 0.7:  # seuil à ajuster
        return "unknown", best_sim
    return best_char, best_sim

# TODO: initialiser embed_model comme dans build_speaker_bank.py

for fr_json in TRANS_FR.glob("*_fr.json"):
    stem = fr_json.stem.replace("_fr", "")
    diar_rttm = DIAR / f"{stem}.rttm"
    wav_path = AUDIO_RAW / f"{stem}_mono16k.wav"

    if not diar_rttm.exists():
        print("Pas de diarisation pour", stem)
        continue

    diar_segments = parse_rttm(diar_rttm)
    diar_segments.sort(key=lambda x: x["start"])

    data = json.loads(fr_json.read_text(encoding="utf-8"))
    text_segments = data["segments"]

    audio = load_audio(wav_path)

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
        char, score = match_character(emb)

        merged.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "character": char,
            "score": float(score),
            "text_fr": seg["text_fr"],
        })

    out_path = SEG_OUT / f"{stem}_segments.json"
    out_path.write_text(json.dumps({"segments": merged}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Segments fusionnés écrits :", out_path)
