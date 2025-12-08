# scripts/07_synthesize_xtts.py
import argparse
from pathlib import Path
import json
from typing import Iterable
import numpy as np
import soundfile as sf
from TTS.api import TTS

from utils_config import PROJECT_ROOT, ensure_directories, get_data_path, load_xtts_config


def build_char_voices(reference_cfg):
    voices = {}
    for char, ref_path in reference_cfg.items():
        if not ref_path:
            continue
        path = Path(ref_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            voices[char] = path
    return voices


def synthesize_episode(seg_file, tts, char_voices, sample_rate, out_dir):
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
            continue

        ref_wav = str(char_voices[char])
        audio = tts.tts(
            text=text,
            speaker_wav=ref_wav,
            language="fr"
        )

        start_sample = int(seg["start"] * sample_rate)
        end_sample = start_sample + len(audio)
        if end_sample > len(mix):
            end_sample = len(mix)
            audio = audio[: end_sample - start_sample]

        mix[start_sample:end_sample] += audio

    out_wav = out_dir / f"{stem}_fr_voices.wav"
    sf.write(out_wav, mix, sample_rate)
    print("Piste voix FR écrite :", out_wav)


def iter_segments(stems_filter: set[str] | None) -> Iterable[str]:
    seg_in = get_data_path("segments_dir")
    for seg_file in sorted(seg_in.glob("*_segments.json")):
        stem = seg_file.stem.replace("_segments", "")
        if stems_filter and stem not in stems_filter:
            continue
        yield stem


def synthesize_all(stems: set[str] | None = None):
    paths = ensure_directories(["dub_audio_dir"])
    seg_in = get_data_path("segments_dir")
    out_dir = paths["dub_audio_dir"]

    xtts_cfg = load_xtts_config()
    model_name = xtts_cfg.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
    device = xtts_cfg.get("device", "cuda")
    sample_rate = int(xtts_cfg.get("sample_rate", 24000))
    char_voices = build_char_voices(xtts_cfg.get("reference_voices", {}))

    tts = TTS(model_name).to(device)

    for stem in iter_segments(stems):
        seg_file = seg_in / f"{stem}_segments.json"
        synthesize_episode(seg_file, tts, char_voices, sample_rate, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthèse XTTS pour un ou plusieurs épisodes")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode à traiter")
    args = parser.parse_args()

    stems_filter = set(args.stem) if args.stem else None
    synthesize_all(stems_filter)
