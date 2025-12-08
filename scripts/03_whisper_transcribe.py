# scripts/03_whisper_transcribe.py
import argparse
import json
from typing import Iterable
from faster_whisper import WhisperModel

from utils_config import ensure_directories, get_data_path


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


def iter_sources(stems_filter: set[str] | None) -> Iterable[str]:
    audio_raw = get_data_path("audio_raw_dir")
    for wav in sorted(audio_raw.glob("*_mono16k.wav")):
        stem = wav.stem.replace("_mono16k", "")
        if stems_filter and stem not in stems_filter:
            continue
        yield stem


def transcribe_all(stems: set[str] | None = None) -> None:
    paths = ensure_directories(["whisper_json_dir", "zh_srt_dir"])
    audio_raw = get_data_path("audio_raw_dir")
    out_json = paths["whisper_json_dir"]
    out_srt = paths["zh_srt_dir"]

    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    for stem in iter_sources(stems):
        wav = audio_raw / f"{stem}_mono16k.wav"
        json_path = out_json / f"{stem}.json"
        srt_path = out_srt / f"{stem}_zh.srt"

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
        print("Whisper OK :", stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcription Whisper d'un ou plusieurs épisodes")
    parser.add_argument("--stem", action="append", help="Nom(s) d'épisode sans suffixe à traiter")
    args = parser.parse_args()

    stems_filter = set(args.stem) if args.stem else None
    transcribe_all(stems_filter)
