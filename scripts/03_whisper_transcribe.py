# scripts/03_whisper_transcribe.py
from pathlib import Path
import json
from faster_whisper import WhisperModel

AUDIO_RAW = Path("data/audio_raw")
OUT_JSON = Path("data/transcripts/whisper_json")
OUT_SRT = Path("data/transcripts/zh_srt")
OUT_JSON.mkdir(parents=True, exist_ok=True)
OUT_SRT.mkdir(parents=True, exist_ok=True)

# Choisis la taille du modèle (tiny, base, small, medium, large-v3...)
model_size = "large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

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

for wav in AUDIO_RAW.glob("*_mono16k.wav"):
    stem = wav.stem.replace("_mono16k", "")
    json_path = OUT_JSON / f"{stem}.json"
    srt_path = OUT_SRT / f"{stem}_zh.srt"

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
