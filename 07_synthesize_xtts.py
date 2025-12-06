# scripts/07_synthesize_xtts.py
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from TTS.api import TTS

SEG_IN = Path("data/segments")
VOICES = Path("data/voices")
OUT = Path("data/dub_audio")
OUT.mkdir(parents=True, exist_ok=True)

# Chargement du modèle XTTS-v2
# (vérifie le nom exact du modèle disponible via TTS.list_models())
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

SAMPLE_RATE = 24000  # XTTS-v2 sort du 24kHz normal.:contentReference[oaicite:14]{index=14}  

# Mapping personnage -> fichier voix de référence (audio en mandarin du perso)
CHAR_VOICES = {
    "char_ning": VOICES / "char_ning" / "ref1.wav",
    "char_heroine": VOICES / "char_heroine" / "ref1.wav",
    # ...
}

def synthesize_episode(seg_file):
    stem = seg_file.stem.replace("_segments", "")
    data = json.loads(seg_file.read_text(encoding="utf-8"))
    segs = data["segments"]

    # Durée totale : on prend la fin du dernier segment + marge
    total_dur = max(s["end"] for s in segs) + 1.0
    total_samples = int(total_dur * SAMPLE_RATE)
    mix = np.zeros((total_samples,), dtype=np.float32)

    for seg in segs:
        text = seg["text_fr"]
        char = seg["character"]
        if char == "unknown" or char not in CHAR_VOICES:
            continue

        ref_wav = str(CHAR_VOICES[char])
        # Génération audio FR
        audio = tts.tts(
            text=text,
            speaker_wav=ref_wav,
            language="fr"
        )  # numpy array float32, 24k

        start_sample = int(seg["start"] * SAMPLE_RATE)
        end_sample = start_sample + len(audio)
        if end_sample > len(mix):
            # on tronque si besoin
            end_sample = len(mix)
            audio = audio[: end_sample - start_sample]

        mix[start_sample:end_sample] += audio

    out_wav = OUT / f"{stem}_fr_voices.wav"
    sf.write(out_wav, mix, SAMPLE_RATE)
    print("Piste voix FR écrite :", out_wav)

for seg_file in SEG_IN.glob("*_segments.json"):
    synthesize_episode(seg_file)
