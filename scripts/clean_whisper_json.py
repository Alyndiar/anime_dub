"""Nettoyage des transcriptions Whisper pour la traduction.

Cette étape filtre les segments bruités, fusionne les doublons
consécutifs et produit un JSON prêt pour la traduction (champs
``text_zh``) ainsi qu'un rapport détaillé de nettoyage.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

DEFAULT_CONFIG = {
    "promo_keywords": [
        "关注",
        "订阅",
        "点赞",
        "转发",
        "打赏",
        "腾讯视频",
        "官方微信",
    ],
    "merge_short": False,
    "merge_min_length": 20,
    "merge_target_length": 35,
    "max_pause": 1.0,
    "allowed_single_chars": {"是", "嗯", "这", "叫"},
}

LATIN_RE = re.compile(r"[A-Za-z]{3,}")
HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
REPEAT_CHAR_RE = re.compile(r"^(.)\1{2,}$")
PUNCT_SPACES_RE = re.compile(r"[\s、，。,.！？!？；;:“”\"'’‘·`~\-\(\)\[\]{}…]+")


def merge_duplicates(segments: list[dict]) -> tuple[list[dict], int]:
    """Fusionne les doublons consécutifs basés sur une normalisation légère."""

    def _normalize(text: str) -> str:
        return PUNCT_SPACES_RE.sub("", text.strip())

    merged: list[dict] = []
    fusion_count = 0

    for seg in segments:
        norm = _normalize(seg["text_zh"])
        if merged:
            prev_norm = merged[-1].setdefault("_norm", _normalize(merged[-1]["text_zh"]))
            if norm and norm == prev_norm:
                merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
                fusion_count += 1
                continue
        seg_copy = dict(seg)
        seg_copy["_norm"] = norm
        merged.append(seg_copy)

    for seg in merged:
        seg.pop("_norm", None)
    return merged, fusion_count


def merge_short_segments(segments: list[dict], config: dict) -> tuple[list[dict], int]:
    """Regroupe les segments trop courts (optionnel)."""
    if not config.get("merge_short"):
        return segments, 0

    max_pause = float(config.get("max_pause", DEFAULT_CONFIG["max_pause"]))
    min_len = int(config.get("merge_min_length", DEFAULT_CONFIG["merge_min_length"]))
    target_len = int(config.get("merge_target_length", DEFAULT_CONFIG["merge_target_length"]))

    merged: list[dict] = []
    merge_count = 0
    i = 0
    while i < len(segments):
        current = dict(segments[i])
        buffer_text = current["text_zh"]
        start = current["start"]
        end = current["end"]
        j = i + 1
        while j < len(segments) and len(buffer_text) < target_len:
            gap = segments[j]["start"] - end
            if gap > max_pause:
                break
            next_text = segments[j]["text_zh"]
            prospective_len = len(buffer_text) + len(next_text)
            if len(buffer_text) >= min_len and prospective_len > target_len:
                break
            buffer_text += next_text
            end = segments[j]["end"]
            j += 1
            merge_count += 1
            if len(buffer_text) >= min_len and gap > max_pause:
                break
        merged.append({"start": start, "end": end, "text_zh": buffer_text, **{k: v for k, v in current.items() if k not in {"start", "end", "text_zh"}}})
        i = j
    return merged, merge_count


def should_drop(text: str, config: dict) -> tuple[bool, str | None]:
    stripped = text.strip()
    promo_keywords: Iterable[str] = config.get("promo_keywords", DEFAULT_CONFIG["promo_keywords"]) or []
    allowed_single = set(config.get("allowed_single_chars", DEFAULT_CONFIG["allowed_single_chars"]))

    if not stripped:
        return True, "empty"
    if any(word in stripped for word in promo_keywords):
        return True, "promo"
    if LATIN_RE.search(stripped):
        return True, "latin"
    if HANGUL_RE.search(stripped):
        return True, "hangul"
    condensed = PUNCT_SPACES_RE.sub("", stripped)
    if len(condensed) == 1 and condensed not in allowed_single:
        return True, "short_fragment"
    if REPEAT_CHAR_RE.match(condensed):
        return True, "repeat"
    return False, None


def clean_whisper_json(in_path: str, out_path: str, report_path: str, config: dict | None = None) -> None:
    """Nettoie un JSON Whisper et écrit un rapport."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    src = Path(in_path)
    data = json.loads(src.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    removed_examples: list[dict] = []
    removed_counts = {k: 0 for k in ["promo", "latin", "hangul", "empty", "short_fragment", "repeat"]}

    cleaned: list[dict] = []
    for seg in segments:
        text = str(seg.get("text", ""))
        drop, reason = should_drop(text, cfg)
        if drop:
            if reason:
                removed_counts[reason] += 1
            if len(removed_examples) < 20:
                removed_examples.append({"text": text, "reason": reason})
            continue
        cleaned.append({
            k: seg[k] for k in seg if k != "text"
        } | {"text_zh": text.strip()})

    deduped, duplicate_fusions = merge_duplicates(cleaned)
    merged, short_merges = merge_short_segments(deduped, cfg)

    output = {"segments": merged}
    Path(out_path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "segments_in": len(segments),
        "segments_after_filter": len(cleaned),
        "segments_out": len(merged),
        "removed": removed_counts,
        "duplicate_fusions": duplicate_fusions,
        "short_merges": short_merges,
        "examples_removed": removed_examples,
    }
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nettoyage des transcriptions Whisper (ZH)")
    parser.add_argument("in_path", help="JSON Whisper brut")
    parser.add_argument("out_path", help="JSON nettoyé (_clean_zh.json)")
    parser.add_argument("report_path", help="Rapport de nettoyage")
    parser.add_argument("--merge-short", action="store_true", dest="merge_short", help="Active le regroupement de segments courts")
    parser.add_argument("--max-pause", type=float, default=DEFAULT_CONFIG["max_pause"], help="Pause max (s) pour fusionner des segments courts")
    parser.add_argument("--merge-min-length", type=int, default=DEFAULT_CONFIG["merge_min_length"], help="Longueur minimale cible pour fusionner les segments courts")
    parser.add_argument("--merge-target-length", type=int, default=DEFAULT_CONFIG["merge_target_length"], help="Longueur maximale approximative après fusion de courts segments")
    args = parser.parse_args()

    config = {
        "merge_short": args.merge_short,
        "merge_min_length": args.merge_min_length,
        "merge_target_length": args.merge_target_length,
        "max_pause": args.max_pause,
    }
    clean_whisper_json(args.in_path, args.out_path, args.report_path, config)


if __name__ == "__main__":
    main()
