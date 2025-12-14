#!/usr/bin/env python3
"""
qa_validate_glossary.py
Validate (and optionally enforce) a Douluo Dalu glossary on translated subtitle JSON.

Input JSON expected:
{
  "segments": [
    {"start": 0.0, "end": 1.23, "text_zh": "...", "text_fr": "..."},
    ...
  ]
}

Glossary format expected:
{
  "entries": [{"type":"name|term|title|phrase","zh":"...","fr":"...","note": "..."}],
  "lock_rules": {...}
}

Modes:
- --mode report   : only report violations (default)
- --mode enforce  : replace/insert canonical FR where possible

Outputs:
- updated JSON (optional) when enforce
- a report JSON listing violations

Usage:
  python qa_validate_glossary.py in_fr.json glossary_master.json report.json --mode enforce --out out_fr_fixed.json
"""
from __future__ import annotations
import json, re, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

@dataclass
class Entry:
    typ: str
    zh: str
    fr: str
    note: str = ""

def build_entries(gloss: Dict[str, Any]) -> List[Entry]:
    entries = []
    for e in gloss.get("entries", []):
        entries.append(Entry(e.get("type","term"), e["zh"], e["fr"], e.get("note","")))
    # Prefer longer zh first to avoid partial matches (e.g., 武魂殿 vs 武魂)
    entries.sort(key=lambda x: len(x.zh), reverse=True)
    return entries

def contains_zh(hay: str, needle: str) -> bool:
    return needle in (hay or "")

def normalize_fr(s: str) -> str:
    return (s or "").strip()

def ensure_contains(fr: str, must: str) -> bool:
    # simple contains check; can be extended with regex variants
    return must in (fr or "")

def enforce_term(fr: str, must: str) -> Tuple[str, str]:
    """
    Best-effort enforcement:
    - If missing, append with separator.
    This avoids inventing context. You can customize for your style.
    Returns (new_fr, action)
    """
    fr = normalize_fr(fr)
    if not fr:
        return must, "inserted"
    if must in fr:
        return fr, "ok"
    # Append in a fansub-friendly way
    sep = " — " if not fr.endswith((".", "!", "?", "…")) else " "
    return fr + sep + must, "appended"

def enforce_name(fr: str, name: str) -> Tuple[str, str]:
    fr = normalize_fr(fr)
    if name in fr:
        return fr, "ok"
    if not fr:
        return name, "inserted"
    # Replace common mistranslations or spacing variants could go here.
    # For safety, append rather than replace.
    return fr + " (" + name + ")", "appended"

def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    in_path = sys.argv[1]
    gloss_path = sys.argv[2]
    report_path = sys.argv[3]

    mode = "report"
    out_path = None
    args = sys.argv[4:]
    for i, a in enumerate(args):
        if a == "--mode" and i+1 < len(args):
            mode = args[i+1]
        if a == "--out" and i+1 < len(args):
            out_path = args[i+1]

    data = load_json(in_path)
    gloss = load_json(gloss_path)
    segs: List[Dict[str, Any]] = data.get("segments", [])
    entries = build_entries(gloss)

    violations = []
    changed = 0

    for idx, seg in enumerate(segs):
        zh = seg.get("text_zh","") or seg.get("text","") or ""
        fr = seg.get("text_fr","") or ""
        original_fr = fr

        for ent in entries:
            if not contains_zh(zh, ent.zh):
                continue

            if ent.typ == "name":
                if not ensure_contains(fr, ent.fr):
                    if mode == "enforce":
                        fr, action = enforce_name(fr, ent.fr)
                        if action != "ok":
                            changed += 1
                    else:
                        action = "missing"
                    violations.append({
                        "segment_index": idx,
                        "zh": zh,
                        "fr": original_fr,
                        "expected": ent.fr,
                        "type": ent.typ,
                        "action": action,
                        "term": ent.zh,
                        "note": ent.note
                    })
            else:
                if not ensure_contains(fr, ent.fr):
                    if mode == "enforce":
                        fr, action = enforce_term(fr, ent.fr)
                        if action != "ok":
                            changed += 1
                    else:
                        action = "missing"
                    violations.append({
                        "segment_index": idx,
                        "zh": zh,
                        "fr": original_fr,
                        "expected": ent.fr,
                        "type": ent.typ,
                        "action": action,
                        "term": ent.zh,
                        "note": ent.note
                    })

        if mode == "enforce" and fr != original_fr:
            seg["text_fr"] = fr

    report = {
        "mode": mode,
        "segments_total": len(segs),
        "violations": violations,
        "violations_total": len(violations),
        "segments_changed": changed
    }
    save_json(report_path, report)
    if mode == "enforce":
        if not out_path:
            print("ERROR: --out is required in enforce mode", file=sys.stderr)
            return 2
        save_json(out_path, data)
    print(f"Wrote report: {report_path} (violations={len(violations)}; changed={changed})")
    if mode == "enforce":
        print(f"Wrote fixed JSON: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
