#!/usr/bin/env python3
"""
mine_new_terms.py
Detect recurring new Chinese terms from cleaned Whisper JSON (Chinese) across episodes,
then propose candidates to enrich your master glossary with minimal manual validation.

Inputs
- One or more cleaned zh JSON files (produced by clean_whisper_json.py):
  {"segments":[{"start":..,"end":..,"text_zh":"..."}, ...]}
- A master glossary JSON:
  {"entries":[{"type":"name|term|title|phrase","zh":"...","fr":"..."}], ...}

Output
- candidates.json + candidates.csv
Each candidate includes:
- zh: term candidate (CJK substring)
- count: occurrences (approx., based on n-gram scan)
- files: number of files where it appears
- contexts: up to N example contexts (zh text snippets)

Local-first design
- No ML required.
- Uses character n-gram mining (2..6) over CJK text, then filters aggressively.

Usage
  python mine_new_terms.py --input-dir /path/to/clean_jsons --glossary douluo_glossary_master_fr.json --out-dir out --top 500

Tips
- Run on your whole season/batch. Sort by "files" then "count".
- Validate a small set each batch; append to glossary; rerun.
"""
from __future__ import annotations
import argparse, json, re, csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

# Very small stoplist of ultra-common function characters/words that create noisy n-grams.
# Extend as needed once you see your candidate list.
STOP_TOKENS = set([
    "我们","你们","他们","这个","那个","这样","那样","不是","可以","不会","什么","怎么","为什么",
    "一定","已经","现在","但是","因为","所以","如果","然后","还是","就是","还有","没有","不要",
    "知道","觉得","真的","当然","可能","时候","一样","一点","一起","这里","那里","出来","进去",
    "一个","两个","三个","一下","一些","这种","那种","为了","自己","别人","之前","之后"
])

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_glossary_zh_set(gloss_path: Path) -> set[str]:
    g = load_json(gloss_path)
    entries = g.get("entries", g if isinstance(g, list) else [])
    zh_set = set()
    if isinstance(entries, list):
        for e in entries:
            zh = e.get("zh")
            if zh:
                zh_set.add(zh)
    return zh_set

def iter_clean_files(input_dir: Path) -> List[Path]:
    files = sorted([p for p in input_dir.rglob("*.json") if p.is_file()])
    return files

def extract_texts(clean_json: Dict[str, Any]) -> List[str]:
    segs = clean_json.get("segments", [])
    texts = []
    for s in segs:
        t = s.get("text_zh") or s.get("text") or ""
        t = t.strip()
        if t:
            texts.append(t)
    return texts

def ngrams_from_cjk(text: str, n_min: int, n_max: int) -> List[str]:
    out = []
    for m in CJK_RE.finditer(text):
        s = m.group(0)
        L = len(s)
        for n in range(n_min, n_max+1):
            if L < n:
                continue
            for i in range(0, L-n+1):
                out.append(s[i:i+n])
    return out

def looks_like_noise(term: str) -> bool:
    if not term:
        return True
    # numeric or mixed digits
    if re.search(r"\d", term):
        return True
    # too short handled by n_min
    if term in STOP_TOKENS:
        return True
    # repeated single char patterns (e.g., 魂魂魂)
    if len(set(term)) == 1 and len(term) >= 2:
        return True
    # common endings that create tons of junk (tune later)
    if term.endswith(("的时候","一下","一点","一样","可以","不会","不是")):
        return True
    return False

def collect_candidates(files: List[Path], glossary_zh: set[str], n_min: int, n_max: int, ctx_per_term: int) -> Tuple[Counter, Dict[str,set], Dict[str,List[str]]]:
    counts = Counter()
    term_files: Dict[str, set] = defaultdict(set)
    contexts: Dict[str, List[str]] = defaultdict(list)

    for fp in files:
        data = load_json(fp)
        texts = extract_texts(data)
        seen_in_file = set()
        for t in texts:
            ngrams = ngrams_from_cjk(t, n_min, n_max)
            for ng in ngrams:
                if ng in glossary_zh:
                    continue
                if looks_like_noise(ng):
                    continue
                counts[ng] += 1
                seen_in_file.add(ng)
                # store a few contexts
                if len(contexts[ng]) < ctx_per_term:
                    # keep a short snippet for review
                    snippet = t
                    if len(snippet) > 80:
                        snippet = snippet[:80] + "…"
                    contexts[ng].append(snippet)
        for ng in seen_in_file:
            term_files[ng].add(fp.name)
    return counts, term_files, contexts

def post_filter(counts: Counter, term_files: Dict[str,set], min_count: int, min_files: int) -> List[Tuple[str,int,int]]:
    out = []
    for term, c in counts.items():
        fcount = len(term_files.get(term, set()))
        if c < min_count:
            continue
        if fcount < min_files:
            continue
        out.append((term, c, fcount))
    # Prefer terms seen in many files; then high count; then longer (often more specific)
    out.sort(key=lambda x: (-x[2], -x[1], -len(x[0]), x[0]))
    return out

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["zh","count","files","contexts"])
        for r in rows:
            w.writerow([r["zh"], r["count"], r["files"], " | ".join(r["contexts"])])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing cleaned zh JSON files")
    ap.add_argument("--glossary", required=True, help="Master glossary JSON")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--min-count", type=int, default=8, help="Min occurrences across corpus")
    ap.add_argument("--min-files", type=int, default=2, help="Min distinct files where term appears")
    ap.add_argument("--ctx", type=int, default=4, help="Contexts per term")
    ap.add_argument("--top", type=int, default=500, help="Max candidates to output")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.rglob("*.json") if p.is_file()]
    if not files:
        raise SystemExit(f"No .json files found in {input_dir}")

    glossary_zh = load_glossary_zh_set(Path(args.glossary))

    counts, term_files, contexts = collect_candidates(files, glossary_zh, args.n_min, args.n_max, args.ctx)
    ranked = post_filter(counts, term_files, args.min_count, args.min_files)[:args.top]

    rows = []
    for term, c, fcount in ranked:
        rows.append({
            "zh": term,
            "count": c,
            "files": fcount,
            "contexts": contexts.get(term, [])[:args.ctx]
        })

    save_json(out_dir / "candidates.json", {"candidates": rows})
    write_csv(out_dir / "candidates.csv", rows)
    print(f"Found {len(rows)} candidates. Wrote: candidates.json, candidates.csv in {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
