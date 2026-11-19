#!/usr/bin/env python3
"""
Clean, filter and deduplicate scraped jokes.

  python clean.py                          # reads data/raw/*.jsonl -> data/clean/jokes.jsonl
  python clean.py --near-threshold 0.75    # stricter/looser near-duplicate merging

Steps
  1. Normalise text: Unicode NFC, Latin look-alike letters inside Cyrillic words
     -> Cyrillic, quotes/dashes/whitespace, dialogue dashes.
  2. Filter junk: too short/long, not Bulgarian, URLs/ads, all-caps, spammy lines.
  3. Merge records of the same joke (same site id across categories).
  4. Remove exact duplicates (punctuation/case-insensitive) across sites.
  5. Remove near-duplicates (MinHash + LSH over character 5-grams, verified with
     real Jaccard similarity) — catches retellings with small wording changes.
  6. Flag content: nsfw (dirty/sexual) and sensitive (ethnic/religious targets),
     so build_dataset.py can include or exclude them.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict

import numpy as np

# ----------------------------------------------------------------------------- #
# Normalisation
# ----------------------------------------------------------------------------- #
LAT2CYR = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "k": "к", "m": "м",
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р",
    "C": "С", "T": "Т", "X": "Х", "Y": "У",
})
CYR = re.compile(r"[а-яА-ЯѝЍ]")
LAT = re.compile(r"[a-zA-Z]")
WORD = re.compile(r"\w+", re.UNICODE)


def fix_homoglyphs(text: str) -> str:
    """'пoлицай' with a Latin 'o' -> all Cyrillic. Only touches mostly-Cyrillic words."""
    def fix(m):
        w = m.group(0)
        c, l = len(CYR.findall(w)), len(LAT.findall(w))
        if c and l and c >= l:
            return w.translate(LAT2CYR)
        return w
    return WORD.sub(fix, text)


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\r", "").replace("\xa0", " ").replace("​", "").replace("﻿", "")
    t = re.sub(r"[“”„«»]", '"', t)
    t = re.sub(r"[‘’‚`´]", "'", t)
    t = t.replace("…", "...")
    t = fix_homoglyphs(t)
    lines = []
    for line in t.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            continue
        if re.fullmatch(r"\(\d{1,4}\)|(към\s+)?вица|Преглед|Виж този виц", line, re.I):
            continue  # site chrome that slipped through scraping
        # unify dialogue dashes at line start:  "-Как си?" / "— Как си?" -> "– Как си?"
        line = re.sub(r"^[-–—―]\s*", "– ", line)
        # dash used mid-line as a dialogue/attribution separator
        line = re.sub(r"\s[-—―]\s", " – ", line)
        line = re.sub(r"([!?.]){4,}", r"\1\1\1", line)   # "!!!!!!" -> "!!!"
        line = re.sub(r"\s+([,.!?;:])", r"\1", line)     # space before punctuation
        lines.append(line)
    return "\n".join(lines).strip()


# ----------------------------------------------------------------------------- #
# Filters
# ----------------------------------------------------------------------------- #
URL_RE = re.compile(r"(https?://|www\.|\.(com|bg|net|org|eu|info|biz)\b)", re.I)
SPAM_RE = re.compile(
    r"(сподели(те)?\s+(с|във)|харесай(те)?\s+страницата|абонирай|изпрати\s+sms|"
    r"натисни|кликни|последвай(те)?|copyright|©|всички\s+права)", re.I)
AI_RE = re.compile(r"(като\s+(езиков|AI)\s+модел|ChatGPT|изкуствен\s+интелект\s+генерира)", re.I)


def reject_reason(t: str, min_chars: int, max_chars: int) -> str | None:
    if len(t) < min_chars:
        return "too_short"
    if len(t) > max_chars:
        return "too_long"
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return "no_letters"
    cyr = sum(1 for ch in letters if CYR.match(ch))
    if cyr / len(letters) < 0.75:
        return "not_bulgarian"
    if URL_RE.search(t):
        return "url"
    if SPAM_RE.search(t):
        return "spam"
    if AI_RE.search(t):
        return "ai_text"
    upp = sum(1 for ch in letters if ch.isupper())
    if len(letters) > 40 and upp / len(letters) > 0.6:
        return "all_caps"
    if t.count("\n") > 40:
        return "too_many_lines"
    # Russian rather than Bulgarian: ы э ё are not in Bulgarian
    if len(re.findall(r"[ыэёЫЭЁ]", t)) >= 3:
        return "russian"
    return None


# ----------------------------------------------------------------------------- #
# Flags
# ----------------------------------------------------------------------------- #
NSFW_CATS = re.compile(
    r"(мръсн|с\*?е?кс|\+18|18\+|проститут|онанист|самозадоволяв|импотент|некрофил|"
    r"гинеколог|лесбийк|гейов|обратни|садист|мазохист|първата\s+брачна|"
    r"mrusni|seks|sex)", re.I)
SENSITIVE_CATS = re.compile(
    r"(цигани|негри|евреи|арменци|турци|мюсюлмани|аллах|чукчи|нацисти|фашисти|"
    r"cigani|negri|evrei)", re.I)
NSFW_WORDS = re.compile(
    r"\b(ебе|еба|ебан|путк|пичк|кур(ът|а|ец)\b|чеп\b|задник|свирк|оргазъм|секс|с\*кс|курв)", re.I)


# ----------------------------------------------------------------------------- #
# Dedup helpers
# ----------------------------------------------------------------------------- #
def exact_key(t: str) -> str:
    k = re.sub(r"[^\w]+", "", t.lower())
    return hashlib.md5(k.encode("utf-8")).hexdigest()


def shingles(t: str, k: int = 5) -> np.ndarray:
    s = re.sub(r"[^\w ]+", "", re.sub(r"\s+", " ", t.lower())).strip()
    if len(s) < k:
        s = s.ljust(k)
    hs = {zlib.crc32(s[i:i + k].encode("utf-8")) for i in range(len(s) - k + 1)}
    return np.fromiter(hs, dtype=np.uint64)


class MinHashLSH:
    PRIME = np.uint64((1 << 61) - 1)

    def __init__(self, num_perm=128, bands=16, seed=7):
        assert num_perm % bands == 0
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, (1 << 31) - 1, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, (1 << 31) - 1, size=num_perm, dtype=np.uint64)
        self.bands, self.rows = bands, num_perm // bands

    def signature(self, sh: np.ndarray) -> np.ndarray:
        # (a*x + b) mod 2^32-ish; uint64 overflow wraps, fine for hashing
        v = (np.outer(sh, self.a) + self.b) & np.uint64(0xFFFFFFFF)
        return v.min(axis=0).astype(np.uint32)

    def candidate_pairs(self, sigs: np.ndarray):
        """Yield candidate pairs band by band (no global pair set -> low memory)."""
        for b in range(self.bands):
            buckets = defaultdict(list)
            lo, hi = b * self.rows, (b + 1) * self.rows
            band = np.ascontiguousarray(sigs[:, lo:hi])
            for i in range(len(band)):
                buckets[band[i].tobytes()].append(i)
            del band
            for idx in buckets.values():
                if 1 < len(idx) <= 200:          # skip degenerate mega-buckets
                    for x in range(len(idx)):
                        for y in range(x + 1, len(idx)):
                            yield idx[x], idx[y]
            del buckets


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def quality(t: str) -> tuple:
    """Prefer: ends with proper punctuation, has line breaks for dialogue, then longer."""
    return (t.rstrip()[-1:] in ".!?\")", "\n" in t, min(len(t), 600))


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/*.jsonl")
    ap.add_argument("--out", default="data/clean/jokes.jsonl")
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--near-threshold", type=float, default=0.7,
                    help="Jaccard similarity above which two jokes are treated as the same (0-1)")
    ap.add_argument("--no-near-dedup", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(args.raw))
    if not files:
        sys.exit(f"no input files match {args.raw}")

    # Records are kept as compact lists to fit small machines (e.g. 512 MB):
    #   [text, categories(list), sources(list), urls(list), ids(list), n_variants]
    T, C, S, U, I, V = range(6)

    def merge_into(g, r):
        g[C] = tuple(sorted(set(g[C]) | set(r[C])))
        g[S] = tuple(sorted(set(g[S]) | set(r[S])))
        if len(g[U]) < 5:
            g[U] = tuple(sorted(set(g[U]) | set(r[U])))[:5]
        if len(g[I]) < 10:
            g[I] = tuple(sorted(set(g[I]) | set(r[I])))[:10]
        g[V] += r[V]
        if quality(r[T]) > quality(g[T]):
            g[T] = r[T]

    # 1-3: load, normalise, filter, merge same-id records
    # Input lines may carry "category" (one) or "categories" (list).
    by_id: dict[str, list | None] = {}
    rejects = Counter()
    n_in = 0
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    rejects["bad_json"] += 1
                    continue
                n_in += 1
                cats = [c.strip() for c in (r.get("categories") or [r.get("category") or ""]) if c and c.strip()]
                rid = r["id"]
                if rid in by_id:
                    if by_id[rid] is not None and cats:
                        by_id[rid][C] = tuple(sorted(set(by_id[rid][C]) | set(cats)))
                    continue
                t = normalise(r.get("text", ""))
                why = reject_reason(t, args.min_chars, args.max_chars)
                if why:
                    rejects[why] += 1
                    by_id[rid] = None  # remember rejection so other categories skip too
                    continue
                by_id[rid] = [t, tuple(cats), (sys.intern(r.get("site", "?")),),
                              (r["url"],) if r.get("url") else (), (rid,), 1]
    recs = [v for v in by_id.values() if v is not None]
    n_ids = len(by_id)
    del by_id
    print(f"read {n_in} records from {len(files)} files -> {len(recs)} unique ids after filtering")

    # 4: exact dedup across sites
    groups: dict[str, list] = {}
    for r in recs:
        k = exact_key(r[T])
        g = groups.get(k)
        if g is None:
            groups[k] = r
        else:
            merge_into(g, r)
    recs = list(groups.values())
    del groups
    n_exact = len(recs)
    print(f"exact dedup -> {n_exact}")

    # 5: near dedup. Signatures are uint32 in one array; shingles are discarded right
    # after hashing and similarity is estimated from signature agreement (128 perms
    # -> about +-0.04), which keeps memory around 100 MB per 200k jokes.
    if not args.no_near_dedup and len(recs) > 1:
        lsh = MinHashLSH(num_perm=128, bands=16)
        sigs = np.empty((len(recs), 128), dtype=np.uint32)
        for i, r in enumerate(recs):
            sigs[i] = lsh.signature(shingles(r[T]))
            if i and i % 20000 == 0:
                print(f"  minhash {i}/{len(recs)}", flush=True)
        uf = UF(len(recs))
        n_pairs = 0
        for i, j in lsh.candidate_pairs(sigs):
            if uf.find(i) == uf.find(j):
                continue
            n_pairs += 1
            if np.count_nonzero(sigs[i] == sigs[j]) >= args.near_threshold * sigs.shape[1]:
                uf.union(i, j)
        del sigs
        clusters = defaultdict(list)
        for i in range(len(recs)):
            clusters[uf.find(i)].append(i)
        out = []
        for idx in clusters.values():
            g = recs[idx[0]]
            for i in idx[1:]:
                merge_into(g, recs[i])
            out.append(g)
        recs = out
        print(f"near dedup (similarity >= {args.near_threshold}) -> {len(recs)}  "
              f"({n_pairs} candidate pairs checked)")

    # 6: flags + write
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    flag_counts = Counter()
    cat_counts = Counter()
    recs.sort(key=lambda r: r[I][0])
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in recs:
            cats = sorted(c for c in r[C] if c)
            cat_blob = " ".join(cats)
            nsfw = bool(NSFW_CATS.search(cat_blob) or NSFW_WORDS.search(r[T]))
            sensitive = bool(SENSITIVE_CATS.search(cat_blob))
            flag_counts["nsfw"] += nsfw
            flag_counts["sensitive"] += sensitive
            cat_counts.update(cats)
            jid = hashlib.md5(r[T].encode("utf-8")).hexdigest()[:12]
            fh.write(json.dumps({
                "id": jid, "text": r[T], "categories": cats,
                "sources": list(r[S]), "urls": [u for u in r[U] if u][:5],
                "source_ids": list(r[I][:10]), "n_variants": r[V],
                "nsfw": nsfw, "sensitive": sensitive, "chars": len(r[T]),
            }, ensure_ascii=False) + "\n")

    lens = np.array([len(r[T]) for r in recs]) if recs else np.array([0])
    stats = {
        "input_records": n_in, "after_filter_unique_ids": n_ids - sum(rejects.values()),
        "after_exact_dedup": n_exact, "final": len(recs),
        "rejected": dict(rejects), "flags": dict(flag_counts),
        "chars": {"median": int(np.median(lens)), "p90": int(np.percentile(lens, 90)), "max": int(lens.max())},
        "top_categories": cat_counts.most_common(60),
    }
    with open(os.path.join(os.path.dirname(args.out) or ".", "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)
    print(f"final: {len(recs)} jokes -> {args.out}")
    print(f"rejected: {dict(rejects)}")
    print(f"flags: {dict(flag_counts)}   median length {stats['chars']['median']} chars")


if __name__ == "__main__":
    main()
