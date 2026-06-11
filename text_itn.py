# pyright: reportMissingImports=false

"""Inverse text normalization for ASR output: spoken form -> written form.

Nemotron (and most ASR models) transcribe numbers as words, e.g.
    "two point one"      -> reference is "2.1"
    "eighty thousand"    -> reference is "80,000"
    "PS five"            -> reference is "PS5"
This module rewrites the spoken form into the compact written form so that WER/CER
against digit-style references is not inflated by pure formatting differences.

Dependency-free and rule-based on purpose: the number grammar is small and the
brand-compaction rules are domain-specific (sales/product queries), so a transparent
ruleset you can extend beats a heavy WFST dependency here.

Use as a library:
    from text_itn import inverse_normalize
    inverse_normalize("which two point one sound bar")  -> "which 2.1 sound bar"

Or as a CLI over a predictions JSONL (adds a normalized field, optionally recomputes WER):
    python text_itn.py --in outputs/preds.jsonl --field pred --out outputs/preds_itn.jsonl \
        --ref-field ref --wer
    python text_itn.py --text "do you have a PS five under eighty thousand"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Number vocabulary
# ---------------------------------------------------------------------------

# Words worth at most 99 that simply add to the current accumulator.
_SMALL = {
    "zero": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
# Multiplier words. "hundred" multiplies the running group; the rest flush it.
_HUNDRED = 100
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

# Words allowed inside a number run after a leading number word.
_CONNECTORS = {"and", "point", "hundred", *_SCALES}
_NUMWORDS = set(_SMALL) | {"hundred"} | set(_SCALES)
# Single digits usable after "point" (decimals are read digit-by-digit).
_DIGIT_WORDS = {w: v for w, v in _SMALL.items() if v <= 9}

# Run regex: starts with a real number word, then any number words / connectors,
# separated by spaces or hyphens (so "twenty-five" is one run). Longest-first
# alternation so "nineteen" wins over "nine".
_VOCAB = sorted(_NUMWORDS | {"and", "point"}, key=len, reverse=True)
_WORD = "(?:" + "|".join(re.escape(w) for w in _VOCAB) + ")"
_START = "(?:" + "|".join(re.escape(w) for w in sorted(_NUMWORDS, key=len, reverse=True)) + ")"
_RUN_RE = re.compile(rf"\b{_START}(?:[\s\-]+{_WORD})*\b", re.IGNORECASE)


def _parse_run(words: list[str], *, thousands_sep: bool) -> tuple[str | None, int]:
    """Parse a list of lowercased number words into (written_number, n_consumed).

    Returns (None, 0) if no real number was found. ``n_consumed`` is the number of
    leading tokens that belong to the number; trailing connectors like a dangling
    "and"/"point" are left for the caller to re-append unchanged.
    """
    total = 0          # flushed scales (thousands, millions, ...)
    current = 0        # current group (< 1000)
    saw_number = False
    last_real = -1     # index of the last token that contributed a digit
    i, n = 0, len(words)

    while i < n:
        w = words[i]
        if w == "and":
            i += 1
            continue
        if w == "point":
            break
        if w in _SMALL:
            current += _SMALL[w]
        elif w == "hundred":
            current = (current or 1) * _HUNDRED
        elif w in _SCALES:
            total += (current or 1) * _SCALES[w]
            current = 0
        else:
            break
        saw_number = True
        last_real = i
        i += 1

    int_val = total + current

    # Optional decimal part: "point" followed by single-digit words.
    dec_digits = ""
    if i < n and words[i] == "point":
        j = i + 1
        digs: list[str] = []
        while j < n and words[j] in _DIGIT_WORDS:
            digs.append(str(_DIGIT_WORDS[words[j]]))
            last_real = j
            j += 1
        if digs:
            dec_digits = "".join(digs)

    if not saw_number and not dec_digits:
        return None, 0

    if dec_digits:
        written = f"{int_val}.{dec_digits}"
    elif thousands_sep:
        written = f"{int_val:,}"
    else:
        written = str(int_val)
    return written, last_real + 1


def spoken_numbers_to_digits(text: str, *, thousands_sep: bool = True) -> str:
    """Rewrite spoken number runs in ``text`` into written digits."""

    def repl(m: re.Match) -> str:
        span = m.group(0)
        words = re.split(r"[\s\-]+", span)
        written, consumed = _parse_run([w.lower() for w in words], thousands_sep=thousands_sep)
        if written is None:
            return span
        leftover = words[consumed:]
        return written + (" " + " ".join(leftover) if leftover else "")

    return _RUN_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Brand / model-number compaction ("PS 5" -> "PS5", "RTX 4090" -> "RTX4090")
# ---------------------------------------------------------------------------

# Known brand/model prefixes that glue to a following number with no space.
# Case-insensitive; extend via --brands on the CLI.
_DEFAULT_BRANDS = {
    "PS", "RTX", "GTX", "RX", "GT", "I3", "I5", "I7", "I9",
    "R5", "R7", "R9", "M1", "M2", "M3", "SE", "XR", "XS",
}

_BRAND_GLUE_RE = re.compile(r"\b([A-Za-z]{2,6})[ \t]+(\d[\d,\.]*)\b")


def compact_brand_numbers(text: str, brands: set[str]) -> str:
    """Glue a brand/model letter token to an immediately following number.

    Joins when the letter token is ALL-CAPS (e.g. "PS", "RTX") or is in ``brands``.
    This is conservative on purpose so it does not eat "room 5" / "under 80,000".
    """
    brands_uc = {b.upper() for b in brands}

    def repl(m: re.Match) -> str:
        prefix, number = m.group(1), m.group(2)
        if prefix.isupper() or prefix.upper() in brands_uc:
            # Model numbers like RTX4090 never carry thousands separators.
            return prefix + number.replace(",", "")
        return m.group(0)

    return _BRAND_GLUE_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inverse_normalize(
    text: str,
    *,
    thousands_sep: bool = True,
    compact_brands: bool = True,
    brands: set[str] | None = None,
) -> str:
    """Full spoken -> written normalization pipeline for one string."""
    out = spoken_numbers_to_digits(text, thousands_sep=thousands_sep)
    if compact_brands:
        out = compact_brand_numbers(out, brands if brands is not None else _DEFAULT_BRANDS)
    out = re.sub(r"[ \t]+", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--in", dest="in_path", type=str, help="Input predictions JSONL")
    src.add_argument("--text", type=str, help="Normalize a single string and print it")

    p.add_argument("--field", type=str, default="pred", help="JSONL field to normalize (default: pred)")
    p.add_argument("--out-field", type=str, default="", help="Field to write (default: <field>_itn)")
    p.add_argument("--out", type=str, default="", help="Output JSONL path (default: alongside input)")

    p.add_argument("--no-thousands-sep", action="store_true", help="Emit 80000 instead of 80,000")
    p.add_argument("--no-brand-compaction", action="store_true", help="Do not glue 'PS 5' -> 'PS5'")
    p.add_argument("--brands", type=str, default="", help="Extra brand prefixes, comma-separated")

    p.add_argument("--ref-field", type=str, default="", help="Reference field for optional WER comparison")
    p.add_argument("--wer", action="store_true", help="Report WER/CER before vs after ITN (needs jiwer)")
    return p


def _norm_for_wer(text: str) -> str:
    """Mirror stt_nemotron_eval._text_norm so WER numbers are comparable."""
    try:
        from stt_nemotron_eval import _text_norm  # reuse the exact eval normalization

        return _text_norm(text)
    except Exception:
        text = str(text).strip().lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"_", " ", text)
        return re.sub(r"\s+", " ", text).strip()


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    brands = set(_DEFAULT_BRANDS)
    if args.brands.strip():
        brands |= {b.strip() for b in args.brands.split(",") if b.strip()}

    def itn(text: str) -> str:
        return inverse_normalize(
            text,
            thousands_sep=not args.no_thousands_sep,
            compact_brands=not args.no_brand_compaction,
            brands=brands,
        )

    if args.text is not None:
        print(itn(args.text))
        return 0

    in_path = Path(args.in_path).resolve()
    out_field = args.out_field or f"{args.field}_itn"
    out_path = Path(args.out).resolve() if args.out else in_path.with_name(in_path.stem + "_itn.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    refs: list[str] = []
    preds_raw: list[str] = []
    preds_itn: list[str] = []
    n = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = str(row.get(args.field, ""))
            row[out_field] = itn(raw)
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.wer and args.ref_field:
                refs.append(str(row.get(args.ref_field, "")))
                preds_raw.append(raw)
                preds_itn.append(row[out_field])

    print(f"Wrote {n} rows -> {out_path} (field '{out_field}')")

    if args.wer and args.ref_field and refs:
        try:
            from jiwer import cer, wer
        except Exception:
            print("jiwer not installed; skipping WER. Install: uv pip install jiwer")
            return 0
        r = [_norm_for_wer(x) for x in refs]
        p0 = [_norm_for_wer(x) for x in preds_raw]
        p1 = [_norm_for_wer(x) for x in preds_itn]
        print(json.dumps({
            "n": len(r),
            "wer_before": float(wer(r, p0)),
            "wer_after": float(wer(r, p1)),
            "cer_before": float(cer(r, p0)),
            "cer_after": float(cer(r, p1)),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
