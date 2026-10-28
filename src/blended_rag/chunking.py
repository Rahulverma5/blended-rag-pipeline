"""Chunk contracts into retrievable spans that map back to exact character offsets.

  * naive        fixed 500-char windows, no overlap: the paper's "Naive" baseline
  * structure    pack whole lines (then sentences, then words) up to `max_chars`, never crossing a heading.
                 No overlap: the metric charges precision for every retrieved character, and gold spans are
                 short (median ~330 chars), so overlap only spends the top-k budget on duplicates.

Every chunk carries two extra fields used by the multi-field query templates (IBM's best_fields etc.):
  * section   the last heading ("ARTICLE IV ...", "Section 7.2 Non-Compete.", ALL-CAPS line) before the chunk
  * title     the document name derived from its file name ("Adamas Pharmaceuticals Supernus Pharmaceuticals")
"""

import re
from dataclasses import dataclass
from urllib.parse import unquote

HEADING = re.compile(
    r"(?:^|(?<=\s))(?:ARTICLE|Article)\s+[IVXLC\d]+\b\.?(?:[ \t]+[A-Z][A-Z,&/'-]*\b)*"  # ARTICLE IV  GENERAL UNDERTAKING
    r"|(?:^|(?<=\s))(?:SECTION|Section)\s+\d+(?:\.\d+)*\s+[A-Z][^.\n]{2,80}\."  # Section 1.02 Powers.
    r"|^[ \t]*[A-Z][A-Z0-9 ,&/'-]{4,60}[ \t]*$",                            # MUTUAL NON-DISCLOSURE AGREEMENT
    re.MULTILINE,
)
BOUNDARIES = (re.compile(r"\n"), re.compile(r"(?<=[.;:])\s+"), re.compile(r"\s+"))


@dataclass(frozen=True)
class Chunk:
    doc: str
    start: int
    end: int
    section: str
    title: str


def doc_title(file_path: str) -> str:
    """Readable document name from a file name: "VELCO%20NDA%20rev0" -> "VELCO NDA rev0",
    "ExcelerateStandardNDAFormat" -> "Excelerate Standard NDA Format", "Kenway-NDA-Form" -> "Kenway NDA Form"."""
    name = file_path.split("/", 1)[-1].rsplit(".txt", 1)[0].split("||")[0].removesuffix(".pdf")
    name = unquote(name)
    name = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    return re.sub(r"[_\s]+|(?<=\w)-(?=\w)", " ", name).strip()


def headings(text: str) -> list[tuple[int, str]]:
    return [(m.start(), " ".join(m.group().split())[:100]) for m in HEADING.finditer(text)]


def _split(text: str, start: int, end: int, max_chars: int, level: int = 0) -> list[tuple[int, int]]:
    """Cut [start, end) at the coarsest boundary that keeps pieces <= max_chars, then greedily re-pack."""
    if end - start <= max_chars:
        return [(start, end)]
    if level == len(BOUNDARIES):
        return [(s, min(s + max_chars, end)) for s in range(start, end, max_chars)]
    cuts = [m.end() + start for m in BOUNDARIES[level].finditer(text[start:end])]
    pieces, prev = [], start
    for c in cuts + [end]:
        if c > prev:
            pieces.extend(_split(text, prev, c, max_chars, level + 1) if c - prev > max_chars else [(prev, c)])
            prev = c
    packed: list[tuple[int, int]] = []
    for s, e in pieces:
        if packed and e - packed[-1][0] <= max_chars:
            packed[-1] = (packed[-1][0], e)
        else:
            packed.append((s, e))
    return packed


def _merge_small(spans: list[tuple[int, int]], max_chars: int, min_chars: int = 150) -> list[tuple[int, int]]:
    # Bare headings ("ARTICLE XII") and sentence tails are ~13% of pieces on MAUD; on their own they are
    # unretrievable noise, so attach them to a neighbour (a heading to its body) when the result still fits.
    out: list[tuple[int, int]] = []
    for s, e in spans:
        if out and min(out[-1][1] - out[-1][0], e - s) < min_chars and e - out[-1][0] <= max_chars:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def chunk_document(file_path: str, text: str, strategy: str = "structure", max_chars: int = 500) -> list[Chunk]:
    title = doc_title(file_path)
    if strategy == "naive":
        spans = [(s, min(s + 500, len(text))) for s in range(0, len(text), 500)]
        heads: list[tuple[int, str]] = []
    elif strategy == "structure":
        heads = headings(text)
        bounds = sorted({0, len(text), *(p for p, _ in heads)})
        spans = _merge_small([sp for a, b in zip(bounds, bounds[1:]) for sp in _split(text, a, b, max_chars)],
                             max_chars)
    else:
        raise ValueError(f"unknown chunking strategy {strategy!r}")
    chunks, h = [], 0
    for s, e in spans:
        s, e = _trim(text, s, e)
        if e <= s:
            continue
        while h < len(heads) and heads[h][0] < s:
            h += 1
        # Headings inside the chunk (a merged chunk can hold several), else the one it continues.
        inside = [t for p, t in heads[h:] if p < e] if h < len(heads) and heads[h][0] < e else []
        section = " | ".join(inside) if inside else (heads[h - 1][1] if h else "")
        chunks.append(Chunk(file_path, s, e, section, title))
    return chunks


def chunk_corpus(corpus: dict[str, str], strategy: str = "structure", max_chars: int = 500) -> list[Chunk]:
    return [c for path in sorted(corpus) for c in chunk_document(path, corpus[path], strategy, max_chars)]
