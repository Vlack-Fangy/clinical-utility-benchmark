"""Narrow, auditable formatting sensitivity; never semantic evidence repair.

This is supplementary to the frozen exact-evidence protocol. Quotation glyphs
are interchangeable only for locating an otherwise identical contiguous span.
Returned offsets always address the unmodified output. It does not certify the
judgment or permit paraphrase, ellipses, whitespace changes, or missing words.
"""
from typing import Literal

_QUOTES = str.maketrans({char: '"' for char in "'\"‘’“”"})


def quote_evidence_span(output: str, evidence: str) -> tuple[int, int, Literal['exact', 'quote_glyph_only']] | None:
    if not evidence:
        return None
    start = output.find(evidence)
    if start >= 0:
        return start, start + len(evidence), 'exact'
    start = output.translate(_QUOTES).find(evidence.translate(_QUOTES))
    if start >= 0:
        return start, start + len(evidence), 'quote_glyph_only'
    return None
