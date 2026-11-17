"""Matching forms of field names and values, shared by ingest (which stores them) and
retrieval (which matches the model's words against them).

They live in `store` because both layers may import it (imports go down only): ingest
writes `field_norm` and `value_norm` with these functions, and the agent's tools must
fold a value the same way to find it again.
"""

import re
import unicodedata

from corpusqa.config import VALUE_NORM_MAX_CHARS

# Quotes and sentence punctuation around a value do not change what it means.
_EDGE_CHARS = "\"'`\u2018\u2019\u201c\u201d\u00ab\u00bb.,;:!? \t\n"
_TRUE, _FALSE = {"yes", "true"}, {"no", "false"}
_ANGLE_EMAIL = re.compile(r"\s*<[^<>\s]*@[^<>\s]*>")
_ROLE = re.compile(r"\s*\([^()]*\)")


def fold_key(key: str) -> str:
    """Grouping form of a field name (never replaces the key itself)."""
    return re.sub(r"[-\s]", "_", unicodedata.normalize("NFKC", key).casefold().strip())


def normalise(text: str) -> str:
    """Matching form of a value for exact filters: NFKC, case-folded, `_` as space,
    whitespace collapsed, surrounding quotes and punctuation removed, yes/no as true/false.
    """
    folded = unicodedata.normalize("NFKC", text).casefold().replace("_", " ")
    norm = " ".join(folded.split()).strip(_EDGE_CHARS)
    if norm in _TRUE:
        return "true"
    if norm in _FALSE:
        return "false"
    return norm[:VALUE_NORM_MAX_CHARS]


def strip_person(value: str) -> str:
    """A name without a role in parentheses or an <email> part."""
    return " ".join(_ROLE.sub("", _ANGLE_EMAIL.sub("", value)).split())


def person_norm(value: str) -> str:
    """Matching form of a person value: role and <email> removed, `.` as space (not in a
    bare email address), then the usual `normalise` (case, `_`, punctuation)."""
    text = strip_person(value) or value.strip()
    if "@" not in text:
        text = text.replace(".", " ")
    return normalise(text)
