"""Lightweight text preprocessing for phishing-email classification.

Design goals:
  - fast (pure-regex, no heavy NLP libs)
  - deterministic
  - Windows-friendly (no locale-dependent behaviour)
"""

import re
from typing import List

# Pre-compiled patterns for speed
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
)

_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Core cleaning function
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Return a cleaned version of *text*.

    Steps (in order):
      1. Lowercase
      2. Replace URLs  -> ``<URL>``
      3. Replace email addresses -> ``<EMAIL>``
      4. Collapse / normalise whitespace
      5. Strip leading / trailing whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub("<URL>", text)
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_texts(texts: List[str]) -> List[str]:
    """Apply :func:`clean_text` to every element."""
    return [clean_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "  Visit https://evil.com NOW!!!  ",
        "Contact admin@company.org for details.",
        "URGENT:  send   money   to   http://scam.net/pay?id=3  ",
        "Hello\n\n  World\t\ttabs",
        "",
        None,
        12345,
    ]
    for s in samples:
        print(f"{s!r:55s} -> {clean_text(s)!r}")
