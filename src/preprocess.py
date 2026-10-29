"""Text cleaning used for every model in this repository."""
import html
import re

# UTF-8 bytes (emoji, Kannada script) that were decoded with a single-byte encoding show up
# as a lead character followed by control characters. The text they encode is lost, so we
# drop the sequences. See "Limitations" in the paper for the consequences.
_MOJIBAKE = re.compile(r"[\u00e0\u00f0][\u0080-\u00ff]+")
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+")
_REPEAT = re.compile(r"(.)\1{2,}")  # 3+ identical characters
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    if text is None:
        return ""
    t = str(text)
    t = _MOJIBAKE.sub(" ", t)
    t = html.unescape(t)
    t = _HTML_TAG.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _REPEAT.sub(r"\1\1", t)  # "soooo" -> "soo"
    t = _SPACES.sub(" ", t).strip()
    return t
