"""Token extraction shared by the hash generator and the no-hardcoding test.

The same function runs on the dev questions (once, to make hashes) and on
shipped files (in the test), so both sides see the same token shapes.
"""

import re

KEY = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{2,}\b")  # issue keys such as ABC-123
DOC_ID = re.compile(r"\bdsid_[0-9a-f]{32}\b")
CAPITALIZED_RUN = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")
LEADING_STOPWORDS = {
    "The", "A", "An", "Our", "What", "Which", "How", "When", "Where", "Who", "Why",
    "In", "On", "For", "And", "But", "If", "Is", "Are", "Does", "Did", "Do", "This", "That",
}  # fmt: skip
MAX_NAME_WORDS = 4
MIN_NAME_CHARS = 10  # shorter capitalized pairs are mostly generic words


def candidate_tokens(text: str) -> set[str]:
    """Issue keys, doc ids, and every 2 to 4 word run of capitalized words."""
    tokens = set(KEY.findall(text)) | set(DOC_ID.findall(text))
    for run in CAPITALIZED_RUN.findall(text):
        words = run.split()
        while words and words[0] in LEADING_STOPWORDS:
            words = words[1:]
        for size in range(2, MAX_NAME_WORDS + 1):
            for start in range(len(words) - size + 1):
                name = " ".join(words[start : start + size])
                if len(name) >= MIN_NAME_CHARS:
                    tokens.add(name)
    return tokens
