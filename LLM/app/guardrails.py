from __future__ import annotations
import re
from typing import Tuple

# Listă simplă; poți îmbogăți ulterior cu regex-uri mai bune sau un serviciu extern.
OFFENSIVE = [
    r"\bidiot\b",
    r"\bprost\b",
    r"\b\w+ dracu\b",
]

def check_text(text: str) -> Tuple[bool, str]:
    """
    Returnează (is_blocked, message). Dacă is_blocked=True, nu trimitem la LLM.
    """
    if not text:
        return False, ""
    for pat in OFFENSIVE:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True, "Hai să păstrăm conversația prietenoasă 😊. Reformulează te rog întrebarea."
    return False, ""
