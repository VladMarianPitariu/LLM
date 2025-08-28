from __future__ import annotations
import os
from gtts import gTTS

def speak_text(text: str, filename: str = "smart_librarian_tts.mp3") -> str | None:
    """
    Generează un mp3 cu TTS local (gTTS). Returnează calea fișierului.
    """
    if not text:
        return None
    tts = gTTS(text=text, lang="ro")
    tts.save(filename)
    return filename
