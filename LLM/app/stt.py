from __future__ import annotations
import speech_recognition as sr

def transcribe_from_mic(timeout: int = 5, phrase_time_limit: int = 10) -> str | None:
    """
    Transcrie audio din microfon (dacă e disponibil). Necesită pyaudio/portaudio instalat.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        return r.recognize_google(audio, language="ro-RO")
    except Exception:
        return None
