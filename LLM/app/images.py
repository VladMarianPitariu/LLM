from __future__ import annotations
import base64
from io import BytesIO
from typing import Optional
from openai import OpenAI

def generate_cover_idea(prompt_text: str) -> Optional[bytes]:
    """
    Generează o imagine sugestivă pe baza textului (OpenAI Images).
    Returnează bytes pentru afișare în Streamlit.
    """
    try:
        client = OpenAI()
        img = client.images.generate(
            model="gpt-image-1",
            prompt=f"Minimalist book cover concept, soft lighting, symbolic elements: {prompt_text}",
            size="1024x1024"
        )
        b64 = img.data[0].b64_json
        return base64.b64decode(b64)
    except Exception:
        return None
