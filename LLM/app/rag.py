from __future__ import annotations
import os
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .vectorstore import semantic_search, seed_if_empty
from .tools import TOOL_SPEC, call_tool
from .guardrails import check_text

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Ești Smart Librarian.
Primești intenții de cititor, cauți în vector store după teme/context și propui O SINGURĂ carte potrivită.
Explică în 2–4 fraze DE CE se potrivește.
Apoi apelează tool-ul `get_summary_by_title` pentru a returna REZUMATUL COMPLET.
Evită spoilerele majore, dacă utilizatorul nu cere explicit.
Răspunde în limba în care a fost pusă întrebarea."""

def _format_context(passages: List[Dict[str, Any]]) -> str:
    # Extragem titlul + teme din metadate
    lines = []
    for p in passages:
        title = p.get("metadata", {}).get("title")
        themes = p.get("metadata", {}).get("themes", [])
        lines.append(f"- {title} | teme: {', '.join(themes)}")
    return "\n".join(lines)

def answer(user_query: str, top_k: int = 5) -> Dict[str, Any]:
    # Guardrails
    blocked, msg = check_text(user_query)
    if blocked:
        return {"assistant": msg, "used_tool": None, "tool_result": None}

    # Asigurăm că există date
    seed_if_empty()

    # 1) RAG
    passages = semantic_search(user_query, k=top_k)
    if not passages:
        polite = "Nu am găsit ceva relevant în colecție. Poți specifica genul, tema sau o carte preferată?"
        return {"assistant": polite, "used_tool": None, "tool_result": None}

    context = _format_context(passages)

    client = OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Context (cărți candidate):\n{context}"},
        {"role": "user", "content": user_query},
    ]

    # 2) LLM cu function calling
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=[TOOL_SPEC],
        tool_choice="auto",
        temperature=0.3,
    )

    assistant_msg = resp.choices[0].message
    content_text = assistant_msg.content or ""
    used_tool = None
    tool_result = None

    # 3) Executăm tool-ul dacă a fost cerut
    if assistant_msg.tool_calls:
        for tc in assistant_msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            # `args` e string JSON în unele versiuni; normalizăm:
            import json as _json
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {}
            tool_output = call_tool(name, args or {})
            used_tool = name
            tool_result = tool_output

            # Trimitem înapoi rezultatul tool-ului pentru completarea finală
            messages.append(
                {"role": "assistant", "content": content_text, "tool_calls": [tc]}
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": name, "content": tool_output}
            )
            final = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.2,
            )
            content_text = final.choices[0].message.content or content_text
            break  # un singur tool call e suficient în acest flow

    return {"assistant": content_text, "used_tool": used_tool, "tool_result": tool_result, "candidates": passages}
