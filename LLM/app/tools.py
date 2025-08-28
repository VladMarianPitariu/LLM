from __future__ import annotations
import json
import os
from typing import Optional, Dict, Any, List

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "book_summaries.json")

def _load_index() -> Dict[str, Dict[str, Any]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        books: List[Dict[str, Any]] = json.load(f)
    idx = {b["title"].strip(): b for b in books}
    # also add case-insensitive lookup
    for b in books:
        idx[b["title"].strip().lower()] = b
    return idx

_BOOKS = _load_index()

def get_summary_by_title(title: str) -> str:
    """
    Returnează rezumatul complet pentru titlul exact (case-insensitive).
    """
    if not title:
        return "Titlul este gol."
    book = _BOOKS.get(title) or _BOOKS.get(title.strip().lower())
    if not book:
        return f"Nu am găsit „{title}” în baza locală."
    return book["long_summary"]

# OpenAI tool (function) specification
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_summary_by_title",
        "description": "Returnează rezumatul complet (paragrafe) pentru un titlu de carte exact.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Titlul exact al cărții pentru care vrei rezumatul complet."
                }
            },
            "required": ["title"]
        }
    }
}

# Mapper din tool call → funcția Python
def call_tool(name: str, arguments: dict) -> str:
    if name == "get_summary_by_title":
        return get_summary_by_title(arguments.get("title", ""))
    return "Tool necunoscut."
