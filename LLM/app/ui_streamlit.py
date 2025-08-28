from __future__ import annotations
import os, sys
# adaugă rădăcina proiectului în sys.path, ca să poți importa `app.*`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.vectorstore import seed_if_empty
from app.rag import answer

# opțional, nu blocăm dacă lipsesc
try:
    from app.tts import speak_text
except Exception:
    speak_text = None

try:
    from app.images import generate_cover_idea
except Exception:
    generate_cover_idea = None


st.set_page_config(page_title="Smart Librarian", page_icon="📚", layout="centered")

# Sidebar
st.sidebar.title("⚙️ Setări")
top_k = st.sidebar.slider("Top-K rezultate (RAG)", min_value=1, max_value=10, value=5)
use_tts = st.sidebar.checkbox("Activează TTS (audio)", value=False)
use_img = st.sidebar.checkbox("Generează imagine (opțional)", value=False)

st.title("📚 Smart Librarian")
st.caption("RAG cu ChromaDB + Tool Calling (rezumat complet după recomandare)")

# Seed automat (o singură dată)
seed_if_empty()

with st.form("chat"):
    user_query = st.text_input("Întreabă-mă despre ce ai vrea să citești:", value="")
    submitted = st.form_submit_button("Caută recomandare")

if submitted and user_query.strip():
    with st.spinner("Caut în colecție…"):
        resp = answer(user_query.strip(), top_k=top_k)

    st.markdown("### Răspuns")
    st.write(resp["assistant"])

    # TTS opțional
    if st.session_state.get("tts_enabled", use_tts) and use_tts:
        audio_path = speak_text(resp["assistant"])
        if audio_path:
            st.audio(audio_path)

    # Imagine opțională (încercăm să deducem titlul din tool_result dacă există)
    if use_img and resp.get("tool_result"):
        img_b64 = generate_cover_idea(resp["tool_result"])
        if img_b64:
            st.image(img_b64, caption="Copertă sugestivă generată")
