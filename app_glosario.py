# app_glosario_v01_ui.py
from __future__ import annotations

import re
import html
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# Core extraction utilities
# =========================

STOPWORDS_ES = set("""
a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaba estabais estaban estaban estabas estado estais estamos estan estar estas este esto estos fue fueron fui fuimos ha habiais habian habias han has hasta hay la las le les lo los mas me mi mis mucha muchas mucho muchos muy no nos o os para pero por porque que quien quienes se sea seais sean seas sera seran seras sere seremos seria seriais serian serias si sin sobre sois somos son soy su sus te teneis tenemos tener tenia teniais tenian tenias tengo tiene tienen tienes toda todas todo todos tu tus un una unas unos y ya
""".split())

STOPWORDS_EN = set("""
a an and are as at be been being but by can could did do does doing for from had has have having he her hers him his how i if in into is it its just may might more most much must my no nor not of on or our ours she should so some such than that the their theirs them then there these they this those through to too until up very was we were what when where which who why will with would you your yours
""".split())

COMMON_JUNK = set([
    "chapter", "figure", "table", "copyright", "contents",
    "page", "pages", "version", "release", "guide", "user", "users",
])


@dataclass
class TermItem:
    term: str
    score: float
    definition: str
    tags: List[str]
    source_page: Optional[int] = None  # v0.1: placeholder para v0.2 evidencia


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def read_pdf_text_bytes(pdf_bytes: bytes, max_pages: int | None = None) -> Tuple[str, int, int]:
    """
    Returns: (text, pages_read, total_pages)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    limit = min(total_pages, max_pages) if max_pages else total_pages

    texts = []
    for i in range(limit):
        page = doc.load_page(i)
        texts.append(page.get_text("text") or "")
    doc.close()

    return ("\n".join(texts), limit, total_pages)


def detect_doc_profile(text: str) -> str:
    t = text.lower()
    tech_hits = sum(w in t for w in [
        "configuration", "program", "device", "control unit", "channel", "subsystem",
        "i/o", "keyword", "statement", "report", "appendix"
    ])
    acad_hits = sum(w in t for w in [
        "abstract", "introduction", "methodology", "results", "discussion",
        "references", "bibliography"
    ])
    if tech_hits >= 4 and tech_hits >= acad_hits:
        return "technical_manual"
    if acad_hits >= 3 and acad_hits > tech_hits:
        return "academic"
    return "generic"


def sentence_split(text: str) -> List[str]:
    text = normalize_space(text.replace("\n", " "))
    parts = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 25]


def looks_like_glossary(text: str) -> bool:
    t = text.lower()
    if "glossary" not in t and "glossary of terms" not in t:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    entry_like = 0
    for ln in lines[:2500]:
        if re.match(r"^[A-Za-z0-9][A-Za-z0-9 \-/()']{1,90}:\s*$", ln):
            entry_like += 1
    return entry_like >= 8


def parse_glossary_entries(text: str) -> List[Tuple[str, str]]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    entries: List[Tuple[str, str]] = []

    term = None
    buf: List[str] = []
    entry_re = re.compile(r"^([A-Za-z0-9][A-Za-z0-9 \-/()']{1,110}):\s*$")

    def flush():
        nonlocal term, buf
        if term and buf:
            definition = normalize_space(" ".join(buf))
            definition = re.sub(r"\s+©.*$", "", definition).strip()
            entries.append((term.strip(), definition.strip()))
        term = None
        buf = []

    for ln in lines:
        l = ln.strip()
        if not l:
            continue

        m = entry_re.match(l)
        if m:
            flush()
            term = m.group(1)
            continue

        low = l.lower()
        if "course materials may not be reproduced" in low:
            continue
        if "©" in l and "copyright" in low:
            continue
        if low in ("glossary", "glossary of terms"):
            continue

        if term:
            buf.append(l)

    flush()
    return entries


def extract_acronyms(text: str, min_len: int = 2, max_len: int = 12, min_freq: int = 2) -> Dict[str, float]:
    acr = re.findall(r"\b[A-Z][A-Z0-9]{%d,%d}\b" % (min_len - 1, max_len - 1), text)
    counts: Dict[str, int] = {}
    for a in acr:
        if a.lower() in COMMON_JUNK:
            continue
        counts[a] = counts.get(a, 0) + 1

    zstuff = re.findall(r"\bz\/[A-Z]{2,4}\b", text)
    for z in zstuff:
        counts[z] = counts.get(z, 0) + 1

    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items() if v >= min_freq}


def build_ngrams_candidates(
    text: str,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
) -> List[Tuple[str, float]]:
    raw = normalize_space(text)
    raw = re.sub(r"[•\u2022]", " ", raw)
    raw = re.sub(r"\s+", " ", raw)

    stop = STOPWORDS_ES | STOPWORDS_EN
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        stop_words=list(stop),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-\/]{1,}\b",
    )

    X = vec.fit_transform([raw])
    scores = X.toarray().ravel()
    feats = np.array(vec.get_feature_names_out())

    out = []
    for term, sc in sorted(zip(feats, scores), key=lambda x: x[1], reverse=True):
        t = term.strip().lower()
        if len(t) < 3:
            continue
        if any(j in t for j in ["http", "www"]):
            continue
        if t in COMMON_JUNK:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", t):
            continue
        out.append((term, float(sc)))
    return out


def better_definition_from_context(term: str, sentences: List[str]) -> str:
    term_re = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)

    for i, s in enumerate(sentences):
        if term_re.search(s):
            s2 = normalize_space(s)

            # patrón "Term: definición"
            colonish = re.search(re.escape(term) + r"\s*:\s*", s2, flags=re.IGNORECASE)
            if colonish:
                out = s2[colonish.start():]
                return out if len(out) <= 520 else out[:520].rstrip() + "…"

            # contexto: prev + actual + next
            parts = []
            if i - 1 >= 0:
                parts.append(sentences[i - 1])
            parts.append(s2)
            if i + 1 < len(sentences):
                parts.append(sentences[i + 1])

            merged = normalize_space(" ".join(parts))
            if len(merged) > 520:
                merged = merged[:520].rstrip() + "…"
            return merged

    return ""


def tag_term(term: str, profile: str) -> List[str]:
    tags = []
    t = term.lower()
    if profile in ("technical_manual", "glossary_detected"):
        tags.append("técnico")
    if re.fullmatch(r"[A-Z0-9\/]{2,16}", term):
        tags.append("acrónimo")
    if any(k in t for k in ["statement", "keyword", "parameter", "report", "config", "configuration"]):
        tags.append("config")
    if any(k in t for k in ["device", "control unit", "channel", "i/o", "subsystem"]):
        tags.append("i/o")
    return tags or ["general"]


def merge_terms(
    acronyms: Dict[str, float],
    phrases: List[Tuple[str, float]],
    top_n: int,
    acronym_boost: float,
) -> List[Tuple[str, float]]:
    items: Dict[str, float] = {}

    for k, v in acronyms.items():
        items[k] = max(items.get(k, 0.0), v * acronym_boost)

    take = min(len(phrases), max(1500, top_n * 10))
    for p, sc in phrases[:take]:
        key = p.strip()
        if len(key) < 4:
            continue
        items[key] = max(items.get(key, 0.0), sc)

    merged = sorted(items.items(), key=lambda x: x[1], reverse=True)
    return [(t, float(s)) for t, s in merged[:top_n]]


def make_html(pages: List[Tuple[str, str, int, int, List[TermItem]]], meta: Dict) -> str:
    blocks = []
    for title, profile, pages_read, total_pages, terms in pages:
        items_html = []
        for it in terms:
            tags_html = " ".join(
                f'<span class="inline-flex items-center rounded-full border px-2 py-0.5 text-xs">{html.escape(tag)}</span>'
                for tag in it.tags
            )
            items_html.append(f"""
              <div class="term-card rounded-2xl border bg-white/70 p-4 shadow-sm backdrop-blur"
                   data-term="{html.escape(it.term).lower()} {html.escape(' '.join(it.tags)).lower()}">
                <div class="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h3 class="text-lg font-semibold tracking-tight">{html.escape(it.term)}</h3>
                    <div class="mt-2 flex flex-wrap gap-2">{tags_html}</div>
                  </div>
                  <div class="text-xs text-slate-500">score: {it.score:.4f}</div>
                </div>
                <p class="mt-3 text-sm leading-relaxed text-slate-700">{html.escape(it.definition)}</p>
              </div>
            """)

        badge = "📘 Glosario detectado" if profile == "glossary_detected" else f"📄 Perfil: {profile}"
        blocks.append(f"""
          <section class="mt-10">
            <div class="flex flex-wrap items-end justify-between gap-4">
              <div>
                <h2 class="text-2xl font-bold tracking-tight">{html.escape(title)}</h2>
                <p class="mt-1 text-sm text-slate-600">{html.escape(badge)} · páginas leídas: <span class="font-medium">{pages_read}</span> / {total_pages}</p>
              </div>
              <div class="text-sm text-slate-500">{len(terms)} términos</div>
            </div>
            <div class="mt-5 grid gap-4 md:grid-cols-2">
              {''.join(items_html)}
            </div>
          </section>
        """)

    meta_line = html.escape(json.dumps(meta, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Glosario desde PDFs</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 text-slate-900">
  <header class="mx-auto max-w-6xl px-6 pt-10">
    <div class="rounded-3xl border bg-white/80 p-6 shadow-sm backdrop-blur">
      <div class="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 class="text-3xl font-extrabold tracking-tight">Glosario desde PDFs</h1>
          <p class="mt-2 text-slate-600">
            Términos extraídos automáticamente (frases clave + acrónimos) con definiciones por glosario o contexto.
          </p>
        </div>
        <div class="text-xs text-slate-500">
          <div class="rounded-2xl border bg-white px-3 py-2 shadow-sm">meta: {meta_line}</div>
        </div>
      </div>

      <div class="mt-5 flex flex-col gap-3 md:flex-row md:items-center">
        <input id="q" type="search"
          placeholder="Buscar término (ej. IOCP, ACL, configuration...)"
          class="w-full rounded-2xl border bg-white px-4 py-3 text-sm shadow-sm outline-none ring-0 focus:border-slate-400 md:flex-1"/>
        <button id="clear"
          class="rounded-2xl border bg-white px-4 py-3 text-sm font-medium shadow-sm hover:bg-slate-50">
          Limpiar
        </button>
      </div>

      <p id="count" class="mt-3 text-sm text-slate-500"></p>
    </div>
  </header>

  <main class="mx-auto max-w-6xl px-6 pb-16">
    {''.join(blocks)}
  </main>

  <script>
    const q = document.getElementById('q');
    const clear = document.getElementById('clear');
    const count = document.getElementById('count');

    function applyFilter() {{
      const val = (q.value || '').trim().toLowerCase();
      const cards = Array.from(document.querySelectorAll('.term-card'));
      let shown = 0;
      for (const c of cards) {{
        const hay = (c.getAttribute('data-term') || '');
        const ok = !val || hay.includes(val);
        c.style.display = ok ? '' : 'none';
        if (ok) shown++;
      }}
      count.textContent = `Mostrando ${{shown}} de ${{cards.length}} términos`;
    }}

    q.addEventListener('input', applyFilter);
    clear.addEventListener('click', () => {{ q.value = ''; applyFilter(); q.focus(); }});
    applyFilter();
  </script>
</body>
</html>
"""


@st.cache_data(show_spinner=False)
def build_glossary_cached(
    pdf_name: str,
    pdf_hash: str,
    pdf_bytes: bytes,
    top: int,
    max_pages: int,
    only_acronyms: bool,
    min_term_len: int,
    max_term_len: int,
    min_acronym_freq: int,
    acronym_boost: float,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
) -> Tuple[str, str, int, int, List[TermItem]]:
    text, pages_read, total_pages = read_pdf_text_bytes(pdf_bytes, max_pages=max_pages)

    if looks_like_glossary(text):
        entries = parse_glossary_entries(text)
        terms: List[TermItem] = []
        for (t, d) in entries:
            if len(t) < min_term_len or len(t) > max_term_len:
                continue
            terms.append(TermItem(term=t, score=1.0, definition=d, tags=tag_term(t, "glossary_detected")))
            if len(terms) >= top:
                break
        return (pdf_name, "glossary_detected", pages_read, total_pages, terms)

    profile = detect_doc_profile(text)
    sents = sentence_split(text)

    acr = extract_acronyms(text, min_freq=min_acronym_freq)

    if only_acronyms:
        merged = sorted(acr.items(), key=lambda x: x[1], reverse=True)[:top]
        terms: List[TermItem] = []
        for t, sc in merged:
            if len(t) < min_term_len or len(t) > max_term_len:
                continue
            definition = better_definition_from_context(t, sents) or "Acrónimo detectado; revisa el documento para su definición."
            terms.append(TermItem(term=t, score=float(sc), definition=definition, tags=tag_term(t, profile)))
        return (pdf_name, profile, pages_read, total_pages, terms)

    phrases = build_ngrams_candidates(
        text=text,
        max_features=max_features,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    merged = merge_terms(acr, phrases, top_n=top, acronym_boost=acronym_boost)

    terms: List[TermItem] = []
    for term, score in merged:
        tnorm = term.strip()
        if len(tnorm) < min_term_len or len(tnorm) > max_term_len:
            continue

        low = tnorm.lower()
        if low in COMMON_JUNK:
            continue
        if low.startswith(("table ", "figure ", "chapter ")):
            continue

        definition = better_definition_from_context(tnorm, sents)
        if not definition:
            # fallback suave: primera oración donde aparece
            definition = "No se encontró una definición directa; se muestra contexto cercano donde aparece."
            for s in sents:
                if re.search(r"\b" + re.escape(tnorm) + r"\b", s, flags=re.IGNORECASE):
                    definition = normalize_space(s)
                    break

        terms.append(TermItem(term=tnorm, score=score, definition=definition, tags=tag_term(tnorm, profile)))

        if len(terms) >= top:
            break

    return (pdf_name, profile, pages_read, total_pages, terms)


def terms_to_json(pages: List[Tuple[str, str, int, int, List[TermItem]]], meta: Dict) -> str:
    payload = {
        "meta": meta,
        "files": [
            {
                "file": title,
                "profile": profile,
                "pages_read": pages_read,
                "total_pages": total_pages,
                "terms": [asdict(t) for t in terms],
            }
            for (title, profile, pages_read, total_pages, terms) in pages
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# =========================
# UI v0.1 (product)
# =========================

st.set_page_config(page_title="AutoGlossary", page_icon="📚", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.0rem; }
.small-note { color: rgba(15,23,42,.62); font-size: .92rem; }
.card {
  border: 1px solid rgba(15, 23, 42, .10);
  border-radius: 1rem;
  background: rgba(255,255,255,.72);
  padding: 1rem;
  margin-top: 3rem;
}
.kpi {
  border: 1px solid rgba(15,23,42,.10);
  border-radius: 1rem;
  padding: .8rem 1rem;
  background: rgba(255,255,255,.70);
}
section[data-testid="stFileUploaderDropzone"]{
  padding: 1.2rem;
  border-radius: 1.1rem;
  border: 2px dashed rgba(15, 23, 42, .25);
  background: rgba(255,255,255,.65);
}
</style>
""", unsafe_allow_html=True)

# Session state init
if "favorites" not in st.session_state:
    st.session_state.favorites = set()  # term strings
if "run_history" not in st.session_state:
    st.session_state.run_history = []  # list of dict meta
if "last_output" not in st.session_state:
    st.session_state.last_output = None  # {pages_out, html, json, meta}
if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False

# Header
st.markdown(
    """
<div class="card">
  <div style="display:flex; flex-wrap:wrap; justify-content:space-between; gap:16px; align-items:flex-start;">
    <div>
      <div style="font-size:28px; font-weight:850;">📚 AutoGlossary</div>
      <div class="small-note">Convierte PDFs en glosarios en 1 clic. Explora, guarda favoritos y exporta HTML/JSON.</div>
    </div>
    <div class="small-note">
      <div><b>Tip:</b> PDFs que ya son “Glossary” se extraen con definiciones reales.</div>
      <div>v0.1: UI pro + favoritos + historial. (v0.2: evidencia por página)</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# Sidebar: Basic / Advanced
with st.sidebar:
    st.header("⚙️ Ajustes")
    mode = st.radio("Modo", ["Básico", "Avanzado"], horizontal=True)

    # defaults pro pero seguros
    top = st.slider("Términos por PDF", 20, 2000, 250, 10)
    max_pages = st.slider("Máx. páginas a leer (por PDF)", 10, 2000, 250, 10)

    st.divider()
    only_acronyms = st.toggle("Solo acrónimos", value=False)
    min_term_len = st.slider("Longitud mínima", 2, 20, 3, 1)
    max_term_len = st.slider("Longitud máxima", 20, 200, 90, 5)

    # Advanced knobs
    if mode == "Avanzado":
        st.divider()
        st.subheader("Acrónimos")
        min_acronym_freq = st.slider("Frecuencia mínima acrónimo", 1, 10, 2, 1)
        acronym_boost = st.slider("Boost acrónimos (ranking)", 1.0, 5.0, 2.2, 0.1)

        st.divider()
        st.subheader("TF-IDF")
        max_features = st.slider("Max features", 500, 20000, 4000, 500)
        ngram_min = st.selectbox("N-gram mínimo", [1, 2, 3], index=1)
        ngram_max = st.selectbox("N-gram máximo", [2, 3, 4, 5], index=2)
    else:
        # Basic: valores “buenos”
        min_acronym_freq = 2
        acronym_boost = 2.2
        max_features = 4000
        ngram_min = 2
        ngram_max = 4

    if ngram_max < ngram_min:
        ngram_max = ngram_min

    st.divider()
    include_meta = st.toggle("Incluir metadata en export", value=True)
    preview_on = st.toggle("Preview HTML dentro de la app", value=True)

# Tabs
tab_upload, tab_results, tab_export, tab_history = st.tabs(["📥 Cargar", "📊 Resultados", "⬇️ Exportar", "🕘 Historial"])

# ---- Upload tab
with tab_upload:
    st.subheader("📥 Cargar PDFs")

    # ---------- Card: Fuente ----------
    st.markdown(
        """
        <div class="card">
          <div style="display:flex; justify-content:space-between; gap:14px; align-items:flex-start; flex-wrap:wrap;">
            <div>
              <div style="font-size:18px; font-weight:800;">Fuente del documento</div>
              <div class="small-note">Sube tus PDFs o prueba rápido con <b>demo.pdf</b> (en la carpeta del proyecto).</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Selector armonizado
    source = st.radio(
        " ",
        ["Subir PDFs", "Usar demo.pdf"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.write("")  # espacio

    inputs: List[Tuple[str, bytes, str]] = []

    # ---------- Subir PDFs ----------
    if source == "Subir PDFs":
        files = st.file_uploader(
            "Arrastra y suelta PDFs (o clic para elegir)",
            type=["pdf"],
            accept_multiple_files=True
        )
        st.caption("Tip: si el PDF ya trae sección “Glossary”, las definiciones salen perfectas.")

        if files:
            for f in files:
                b = f.getvalue()
                inputs.append((f.name, b, sha1_bytes(b)))
        else:
            st.info("Sube al menos un PDF (o cambia a Demo).")

    # ---------- Usar demo.pdf ----------
    else:
        # UI tipo “botón principal”
        cA, cB = st.columns([1, 1])
        with cA:
            st.markdown(
                """
                <div class="card" style="padding: 1.2rem;">
                  <div style="font-size:16px; font-weight:800;">Demo listo</div>
                  <div class="small-note">Usa <b>demo.pdf</b> para ver la app funcionando en segundos.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cB:
            use_demo = st.button("✨ Usar demo.pdf", type="primary", use_container_width=True)

        # Opción avanzada: cambiar ruta (colapsada)
        with st.expander("Opciones avanzadas (ruta demo)", expanded=False):
            demo_path = st.text_input("Ruta del PDF demo", value="demo.pdf")

        if use_demo:
            try:
                p = (demo_path if "demo_path" in locals() else "demo.pdf").strip().strip('"')
                with open(p, "rb") as f:
                    demo_bytes = f.read()
                inputs.append(("DEMO - " + p.split("\\")[-1].split("/")[-1], demo_bytes, sha1_bytes(demo_bytes)))
                st.success("Demo cargado ✅")
            except Exception as e:
                st.error(f"No pude abrir {demo_path!r}. Asegúrate de que exista en la carpeta del proyecto. Error: {e}")

        # Si todavía no le dio al botón, pero el archivo existe, le damos un hint.
        st.caption("Nota: coloca **demo.pdf** en la misma carpeta que el script (o ajusta la ruta en Opciones avanzadas).")

    st.divider()

    # ---------- Lista de archivos ----------
    if inputs:
        st.markdown("### 📄 Archivos listos")
        header = st.columns([2.2, 1, 1, 1])
        header[0].markdown("**Archivo**")
        header[1].markdown("**Tamaño**")
        header[2].markdown("**Hash**")
        header[3].markdown("**Estado**")

        for name, b, h in inputs:
            row = st.columns([2.2, 1, 1, 1])
            row[0].write(name)
            row[1].write(f"{len(b)/1024/1024:.2f} MB")
            row[2].code(h[:10], language=None)
            row[3].write("Listo ✅")

        st.divider()
        generate = st.button("⚡ Generar glosario", type="primary", use_container_width=True)
    else:
        generate = False
# ---- Generate when clicked
if "inputs_for_run" not in st.session_state:
    st.session_state.inputs_for_run = None
if generate:
    # Persist inputs into session so other tabs can show results
    st.session_state.inputs_for_run = inputs

    progress = st.progress(0)
    status = st.empty()
    details = st.empty()

    t0 = time.time()
    pages_out: List[Tuple[str, str, int, int, List[TermItem]]] = []

    for idx, (name, b, h) in enumerate(inputs, start=1):
        status.markdown(f"**Procesando:** `{name}` ({idx}/{len(inputs)})")
        details.caption("Leyendo PDF → detectando glosario → extrayendo términos…")

        result = build_glossary_cached(
            pdf_name=name,
            pdf_hash=h,
            pdf_bytes=b,
            top=top,
            max_pages=max_pages,
            only_acronyms=only_acronyms,
            min_term_len=min_term_len,
            max_term_len=max_term_len,
            min_acronym_freq=min_acronym_freq,
            acronym_boost=acronym_boost,
            max_features=max_features,
            ngram_min=int(ngram_min),
            ngram_max=int(ngram_max),
        )
        pages_out.append(result)
        progress.progress(int(idx / len(inputs) * 100))

    elapsed = time.time() - t0
    meta = {
        "generated_at_unix": int(time.time()),
        "elapsed_seconds": round(elapsed, 2),
        "settings": {
            "mode": mode,
            "top": top,
            "max_pages": max_pages,
            "only_acronyms": only_acronyms,
            "min_term_len": min_term_len,
            "max_term_len": max_term_len,
            "min_acronym_freq": min_acronym_freq,
            "acronym_boost": acronym_boost,
            "max_features": max_features,
            "ngram_min": int(ngram_min),
            "ngram_max": int(ngram_max),
        } if include_meta else {},
    }

    out_html = make_html(pages_out, meta if include_meta else {})
    out_json = terms_to_json(pages_out, meta if include_meta else {})

    st.session_state.last_output = {
        "pages_out": pages_out,
        "html": out_html,
        "json": out_json,
        "meta": meta,
    }

    # add run to history
    st.session_state.run_history.insert(0, {
        "ts": meta["generated_at_unix"],
        "elapsed": meta["elapsed_seconds"],
        "files": [p[0] for p in pages_out],
        "total_terms": sum(len(p[4]) for p in pages_out),
        "settings": meta.get("settings", {}),
    })

    status.success("✅ Listo")
    details.caption(f"Tiempo total: {elapsed:.2f}s")


# ---- Results tab
with tab_results:
    st.subheader("Resultados")
    out = st.session_state.last_output

    if not out:
        st.info("Primero genera un glosario en la pestaña **Cargar**.")
    else:
        pages_out = out["pages_out"]
        total_terms = sum(len(p[4]) for p in pages_out)

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='kpi'><div class='small-note'>Total términos</div><div style='font-size:24px; font-weight:800;'>{total_terms}</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'><div class='small-note'>Archivos</div><div style='font-size:24px; font-weight:800;'>{len(pages_out)}</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'><div class='small-note'>Tiempo (s)</div><div style='font-size:24px; font-weight:800;'>{out['meta']['elapsed_seconds']}</div></div>", unsafe_allow_html=True)

        st.divider()

        # Build a combined dataframe for filtering/table view
        rows = []
        for (title, profile, pages_read, total_pages, terms) in pages_out:
            for it in terms:
                rows.append({
                    "file": title,
                    "term": it.term,
                    "definition": it.definition,
                    "tags": ", ".join(it.tags),
                    "score": it.score,
                    "profile": profile,
                    "pages_read": pages_read,
                    "total_pages": total_pages,
                    "favorite": (it.term in st.session_state.favorites),
                })

        df = pd.DataFrame(rows)

        # Filters
        fc1, fc2, fc3, fc4 = st.columns([1.5, 1.2, 1.1, 1.1])
        with fc1:
            q = st.text_input("Buscar", placeholder="Ej. IOCP, ACL, configuration…", value="")
        with fc2:
            file_filter = st.multiselect("Archivos", options=sorted(df["file"].unique().tolist()), default=[])
        with fc3:
            tag_options = sorted({t.strip() for s in df["tags"].tolist() for t in s.split(",") if t.strip()})
            tag_filter = st.multiselect("Tags", options=tag_options, default=[])
        with fc4:
            view = st.selectbox("Vista", ["Tarjetas", "Tabla"], index=0)

        fav_only = st.toggle("Solo favoritos ⭐", value=False)

        # Apply filters
        fdf = df.copy()
        if q.strip():
            qq = q.strip().lower()
            fdf = fdf[
                fdf["term"].str.lower().str.contains(qq, na=False)
                | fdf["definition"].str.lower().str.contains(qq, na=False)
                | fdf["tags"].str.lower().str.contains(qq, na=False)
            ]
        if file_filter:
            fdf = fdf[fdf["file"].isin(file_filter)]
        if tag_filter:
            # tag match if any selected tag in row tags
            mask = []
            for _, r in fdf.iterrows():
                row_tags = {t.strip() for t in str(r["tags"]).split(",") if t.strip()}
                mask.append(bool(row_tags.intersection(set(tag_filter))))
            fdf = fdf[pd.Series(mask, index=fdf.index)]
        if fav_only:
            fdf = fdf[fdf["favorite"] == True]

        st.caption(f"Mostrando **{len(fdf)}** de **{len(df)}** términos")

        # Favorites controls
        cfa, cfb = st.columns([1, 1])
        with cfa:
            if st.button("🧹 Limpiar favoritos", use_container_width=True):
                st.session_state.favorites = set()
                st.rerun()
        with cfb:
            if st.button("⭐ Marcar visibles como favoritos", use_container_width=True):
                for t in fdf["term"].tolist():
                    st.session_state.favorites.add(t)
                st.rerun()

        st.divider()

        if view == "Tabla":
            # Table with a "favorite" column toggled by selection
            show_cols = ["favorite", "term", "tags", "file", "score", "definition"]
            st.dataframe(
                fdf[show_cols].sort_values(["favorite", "score"], ascending=[False, False]),
                use_container_width=True,
                height=560
            )
            st.caption("En v0.2 agregamos click para togglear ⭐ directo desde la tabla.")
        else:
            # Cards: show per file sections
            for file_name in sorted(fdf["file"].unique().tolist()):
                sub = fdf[fdf["file"] == file_name].sort_values("score", ascending=False)

                with st.expander(f"{file_name} — {len(sub)} términos", expanded=True if len(pages_out) == 1 else False):
                    for _, r in sub.head(200).iterrows():  # límite visual
                        term = r["term"]
                        fav = (term in st.session_state.favorites)

                        left, right = st.columns([0.88, 0.12])
                        with left:
                            st.markdown(f"### {term} {'⭐' if fav else ''}")
                            st.caption(r["tags"])
                            st.write(r["definition"])
                        with right:
                            if st.button("⭐" if not fav else "✅", key=f"fav_{file_name}_{term}"):
                                if fav:
                                    st.session_state.favorites.discard(term)
                                else:
                                    st.session_state.favorites.add(term)
                                st.rerun()
                        st.divider()


# ---- Export tab
with tab_export:
    st.subheader("Exportar")
    out = st.session_state.last_output

    if not out:
        st.info("Primero genera un glosario.")
    else:
        colx, coly = st.columns([1, 1])
        with colx:
            st.download_button(
                label="⬇️ Descargar glosario.html",
                data=out["html"].encode("utf-8"),
                file_name="glosario.html",
                mime="text/html",
                use_container_width=True
            )
        with coly:
            st.download_button(
                label="⬇️ Descargar glosario.json",
                data=out["json"].encode("utf-8"),
                file_name="glosario.json",
                mime="application/json",
                use_container_width=True
            )

        st.divider()
        st.subheader("Preview")
        if preview_on:
            st.components.v1.html(out["html"], height=820, scrolling=True)
        else:
            st.info("Activa el toggle “Preview HTML dentro de la app” (sidebar).")


# ---- History tab
with tab_history:
    st.subheader("Historial (sesión actual)")
    if not st.session_state.run_history:
        st.info("Aún no hay corridas. Genera un glosario y aparecerán aquí.")
    else:
        for i, r in enumerate(st.session_state.run_history[:20], start=1):
            with st.expander(f"#{i} — {r['total_terms']} términos — {r['elapsed']}s — {len(r['files'])} archivos", expanded=False):
                st.write("**Archivos:**")
                for fn in r["files"]:
                    st.write(f"• {fn}")
                if r.get("settings"):
                    st.write("**Settings:**")
                    st.json(r["settings"])