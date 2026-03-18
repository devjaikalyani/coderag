"""
app.py — CodeRAG Streamlit UI
"""

import re
import time
import httpx
import streamlit as st
from datetime import datetime

API_BASE = "http://localhost:8001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_markdown(text: str) -> str:
    """
    Normalise LLM markdown for clean Streamlit rendering.
    - ### Heading  →  **Heading**  (no giant h3 in chat bubbles)
    - * bullet     →  - bullet
    - Ensure blank lines before sections and lists
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        # Headings → bold label
        m = re.match(r"^#{1,4}\s+(.+)$", line)
        if m:
            line = f"**{m.group(1)}**"
        # * bullet → - bullet
        if line.startswith("* "):
            line = "- " + line[2:]
        # Blank line before bold section labels
        if line.startswith("**") and line.endswith("**") and result and result[-1].strip():
            result.append("")
        # Blank line before bullet lists
        if (line.startswith("- ") and result
                and result[-1].strip()
                and not result[-1].startswith("- ")):
            result.append("")
        result.append(line)
    return "\n".join(result)


def _safe_error(r) -> str:
    try:
        return r.json().get("detail", r.text or "Unknown error")
    except Exception:
        return r.text or f"HTTP {r.status_code}"


def _get_stats():
    for attempt in range(3):
        try:
            return httpx.get(f"{API_BASE}/index/stats", timeout=5).json()
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return None


def _faithfulness_badge(score: float):
    if score >= 0.75:
        return f"🟢 Answer supported by source code ({score:.0%})"
    elif score >= 0.5:
        return f"🟡 Partially supported ({score:.0%})"
    return None


def _render_sources(sources: list):
    if not sources:
        return
    label = f"📄 {len(sources)} source{'s' if len(sources) != 1 else ''} retrieved"
    with st.expander(label, expanded=False):
        for i, s in enumerate(sources, 1):
            score = s.get("rerank_score", 0)
            lang  = s.get("language", "text") or "text"
            st.markdown(
                f"**[{i}]** `{s['source']}` &nbsp;·&nbsp; "
                f"lines {s['start_line']}–{s['end_line']} &nbsp;·&nbsp; "
                f"`{lang}` &nbsp;·&nbsp; relevance **{score:.0%}**"
            )
            preview = (s.get("text_preview") or "").strip()
            if preview:
                st.code(preview + "…", language=lang)
            if i < len(sources):
                st.divider()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CodeRAG", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages"   not in st.session_state: st.session_state.messages   = []
if "active_key" not in st.session_state: st.session_state.active_key = None

# ---------------------------------------------------------------------------
# Fetch live stats
# ---------------------------------------------------------------------------

stats      = _get_stats()
sources_db = stats.get("ingested_sources", []) if stats else []
active_key = stats.get("active_key")           if stats else None
if active_key:
    st.session_state.active_key = active_key

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("CodeRAG — Code & Documentation Assistant")
if active_key:
    src = next((s for s in sources_db if s["key"] == active_key), None)
    name   = src.get("display_name", active_key) if src else active_key
    chunks = src.get("chunk_count", "?")          if src else "?"
    st.caption(f"🟢 **{name}** · {chunks} chunks indexed")
else:
    st.caption("⚪ No repo selected — add one from the sidebar")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    if stats is None:
        st.error("⚠️ API not reachable")
        st.code("OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \\\nuvicorn src.api.main:app --port 8001 --workers 1")
        if st.button("🔄 Retry", type="primary"):
            st.rerun()
        st.stop()

    # ── Repo selector ──────────────────────────────────────────────────
    st.header("Your Repositories")

    if sources_db:
        options = {s["key"]: s for s in sources_db}
        labels  = {
            k: f"{'🟢' if k == active_key else '⚪'} {v['display_name']} ({v.get('chunk_count','?')} chunks)"
            for k, v in options.items()
        }
        selected = st.radio(
            "Select active repo",
            list(options.keys()),
            format_func=lambda k: labels[k],
            index=list(options.keys()).index(active_key) if active_key in options else 0,
            key="repo_selector",
        )
        if selected != active_key:
            with st.spinner("Switching…"):
                r = httpx.post(f"{API_BASE}/repo/switch", json={"key": selected}, timeout=30)
            if r.status_code == 200:
                st.session_state.active_key = selected
                st.session_state.messages   = []
                st.rerun()
            else:
                st.error(_safe_error(r))

        if active_key and active_key in options:
            src = options[active_key]
            ts  = src.get("ingested_at", "")
            try:
                ts = datetime.fromisoformat(ts).strftime("%b %d %H:%M")
            except Exception:
                pass
            st.caption(f"Ingested: {ts}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                try: httpx.delete(f"{API_BASE}/history", timeout=5)
                except: pass
                st.rerun()
        with col2:
            if active_key and st.button("Delete repo", use_container_width=True):
                httpx.delete(f"{API_BASE}/repo/{active_key}", timeout=10)
                st.session_state.messages   = []
                st.session_state.active_key = None
                st.rerun()
        if st.button("Delete ALL repos", use_container_width=True):
            httpx.delete(f"{API_BASE}/index", timeout=10)
            st.session_state.messages   = []
            st.session_state.active_key = None
            st.rerun()
    else:
        st.info("No repos yet — add one below.")

    st.divider()

    # ── Add Repo ────────────────────────────────────────────────────────
    st.header("Add a Repository")

    tab = st.radio("Source", ["GitHub URL","Local Directory","Paste Text","Upload File"],
                   key="ingest_tab")

    if tab == "GitHub URL":
        gh_url    = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo", key="gh_url")
        gh_branch = st.text_input("Branch", value="main", key="gh_branch")
        if st.button("Ingest GitHub Repo", type="primary", use_container_width=True):
            url = gh_url.strip().strip("_")
            if not url.startswith("http"):
                st.error("Enter a valid URL starting with https://")
            else:
                with st.spinner(f"Cloning {url.split('/')[-1]}…"):
                    try:
                        r = httpx.post(f"{API_BASE}/ingest/github",
                                       json={"url": url, "branch": gh_branch.strip() or "main"},
                                       timeout=600)
                        if r.status_code == 200:
                            data = r.json()
                            if data.get("status") == "skipped":
                                st.info(f"ℹ️ {data['reason']}")
                            else:
                                st.success(f"✅ {data.get('new_chunks','?')} chunks indexed")
                            st.session_state.messages = []
                            st.rerun()
                        else:
                            st.error(_safe_error(r))
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif tab == "Local Directory":
        lpath = st.text_input("Directory path", placeholder="/path/to/project", key="lpath")
        if st.button("Ingest Directory", type="primary", use_container_width=True):
            path = lpath.strip()
            if not path or path == "/path/to/project":
                st.error("Enter a valid path")
            else:
                with st.spinner("Ingesting…"):
                    try:
                        r = httpx.post(f"{API_BASE}/ingest/directory",
                                       json={"path": path}, timeout=300)
                        if r.status_code == 200:
                            data = r.json()
                            if data.get("status") == "skipped":
                                st.info(f"ℹ️ {data['reason']}")
                            else:
                                st.success(f"✅ {data.get('new_chunks','?')} chunks indexed")
                            st.session_state.messages = []
                            st.rerun()
                        else:
                            st.error(_safe_error(r))
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif tab == "Paste Text":
        raw  = st.text_area("Paste code or docs", height=150, key="paste_raw")
        name = st.text_input("Source name", value="pasted_code.py", key="paste_name")
        if st.button("Ingest Text", type="primary", use_container_width=True):
            with st.spinner("Indexing…"):
                try:
                    r = httpx.post(f"{API_BASE}/ingest/text",
                                   json={"text": raw, "source_name": name}, timeout=120)
                    if r.status_code == 200:
                        st.success("✅ Indexed")
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error(_safe_error(r))
                except Exception as e:
                    st.error(f"Error: {e}")

    elif tab == "Upload File":
        up = st.file_uploader("Upload file",
                              type=["py","js","ts","md","txt","go","rs","java","css","html"])
        if up and st.button("Ingest File", type="primary", use_container_width=True):
            with st.spinner("Indexing…"):
                try:
                    r = httpx.post(f"{API_BASE}/ingest/file",
                                   files={"file": (up.name, up.getvalue(), "text/plain")},
                                   timeout=120)
                    if r.status_code == 200:
                        st.success("✅ Indexed")
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error(_safe_error(r))
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.markdown(
        "**Stack used**\n"
        "- 🧠 CodeBERT embeddings\n"
        "- 🔍 FAISS + BM25 hybrid search\n"
        "- ⚡ Llama 3 70B via Groq\n"
        "- 🛡️ NLI faithfulness check\n"
        "- 📊 MLflow experiment tracking\n"
    )

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(_fix_markdown(msg["content"]))
        if msg.get("sources"):
            _render_sources(msg["sources"])
        if msg.get("faith") is not None:
            badge = _faithfulness_badge(msg["faith"])
            if badge:
                st.caption(badge)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

is_ready = active_key is not None

if prompt := st.chat_input(
    "Ask anything about the active repo…" if is_ready else "Select or ingest a repo first…",
    disabled=not is_ready,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Non-streaming — ensures _fix_markdown always runs on complete response
        with st.spinner("Thinking…"):
            try:
                r = httpx.post(
                    f"{API_BASE}/query",
                    json={"question": prompt, "check_faithfulness": True},
                    timeout=120,
                )
                if r.status_code == 200:
                    data          = r.json()
                    full_response = data["answer"]
                    sources       = data.get("sources", [])
                    faith         = data.get("faithfulness_score")
                else:
                    full_response = f"Error: {_safe_error(r)}"
                    sources, faith = [], None
            except Exception as e:
                full_response  = f"Connection error: {e}"
                sources, faith = [], None

        placeholder.markdown(_fix_markdown(full_response))
        _render_sources(sources)
        if faith is not None:
            badge = _faithfulness_badge(faith)
            if badge:
                st.caption(badge)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "sources": sources,
        "faith":   faith,
    })

if not is_ready:
    st.info("👈 Add a GitHub repo from the sidebar to get started.")