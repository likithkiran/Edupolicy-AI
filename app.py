"""
app.py
EduPolicy AI — Intelligent Academic Policy Search & RAG System.
A modern, premium SaaS interface for searching, querying, and verifying
academic policies and regulations with verifiable citations.
"""

import os
import tempfile
import time
from typing import List, Optional
import streamlit as st
from dotenv import load_dotenv

from pdf_processor import extract_chunks_from_pdf, generate_sample_policy_pdf, PageChunk
from search_engine import SemanticSearchEngine, DEFAULT_MODEL
from rag_engine import RAGEngine, extract_concise_policy_rules

# Load local environment variables if available
load_dotenv()

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="EduPolicy AI · Academic Policy Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Premium SaaS CSS Styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts & Base Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1280px;
    }

    /* Gradient Brand Header */
    .brand-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e1b4b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2.2rem 2.4rem;
        color: #ffffff;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.35);
        position: relative;
        overflow: hidden;
    }
    .brand-hero::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-subtitle {
        font-size: 1.02rem;
        color: #94a3b8;
        max-width: 720px;
        line-height: 1.55;
        margin-bottom: 0;
    }

    /* KPI Metrics Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1rem;
        margin-bottom: 1.8rem;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0,0,0,0.08);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-badge {
        display: inline-block;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 9999px;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .badge-success { background: #dcfce7; color: #166534; }
    .badge-info    { background: #e0f2fe; color: #0369a1; }
    .badge-warn    { background: #fef3c7; color: #92400e; }

    /* Answer Container */
    .answer-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin: 0.8rem 0 1.2rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        line-height: 1.65;
    }
    .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #eef2ff;
        color: #4338ca;
        border: 1px solid #c7d2fe;
        border-radius: 6px;
        padding: 3px 9px;
        margin: 3px 4px 3px 0;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Result Passage Box */
    .passage-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 9px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .passage-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.45rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
    }

    /* Topic Pill */
    .topic-chip {
        display: inline-block;
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 5px 12px;
        margin: 3px;
        font-size: 0.82rem;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .topic-chip:hover {
        background: #e2e8f0;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Cached Resources
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_semantic_engine() -> SemanticSearchEngine:
    """Load and cache the SentenceTransformer search engine."""
    return SemanticSearchEngine(DEFAULT_MODEL)


# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------
if "search_engine" not in st.session_state:
    st.session_state.search_engine = get_semantic_engine()

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []

if "searches_run_count" not in st.session_state:
    st.session_state.searches_run_count = 0

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

# Auto-load Gemini API key from environment if available
default_env_key = os.environ.get("GEMINI_API_KEY", "").strip()
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = default_env_key

if "gemini_model_name" not in st.session_state:
    st.session_state.gemini_model_name = "gemini-1.5-flash"

if "top_k_val" not in st.session_state:
    st.session_state.top_k_val = 5

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(
        search_engine=st.session_state.search_engine,
        gemini_api_key=st.session_state.gemini_api_key,
        model_name=st.session_state.gemini_model_name,
        top_k=st.session_state.top_k_val,
    )

# ------------------------------------------------------------------
# Auto-index Sample Policy Handbook on First Run if empty
# ------------------------------------------------------------------
if not st.session_state.search_engine.is_ready:
    sample_pdf_path = os.path.join(os.getcwd(), "University_Academic_Policy_Handbook.pdf")
    if not os.path.exists(sample_pdf_path):
        generate_sample_policy_pdf(sample_pdf_path)

    if os.path.exists(sample_pdf_path):
        try:
            sample_chunks = extract_chunks_from_pdf(sample_pdf_path)
            if sample_chunks:
                st.session_state.search_engine.build_index(sample_chunks)
                st.session_state.indexed_files = [os.path.basename(sample_pdf_path)]
                st.session_state.total_pages = len(sample_chunks)
        except Exception:
            pass

# ------------------------------------------------------------------
# Sidebar: Configuration & Document Management
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 **EduPolicy AI**")
    st.caption("Verifiable Academic Policy Intelligence")
    st.divider()

    st.markdown("#### 🔑 **Gemini API Configuration**")
    api_key_input = st.text_input(
        "API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Paste your free Google AI Studio key to unlock generative synthesis with citations.",
    )

    if api_key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key_input.strip()
        st.session_state.rag_engine.update_key(
            api_key_input.strip(), st.session_state.gemini_model_name
        )

    model_choice = st.selectbox(
        "Model Selection",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
    )
    if model_choice != st.session_state.gemini_model_name:
        st.session_state.gemini_model_name = model_choice
        st.session_state.rag_engine.update_key(
            st.session_state.gemini_api_key, model_choice
        )

    if st.session_state.gemini_api_key:
        st.markdown(
            '<span class="kpi-badge badge-success">✓ Gemini Active</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="kpi-badge badge-info">ℹ Local Semantic Mode</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("#### 📑 **Policy Library**")
    st.caption(
        f"**{len(st.session_state.indexed_files)}** document(s) · **{st.session_state.total_pages}** total pages"
    )

    uploaded_files = st.file_uploader(
        "Upload Policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic handbooks, bylaws, or exam regulations.",
    )

    if uploaded_files:
        if st.button("📥 Index Uploaded PDFs", type="primary", use_container_width=True):
            with st.spinner("Extracting text and generating embeddings..."):
                new_chunks: List[PageChunk] = []
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uf in uploaded_files:
                        tmp_path = os.path.join(temp_dir, uf.name)
                        with open(tmp_path, "wb") as f:
                            f.write(uf.read())
                        try:
                            chunks = extract_chunks_from_pdf(tmp_path)
                            new_chunks.extend(chunks)
                            if uf.name not in st.session_state.indexed_files:
                                st.session_state.indexed_files.append(uf.name)
                        except Exception as e:
                            st.error(f"Error parsing '{uf.name}': {e}")

                if new_chunks:
                    if st.session_state.search_engine.is_ready:
                        st.session_state.search_engine.add_chunks(new_chunks)
                    else:
                        st.session_state.search_engine.build_index(new_chunks)

                    st.session_state.total_pages = len(st.session_state.search_engine.chunks)
                    st.success(f"Indexed {len(new_chunks)} pages successfully!")
                    time.sleep(0.8)
                    st.rerun()

    # Reset or reload sample
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🔄 Sample Policy", use_container_width=True, help="Load official 7-page university handbook"):
            sample_pdf_path = os.path.join(os.getcwd(), "University_Academic_Policy_Handbook.pdf")
            generate_sample_policy_pdf(sample_pdf_path)
            sample_chunks = extract_chunks_from_pdf(sample_pdf_path)
            st.session_state.search_engine.build_index(sample_chunks)
            st.session_state.indexed_files = [os.path.basename(sample_pdf_path)]
            st.session_state.total_pages = len(sample_chunks)
            st.session_state.chat_history = []
            st.rerun()

    with col_s2:
        if st.button("🗑️ Reset All", use_container_width=True, help="Clear index"):
            st.session_state.search_engine = get_semantic_engine()
            st.session_state.search_engine.chunks = []
            st.session_state.search_engine.embeddings = None
            st.session_state.indexed_files = []
            st.session_state.total_pages = 0
            st.session_state.chat_history = []
            st.session_state.recent_queries = []
            st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size: 0.78rem; color: #64748b; text-align: center;'>"
        "EduPolicy AI v2.0 · Enterprise Edition<br>"
        "Powered by PyMuPDF · MiniLM · Gemini"
        "</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Header Hero Section
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-hero">
        <div class="brand-title">🎓 EduPolicy AI</div>
        <p class="brand-subtitle">
            Instant semantic search and cited AI question answering across academic regulations,
            course policies, attendance rules, and graduation requirements.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Navigation Tabs
# ------------------------------------------------------------------
tab_dash, tab_chat, tab_search, tab_docs, tab_settings = st.tabs(
    [
        "📊 Dashboard",
        "💬 AI Policy Advisor",
        "🔍 Semantic Search",
        "📁 Document Library",
        "⚙️ Engine & API",
    ]
)

# ==================================================================
# TAB 1: DASHBOARD
# ==================================================================
with tab_dash:
    # KPI Grid
    gemini_status_badge = (
        '<span class="kpi-badge badge-success">● Connected</span>'
        if st.session_state.gemini_api_key
        else '<span class="kpi-badge badge-info">● Local Mode</span>'
    )
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Indexed Documents</div>
                <div class="kpi-value">{len(st.session_state.indexed_files)}</div>
                <span class="kpi-badge badge-info">PDF Repositories</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Policy Pages</div>
                <div class="kpi-value">{st.session_state.total_pages}</div>
                <span class="kpi-badge badge-info">Vector Chunks</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Queries Run</div>
                <div class="kpi-value">{st.session_state.searches_run_count}</div>
                <span class="kpi-badge badge-info">This Session</span>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">AI RAG Status</div>
                <div class="kpi-value" style="font-size: 1.25rem; font-weight: 700; margin-top: 0.35rem;">
                    {st.session_state.gemini_model_name}
                </div>
                {gemini_status_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Search Bar in Dashboard
    st.markdown("### ⚡ **Quick Policy Search**")
    dash_col_input, dash_col_btn = st.columns([5, 1])

    with dash_col_input:
        dash_query = st.text_input(
            "Search regulations",
            placeholder="e.g. What is the penalty for late assignment submission?",
            key="dash_search_input",
            label_visibility="collapsed",
        )
    with dash_col_btn:
        dash_search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Starter Topics & Questions
    st.markdown("##### 📌 **Popular Academic Topics**")
    starter_topics = [
        "What is the minimum attendance required to avoid FA?",
        "How are late assignments penalized per day?",
        "Can I use ChatGPT or AI tools for coursework?",
        "What are the requirements for Dean's Honor Roll?",
        "What happens if cumulative GPA drops below 2.00?",
        "How many course withdrawals (W) are permitted?",
    ]

    selected_chip_col1, selected_chip_col2, selected_chip_col3 = st.columns(3)
    for idx, prompt_text in enumerate(starter_topics):
        col = [selected_chip_col1, selected_chip_col2, selected_chip_col3][idx % 3]
        if col.button(f"👉 {prompt_text}", key=f"chip_{idx}", use_container_width=True):
            dash_query = prompt_text
            dash_search_clicked = True

    # Execute Quick Search from Dashboard
    if dash_search_clicked and dash_query.strip():
        st.session_state.searches_run_count += 1
        if dash_query not in st.session_state.recent_queries:
            st.session_state.recent_queries.insert(0, dash_query)
            st.session_state.recent_queries = st.session_state.recent_queries[:6]

        with st.spinner("Searching policies and generating answer..."):
            ans_result = st.session_state.rag_engine.answer(
                dash_query, chat_history=st.session_state.chat_history
            )

        st.markdown(f"#### 💡 Policy Answer for: *\"{dash_query}\"*")
        with st.container(border=True):
            st.markdown(ans_result["answer"])

        if ans_result["sources"]:
            st.markdown(
                "**Verified Policy Citations:** "
                + " ".join(
                    f'<span class="source-tag">📄 {s}</span>'
                    for s in ans_result["sources"]
                ),
                unsafe_allow_html=True,
            )

        with st.expander("🔎 View Grounding Passages & Citations"):
            for chunk, score in ans_result["passages"]:
                takeaways = extract_concise_policy_rules(chunk.text, query=dash_query, max_bullets=3)
                takeaways_html = "".join(f"<li style='margin-bottom: 4px;'>{t}</li>" for t in takeaways)
                st.markdown(
                    f"""
                    <div class="passage-card">
                        <div class="passage-header">
                            <span>📄 <strong>[{chunk.citation}]</strong></span>
                            <span>Relevance: <strong>{score * 100:.1f}%</strong></span>
                        </div>
                        <div style="font-size: 0.88rem; color: #1e293b; line-height: 1.5; margin-top: 4px;">
                            <ul style="padding-left: 1.2rem; margin-bottom: 0.2rem;">
                                {takeaways_html}
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"📄 Full Original Page Text ({chunk.citation})"):
                    st.text(chunk.text)

    # Recent Searches
    if st.session_state.recent_queries:
        st.markdown("---")
        st.markdown("##### 🕒 **Recent Searches**")
        r_cols = st.columns(len(st.session_state.recent_queries))
        for i, q_text in enumerate(st.session_state.recent_queries):
            if r_cols[i].button(f"🔍 {q_text[:28]}…", key=f"rec_{i}", use_container_width=True):
                st.session_state.pending_query = q_text
                st.rerun()

# ==================================================================
# TAB 2: AI POLICY ADVISOR (CHATBOT)
# ==================================================================
with tab_chat:
    st.markdown("### 💬 **Interactive AI Policy Advisor**")
    st.caption("Ask multi-turn questions about graduation rules, grade appeals, leaves of absence, and academic standing.")

    chat_actions_col1, chat_actions_col2 = st.columns([6, 1])
    with chat_actions_col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Display past conversation
    for chat_turn in st.session_state.chat_history:
        with st.chat_message("user", avatar="🎓"):
            st.markdown(chat_turn["q"])

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(chat_turn["r"]["answer"])
            if chat_turn["r"].get("sources"):
                source_pills = " ".join(
                    f'<span class="source-tag">📄 {src}</span>'
                    for src in chat_turn["r"]["sources"]
                )
                st.markdown(
                    f"<div style='margin-top: 8px;'><strong>Citations:</strong> {source_pills}</div>",
                    unsafe_allow_html=True,
                )

    # Chat input box
    user_chat_query = st.chat_input("Ask a question about university policies...")

    # Handle pending query from chips/recent searches
    if st.session_state.pending_query and not user_chat_query:
        user_chat_query = st.session_state.pending_query
        st.session_state.pending_query = ""

    if user_chat_query:
        st.session_state.searches_run_count += 1
        if user_chat_query not in st.session_state.recent_queries:
            st.session_state.recent_queries.insert(0, user_chat_query)
            st.session_state.recent_queries = st.session_state.recent_queries[:6]

        with st.chat_message("user", avatar="🎓"):
            st.markdown(user_chat_query)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing regulations..."):
                chat_res = st.session_state.rag_engine.answer(
                    user_chat_query, chat_history=st.session_state.chat_history
                )
                st.markdown(chat_res["answer"])

                if chat_res.get("sources"):
                    source_pills = " ".join(
                        f'<span class="source-tag">📄 {src}</span>'
                        for src in chat_res["sources"]
                    )
                    st.markdown(
                        f"<div style='margin-top: 8px;'><strong>Citations:</strong> {source_pills}</div>",
                        unsafe_allow_html=True,
                    )

                with st.expander("🔎 View Grounding Passages & Evidence"):
                    for chunk, score in chat_res.get("passages", []):
                        takeaways = extract_concise_policy_rules(chunk.text, query=user_chat_query, max_bullets=3)
                        takeaways_html = "".join(f"<li style='margin-bottom: 4px;'>{t}</li>" for t in takeaways)
                        st.markdown(
                            f"""
                            <div class="passage-card">
                                <div class="passage-header">
                                    <span>📄 <strong>[{chunk.citation}]</strong></span>
                                    <span>Relevance: <strong>{score * 100:.1f}%</strong></span>
                                </div>
                                <div style="font-size: 0.88rem; color: #1e293b; line-height: 1.5; margin-top: 4px;">
                                    <ul style="padding-left: 1.2rem; margin-bottom: 0.2rem;">
                                        {takeaways_html}
                                    </ul>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"📄 Full Original Page Text ({chunk.citation})"):
                            st.text(chunk.text)

        # Append to chat history
        st.session_state.chat_history.append({"q": user_chat_query, "r": chat_res})

# ==================================================================
# TAB 3: DEEP SEMANTIC SEARCH
# ==================================================================
with tab_search:
    st.markdown("### 🔍 **Semantic Regulation Search**")
    st.caption("Search policies by conceptual meaning rather than exact keyword matches.")

    s_col_query, s_col_filter = st.columns([4, 2])

    with s_col_query:
        sem_query = st.text_input(
            "Search Query",
            placeholder="e.g. What happens if I miss a midterm due to illness?",
            key="sem_search_box",
        )

    with s_col_filter:
        doc_options = ["All Documents"] + st.session_state.search_engine.document_names
        selected_doc = st.selectbox("Filter Document", doc_options, index=0)

    s_tune_col1, s_tune_col2 = st.columns(2)
    with s_tune_col1:
        sem_top_k = st.slider("Results to return", min_value=1, max_value=12, value=5)
    with s_tune_col2:
        sem_min_score = st.slider(
            "Minimum Similarity Match",
            min_value=0.0,
            max_value=0.8,
            value=0.25,
            step=0.05,
            format="%d%%",
        )

    if sem_query.strip():
        with st.spinner("Computing cosine similarities across vector embeddings..."):
            raw_results = st.session_state.search_engine.search(
                sem_query,
                top_k=sem_top_k,
                doc_filter=selected_doc,
                min_score=sem_min_score,
            )

        st.markdown(f"#### 📄 Found **{len(raw_results)}** matching policy section(s)")

        if not raw_results:
            st.info("No policy passages met the minimum similarity score. Try lowering the threshold.")
        else:
            for rank, (chunk, score) in enumerate(raw_results, start=1):
                pct = score * 100
                takeaways = extract_concise_policy_rules(chunk.text, query=sem_query, max_bullets=3)
                takeaways_html = "".join(f"<li style='margin-bottom: 5px;'>{t}</li>" for t in takeaways)
                st.markdown(
                    f"""
                    <div class="passage-card" style="border-left: 4px solid {'#10b981' if pct > 65 else '#6366f1'};">
                        <div class="passage-header">
                            <span>#{rank} &nbsp; 📄 <strong>[{chunk.citation}]</strong></span>
                            <span>Cosine Similarity: <strong>{pct:.1f}%</strong></span>
                        </div>
                        <div style="margin-top: 0.6rem; font-size: 0.9rem; color: #0f172a; line-height: 1.55;">
                            <div style="font-weight: 600; color: #4338ca; margin-bottom: 4px; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em;">
                                📋 Key Policy Takeaways:
                            </div>
                            <ul style="padding-left: 1.25rem; margin-bottom: 0.3rem;">
                                {takeaways_html}
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"📄 View Original PDF Page Excerpt ({chunk.citation})"):
                    st.text(chunk.text)
                st.progress(min(max(score, 0.0), 1.0))

# ==================================================================
# TAB 4: DOCUMENT LIBRARY & PAGE INSPECTOR
# ==================================================================
with tab_docs:
    st.markdown("### 📁 **Academic Policy Repository**")
    st.caption("Manage indexed PDF policy handbooks and inspect extracted text page-by-page.")

    if not st.session_state.search_engine.chunks:
        st.warning("No documents are currently indexed. Upload a PDF or click 'Sample Policy' in the sidebar.")
    else:
        # Document Summary Table
        docs_summary = {}
        for c in st.session_state.search_engine.chunks:
            if c.doc_name not in docs_summary:
                docs_summary[c.doc_name] = {"pages": 0, "chars": 0, "words": 0}
            docs_summary[c.doc_name]["pages"] += 1
            docs_summary[c.doc_name]["chars"] += c.char_count
            docs_summary[c.doc_name]["words"] += c.word_count

        table_data = [
            {
                "Document Name": doc_name,
                "Pages Indexed": data["pages"],
                "Total Words": f"{data['words']:,}",
                "Character Count": f"{data['chars']:,}",
                "Status": "✅ Fully Vectorized",
            }
            for doc_name, data in docs_summary.items()
        ]

        st.dataframe(table_data, use_container_width=True)

        st.divider()
        st.markdown("#### 📖 **Page Inspector**")
        st.caption("Read the exact text extracted from any indexed page to verify citations.")

        insp_col1, insp_col2 = st.columns([2, 1])
        with insp_col1:
            inspect_doc = st.selectbox(
                "Select Document to Inspect",
                list(docs_summary.keys()),
                key="doc_inspect_sel",
            )
        with insp_col2:
            max_pages = docs_summary[inspect_doc]["pages"] if inspect_doc in docs_summary else 1
            inspect_page = st.number_input(
                f"Page (1 to {max_pages})",
                min_value=1,
                max_value=max_pages,
                value=1,
                step=1,
            )

        matching_chunks = [
            c
            for c in st.session_state.search_engine.chunks
            if c.doc_name == inspect_doc and c.page_number == inspect_page
        ]

        if matching_chunks:
            chunk = matching_chunks[0]
            st.markdown(
                f"""
                <div class="answer-card" style="border-left: 4px solid #0284c7;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #0284c7; text-transform: uppercase;">
                        {chunk.citation} · {chunk.word_count} words
                    </div>
                    <div style="font-size: 0.95rem; color: #1e293b; line-height: 1.65; margin-top: 0.6rem; white-space: pre-wrap;">
{chunk.text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No text content found for this specific page.")

# ==================================================================
# TAB 5: ENGINE & API SETTINGS
# ==================================================================
with tab_settings:
    st.markdown("### ⚙️ **Engine & API Settings**")
    st.caption("Fine-tune embedding vectors, retrieval parameters, and Gemini generation.")

    set_col1, set_col2 = st.columns(2)

    with set_col1:
        st.markdown("#### 🧠 **Vector Embedding Model**")
        st.markdown(
            f"""
            - **Model Architecture:** `{DEFAULT_MODEL}` (MiniLM-L6-v2)
            - **Vector Dimensions:** `384` floating-point dimensions
            - **Similarity Metric:** Normalized Cosine Similarity (Dot Product)
            - **Execution Hardware:** CPU Vector Operations via NumPy
            - **Latency:** ~5ms per search query
            """
        )

    with set_col2:
        st.markdown("#### 🤖 **Generative Model & RAG**")
        st.markdown(
            f"""
            - **Active Generative Model:** `{st.session_state.gemini_model_name}`
            - **API Connection:** {'✅ Configured & Authenticated' if st.session_state.gemini_api_key else '⚠️ Local Extractive Mode (Key Missing)'}
            - **Strict Source Grounding:** Enforced via System Instructions
            - **Context Window:** Top {st.session_state.top_k_val} retrieved chunks
            """
        )

    st.divider()
    st.markdown("#### 🛠️ **System Diagnostics**")
    diag_c1, diag_c2, diag_c3 = st.columns(3)
    diag_c1.metric("Indexed Chunks", len(st.session_state.search_engine.chunks))
    diag_c2.metric(
        "Embedding Matrix",
        f"{st.session_state.search_engine.embeddings.shape if st.session_state.search_engine.embeddings is not None else 'None'}",
    )
    diag_c3.metric("Chat Messages Logged", len(st.session_state.chat_history))
