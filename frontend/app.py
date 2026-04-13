"""
frontend/app.py
────────────────
RecommendAI — Streamlit Frontend

Entrypoint:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import frontend.api_client as api
from frontend.components import (
    render_empty_state,
    render_error,
    render_metric_row,
    render_query_card,
    render_recommendation_card,
    render_sidebar_stats,
)

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RecommendAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0d1117;
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f1923 !important;
        border-right: 1px solid #1e293b;
    }

    /* Header area */
    .hero-header {
        background: linear-gradient(135deg, #0f2027 0%, #1a1a2e 50%, #16213e 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 32px;
    }

    /* Cards */
    [data-testid="stContainer"] {
        background: #0f1923;
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 12px 28px !important;
        font-size: 16px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(59,130,246,0.4) !important;
    }

    /* Selectbox / Input */
    .stSelectbox > div > div,
    .stTextInput > div > div {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    /* Slider */
    .stSlider > div {
        color: #94a3b8;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #1e293b;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #334155;
    }

    /* Divider */
    hr {
        border-color: #1e293b !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e293b;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.session_state.results = None
if "query_meta" not in st.session_state:
    st.session_state.query_meta = None
if "all_titles" not in st.session_state:
    st.session_state.all_titles = []
if "categories" not in st.session_state:
    st.session_state.categories = []

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style='padding:16px 0 8px'>
            <div style='font-size:28px;font-weight:800;color:#f1f5f9'> AI Recommend System</div>
            <div style='color:#64748b;font-size:13px'>Cosine Similarity Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Parameters ─────────────────────────────────────────────
    st.markdown("### ⚙️ Parameters")

    top_n = st.slider("Number of recommendations", 3, 20, 8)

    mode = st.selectbox(
        "Recommendation mode",
        options=["hybrid", "content"],
        format_func=lambda x: "🔀 Hybrid (Content + Collaborative)" if x == "hybrid" else "📄 Content-Based Only",
    )

    # Category filter
    if not st.session_state.categories:
        with st.spinner("Loading categories…"):
            try:
                st.session_state.categories = api.get_categories()
            except Exception:
                st.session_state.categories = []

    cat_options = ["All genres"] + st.session_state.categories
    cat_selection = st.selectbox("Filter by genre", cat_options)
    category_filter = None if cat_selection == "All genres" else cat_selection

    # Optional user ID for hybrid
    if mode == "hybrid":
        user_id = st.text_input("User ID (optional)", placeholder="e.g. user_42")
        user_id = user_id.strip() or None
    else:
        user_id = None

    st.divider()

    # ── Similarity calculator ───────────────────────────────────
    st.markdown("### 🔬 Similarity Calculator")
    with st.expander("Compare two items"):
        if not st.session_state.all_titles:
            with st.spinner("Loading titles…"):
                try:
                    st.session_state.all_titles = api.get_all_titles()
                except Exception:
                    pass

        titles = st.session_state.all_titles or []
        sim_a = st.selectbox("Item A", titles, key="sim_a")
        sim_b = st.selectbox("Item B", titles, index=min(1, len(titles) - 1), key="sim_b")
        if st.button("Calculate", use_container_width=True, key="calc_sim"):
            if sim_a and sim_b:
                with st.spinner("Computing…"):
                    try:
                        score = api.get_similarity(sim_a, sim_b)
                        st.metric("Cosine Similarity", f"{score:.4f}")
                        st.progress(float(score))
                    except Exception as e:
                        st.error(str(e))

    # ── Health status ──────────────────────────────────────────
    health = api.health_check()
    render_sidebar_stats(health)

# ══════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════

# Hero header
st.markdown(
    """
    <div class='hero-header'>
        <h1 style='font-size:36px;font-weight:800;color:#f1f5f9;margin:0 0 8px'>
            AI Recommendation System
        </h1>
        <p style='color:#94a3b8;font-size:16px;margin:0'>
            Personalized recommendations powered by cosine similarity &amp; hybrid filtering.
            Select a movie below to discover what to watch next.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────
tab_rec, tab_browse, tab_about = st.tabs(["🎯 Recommendations", "📚 Browse Catalog", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════
#  TAB 1 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
with tab_rec:
    # Load all titles if needed
    if not st.session_state.all_titles:
        with st.spinner("Connecting to recommendation engine…"):
            try:
                st.session_state.all_titles = api.get_all_titles()
            except Exception as e:
                render_error(f"Cannot connect to backend: {e}")
                st.stop()

    titles = st.session_state.all_titles

    # Search / select row
    col_search, col_btn = st.columns([4, 1])

    with col_search:
        # Text search → filter dropdown
        search_query = st.text_input(
            "Search for a movie",
            placeholder="e.g. Inception, Nolan, Sci-Fi…",
            label_visibility="collapsed",
        )

        if search_query.strip():
            try:
                filtered = api.search_titles(search_query.strip(), limit=20)
            except Exception:
                filtered = [t for t in titles if search_query.lower() in t.lower()][:20]
        else:
            filtered = titles

        selected_title = st.selectbox(
            "Select movie",
            options=filtered if filtered else titles,
            label_visibility="collapsed",
        )

    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        get_btn = st.button("🔍 Get Recommendations", use_container_width=True)

    # Trigger recommendation
    if get_btn and selected_title:
        with st.spinner(f"Finding recommendations for **{selected_title}**…"):
            try:
                response = api.get_recommendations(
                    item_title=selected_title,
                    top_n=top_n,
                    category_filter=category_filter,
                    user_id=user_id,
                    mode=mode,
                )
                st.session_state.results = response
                # Fetch query item metadata
                try:
                    st.session_state.query_meta = api.get_item_metadata(selected_title)
                except Exception:
                    st.session_state.query_meta = {"title": selected_title}
            except Exception as e:
                render_error(str(e))

    # Display results
    if st.session_state.results:
        data = st.session_state.results
        recs = data.get("recommendations", [])

        # Query item banner
        if st.session_state.query_meta:
            render_query_card(st.session_state.query_meta)

        # Metrics row
        render_metric_row(
            total=data.get("total", len(recs)),
            mode=data.get("mode", mode),
            cached=data.get("cached", False),
        )
        st.markdown("")

        if not recs:
            st.info("No recommendations found. Try adjusting the genre filter or mode.")
        else:
            for item in recs:
                render_recommendation_card(item, rank=item.get("rank", 0))
    else:
        render_empty_state()

# ══════════════════════════════════════════════════════════════
#  TAB 2 — BROWSE CATALOG
# ══════════════════════════════════════════════════════════════
with tab_browse:
    st.markdown("### 📚 Full Movie Catalog")

    if not st.session_state.all_titles:
        st.info("Load the recommendation engine first (use the main tab).")
    else:
        browse_query = st.text_input("🔍 Search catalog", placeholder="Filter by title…")

        display_titles = (
            [t for t in st.session_state.all_titles if browse_query.lower() in t.lower()]
            if browse_query
            else st.session_state.all_titles
        )

        st.caption(f"Showing {len(display_titles)} of {len(st.session_state.all_titles)} items")

        # Display in a 3-column grid
        cols = st.columns(3)
        for i, title in enumerate(display_titles):
            with cols[i % 3]:
                if st.button(f"🎬 {title}", use_container_width=True, key=f"browse_{i}"):
                    # Switch to recommendations tab and pre-fill
                    st.session_state["selected_from_browse"] = title
                    st.info(f"Go to the **Recommendations** tab and search for: **{title}**")

# ══════════════════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(
        """
        ## 🎯 RecommendAI — Architecture Overview

        ### How it works

        ```
        User Input (Streamlit)
               ↓
        FastAPI Backend  ←→  Redis Cache
               ↓
        Recommendation Engine
          ├── Content-Based (TF-IDF + Cosine Similarity)
          ├── Collaborative (User-Item Matrix)
          └── Hybrid (α × content + (1-α) × collab)
               ↓
        Pre-processed Feature Matrix (sklearn)
               ↓
        MongoDB (optional persistence)
        ```

        ### Recommendation Modes

        | Mode | Description | Best For |
        |------|-------------|----------|
        | **Content-Based** | Uses item features (genre, director, cast) | New users / cold-start |
        | **Hybrid** | Blends content + collaborative signals | Returning users with history |

        ### Similarity Score

        Cosine similarity ranges from **0.0** (no relation) to **1.0** (identical).
        The threshold is configurable via `SIMILARITY_THRESHOLD` in `.env`.

        ### Tech Stack

        - **ML**: scikit-learn, pandas, numpy
        - **Backend**: FastAPI + Uvicorn
        - **Cache**: Redis (in-memory fallback)
        - **Frontend**: Streamlit
        - **Deployment**: Docker + Nginx

        ---
        Built with ❤️ as a production-grade recommendation system template.
        """
    )
