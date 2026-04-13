"""
streamlit_app.py  ← Streamlit Cloud entry point (root of repo)
───────────────────────────────────────────────────────────────
Self-contained app: embeds the ML engine directly.
No FastAPI process needed — works on Streamlit Cloud.

For local/Docker deployment with FastAPI, use frontend/app.py instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RecommendAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0f1923 !important; border-right: 1px solid #1e293b; }
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 10px 24px !important;
}
[data-testid="stMetric"] {
    background: #1e293b; border-radius: 8px;
    padding: 12px; border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  ML ENGINE (embedded — no FastAPI needed)
# ══════════════════════════════════════════════════════════════

DATA_PATH = ROOT / "data" / "raw" / "movies.csv"

@st.cache_resource(show_spinner="🔄 Building recommendation engine…")
def load_engine():
    """Load data and build cosine similarity matrix. Cached across sessions."""
    df = pd.read_csv(DATA_PATH)

    # Fill missing values
    for col in ["genres", "director", "cast"]:
        df[col] = df[col].fillna("").astype(str)
    for col in ["rating", "votes", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    # Feature engineering: text soup
    def expand(text):
        return " ".join(p.strip().replace(" ", "_") for p in text.split("|"))

    df["_soup"] = (
        df["genres"].apply(expand) + " " +
        df["director"].str.lower() + " " +
        df["cast"].apply(expand)
    )

    # TF-IDF on text features
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    text_matrix = tfidf.fit_transform(df["_soup"]).toarray() * 0.7

    # Scaled numeric features
    scaler = MinMaxScaler()
    num_matrix = scaler.fit_transform(df[["rating", "votes", "year"]].values) * 0.3

    feature_matrix = np.hstack([text_matrix, num_matrix])
    sim_matrix = cosine_similarity(feature_matrix, feature_matrix)

    return df.reset_index(drop=True), sim_matrix


def get_recommendations(df, sim_matrix, title, top_n=10, genre_filter=None):
    """Return top-N recommendations for a given title."""
    lower = df["title"].str.lower()
    mask = lower == title.lower()
    if not mask.any():
        return []
    idx = int(mask.idxmax())

    scores = sim_matrix[idx].copy()
    result_df = df.copy()
    result_df["_score"] = scores
    result_df = result_df[result_df.index != idx]
    result_df = result_df[result_df["_score"] >= 0.05]

    if genre_filter and genre_filter != "All":
        result_df = result_df[result_df["genres"].str.contains(genre_filter, case=False, na=False)]

    result_df = result_df.sort_values("_score", ascending=False).head(top_n)
    return result_df.to_dict(orient="records")


def get_categories(df):
    cats = set()
    for g in df["genres"].dropna():
        for c in g.split("|"):
            cats.add(c.strip())
    return sorted(cats)


# ══════════════════════════════════════════════════════════════
#  LOAD ENGINE
# ══════════════════════════════════════════════════════════════
try:
    df, sim_matrix = load_engine()
    engine_ok = True
except Exception as e:
    st.error(f"❌ Failed to load engine: {e}")
    engine_ok = False
    st.stop()

titles = df["title"].tolist()
categories = get_categories(df)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
        <div style='font-size:26px;font-weight:800;color:#f1f5f9'>🎯 RecommendAI</div>
        <div style='color:#64748b;font-size:13px'>Cosine Similarity Engine</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Parameters")
    top_n = st.slider("Number of recommendations", 3, 20, 8)
    genre_filter = st.selectbox("Filter by genre", ["All"] + categories)

    st.divider()

    # Similarity calculator
    st.markdown("### 🔬 Similarity Calculator")
    with st.expander("Compare two movies"):
        sim_a = st.selectbox("Movie A", titles, key="sa")
        sim_b = st.selectbox("Movie B", titles, index=1, key="sb")
        if st.button("Calculate", key="calc"):
            lower = df["title"].str.lower()
            ia = int((lower == sim_a.lower()).idxmax())
            ib = int((lower == sim_b.lower()).idxmax())
            score = float(sim_matrix[ia, ib])
            st.metric("Cosine Similarity", f"{score:.4f}")
            st.progress(score)

    st.divider()
    st.success(f"✅ Engine ready · {len(titles)} movies loaded")

# ══════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#1e3a5f,#0f2027);
            border:1px solid #1e3a5f;border-radius:16px;
            padding:28px 36px;margin-bottom:28px'>
    <h1 style='font-size:32px;font-weight:800;color:#f1f5f9;margin:0 0 8px'>
        🎯 RecommendAI
    </h1>
    <p style='color:#94a3b8;font-size:15px;margin:0'>
        Personalized movie recommendations powered by cosine similarity.
        Select a movie and discover what to watch next.
    </p>
</div>
""", unsafe_allow_html=True)

tab_rec, tab_browse, tab_about = st.tabs(["🎯 Recommendations", "📚 Browse", "ℹ️ About"])

# ── Tab 1: Recommendations ────────────────────────────────────
with tab_rec:
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        search = st.text_input("Search for a movie", placeholder="e.g. Inception, Nolan, Sci-Fi…",
                                label_visibility="collapsed")
        filtered = [t for t in titles if search.lower() in t.lower()] if search.strip() else titles
        selected = st.selectbox("Select movie", filtered if filtered else titles,
                                label_visibility="collapsed")
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        go = st.button("🔍 Get Recommendations", use_container_width=True)

    if go and selected:
        with st.spinner(f"Finding recommendations for **{selected}**…"):
            results = get_recommendations(
                df, sim_matrix, selected, top_n=top_n,
                genre_filter=None if genre_filter == "All" else genre_filter
            )

        if not results:
            st.info("No results found. Try adjusting the genre filter.")
        else:
            # Query item banner
            q_row = df[df["title"].str.lower() == selected.lower()].iloc[0]
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1e3a5f,#0f2027);
                        border:1px solid #334155;border-radius:12px;
                        padding:20px;margin-bottom:20px'>
                <div style='color:#94a3b8;font-size:12px;letter-spacing:2px;
                            text-transform:uppercase;margin-bottom:6px'>Showing results for</div>
                <div style='font-size:26px;font-weight:800;color:#f1f5f9'>
                    🎬 {q_row['title']}
                </div>
                <div style='color:#64748b;margin-top:6px'>
                    {str(q_row.get('genres','')).replace('|',' · ')} &nbsp;·&nbsp;
                    {int(q_row['year']) if pd.notna(q_row.get('year')) else ''} &nbsp;·&nbsp;
                    ⭐ {q_row.get('rating','')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            c1, c2, c3 = st.columns(3)
            c1.metric("Results found", len(results))
            c2.metric("Mode", "Content-based")
            genre_label = genre_filter if genre_filter != "All" else "All genres"
            c3.metric("Genre filter", genre_label)
            st.markdown("")

            # Recommendation cards
            for item in results:
                score = float(item.get("_score", 0))
                score_pct = int(score * 100)
                score_color = "#22c55e" if score >= 0.7 else "#f59e0b" if score >= 0.4 else "#ef4444"

                col_poster, col_info = st.columns([1, 4])
                with col_poster:
                    poster = str(item.get("poster_url", ""))
                    if poster.startswith("http"):
                        st.image(poster, width=100)
                    else:
                        st.markdown("""<div style='width:80px;height:120px;background:#1e293b;
                            border-radius:8px;display:flex;align-items:center;
                            justify-content:center;font-size:28px'>🎬</div>""",
                            unsafe_allow_html=True)

                with col_info:
                    rank = results.index(item) + 1
                    st.markdown(f"### {rank}. {item.get('title','?')}")
                    parts = []
                    if item.get("year"): parts.append(f"📅 {int(item['year'])}")
                    if item.get("genres"): parts.append(f"🎭 {str(item['genres']).replace('|',' · ')}")
                    if item.get("director"): parts.append(f"🎬 {item['director']}")
                    if parts: st.caption("  |  ".join(parts))

                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        if item.get("rating"): st.metric("IMDb", f"⭐ {float(item['rating']):.1f}")
                    with mc2:
                        st.markdown(f"""<div style='padding:4px 0'>
                            <small style='color:#94a3b8'>Similarity</small><br>
                            <span style='font-size:20px;font-weight:700;color:{score_color}'>{score_pct}%</span>
                        </div>""", unsafe_allow_html=True)
                    with mc3:
                        if item.get("votes"): st.metric("Votes", f"{int(item['votes']):,}")

                    st.progress(score)
                st.divider()
    elif not go:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#475569'>
            <div style='font-size:64px'>🎯</div>
            <h3 style='color:#94a3b8'>No recommendations yet</h3>
            <p>Select a movie above and click <strong>Get Recommendations</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: Browse ─────────────────────────────────────────────
with tab_browse:
    st.markdown("### 📚 Full Movie Catalog")
    bq = st.text_input("🔍 Filter catalog", placeholder="Search titles…")
    display = [t for t in titles if bq.lower() in t.lower()] if bq else titles
    st.caption(f"Showing {len(display)} of {len(titles)} movies")
    cols = st.columns(3)
    for i, t in enumerate(display):
        with cols[i % 3]:
            st.markdown(f"🎬 {t}")

# ── Tab 3: About ──────────────────────────────────────────────
with tab_about:
    st.markdown("""
    ## 🎯 RecommendAI

    ### How recommendations work

    ```
    1. Load movies.csv
    2. Build feature vector per movie:
       • TF-IDF on genres + director + cast  (weight: 70%)
       • MinMax scaled rating, votes, year    (weight: 30%)
    3. Compute N×N cosine similarity matrix
    4. For query movie i:
       scores = sim_matrix[i]
       return top-N by score (excluding self)
    ```

    ### Cosine Similarity Formula

    ```
    cos(θ) = (A · B) / (‖A‖ × ‖B‖)
    ```

    Ranges from **0.0** (unrelated) to **1.0** (identical feature vectors).

    ### Tech Stack
    - **ML**: scikit-learn, pandas, numpy
    - **UI**: Streamlit
    - **Full stack**: FastAPI + Redis + MongoDB + Docker (local/VPS)
    """)