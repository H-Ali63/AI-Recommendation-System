"""
frontend/components.py
───────────────────────
Reusable Streamlit UI components for the recommendation app.
"""
from __future__ import annotations

import streamlit as st


def render_recommendation_card(item: dict, rank: int) -> None:
    """Render a single recommendation as a styled card."""
    score = item.get("similarity_score", 0)
    score_pct = int(score * 100)
    score_color = _score_color(score)

    with st.container():
        col_poster, col_info = st.columns([1, 4])

        with col_poster:
            poster = item.get("poster_url") or ""
            if poster and poster.startswith("http"):
                st.image(poster, width=100)
            else:
                st.markdown(
                    f"""<div style='
                        width:80px;height:120px;background:#1e293b;
                        border-radius:8px;display:flex;align-items:center;
                        justify-content:center;color:#64748b;font-size:24px;
                    '>🎬</div>""",
                    unsafe_allow_html=True,
                )

        with col_info:
            st.markdown(
                f"### {rank}. {item.get('title', 'Unknown')}",
            )
            genres = item.get("genres", "")
            year = item.get("year", "")
            director = item.get("director", "")

            meta_parts = []
            if year:
                meta_parts.append(f"{int(year)}")
            if genres:
                meta_parts.append(f"{genres.replace('|', ' · ')}")
            if director:
                meta_parts.append(f"{director}")

            if meta_parts:
                st.caption("  |  ".join(meta_parts))

            # Rating + similarity row
            rating = item.get("rating")
            col_r, col_s, col_v = st.columns(3)
            with col_r:
                if rating:
                    st.metric("IMDb Rating", f"{rating:.1f}")
            with col_s:
                st.markdown(
                    f"""<div style='padding:4px 0'>
                        <small style='color:#94a3b8'>Similarity</small><br>
                        <span style='font-size:20px;font-weight:700;color:{score_color}'>{score_pct}%</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col_v:
                votes = item.get("votes")
                if votes:
                    st.metric("Votes", f"{int(votes):,}")

            # Similarity progress bar
            st.progress(score, text="")

        st.divider()


def render_query_card(item: dict) -> None:
    """Render the query item at the top of results."""
    st.markdown(
        f"""
        <div style='
            background: linear-gradient(135deg, #1e3a5f 0%, #0f2027 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        '>
            <div style='color:#94a3b8;font-size:12px;text-transform:uppercase;
                        letter-spacing:2px;margin-bottom:8px'>Showing results for</div>
            <div style='font-size:28px;font-weight:800;color:#f1f5f9'>
                🎬 {item.get('title','?')}
            </div>
            <div style='color:#64748b;margin-top:8px'>
                {item.get('genres','').replace('|',' · ')} &nbsp;·&nbsp;
                {int(item.get('year',0)) if item.get('year') else ''} &nbsp;·&nbsp;
                {item.get('rating','')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(total: int, mode: str, cached: bool) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Results", total)
    c2.metric("Mode", mode.title())
    c3.metric("Source", "⚡ Cache" if cached else "Fresh")


def render_empty_state() -> None:
    st.markdown(
        """
        <div style='text-align:center;padding:60px 20px;color:#475569'>
            <div style='font-size:64px'></div>
            <h3 style='color:#94a3b8'>No recommendations yet</h3>
            <p>Select a movie above and click <strong>Get Recommendations</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error(message: str) -> None:
    st.error(f"{message}", icon="")


def render_sidebar_stats(health: dict) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    if health.get("engine_ready"):
        st.sidebar.success(f"Engine ready · {health.get('items_loaded', 0)} items")
    else:
        st.sidebar.error("Engine not ready")
    st.sidebar.caption(f"v{health.get('version','?')} · {health.get('status','?')}")


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "#22c55e"   # green
    if score >= 0.4:
        return "#f59e0b"   # amber
    return "#ef4444"       # red
