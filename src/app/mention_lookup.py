"""Streamlit app for interactive mention market mispricing lookup.

Run with: make mention-app
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when run via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import duckdb
import streamlit as st

from src.analysis.kalshi.mention_queries import query_word_by_speaker, query_word_mispricing

BASE_DIR = Path(__file__).parent.parent.parent
TRADES_DIR = BASE_DIR / "data" / "kalshi" / "trades"
MARKETS_DIR = BASE_DIR / "data" / "kalshi" / "markets"


@st.cache_data(ttl=3600)
def load_word_data() -> dict:
    """Load all word mispricing data (cached for 1 hour)."""
    con = duckdb.connect()
    try:
        word_df = query_word_mispricing(con, TRADES_DIR, MARKETS_DIR, min_markets=1)
    except duckdb.BinderException:
        word_df = None
    try:
        speaker_df = query_word_by_speaker(con, TRADES_DIR, MARKETS_DIR, min_markets=1)
    except duckdb.BinderException:
        speaker_df = None
    return {"words": word_df, "speakers": speaker_df}


def main():
    st.set_page_config(page_title="Mention Mispricing Lookup", page_icon="📊", layout="wide")
    st.title("Mention Market Mispricing Lookup")

    # Sidebar controls
    with st.sidebar:
        st.header("Search")
        search_term = st.text_input("Word or phrase", placeholder="e.g. Tariff, China, Risk")
        min_markets = st.slider("Min markets threshold", min_value=1, max_value=50, value=10)
        show_low_sample = st.checkbox("Include low-sample words in browse table", value=False)

    # Load data
    data = load_word_data()
    word_df = data["words"]
    speaker_df = data["speakers"]

    if word_df is None or word_df.empty:
        st.error("No mention market data found. Run `make setup` to download data first.")
        return

    # Search result section
    if search_term:
        term_lower = search_term.strip().lower()

        # Exact match first, then partial
        exact = word_df[word_df["word"].str.lower() == term_lower]
        if not exact.empty:
            match_df = exact
        else:
            match_df = word_df[word_df["word"].str.lower().str.contains(term_lower, na=False)]

        if match_df.empty:
            st.warning(f"No results for '{search_term}'.")
        else:
            # Show first exact/best match in detail
            row = match_df.iloc[0]

            if row["n_markets"] < 10:
                st.warning(f"Low sample size: only {int(row['n_markets'])} markets. Statistical confidence is limited.")

            st.subheader(f"**{row['word']}**")

            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Edge (pp)", f"{row['mispricing_pp']:+.1f}")
            col2.metric("Actual YES Rate", f"{row['actual_yes_rate']:.1f}%")
            col3.metric("Market Price", f"{row['vol_weighted_price']:.1f}¢")
            col4.metric("Markets / Contracts", f"{int(row['n_markets'])} / {int(row['total_contracts']):,}")

            # Trade signal
            edge = row["mispricing_pp"]
            if edge > 0:
                st.success(
                    f"**BUY YES** — Market prices this at {row['vol_weighted_price']:.1f}¢ "
                    f"but it resolves YES {row['actual_yes_rate']:.1f}% of the time. "
                    f"Edge: {edge:+.1f}pp."
                )
            else:
                st.error(
                    f"**BUY NO** — Market prices this at {row['vol_weighted_price']:.1f}¢ "
                    f"but it resolves YES only {row['actual_yes_rate']:.1f}% of the time. "
                    f"Edge: {edge:+.1f}pp."
                )

            # Speaker breakdown
            if speaker_df is not None and not speaker_df.empty:
                word_speakers = speaker_df[speaker_df["word"].str.lower() == row["word"].lower()]
                if not word_speakers.empty:
                    st.subheader("Speaker Breakdown")
                    display_cols = [
                        "speaker",
                        "n_markets",
                        "total_contracts",
                        "vol_weighted_price",
                        "actual_yes_rate",
                        "mispricing_pp",
                    ]
                    available_cols = [c for c in display_cols if c in word_speakers.columns]
                    st.dataframe(
                        word_speakers[available_cols].sort_values("mispricing_pp", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

            # Show other partial matches if any
            if len(match_df) > 1:
                st.subheader("Other matches")
                st.dataframe(
                    match_df.iloc[1:][
                        [
                            "word",
                            "n_markets",
                            "total_contracts",
                            "vol_weighted_price",
                            "actual_yes_rate",
                            "mispricing_pp",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

    # Full browseable table
    st.subheader("All Words")
    threshold = 1 if show_low_sample else min_markets
    browse_df = word_df[word_df["n_markets"] >= threshold].copy()
    browse_df = browse_df.sort_values("mispricing_pp", ascending=False)

    st.dataframe(
        browse_df[["word", "n_markets", "total_contracts", "vol_weighted_price", "actual_yes_rate", "mispricing_pp"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing {len(browse_df)} words with >= {threshold} markets")


if __name__ == "__main__":
    main()
