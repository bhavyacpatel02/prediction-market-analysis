"""Word-level mispricing analysis for Kalshi mention markets.

Drills into individual words/phrases (yes_sub_title) within mention markets to find
which specific words are systematically overpriced or underpriced. Builds on the
subcategory-level findings from mention_market_analysis.py by providing word-level
granularity for tradeable edges.

Key metrics per word:
- vol_weighted_price: contract-weighted average YES price (what the market believed)
- actual_yes_rate: market-weighted YES resolution rate * 100 (how often the word was mentioned)
- mispricing_pp: actual_yes_rate - vol_weighted_price (positive = underpriced YES)

Statistical approach: actual_yes_rate is computed per-market (each market = one independent
Bernoulli trial), while vol_weighted_price is contract-weighted (reflects dollar-weighted
consensus). Minimum threshold is n_markets >= 10 for statistical significance.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.kalshi.mention_queries import (
    query_word_by_price_bucket,
    query_word_by_speaker,
    query_word_mispricing,
)
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class MentionWordMispricingAnalysis(Analysis):
    """Analyze word-level mispricing in Kalshi mention markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="mention_word_mispricing",
            description="Word-level mispricing analysis for mention markets",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        with self.progress("Querying word-level mispricing"):
            word_df = self._safe_query(con, self._query_word_mispricing)
        with self.progress("Querying word × speaker mispricing"):
            speaker_df = self._safe_query(con, self._query_word_by_speaker)
        with self.progress("Querying word × price bucket mispricing"):
            bucket_df = self._safe_query(con, self._query_word_by_price_bucket)

        fig = self._create_figure(word_df, speaker_df, bucket_df)
        chart = self._create_chart(word_df)

        return AnalysisOutput(figure=fig, data=word_df, chart=chart)

    @staticmethod
    def _safe_query(con: duckdb.DuckDBPyConnection, query_fn) -> pd.DataFrame:
        """Run a query, returning empty DataFrame if columns are missing (e.g. test fixtures)."""
        try:
            return query_fn(con)
        except duckdb.BinderException:
            return pd.DataFrame()

    def _query_word_mispricing(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Core query: mispricing grouped by yes_sub_title (the word/phrase)."""
        return query_word_mispricing(con, self.trades_dir, self.markets_dir)

    def _query_word_by_speaker(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Mispricing grouped by (word, speaker) to reveal context dependence."""
        return query_word_by_speaker(con, self.trades_dir, self.markets_dir)

    def _query_word_by_price_bucket(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Mispricing by price bucket for top 20 words (by n_markets)."""
        return query_word_by_price_bucket(con, self.trades_dir, self.markets_dir)

    def _create_figure(
        self,
        word_df: pd.DataFrame,
        speaker_df: pd.DataFrame,
        bucket_df: pd.DataFrame,
    ) -> plt.Figure:
        """Create the 2x3 panel figure."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        self._plot_overpriced(axes[0, 0], word_df)
        self._plot_underpriced(axes[0, 1], word_df)
        self._plot_mispricing_vs_markets(axes[0, 2], word_df)
        self._plot_speaker_heatmap(axes[1, 0], speaker_df)
        self._plot_bucket_heatmap(axes[1, 1], bucket_df)
        self._plot_summary(axes[1, 2], word_df)

        fig.suptitle("Mention Market Word-Level Mispricing Analysis", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _plot_overpriced(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (0,0): Top 20 words where YES is overpriced (+EV NO bets)."""
        ax.set_title("Top Overpriced Words (+EV NO)")

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        overpriced = df[df["mispricing_pp"] < 0].nsmallest(20, "mispricing_pp")
        if overpriced.empty:
            ax.text(0.5, 0.5, "No overpriced words found", ha="center", va="center", transform=ax.transAxes)
            return

        y_pos = range(len(overpriced))
        ax.barh(list(y_pos), overpriced["mispricing_pp"].values, color="#e74c3c", alpha=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(overpriced["word"].values, fontsize=7)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Mispricing (pp)")
        ax.invert_yaxis()

    def _plot_underpriced(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (0,1): Top 20 words where YES is underpriced (+EV YES bets)."""
        ax.set_title("Top Underpriced Words (+EV YES)")

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        underpriced = df[df["mispricing_pp"] > 0].nlargest(20, "mispricing_pp")
        if underpriced.empty:
            ax.text(0.5, 0.5, "No underpriced words found", ha="center", va="center", transform=ax.transAxes)
            return

        y_pos = range(len(underpriced))
        ax.barh(list(y_pos), underpriced["mispricing_pp"].values, color="#2ecc71", alpha=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(underpriced["word"].values, fontsize=7)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Mispricing (pp)")
        ax.invert_yaxis()

    def _plot_mispricing_vs_markets(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (0,2): Scatter of mispricing vs n_markets, sized by total_contracts."""
        ax.set_title("Mispricing vs # Markets")

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        sizes = np.clip(np.log10(df["total_contracts"].values) * 15, 10, 200)
        colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in df["mispricing_pp"]]

        ax.scatter(
            df["n_markets"],
            df["mispricing_pp"],
            s=sizes,
            c=colors,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("# Independent Markets")
        ax.set_ylabel("Mispricing (pp)")
        ax.grid(True, alpha=0.3)

        # Label extreme points
        for _, row in df.nlargest(3, "mispricing_pp").iterrows():
            ax.annotate(
                row["word"],
                (row["n_markets"], row["mispricing_pp"]),
                fontsize=6,
                ha="center",
                va="bottom",
            )
        for _, row in df.nsmallest(3, "mispricing_pp").iterrows():
            ax.annotate(
                row["word"],
                (row["n_markets"], row["mispricing_pp"]),
                fontsize=6,
                ha="center",
                va="top",
            )

    def _plot_speaker_heatmap(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (1,0): Heatmap of mispricing by word × speaker."""
        ax.set_title("Word × Speaker Mispricing (pp)")

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        # Top 15 words by total contracts, top 8 speakers
        word_totals = df.groupby("word")["total_contracts"].sum().nlargest(15)
        speaker_totals = df.groupby("speaker")["total_contracts"].sum().nlargest(8)
        top_words = word_totals.index.tolist()
        top_speakers = speaker_totals.index.tolist()

        filtered = df[df["word"].isin(top_words) & df["speaker"].isin(top_speakers)]
        if filtered.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        pivot = filtered.pivot_table(index="word", columns="speaker", values="mispricing_pp", aggfunc="first")
        pivot = pivot.reindex(index=top_words, columns=top_speakers)

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-50, vmax=50)
        ax.set_xticks(range(len(top_speakers)))
        ax.set_xticklabels(top_speakers, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words, fontsize=7)
        plt.colorbar(im, ax=ax, label="Mispricing (pp)", shrink=0.8)

        # Annotate cells with values
        for i in range(len(top_words)):
            for j in range(len(top_speakers)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5, color="black")

    def _plot_bucket_heatmap(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (1,1): Heatmap of mispricing by word × price bucket."""
        ax.set_title("Word × Price Bucket Mispricing (pp)")

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        bucket_order = ["1-20", "21-40", "41-60", "61-80", "81-99"]
        word_totals = df.groupby("word")["total_contracts"].sum().nlargest(15)
        top_words = word_totals.index.tolist()

        filtered = df[df["word"].isin(top_words)]
        if filtered.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        pivot = filtered.pivot_table(index="word", columns="price_bucket", values="mispricing_pp", aggfunc="first")
        pivot = pivot.reindex(index=top_words, columns=bucket_order)

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-50, vmax=50)
        ax.set_xticks(range(len(bucket_order)))
        ax.set_xticklabels(bucket_order, fontsize=8)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words, fontsize=7)
        ax.set_xlabel("Price Bucket (cents)")
        plt.colorbar(im, ax=ax, label="Mispricing (pp)", shrink=0.8)

        for i in range(len(top_words)):
            for j in range(len(bucket_order)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5, color="black")

    def _plot_summary(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (1,2): Text summary of key stats and top actionable trades."""
        ax.set_title("Summary & Top Trades")
        ax.axis("off")

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        lines = []
        lines.append(f"Words analyzed: {len(df)}")
        lines.append(f"Avg mispricing: {df['mispricing_pp'].mean():+.1f} pp")
        n_over = (df["mispricing_pp"] < 0).sum()
        n_under = (df["mispricing_pp"] > 0).sum()
        lines.append(f"Overpriced (YES): {n_over} ({100 * n_over / len(df):.0f}%)")
        lines.append(f"Underpriced (YES): {n_under} ({100 * n_under / len(df):.0f}%)")
        lines.append("")
        lines.append("TOP 5 ACTIONABLE TRADES:")
        lines.append("-" * 40)

        # Best trades by absolute mispricing
        top_trades = df.reindex(df["mispricing_pp"].abs().nlargest(5).index)
        for _, row in top_trades.iterrows():
            direction = "YES" if row["mispricing_pp"] > 0 else "NO"
            edge = abs(row["mispricing_pp"])
            lines.append(f"  {row['word']}: {direction} ({edge:+.1f}pp edge)")
            lines.append(f"    {row['total_contracts']:,.0f} contracts, {row['n_markets']:.0f} markets")

        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
        )

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create bar chart config for top overpriced + underpriced words."""
        if df.empty:
            return ChartConfig(
                type=ChartType.BAR,
                data=[],
                xKey="word",
                yKeys=["mispricing_pp"],
                title="Word-Level Mention Market Mispricing",
            )

        overpriced = df[df["mispricing_pp"] < 0].nsmallest(10, "mispricing_pp")
        underpriced = df[df["mispricing_pp"] > 0].nlargest(10, "mispricing_pp")
        combined = pd.concat([underpriced, overpriced])

        chart_data = [
            {"word": row["word"], "mispricing_pp": round(float(row["mispricing_pp"]), 1)}
            for _, row in combined.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="word",
            yKeys=["mispricing_pp"],
            title="Word-Level Mention Market Mispricing (pp)",
            yUnit=UnitType.NUMBER,
            xLabel="Word/Phrase",
            yLabel="Mispricing (pp)",
        )
