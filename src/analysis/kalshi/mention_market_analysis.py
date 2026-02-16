"""Analyze mention market mispricing and calibration.

Mention markets (TRUMPMENTION, FEDMENTION, SWIFTMENTION, etc.) are a distinct
Kalshi category. This analysis compares their calibration to all other markets,
identifies mispricing by price level, checks YES vs NO EV asymmetry, and breaks
down mispricing by subcategory and broad group.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.kalshi.util.categories import CATEGORY_SQL, get_hierarchy
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType

# Mention subcategories: ticker prefix -> short label
MENTION_SUBCATEGORIES = {
    "TRUMPMENTION": "Trump",
    "TRUMPMENTIONB": "Trump B",
    "FEDMENTION": "Fed",
    "POWELLMENTION": "Powell",
    "SWIFTMENTION": "Swift",
    "KIMMELMENTION": "Kimmel",
    "SOUTHPARKMENTION": "South Park",
    "MRBEASTMENTION": "MrBeast",
    "SNFMENTION": "SNF",
    "TNFMENTION": "TNF",
    "SECPRESSMENTION": "Sec Press",
    "MAMDANIMENTION": "Mamdani",
    "EARNINGSMENTIONTSLA": "TSLA Earnings",
    "NYCMAYORDEBATEMENTION": "NYC Debate",
}

# Broad groups for mention subcategories
MENTION_BROAD_GROUPS = {
    "Trump": "Politics",
    "Trump B": "Politics",
    "Fed": "Finance",
    "Powell": "Finance",
    "Swift": "Entertainment",
    "Kimmel": "Entertainment",
    "South Park": "Entertainment",
    "MrBeast": "Entertainment",
    "SNF": "Entertainment",
    "TNF": "Entertainment",
    "Sec Press": "Politics",
    "Mamdani": "Politics",
    "TSLA Earnings": "Finance",
    "NYC Debate": "Politics",
}


def _classify_mention(event_ticker: str) -> tuple[str, str]:
    """Return (broad_group, label) for a mention market event ticker."""
    upper = event_ticker.upper() if event_ticker else ""
    for prefix, label in sorted(MENTION_SUBCATEGORIES.items(), key=lambda x: -len(x[0])):
        if prefix in upper:
            return MENTION_BROAD_GROUPS.get(label, "Other"), label
    # Fallback: use the categories hierarchy
    group, _, _ = get_hierarchy(event_ticker or "")
    return group, "Other Mention"


class MentionMarketAnalysis(Analysis):
    """Analyze mention market mispricing and calibration on Kalshi."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="mention_market_analysis",
            description="Mention market mispricing and calibration analysis",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        con = duckdb.connect()

        with self.progress("Querying calibration comparison"):
            cal_df = self._query_calibration(con)
        with self.progress("Querying mispricing by price"):
            mis_df = self._query_mispricing(con)
        with self.progress("Querying YES vs NO EV"):
            ev_df = self._query_ev_yes_vs_no(con)
        with self.progress("Querying subcategory mispricing"):
            sub_df = self._query_subcategory_mispricing(con)

        fig = self._create_figure(cal_df, mis_df, ev_df, sub_df)
        chart = self._create_chart(cal_df)
        combined_df = self._build_combined_df(cal_df, mis_df, ev_df, sub_df)

        return AnalysisOutput(figure=fig, data=combined_df, chart=chart)

    def _query_calibration(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Win rate by price for mention vs all other markets."""
        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, event_ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
            ),
            all_positions AS (
                SELECT
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won,
                    CASE WHEN UPPER(m.event_ticker) LIKE '%MENTION%' THEN 1 ELSE 0 END AS is_mention,
                    t.count AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker

                UNION ALL

                SELECT
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won,
                    CASE WHEN UPPER(m.event_ticker) LIKE '%MENTION%' THEN 1 ELSE 0 END AS is_mention,
                    t.count AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT
                price,
                SUM(CASE WHEN is_mention = 1 THEN contracts ELSE 0 END) AS mention_trades,
                CASE WHEN SUM(CASE WHEN is_mention = 1 THEN contracts ELSE 0 END) > 0
                    THEN 100.0 * SUM(CASE WHEN is_mention = 1 THEN won * contracts ELSE 0 END)
                        / SUM(CASE WHEN is_mention = 1 THEN contracts ELSE 0 END)
                    ELSE NULL END AS mention_win_rate,
                SUM(CASE WHEN is_mention = 0 THEN contracts ELSE 0 END) AS other_trades,
                CASE WHEN SUM(CASE WHEN is_mention = 0 THEN contracts ELSE 0 END) > 0
                    THEN 100.0 * SUM(CASE WHEN is_mention = 0 THEN won * contracts ELSE 0 END)
                        / SUM(CASE WHEN is_mention = 0 THEN contracts ELSE 0 END)
                    ELSE NULL END AS other_win_rate
            FROM all_positions
            WHERE price BETWEEN 1 AND 99
            GROUP BY price
            ORDER BY price
            """
        ).df()
        return df

    def _query_mispricing(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Taker/maker/combined mispricing by price for mention markets only."""
        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, event_ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
                  AND UPPER(event_ticker) LIKE '%MENTION%'
            ),
            taker_positions AS (
                SELECT
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            maker_positions AS (
                SELECT
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            ),
            taker_stats AS (
                SELECT price, COUNT(*) AS trades, 100.0 * SUM(won) / COUNT(*) AS win_rate
                FROM taker_positions GROUP BY price
            ),
            maker_stats AS (
                SELECT price, COUNT(*) AS trades, 100.0 * SUM(won) / COUNT(*) AS win_rate
                FROM maker_positions GROUP BY price
            ),
            combined_stats AS (
                SELECT price, COUNT(*) AS trades, 100.0 * SUM(won) / COUNT(*) AS win_rate
                FROM (SELECT * FROM taker_positions UNION ALL SELECT * FROM maker_positions)
                GROUP BY price
            )
            SELECT
                COALESCE(t.price, m.price, c.price) AS price,
                t.win_rate AS taker_win_rate,
                m.win_rate AS maker_win_rate,
                c.win_rate AS combined_win_rate,
                t.trades AS taker_trades,
                m.trades AS maker_trades,
                c.trades AS combined_trades
            FROM taker_stats t
            FULL OUTER JOIN maker_stats m ON t.price = m.price
            FULL OUTER JOIN combined_stats c ON COALESCE(t.price, m.price) = c.price
            WHERE COALESCE(t.price, m.price, c.price) BETWEEN 1 AND 99
            ORDER BY COALESCE(t.price, m.price, c.price)
            """
        ).df()

        if not df.empty:
            df["taker_mispricing_pp"] = df["taker_win_rate"] - df["price"].astype(float)
            df["maker_mispricing_pp"] = df["maker_win_rate"] - df["price"].astype(float)
            df["combined_mispricing_pp"] = df["combined_win_rate"] - df["price"].astype(float)

        return df

    def _query_ev_yes_vs_no(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """YES and NO EV at each price for mention markets."""
        yes_df = con.execute(
            f"""
            SELECT
                t.yes_price AS price,
                SUM(CASE WHEN m.result = 'yes' THEN t.count ELSE 0 END) * 1.0
                    / SUM(t.count) AS win_rate,
                SUM(t.count) AS total_contracts
            FROM '{self.trades_dir}/*.parquet' t
            INNER JOIN '{self.markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
              AND t.yes_price BETWEEN 1 AND 99
              AND UPPER(m.event_ticker) LIKE '%MENTION%'
            GROUP BY t.yes_price
            ORDER BY t.yes_price
            """
        ).df()

        no_df = con.execute(
            f"""
            SELECT
                t.no_price AS price,
                SUM(CASE WHEN m.result = 'no' THEN t.count ELSE 0 END) * 1.0
                    / SUM(t.count) AS win_rate,
                SUM(t.count) AS total_contracts
            FROM '{self.trades_dir}/*.parquet' t
            INNER JOIN '{self.markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE m.result IN ('yes', 'no')
              AND t.no_price BETWEEN 1 AND 99
              AND UPPER(m.event_ticker) LIKE '%MENTION%'
            GROUP BY t.no_price
            ORDER BY t.no_price
            """
        ).df()

        # Build combined EV dataframe
        combined = pd.DataFrame({"price": range(1, 100)})

        if not yes_df.empty:
            yes_df["ev"] = 100 * yes_df["win_rate"] - yes_df["price"]
            combined = combined.merge(
                yes_df[["price", "ev"]].rename(columns={"ev": "yes_ev"}),
                on="price",
                how="left",
            )
        else:
            combined["yes_ev"] = np.nan

        if not no_df.empty:
            no_df["ev"] = 100 * no_df["win_rate"] - no_df["price"]
            combined = combined.merge(
                no_df[["price", "ev"]].rename(columns={"ev": "no_ev"}),
                on="price",
                how="left",
            )
        else:
            combined["no_ev"] = np.nan

        combined["best_bet"] = np.where(
            combined["yes_ev"].fillna(-100) > combined["no_ev"].fillna(-100),
            "YES",
            "NO",
        )

        return combined

    def _query_subcategory_mispricing(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Taker excess returns grouped by mention subcategory."""
        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, event_ticker, result
                FROM '{self.markets_dir}/*.parquet'
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
                  AND UPPER(event_ticker) LIKE '%MENTION%'
            ),
            taker_positions AS (
                SELECT
                    {CATEGORY_SQL.replace("event_ticker", "m.event_ticker")} AS category,
                    m.event_ticker,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
                    t.count AS contracts
                FROM '{self.trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT
                category,
                SUM(contracts) AS total_contracts,
                AVG(won - price / 100.0) * 100 AS excess_return_pp,
                AVG(won) * 100 - AVG(price) AS mispricing_pp
            FROM taker_positions
            GROUP BY category
            ORDER BY SUM(contracts) DESC
            """
        ).df()

        if not df.empty:
            classifications = df["category"].apply(_classify_mention)
            df["broad_group"] = classifications.apply(lambda x: x[0])
            df["label"] = classifications.apply(lambda x: x[1])

        return df

    def _create_figure(
        self,
        cal_df: pd.DataFrame,
        mis_df: pd.DataFrame,
        ev_df: pd.DataFrame,
        sub_df: pd.DataFrame,
    ) -> plt.Figure:
        """Create the 2x3 panel figure."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        self._plot_calibration(axes[0, 0], cal_df)
        self._plot_mispricing(axes[0, 1], mis_df)
        self._plot_ev_yes_no(axes[0, 2], ev_df)
        self._plot_subcategory(axes[1, 0], sub_df)
        self._plot_broad_group(axes[1, 1], sub_df)
        self._plot_deviation_distribution(axes[1, 2], cal_df)

        fig.suptitle("Mention Market Mispricing & Calibration Analysis", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _plot_calibration(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (0,0): Calibration comparison — mention vs other vs perfect."""
        ax.plot([0, 100], [0, 100], "--", color="gray", linewidth=1.5, label="Perfect calibration")

        has_other = not df.empty and df["other_win_rate"].notna().any()
        has_mention = not df.empty and df["mention_win_rate"].notna().any()

        if has_other:
            other = df[df["other_win_rate"].notna()]
            ax.scatter(other["price"], other["other_win_rate"], s=20, alpha=0.5, color="gray", label="Other markets")

        if has_mention:
            mention = df[df["mention_win_rate"].notna()]
            ax.scatter(
                mention["price"], mention["mention_win_rate"], s=30, alpha=0.8, color="#e74c3c", label="Mention markets"
            )

        ax.set_xlabel("Contract Price (cents)")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Calibration: Mention vs Other")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_aspect("equal")

    def _plot_mispricing(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (0,1): Mispricing by price for taker/maker/combined."""
        ax.axhline(y=0, linestyle="--", color="gray", linewidth=1)

        if not df.empty and "taker_mispricing_pp" in df.columns:
            ax.scatter(df["price"], df["taker_mispricing_pp"], s=20, alpha=0.7, color="#e74c3c", label="Taker")
            ax.scatter(df["price"], df["maker_mispricing_pp"], s=20, alpha=0.7, color="#2ecc71", label="Maker")
            ax.scatter(df["price"], df["combined_mispricing_pp"], s=20, alpha=0.7, color="#4C72B0", label="Combined")

        ax.set_xlabel("Contract Price (cents)")
        ax.set_ylabel("Mispricing (pp)")
        ax.set_title("Mention Mispricing by Price")
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=8)

    def _plot_ev_yes_no(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (0,2): YES vs NO EV with fill_between."""
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)

        has_yes = df["yes_ev"].notna().any()
        has_no = df["no_ev"].notna().any()

        if has_yes:
            yes_data = df.dropna(subset=["yes_ev"])
            ax.plot(yes_data["price"], yes_data["yes_ev"], color="#2ecc71", linewidth=2, label="YES EV")
            ax.fill_between(yes_data["price"], yes_data["yes_ev"], 0, alpha=0.2, color="#2ecc71")

        if has_no:
            no_data = df.dropna(subset=["no_ev"])
            ax.plot(no_data["price"], no_data["no_ev"], color="#e74c3c", linewidth=2, label="NO EV")
            ax.fill_between(no_data["price"], no_data["no_ev"], 0, alpha=0.2, color="#e74c3c")

        ax.set_xlabel("Purchase Price (cents)")
        ax.set_ylabel("Expected Value (cents)")
        ax.set_title("Mention Markets: YES vs NO EV")
        ax.set_xlim(1, 99)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_subcategory(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (1,0): Horizontal bar chart of subcategory excess returns."""
        if df.empty:
            ax.set_title("Subcategory Excess Returns")
            ax.text(0.5, 0.5, "No mention data", ha="center", va="center", transform=ax.transAxes)
            return

        top = df.nlargest(15, "total_contracts")
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top["excess_return_pp"]]
        y_pos = range(len(top))

        ax.barh(list(y_pos), top["excess_return_pp"].values, color=colors, alpha=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top["label"].values, fontsize=8)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Taker Excess Return (pp)")
        ax.set_title("Subcategory Excess Returns")
        ax.invert_yaxis()

    def _plot_broad_group(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (1,1): Broad group (Politics/Finance/Entertainment) mispricing."""
        if df.empty or "broad_group" not in df.columns:
            ax.set_title("Broad Group Mispricing")
            ax.text(0.5, 0.5, "No mention data", ha="center", va="center", transform=ax.transAxes)
            return

        group_stats = []
        for group in df["broad_group"].unique():
            gdata = df[df["broad_group"] == group]
            total = gdata["total_contracts"].sum()
            if total > 0:
                weighted = (gdata["excess_return_pp"] * gdata["total_contracts"]).sum() / total
                group_stats.append({"group": group, "excess_return_pp": weighted, "total_contracts": total})

        if not group_stats:
            ax.set_title("Broad Group Mispricing")
            ax.text(0.5, 0.5, "No mention data", ha="center", va="center", transform=ax.transAxes)
            return

        gdf = pd.DataFrame(group_stats).sort_values("total_contracts", ascending=False)
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in gdf["excess_return_pp"]]

        ax.bar(range(len(gdf)), gdf["excess_return_pp"].values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(gdf)))
        ax.set_xticklabels(gdf["group"].values, rotation=45, ha="right", fontsize=9)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Vol-Weighted Excess Return (pp)")
        ax.set_title("Broad Group Mispricing")

    def _plot_deviation_distribution(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Panel (1,2): Histogram of calibration deviation — mention vs other."""
        has_mention = not df.empty and df["mention_win_rate"].notna().any()
        has_other = not df.empty and df["other_win_rate"].notna().any()

        if has_mention:
            mention = df[df["mention_win_rate"].notna()]
            mention_dev = mention["mention_win_rate"] - mention["price"].astype(float)
            ax.hist(mention_dev, bins=20, alpha=0.6, color="#e74c3c", label="Mention", density=True)

        if has_other:
            other = df[df["other_win_rate"].notna()]
            other_dev = other["other_win_rate"] - other["price"].astype(float)
            ax.hist(other_dev, bins=20, alpha=0.4, color="gray", label="Other", density=True)

        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Calibration Deviation (pp)")
        ax.set_ylabel("Density")
        ax.set_title("Calibration Deviation Distribution")
        ax.legend(fontsize=8)

    def _create_chart(self, cal_df: pd.DataFrame) -> ChartConfig:
        """Create chart config for the calibration comparison."""
        chart_data = []
        for _, row in cal_df.iterrows():
            price = int(row["price"])
            entry = {"price": price, "perfect": price}
            if pd.notna(row.get("mention_win_rate")):
                entry["mention"] = round(float(row["mention_win_rate"]), 2)
            if pd.notna(row.get("other_win_rate")):
                entry["other"] = round(float(row["other_win_rate"]), 2)
            chart_data.append(entry)

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="price",
            yKeys=["mention", "other", "perfect"],
            title="Calibration: Mention vs Other Markets",
            strokeDasharrays=[None, None, "5 5"],
            yUnit=UnitType.PERCENT,
            xLabel="Contract Price (cents)",
            yLabel="Win Rate (%)",
            colors={"mention": "#e74c3c", "other": "#999999", "perfect": "#333333"},
        )

    def _build_combined_df(
        self,
        cal_df: pd.DataFrame,
        mis_df: pd.DataFrame,
        ev_df: pd.DataFrame,
        sub_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build combined output dataframe with section labels."""
        sections = []

        if not cal_df.empty:
            cal_out = cal_df.copy()
            cal_out["section"] = "calibration"
            sections.append(cal_out)

        if not mis_df.empty:
            mis_out = mis_df.copy()
            mis_out["section"] = "mispricing"
            sections.append(mis_out)

        if not ev_df.empty:
            ev_out = ev_df.copy()
            ev_out["section"] = "ev_yes_vs_no"
            sections.append(ev_out)

        if not sub_df.empty:
            sub_out = sub_df.copy()
            sub_out["section"] = "subcategory"
            sections.append(sub_out)

        # Summary section
        summary_rows = []
        if not cal_df.empty:
            mention_cal = cal_df[cal_df["mention_win_rate"].notna()]
            other_cal = cal_df[cal_df["other_win_rate"].notna()]

            if not mention_cal.empty:
                mention_mae = (mention_cal["mention_win_rate"] - mention_cal["price"].astype(float)).abs().mean()
                summary_rows.append({"metric": "mention_mean_abs_cal_error", "value": round(mention_mae, 2)})
                summary_rows.append({"metric": "mention_total_trades", "value": int(cal_df["mention_trades"].sum())})

            if not other_cal.empty:
                other_mae = (other_cal["other_win_rate"] - other_cal["price"].astype(float)).abs().mean()
                summary_rows.append({"metric": "other_mean_abs_cal_error", "value": round(other_mae, 2)})

        if not ev_df.empty:
            avg_yes = ev_df["yes_ev"].mean()
            avg_no = ev_df["no_ev"].mean()
            if pd.notna(avg_yes):
                summary_rows.append({"metric": "avg_yes_ev", "value": round(avg_yes, 2)})
            if pd.notna(avg_no):
                summary_rows.append({"metric": "avg_no_ev", "value": round(avg_no, 2)})

        if not sub_df.empty:
            best = sub_df.loc[sub_df["excess_return_pp"].idxmax()]
            worst = sub_df.loc[sub_df["excess_return_pp"].idxmin()]
            summary_rows.append({"metric": "best_subcategory", "value": best["label"]})
            summary_rows.append({"metric": "worst_subcategory", "value": worst["label"]})

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df["section"] = "summary"
            sections.append(summary_df)

        if sections:
            return pd.concat(sections, ignore_index=True)
        return pd.DataFrame(columns=["section"])
