"""Shared DuckDB queries for Kalshi mention market mispricing analysis.

These functions are used by both the static analysis (MentionWordMispricingAnalysis)
and the interactive Streamlit app (mention_lookup.py).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def query_word_mispricing(
    con: duckdb.DuckDBPyConnection,
    trades_dir: Path,
    markets_dir: Path,
    min_markets: int = 10,
) -> pd.DataFrame:
    """Word-level mispricing: actual YES rate vs volume-weighted market price.

    actual_yes_rate is market-weighted (each market = one independent trial).
    vol_weighted_price is contract-weighted (reflects dollar-weighted consensus).
    """
    return con.execute(
        f"""
        WITH mention_markets AS (
            SELECT ticker, event_ticker, yes_sub_title, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
              AND UPPER(event_ticker) LIKE '%MENTION%'
              AND UPPER(event_ticker) NOT LIKE '%MENTIONSSINGLE%'
              AND yes_sub_title IS NOT NULL
              AND yes_sub_title != ''
        ),
        market_outcomes AS (
            SELECT
                LOWER(m.yes_sub_title) AS word_key,
                m.yes_sub_title,
                m.ticker,
                m.result,
                CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS resolved_yes
            FROM mention_markets m
        ),
        market_yes_rates AS (
            SELECT
                word_key,
                mode(yes_sub_title) AS display_word,
                COUNT(*) AS n_markets,
                AVG(resolved_yes) * 100.0 AS actual_yes_rate
            FROM market_outcomes
            GROUP BY word_key
            HAVING COUNT(*) >= {min_markets}
        )
        SELECT
            r.display_word AS word,
            r.n_markets,
            SUM(t.count) AS total_contracts,
            SUM(t.yes_price * t.count) * 1.0 / SUM(t.count) AS vol_weighted_price,
            r.actual_yes_rate,
            r.actual_yes_rate
                - SUM(t.yes_price * t.count) * 1.0 / SUM(t.count) AS mispricing_pp
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN mention_markets m ON t.ticker = m.ticker
        INNER JOIN market_yes_rates r ON LOWER(m.yes_sub_title) = r.word_key
        GROUP BY r.display_word, r.n_markets, r.actual_yes_rate
        ORDER BY mispricing_pp DESC
        """
    ).df()


def query_word_by_speaker(
    con: duckdb.DuckDBPyConnection,
    trades_dir: Path,
    markets_dir: Path,
    min_markets: int = 5,
) -> pd.DataFrame:
    """Mispricing grouped by (word, speaker) to reveal context dependence."""
    return con.execute(
        f"""
        WITH mention_markets AS (
            SELECT
                ticker, event_ticker, yes_sub_title, result,
                regexp_extract(UPPER(event_ticker), 'KX([A-Z]+)MENTION', 1) AS speaker
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
              AND UPPER(event_ticker) LIKE '%MENTION%'
              AND UPPER(event_ticker) NOT LIKE '%MENTIONSSINGLE%'
              AND yes_sub_title IS NOT NULL
              AND yes_sub_title != ''
        ),
        market_outcomes AS (
            SELECT
                LOWER(m.yes_sub_title) AS word_key,
                m.yes_sub_title,
                m.speaker,
                m.ticker,
                CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS resolved_yes
            FROM mention_markets m
            WHERE m.speaker IS NOT NULL AND m.speaker != ''
        ),
        market_yes_rates AS (
            SELECT
                word_key,
                mode(yes_sub_title) AS display_word,
                speaker,
                COUNT(*) AS n_markets,
                AVG(resolved_yes) * 100.0 AS actual_yes_rate
            FROM market_outcomes
            GROUP BY word_key, speaker
            HAVING COUNT(*) >= {min_markets}
        )
        SELECT
            r.display_word AS word,
            r.speaker,
            r.n_markets,
            SUM(t.count) AS total_contracts,
            SUM(t.yes_price * t.count) * 1.0 / SUM(t.count) AS vol_weighted_price,
            r.actual_yes_rate,
            r.actual_yes_rate
                - SUM(t.yes_price * t.count) * 1.0 / SUM(t.count) AS mispricing_pp
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN mention_markets m ON t.ticker = m.ticker
        INNER JOIN market_yes_rates r
            ON LOWER(m.yes_sub_title) = r.word_key
            AND regexp_extract(UPPER(m.event_ticker), 'KX([A-Z]+)MENTION', 1) = r.speaker
        GROUP BY r.display_word, r.speaker, r.n_markets, r.actual_yes_rate
        ORDER BY ABS(mispricing_pp) DESC
        """
    ).df()


def query_word_by_price_bucket(
    con: duckdb.DuckDBPyConnection,
    trades_dir: Path,
    markets_dir: Path,
    min_markets: int = 10,
    top_n: int = 20,
) -> pd.DataFrame:
    """Mispricing by price bucket for top N words (by n_markets).

    Uses market-weighted YES rate per word (not per bucket), since buckets
    split a single market's trades and don't represent independent trials.
    """
    return con.execute(
        f"""
        WITH mention_markets AS (
            SELECT ticker, event_ticker, yes_sub_title, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
              AND UPPER(event_ticker) LIKE '%MENTION%'
              AND UPPER(event_ticker) NOT LIKE '%MENTIONSSINGLE%'
              AND yes_sub_title IS NOT NULL
              AND yes_sub_title != ''
        ),
        market_outcomes AS (
            SELECT
                LOWER(yes_sub_title) AS word_key,
                yes_sub_title,
                ticker,
                CASE WHEN result = 'yes' THEN 1.0 ELSE 0.0 END AS resolved_yes
            FROM mention_markets
        ),
        market_yes_rates AS (
            SELECT
                word_key,
                mode(yes_sub_title) AS display_word,
                COUNT(*) AS n_markets,
                AVG(resolved_yes) * 100.0 AS actual_yes_rate
            FROM market_outcomes
            GROUP BY word_key
            HAVING COUNT(*) >= {min_markets}
        ),
        top_words AS (
            SELECT word_key, display_word
            FROM market_yes_rates
            ORDER BY n_markets DESC
            LIMIT {top_n}
        )
        SELECT
            tw.display_word AS word,
            CASE
                WHEN t.yes_price BETWEEN 1 AND 20 THEN '1-20'
                WHEN t.yes_price BETWEEN 21 AND 40 THEN '21-40'
                WHEN t.yes_price BETWEEN 41 AND 60 THEN '41-60'
                WHEN t.yes_price BETWEEN 61 AND 80 THEN '61-80'
                WHEN t.yes_price BETWEEN 81 AND 99 THEN '81-99'
            END AS price_bucket,
            SUM(t.count) AS total_contracts,
            r.actual_yes_rate,
            SUM(t.yes_price * t.count) * 1.0 / SUM(t.count) AS vol_weighted_price,
            r.actual_yes_rate
                - SUM(t.yes_price * t.count) * 1.0 / SUM(t.count) AS mispricing_pp
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN mention_markets m ON t.ticker = m.ticker
        INNER JOIN top_words tw ON LOWER(m.yes_sub_title) = tw.word_key
        INNER JOIN market_yes_rates r ON LOWER(m.yes_sub_title) = r.word_key
        WHERE t.yes_price BETWEEN 1 AND 99
        GROUP BY tw.display_word, price_bucket, r.actual_yes_rate
        HAVING SUM(t.count) > 0
        ORDER BY tw.display_word, price_bucket
        """
    ).df()
