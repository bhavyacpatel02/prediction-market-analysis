# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python framework for analyzing prediction market data from Polymarket and Kalshi. Collects trade/market data as chunked Parquet files, then runs DuckDB-based analyses that produce figures, CSVs, and chart configs.

## Commands

```bash
make analyze              # Interactive analysis menu
make run <analysis_name>  # Run specific analysis by snake_case name
make index                # Interactive indexer menu (data collection)
make test                 # Run all tests
make lint                 # Check linting (ruff check + ruff format --check)
make format               # Auto-fix lint and format
make setup                # Download dataset (36 GiB compressed) from Cloudflare R2
make package              # Package data/ into data.tar.zst
```

Run a single test: `uv run pytest tests/test_compile.py -v -k "test_name"`
Skip slow tests: `uv run pytest tests/ -v -m "not slow"`

**Uses `uv` for all dependency management — never use pip.** All make targets use `uv run`.

## Architecture

### Auto-Discovery

`Analysis.load()` and `Indexer.load()` glob `**/*.py` in their respective directories, import each module, and collect concrete non-abstract subclasses. Files starting with `_` are skipped. `main.py` uses this to populate interactive menus.

### Analysis Classes (`src/analysis/`)

Organized by platform: `kalshi/`, `polymarket/`, `comparison/`. Each analysis:
- Inherits from `Analysis` (defined in `src/common/analysis.py`)
- Sets `name` (snake_case, becomes output filename) and `description`
- Implements `run()` returning `AnalysisOutput(figure, data, chart)`
- Constructor accepts optional `Path` params (e.g. `trades_dir`, `markets_dir`) for test injection, defaulting to `data/<platform>/<type>` via `Path(__file__).parent.parent.parent.parent`
- Uses DuckDB glob queries (`'{dir}/*.parquet'`) to read chunked Parquet data
- `FuncAnimation` outputs save as GIF; static `Figure` outputs save as PNG/PDF/SVG

### Indexer Classes (`src/indexers/`)

Organized by platform. Each inherits from `Indexer` (`src/common/indexer.py`). Uses `ParquetStorage` for chunked writes (10k records/chunk) with deduplication.

### Key Shared Components (`src/common/`)

- **`storage.py`** — `ParquetStorage`: chunked Parquet file management with DuckDB-based dedup
- **`client.py`** — `@retry_request()` decorator: 5 attempts, exponential backoff, retries on 429/5xx
- **`interfaces/chart.py`** — `ChartConfig` dataclass + factory functions (`line_chart()`, `bar_chart()`, etc.) for JSON chart configs
- **`analysis.py`** — `Analysis` base class + `AnalysisOutput` dataclass; `self.progress(msg)` context manager for tqdm spinners

### Data Conventions

- Kalshi prices are cents (1-99); Polymarket prices are decimals (0.0-1.0)
- `_fetched_at` column added to all stored records
- Large integers (Polymarket asset IDs) stored as strings to avoid Parquet overflow
- Parquet files named `{type}_N_M.parquet` where N and M are row index bounds

## Testing

Tests use session-scoped fixtures (`tests/conftest.py`) that build in-memory Parquet test data. Kalshi fixtures have 2100 trade rows to meet minimum-sample thresholds in analyses. `test_analysis_run.py` uses `inspect.signature` to map fixture paths to constructor parameters based on module path.

`matplotlib.use("Agg")` is set in conftest for headless CI.

## Linting

Ruff with Python 3.9 target, 120 char line length. Rules: E, W, F, I (isort), B (bugbear), C4, UP. `src` is first-party for isort.

## CI

- **`ci.yml`**: Lint + test on ubuntu-latest with Python 3.9
- **`pr-validation.yml`**: PR titles must follow conventional commits: `<type>(<scope>): <description>` — types: `feat`, `fix`, `perf`, `chore`, `refactor`, `deps`, `docs`, `test`, `ci`, `build`, `style`, `revert`

## Environment Variables

`POLYGON_RPC` — required for Polymarket blockchain indexer. Copy `.env.example` to `.env`.
