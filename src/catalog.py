import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone

import click
from huggingface_hub.utils import filter_repo_objects

from src.simready import CACHE_DIR, download_patterns, repo_files

DB_PATH = CACHE_DIR / "catalog.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    name           TEXT PRIMARY KEY,  -- folder name, e.g. IV_pole_a01
    family         TEXT,              -- name without the variant suffix, e.g. IV_pole
    variant        TEXT,              -- variant suffix, e.g. a01
    files          INTEGER NOT NULL,  -- file count
    total_bytes    INTEGER NOT NULL,  -- size of the whole folder
    download_bytes INTEGER NOT NULL,  -- size download_asset() downloads by default
    has_textures   INTEGER NOT NULL,
    has_thumbnail  INTEGER NOT NULL,
    synced_at      TEXT NOT NULL      -- UTC ISO timestamp
);
CREATE INDEX IF NOT EXISTS assets_family ON assets(family);
"""

VARIANT = re.compile(r"^(.*)_([a-z]\d+)$")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def fetch_rows() -> list[tuple]:
    """Recursively list the whole repo and aggregate it into one row per asset (top-level folder)."""
    files: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with click.progressbar(repo_files(), label="Listing repo files", show_pos=True) as entries:
        for e in entries:
            if "/" in e.path:  # only files inside asset folders
                files[e.path.split("/", 1)[0]].append((e.path, e.size))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for name, items in files.items():
        allow, ignore = download_patterns(name)
        picked = filter_repo_objects(items, allow_patterns=allow, ignore_patterns=ignore, key=lambda x: x[0])
        m = VARIANT.match(name)
        rows.append((
            name, m[1] if m else name, m[2] if m else None,
            len(items), sum(s for _, s in items), sum(s for _, s in picked),
            any("/Textures/" in p for p, _ in items),
            any(p.endswith("_thumbnail.png") for p, _ in items),
            now,
        ))
    return rows


SORTS = {"name": "name", "size": "total_bytes DESC", "download": "download_bytes DESC"}


def query(keyword: str = "", sort: str = "name", limit: int | None = None) -> tuple[list[str], list[tuple]]:
    """Rows of the assets table whose name contains keyword (case-insensitive): (column names, rows)."""
    with closing(connect()) as con:
        cur = con.execute(f"SELECT * FROM assets WHERE instr(lower(name), lower(?)) > 0 ORDER BY {SORTS[sort]} LIMIT ?",
                          (keyword, -1 if limit is None else limit))
        return [d[0] for d in cur.description], cur.fetchall()


@click.group()
def cli():
    """Local asset catalog (SQLite) for nvidia/simready-assets."""


@cli.command()
def sync():
    """Re-fetch the list from Hugging Face and overwrite catalog.db."""
    rows = fetch_rows()
    with connect() as con:
        con.execute("DELETE FROM assets")  # rebuild everything so assets removed from the repo disappear too
        con.executemany("INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    click.echo(f"{len(rows)} assets -> {DB_PATH}")


@cli.command()
@click.argument("keyword", default="")
@click.option("--sort", type=click.Choice(list(SORTS)), default="name", show_default=True)
@click.option("--limit", type=int, default=None, help="Maximum number of rows to list")
def search(keyword: str, sort: str, limit: int | None):
    """Search by name keyword (case-insensitive)."""
    cols, rows = query(keyword, sort, limit)
    rows = [dict(zip(cols, r)) for r in rows]
    if not rows:
        click.echo("No results (if you haven't synced yet, run `python -m src.catalog sync` first)")
        return
    click.echo(f"  {'name':<45} {'files':>6} {'total MB':>10} {'download MB':>12}")
    for r in rows:
        mark = "*" if (CACHE_DIR / r["name"]).is_dir() else " "
        click.echo(f"{mark} {r['name']:<45} {r['files']:>6} {r['total_bytes'] / 1e6:>10.1f} "
                   f"{r['download_bytes'] / 1e6:>12.1f}")
    click.echo(f"\n{len(rows)} rows (* = already downloaded to {CACHE_DIR}/)")


if __name__ == "__main__":
    cli()
