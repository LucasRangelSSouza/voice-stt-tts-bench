"""Shared helpers for the analysis scripts: data loading, GPU naming, figure and summary output."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
SUMMARY = ROOT / "results" / "summary.json"
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
BLUE, ORANGE, GREEN, GREY, PURPLE, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#718096", "#805ad5", "#e53e3e"


def gpu_name(g) -> str:
    """Normalize the many spellings of a GPU model found in the raw logs."""
    g = str(g)
    g = re.sub(r"NVIDIA (GeForce )?", "", g).strip()
    if "PRO 6000" in g:
        return "RTX PRO 6000"
    if "A100" in g:
        return "A100"
    return g


def read(name: str) -> pd.DataFrame:
    """Read a curated CSV from data/."""
    return pd.read_csv(DATA / name)


def save_fig(fig, name: str) -> str:
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f"results/figures/{name}"


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    """Write a table as CSV and as a GitHub-flavoured markdown file."""
    df.to_csv(TAB / f"{name}.csv", index=index)
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r.tolist()) + " |")
    (TAB / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_summary(values: dict) -> None:
    """Merge key facts into results/summary.json (read by the README builder and by the tests)."""
    cur = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    cur.update({k: (float(v) if hasattr(v, "item") else v) for k, v in values.items()})
    SUMMARY.write_text(json.dumps(cur, indent=1, sort_keys=True), encoding="utf-8")
