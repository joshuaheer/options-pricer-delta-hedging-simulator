"""Shared chart style: one consistent, colour-blind-checked palette for every figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"   # secondary text, reference lines
GRID = "#e4e3df"

# Categorical slots, used in this fixed order (validated for adjacent-pair CVD separation).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Ordinal blue ramp for ordered levels (expiries, cost levels), light to dark.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#104281", "#0d366b"]


def ramp(n):
    """n evenly spaced steps from the ordinal ramp."""
    if n == 1:
        return [RAMP[3]]
    return [RAMP[round(i * (len(RAMP) - 1) / (n - 1))] for i in range(n)]


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "font.family": "DejaVu Sans", "text.color": INK,
    })


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
