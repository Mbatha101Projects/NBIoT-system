import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_METHODS = [
    "Plain Q-learning",
    "SARSA",
    "UCB",
    "Monte Carlo",
    "Double Q-learning (production)",
    "Contextual Bandit",
]

DISPLAY_NAMES = {
    "Plain Q-learning": "Plain\nQ-learning",
    "SARSA": "SARSA",
    "UCB": "UCB",
    "Monte Carlo": "Monte\nCarlo",
    "Double Q-learning (production)": "Double Q-learning\n(production)",
    "Contextual Bandit": "Contextual\nBandit",
}

# These colors are only for distinguishing methods; the palette is deliberately
# restrained so the graph remains readable in a report or presentation.
COLORS = [
    "#D3E2F1",
    "#D3E2F1",
    "#D3E2F1",
    "#D3E2F1",
    "#4119C5",
    "#03E91A",
]


def locate_csv() -> Path:
    if len(sys.argv) > 1:
        supplied = Path(sys.argv[1])
        if supplied.exists():
            return supplied
        raise FileNotFoundError(f"CSV not found: {supplied}")

    possible = [
        Path("results/rl_comparison_raw.csv"),
        Path(__file__).resolve().parent / "results" / "rl_comparison_raw.csv",
    ]

    for path in possible:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find results/rl_comparison_raw.csv. "
        "Run the RL training script first, or pass the CSV path as an argument."
    )


def normalise_method(value: str) -> str:
    # The original CSV contains quoted strings with embedded newlines for two
    # method names. Convert those to the one-line labels used for plotting.
    return " ".join(str(value).split())


def main() -> None:
    csv_path = locate_csv()
    df = pd.read_csv(csv_path)

    required = {"method", "seed", "states_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    df["method"] = df["method"].map(normalise_method)
    df["states_correct"] = pd.to_numeric(
        df["states_correct"], errors="coerce"
    )
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.dropna(subset=["method", "states_correct", "seed"])

    methods = [m for m in EXPECTED_METHODS if m in set(df["method"])]
    if not methods:
        raise ValueError("None of the expected RL methods were found in the CSV.")

    stats = (
        df[df["method"].isin(methods)]
        .groupby("method")["states_correct"]
        .agg(["mean", "median", "min", "max", "count"])
        .reindex(methods)
    )

    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=160)
    fig.patch.set_facecolor("#F7F8FA")
    ax.set_facecolor("#FFFFFF")

    means = stats["mean"].to_numpy()
    medians = stats["median"].to_numpy()
    mins = stats["min"].to_numpy()
    maxs = stats["max"].to_numpy()

    # Draw bars for the mean values.
    bars = ax.bar(
        x,
        means,
        width=0.62,
        color=COLORS[:len(methods)],
        edgecolor="#263238",
        linewidth=0.8,
        alpha=0.88,
        zorder=2,
    )

    # Full min-to-max range for each method.
    lower = means - mins
    upper = maxs - means

    ax.errorbar(
        x,
        means,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#263238",
        elinewidth=2.0,
        capsize=7,
        capthick=2.0,
        zorder=4,
    )

    # Median marker gives another useful summary without adding seed dots.
    ax.scatter(
        x,
        medians,
        marker="D",
        s=58,
        facecolor="#FFFFFF",
        edgecolor="#263238",
        linewidth=1.5,
        zorder=5,
        label="Median",
    )

    # Mean value on top of each bar.
    for i, (bar, mean_value, min_value, max_value) in enumerate(
        zip(bars, means, mins, maxs)
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_value + 0.45,
            f"{mean_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#263238",
            zorder=6,
        )

        # Show the full observed seed range underneath the mean label only
        # when there is enough variation to make it informative.
        if max_value != min_value:
            ax.text(
                x[i],
                max_value + 0.18,
                f"{int(min_value)}–{int(max_value)}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#5F6B72",
                zorder=6,
            )

    # Maximum theoretical score is 30 in this experiment, based on the
    # "out of 30" definition in the source data.
    ax.axhline(
        30,
        color="#7E8A97",
        linestyle="--",
        linewidth=1.2,
        alpha=0.55,
        zorder=1,
    )


    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(m, m) for m in methods],
        fontsize=10.5,
        fontweight="bold",
        color="#263238",
    )

    ax.set_ylabel(
        "States matching true-optimal action (out of 30)",
        fontsize=14,
        fontweight="bold",
        color="#263238",
        labelpad=12,
    )

    ax.set_xlabel(
        "RL Method",
        fontsize=12,
        fontweight="bold",
        color="#263238",
        labelpad=12,
    )

    ax.set_title(
        "RL Method Comparison",
        fontsize=20,
        fontweight="bold",
        color="#172026",
        pad=18,
    )



    ax.set_ylim(0, 32.8)
    ax.set_xlim(-0.55, len(methods) - 0.45)

    # Cleaner presentation/report styling.
    ax.grid(
        axis="y",
        color="#D8DDE2",
        linewidth=0.9,
        alpha=0.7,
        zorder=0,
    )
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB3BA")
    ax.spines["bottom"].set_color("#AAB3BA")

    ax.tick_params(axis="y", colors="#46515A", labelsize=10)
    ax.tick_params(axis="x", length=0)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9.5,
        labelcolor="#46515A",
    )



    plt.tight_layout(rect=(0, 0.035, 1, 0.97))

    output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    png = output_dir / "figure8_rl_comparison_clean.png"
    pdf = output_dir / "figure8_rl_comparison_clean.pdf"

    fig.savefig(
        png,
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    print(f"[READ]  {csv_path}")
    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")

    print("\nSummary used for the graph:")
    print(stats.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()