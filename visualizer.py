"""
visualizer.py — Draw NBA playoff bracket and analytics charts.

Bracket is drawn with coordinate-based matplotlib (no bracket library).
Seaborn is used for heatmap, feature importance, and calibration charts.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.calibration import CalibrationDisplay

log = logging.getLogger(__name__)

# Team abbreviations keyed by team ID
TEAM_ABBREVS = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN",
    1610612766: "CHA", 1610612741: "CHI", 1610612739: "CLE",
    1610612742: "DAL", 1610612743: "DEN", 1610612765: "DET",
    1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM",
    1610612748: "MIA", 1610612749: "MIL", 1610612750: "MIN",
    1610612740: "NOP", 1610612752: "NYK", 1610612760: "OKC",
    1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS",
    1610612761: "TOR", 1610612762: "UTA", 1610612764: "WAS",
}

CONF_COLORS = {
    "West": "#C8102E",
    "East": "#006BB6",
}


def _team_label(team_id: int, seed: int, team_names: dict) -> str:
    name = team_names.get(team_id, TEAM_ABBREVS.get(team_id, str(team_id)))
    return f"({seed}) {name}"


def _get_seed_for_team(team_id: int, seedings: list) -> int:
    for seed, tid in seedings:
        if tid == team_id:
            return seed
    return 0


def draw_conference_bracket(
    conference: str,
    conference_results: dict,
    team_names: dict,
    ax: plt.Axes,
) -> None:
    """
    Draw a single conference bracket (4 rounds) on the given Axes.
    Left-to-right: R1 -> R2 -> CF -> Champion.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    conf_color = CONF_COLORS.get(conference, "#333333")
    ax.set_title(f"{conference}ern Conference", fontsize=16, fontweight="bold",
                 color=conf_color, pad=12)

    seedings = conference_results.get("seedings", [])
    seed_map = conference_results.get("seed_map_original", {s: t for s, t in seedings})

    # x-positions for each round column
    x_r1 = 0.02
    x_r2 = 0.30
    x_cf = 0.58
    x_champ = 0.82
    line_len = 0.16  # horizontal line length

    # 8 vertical slots for 8 teams; each matchup occupies 2 adjacent slots
    # Slots from top to bottom: slot indices 0..7 mapped to y in [0.9, 0.1]
    def slot_y(slot_idx, n_slots=8):
        return 0.95 - slot_idx * (0.85 / (n_slots - 1))

    # Round 1 matchup positions: (top_slot, bottom_slot) for each matchup
    r1_matchups_order = [(0, 1), (2, 3), (4, 5), (6, 7)]
    r1_seed_pairs = [(1, 8), (2, 7), (3, 6), (4, 5)]

    def draw_matchup(x, top_y, bot_y, team_a_id, team_b_id, seed_a, seed_b,
                     winner_id, series_result, next_x):
        mid_y = (top_y + bot_y) / 2

        for y, tid, seed in [(top_y, team_a_id, seed_a), (bot_y, team_b_id, seed_b)]:
            label = _team_label(tid, seed, team_names)
            is_winner = (tid == winner_id)
            color = conf_color if is_winner else "#999999"
            weight = "bold" if is_winner else "normal"
            ax.text(x, y, label, fontsize=7.5, color=color, fontweight=weight,
                    va="center", ha="left", clip_on=True)
            # Horizontal line from team name to connector
            ax.plot([x + line_len * 0.55, x + line_len],
                    [y, y], color=color, lw=1.2, solid_capstyle="round")

        # Vertical connector
        ax.plot([x + line_len, x + line_len], [top_y, bot_y],
                color="#cccccc", lw=1.2)

        # Line from connector midpoint to next round
        if next_x is not None:
            ax.plot([x + line_len, next_x], [mid_y, mid_y],
                    color=conf_color, lw=1.5, alpha=0.7)

        # Series result annotation
        if series_result:
            ax.text(x + line_len + 0.01, mid_y + 0.018, series_result,
                    fontsize=6.5, color="#555555", va="bottom")

    # Collect R1 winner positions (midpoints) for R2 line targeting
    r1_mid_ys = []
    for (top_slot, bot_slot), (seed_a, seed_b) in zip(r1_matchups_order, r1_seed_pairs):
        top_y = slot_y(top_slot)
        bot_y = slot_y(bot_slot)
        mid_y = (top_y + bot_y) / 2
        r1_mid_ys.append(mid_y)

        key = f"{seed_a}v{seed_b}"
        r1_data = conference_results["round1"].get(key, {})
        tid_a = seed_map.get(seed_a, 0)
        tid_b = seed_map.get(seed_b, 0)
        winner_id = r1_data.get("winner_id", tid_a)
        series_result = r1_data.get("series_result", "")

        draw_matchup(x_r1, top_y, bot_y, tid_a, tid_b, seed_a, seed_b,
                     winner_id, series_result, x_r2)

    # Round 2: winners of (M1,M2) -> top half; (M3,M4) -> bottom half
    r2_pairs = conference_results.get("round2", {})
    r2_keys = sorted(r2_pairs.keys())
    r2_mid_ys = []

    for i, key in enumerate(r2_keys):
        r2_data = r2_pairs[key]
        tid_a = r2_data.get("team_a_id", 0)
        tid_b = r2_data.get("team_b_id", 0)
        winner_id = r2_data.get("winner_id", tid_a)
        series_result = r2_data.get("series_result", "")

        # y positions: top half at avg of r1_mid_ys[0,1], bottom half at avg of r1_mid_ys[2,3]
        if i == 0:
            top_y = r1_mid_ys[0]
            bot_y = r1_mid_ys[1]
        else:
            top_y = r1_mid_ys[2]
            bot_y = r1_mid_ys[3]
        mid_y = (top_y + bot_y) / 2
        r2_mid_ys.append(mid_y)

        seed_a = _get_seed_for_team(tid_a, seedings)
        seed_b = _get_seed_for_team(tid_b, seedings)
        draw_matchup(x_r2, top_y, bot_y, tid_a, tid_b, seed_a, seed_b,
                     winner_id, series_result, x_cf)

    # Conference Final
    cf_data_dict = conference_results.get("conference_final", {})
    cf_key = list(cf_data_dict.keys())[0] if cf_data_dict else None
    cf_data = cf_data_dict.get(cf_key, {}) if cf_key else {}

    if cf_data and len(r2_mid_ys) >= 2:
        tid_a = cf_data.get("team_a_id", 0)
        tid_b = cf_data.get("team_b_id", 0)
        winner_id = cf_data.get("winner_id", tid_a)
        series_result = cf_data.get("series_result", "")
        top_y = r2_mid_ys[0]
        bot_y = r2_mid_ys[1] if len(r2_mid_ys) > 1 else r2_mid_ys[0] - 0.2
        mid_y = (top_y + bot_y) / 2

        seed_a = _get_seed_for_team(tid_a, seedings)
        seed_b = _get_seed_for_team(tid_b, seedings)
        draw_matchup(x_cf, top_y, bot_y, tid_a, tid_b, seed_a, seed_b,
                     winner_id, series_result, None)

        # Champion label
        champ_id = conference_results.get("conference_champion_id", winner_id)
        champ_seed = _get_seed_for_team(champ_id, seedings)
        champ_label = _team_label(champ_id, champ_seed, team_names)
        ax.text(x_champ, mid_y, champ_label, fontsize=9, fontweight="bold",
                color=conf_color, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                          edgecolor=conf_color, alpha=0.8))
        ax.plot([x_cf + line_len, x_champ - 0.01], [mid_y, mid_y],
                color=conf_color, lw=2, alpha=0.8)


def draw_finals_bracket(
    finals_result: dict,
    west_champion_id: int,
    east_champion_id: int,
    team_names: dict,
    ax: plt.Axes,
    west_seed: int = 1,
    east_seed: int = 2,
) -> None:
    """
    Draw the NBA Finals matchup in the center panel.
    West enters from left, East from right. Champion shown at top.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("NBA Finals", fontsize=16, fontweight="bold",
                 color="#333333", pad=12)

    west_label = _team_label(west_champion_id, west_seed, team_names)
    east_label = _team_label(east_champion_id, east_seed, team_names)
    champion_id = finals_result.get("winner_id", west_champion_id)
    series_result = finals_result.get("series_result", "")

    mid_y = 0.45

    # West team (left side)
    west_color = CONF_COLORS["West"]
    east_color = CONF_COLORS["East"]
    west_is_champ = (champion_id == west_champion_id)

    ax.text(0.05, mid_y + 0.12, west_label, fontsize=9,
            color=west_color if west_is_champ else "#999999",
            fontweight="bold" if west_is_champ else "normal",
            ha="left", va="center")
    ax.plot([0.05, 0.38], [mid_y + 0.12, mid_y + 0.12],
            color=west_color, lw=1.5, alpha=0.6)
    ax.plot([0.38, 0.38], [mid_y + 0.12, mid_y],
            color="#cccccc", lw=1.5)

    # East team (right side)
    ax.text(0.95, mid_y + 0.12, east_label, fontsize=9,
            color=east_color if not west_is_champ else "#999999",
            fontweight="bold" if not west_is_champ else "normal",
            ha="right", va="center")
    ax.plot([0.62, 0.95], [mid_y + 0.12, mid_y + 0.12],
            color=east_color, lw=1.5, alpha=0.6)
    ax.plot([0.62, 0.62], [mid_y + 0.12, mid_y],
            color="#cccccc", lw=1.5)

    # Horizontal connector
    ax.plot([0.38, 0.62], [mid_y, mid_y], color="#cccccc", lw=1.5)

    # Series result
    if series_result:
        ax.text(0.5, mid_y - 0.04, f"Series: {series_result}",
                fontsize=9, ha="center", va="center", color="#555555")

    # Champion label
    champ_label = _team_label(champion_id, 1, team_names)
    champ_color = west_color if west_is_champ else east_color
    ax.text(0.5, 0.75, "🏆 NBA Champion", fontsize=11, ha="center",
            va="center", color="#FFD700", fontweight="bold")
    ax.text(0.5, 0.65, champ_label, fontsize=13, ha="center",
            va="center", color=champ_color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9e6",
                      edgecolor="#FFD700", linewidth=2))

    # Finals avg win probability
    avg_prob = finals_result.get("winner_win_prob_avg", 0.5)
    ax.text(0.5, 0.28, f"Model confidence: {avg_prob:.1%}",
            fontsize=8, ha="center", va="center", color="#777777",
            style="italic")


def plot_full_bracket(
    bracket_results: dict,
    team_names: dict,
    output_path: str = "output/full_bracket.png",
    figsize: tuple = (26, 12),
) -> None:
    """
    Create 3-panel figure: West bracket | Finals | East bracket.
    Saves to output_path.
    """
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [5, 2, 5]},
    )
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("#f8f9fa")

    # West bracket
    draw_conference_bracket("West", bracket_results["west"], team_names, axes[0])

    # Finals center panel
    draw_finals_bracket(
        bracket_results["finals"],
        bracket_results["west_champion_id"],
        bracket_results["east_champion_id"],
        team_names,
        axes[1],
    )

    # East bracket
    draw_conference_bracket("East", bracket_results["east"], team_names, axes[2])

    fig.suptitle("2024-25 NBA Playoff Bracket Prediction",
                 fontsize=18, fontweight="bold", y=1.01, color="#1a1a2e")
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Saved bracket to {output_path}")


def plot_win_probability_heatmap(
    bracket_results: dict,
    team_names: dict,
    output_path: str = "output/win_probabilities.png",
) -> None:
    """
    Seaborn heatmap showing predicted win probabilities per series game.
    Rows = matchups, columns = game 1-7.
    """
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    rows = []
    labels = []

    def _extract(conf_results, conf_label):
        for rnd_key in ["round1", "round2", "conference_final"]:
            for matchup_key, series in conf_results.get(rnd_key, {}).items():
                tid_a = series.get("team_a_id", 0)
                tid_b = series.get("team_b_id", 0)
                name_a = team_names.get(tid_a, TEAM_ABBREVS.get(tid_a, str(tid_a)))
                name_b = team_names.get(tid_b, TEAM_ABBREVS.get(tid_b, str(tid_b)))
                label = f"{conf_label}: {name_a} vs {name_b}"
                probs = [g.get("win_prob_a", 0.5) for g in series.get("game_by_game", [])]
                # Pad to 7 games with NaN
                probs += [np.nan] * (7 - len(probs))
                rows.append(probs[:7])
                labels.append(label)

    _extract(bracket_results["west"], "W")
    _extract(bracket_results["east"], "E")

    # Finals
    finals = bracket_results.get("finals", {})
    tid_a = finals.get("team_a_id", 0)
    tid_b = finals.get("team_b_id", 0)
    name_a = team_names.get(tid_a, TEAM_ABBREVS.get(tid_a, str(tid_a)))
    name_b = team_names.get(tid_b, TEAM_ABBREVS.get(tid_b, str(tid_b)))
    probs = [g.get("win_prob_a", 0.5) for g in finals.get("game_by_game", [])]
    probs += [np.nan] * (7 - len(probs))
    rows.append(probs[:7])
    labels.append(f"Finals: {name_a} vs {name_b}")

    heatmap_data = pd.DataFrame(rows, index=labels,
                                 columns=[f"G{i}" for i in range(1, 8)])

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.55)))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="RdYlBu_r",
        center=0.5,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Win Probability (Team A)"},
        mask=heatmap_data.isna(),
    )
    ax.set_title("Predicted Win Probabilities by Series & Game\n(Team A = first-named team)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Game Number")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved win probability heatmap to {output_path}")


def plot_feature_importance(
    model,
    feature_names: list,
    output_path: str = "output/feature_importance.png",
) -> None:
    """
    Horizontal bar chart of logistic regression coefficients.
    Green = positive (favors win), Red = negative (hurts win prob).
    """
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    coef = model.named_steps["logisticregression"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
    }).sort_values("coefficient")

    colors = ["#e74c3c" if c < 0 else "#27ae60" for c in coef_df["coefficient"]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(feature_names) * 0.5)))
    bars = ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Logistic Regression Coefficient", fontsize=11)
    ax.set_title("Feature Importance\n(Positive = favors win, Negative = hurts win probability)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    green_patch = mpatches.Patch(color="#27ae60", label="Favors win")
    red_patch = mpatches.Patch(color="#e74c3c", label="Hurts win prob")
    ax.legend(handles=[green_patch, red_patch], loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved feature importance chart to {output_path}")


def plot_model_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str = "output/calibration_curve.png",
) -> None:
    """
    Reliability diagram (calibration curve) + KDE of predicted probabilities.
    A well-calibrated model's curve should track the diagonal.
    """
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    CalibrationDisplay.from_predictions(
        y_true, y_prob, n_bins=10, ax=ax1, name="Logistic Regression"
    )
    ax1.set_title("Calibration Curve\n(Reliability Diagram)", fontsize=12, fontweight="bold")
    ax1.set_facecolor("#f8f9fa")

    # KDE of predicted probabilities
    prob_df = pd.DataFrame({"probability": y_prob, "actual": y_true.astype(str)})
    sns.kdeplot(data=prob_df, x="probability", hue="actual",
                ax=ax2, fill=True, alpha=0.4,
                palette={"0": "#e74c3c", "1": "#27ae60"})
    ax2.set_xlabel("Predicted Win Probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Distribution of Predicted Probabilities\nby Actual Outcome",
                  fontsize=12, fontweight="bold")
    ax2.set_facecolor("#f8f9fa")
    ax2.legend(title="Actual Outcome", labels=["Win (1)", "Loss (0)"])

    fig.patch.set_facecolor("#f8f9fa")
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved calibration chart to {output_path}")
