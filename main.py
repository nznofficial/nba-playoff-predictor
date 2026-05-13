"""
main.py — NBA Playoff Predictor pipeline orchestration.

Run with: python main.py
Uses pre-fetched playoff game CSVs and cached team stats — no API wait.
Output: models/logistic_model.joblib + output/*.png

Output: models/logistic_model.joblib + output/*.png
"""

import logging
import os
import sys
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("nba_predictor.log"),
    ],
)
log = logging.getLogger(__name__)

# ─── 2024-25 PLAYOFF SEEDINGS ────────────────────────────────────────────────
# Western Conference
# Based on 2024-25 NBA playoff bracket. Update if seedings have changed.
WEST_SEEDINGS = [
    (1, 1610612760),  # Oklahoma City Thunder
    (2, 1610612745),  # Houston Rockets
    (3, 1610612744),  # Golden State Warriors
    (4, 1610612746),  # LA Clippers
    (5, 1610612747),  # Los Angeles Lakers
    (6, 1610612763),  # Memphis Grizzlies
    (7, 1610612743),  # Denver Nuggets
    (8, 1610612750),  # Minnesota Timberwolves
]

# Eastern Conference
EAST_SEEDINGS = [
    (1, 1610612739),  # Cleveland Cavaliers
    (2, 1610612738),  # Boston Celtics
    (3, 1610612752),  # New York Knicks
    (4, 1610612749),  # Milwaukee Bucks
    (5, 1610612748),  # Miami Heat
    (6, 1610612753),  # Orlando Magic
    (7, 1610612755),  # Philadelphia 76ers
    (8, 1610612754),  # Indiana Pacers
]

CURRENT_SEASON = "2024-25"
DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/logistic_model.joblib"


def load_team_names(data_dir: str = DATA_DIR) -> dict:
    """Load team ID -> name mapping from cached team_ids.csv."""
    path = os.path.join(data_dir, "team_ids.csv")
    if not os.path.exists(path):
        log.warning("team_ids.csv not found; team names will show as IDs")
        return {}
    df = pd.read_csv(path)
    if "TEAM_ID" in df.columns and "TEAM_NAME" in df.columns:
        return dict(zip(df["TEAM_ID"].tolist(), df["TEAM_NAME"].tolist()))
    return {}


def main():
    log.info("=" * 60)
    log.info("NBA PLAYOFF PREDICTOR")
    log.info("=" * 60)

    # ── Step 1: Ensure team stats are cached (game logs not needed) ──────────
    log.info("\n[1/8] Loading team stats (cached)...")
    from data_fetcher import (
        fetch_team_advanced_stats, fetch_team_basic_stats, fetch_all_team_ids, SEASONS
    )
    fetch_all_team_ids()
    for _season in SEASONS + [CURRENT_SEASON]:
        fetch_team_advanced_stats(_season, "Regular Season")
        fetch_team_basic_stats(_season, "Regular Season")

    # ── Step 2: Build training DataFrame ─────────────────────────────────────
    log.info("\n[2/8] Building training DataFrame...")
    from feature_engineering import build_training_dataframe
    training_path = os.path.join(PROCESSED_DIR, "training_data.csv")
    training_df = build_training_dataframe(
        seasons=SEASONS,
        data_dir=DATA_DIR,
        output_path=training_path,
    )
    log.info(f"  Training data: {len(training_df)} rows, win rate={training_df['won'].mean():.3f}")

    # ── Step 3: Train model + cross-validation ────────────────────────────────
    log.info("\n[3/8] Training logistic regression model...")
    from model import train_model, cross_validate_model, print_model_report, MODEL_FEATURES
    pipeline, metrics = train_model(training_df, model_path=MODEL_PATH)

    log.info("\n[4/8] Cross-validating model (TimeSeriesSplit)...")
    cv_results = cross_validate_model(training_df)
    metrics["cv_results"] = cv_results

    print_model_report(pipeline, metrics)

    # ── Step 5: Build current-season features ─────────────────────────────────
    log.info("\n[5/8] Building 2024-25 playoff features...")
    from feature_engineering import build_current_season_features
    playoff_seedings = {"West": WEST_SEEDINGS, "East": EAST_SEEDINGS}
    current_path = os.path.join(PROCESSED_DIR, "current_season.csv")

    current_df = build_current_season_features(
        season=CURRENT_SEASON,
        playoff_seedings=playoff_seedings,
        data_dir=DATA_DIR,
        output_path=current_path,
        all_seasons=SEASONS + [CURRENT_SEASON],
    )
    log.info(f"  Current season features: {len(current_df)} rows")

    # ── Step 6: Simulate full bracket ─────────────────────────────────────────
    log.info("\n[6/8] Simulating 2024-25 playoff bracket...")
    from predictor import run_full_bracket
    team_names = load_team_names(DATA_DIR)

    bracket_results = run_full_bracket(
        west_seedings=WEST_SEEDINGS,
        east_seedings=EAST_SEEDINGS,
        current_features=current_df,
        model=pipeline,
        team_names=team_names,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BRACKET PREDICTION SUMMARY")
    print("=" * 60)
    west_champ = team_names.get(bracket_results["west_champion_id"],
                                str(bracket_results["west_champion_id"]))
    east_champ = team_names.get(bracket_results["east_champion_id"],
                                str(bracket_results["east_champion_id"]))
    champion = bracket_results["champion_name"] or str(bracket_results["champion_id"])
    print(f"Western Champion: {west_champ}")
    print(f"Eastern Champion: {east_champ}")
    print(f"Finals Result:    {bracket_results['finals']['series_result']}")
    print(f"NBA CHAMPION:     {champion}")
    print("=" * 60 + "\n")

    # ── Step 7: Visualizations ────────────────────────────────────────────────
    log.info("\n[7/8] Generating visualizations...")
    from visualizer import (
        plot_full_bracket,
        plot_win_probability_heatmap,
        plot_feature_importance,
        plot_model_calibration,
    )

    plot_full_bracket(bracket_results, team_names, output_path="output/full_bracket.png")
    plot_win_probability_heatmap(bracket_results, team_names, output_path="output/win_probabilities.png")
    plot_feature_importance(pipeline, MODEL_FEATURES, output_path="output/feature_importance.png")
    plot_model_calibration(
        metrics["y_test"],
        metrics["y_prob"],
        output_path="output/calibration_curve.png",
    )

    log.info("\n[8/8] All outputs saved to output/")
    log.info("Done.")

    return bracket_results


if __name__ == "__main__":
    main()
