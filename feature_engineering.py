"""
feature_engineering.py — Build model-ready training and prediction DataFrames.

Each playoff game produces two rows (one per team perspective) with differential
features derived from regular-season stats. Playoff series structure and rest days
are derived from pre-fetched playoff_games CSV files (LeagueGameFinder format).
No additional API calls are needed.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

BUBBLE_SEASON = "2019-20"
REST_CAP = 10
REST_DEFAULT = 5

# Seasons with available playoff game CSVs (LeagueGameFinder data)
TRAINING_SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYOFF_DATA_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "emv_proj1_Playoff_Predictor", "data", "raw")
)


def _load_playoff_games(season: str, playoff_dir: str = None) -> pd.DataFrame:
    """Load a season's playoff games CSV (two rows per game, one per team)."""
    if playoff_dir is None:
        playoff_dir = PLAYOFF_DATA_DIR
    path = os.path.join(playoff_dir, f"playoff_games_{season}.csv")
    if not os.path.exists(path):
        log.warning(f"Playoff games file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def build_series_records(
    season: str,
    data_dir: str = "data/raw",
    playoff_dir: str = None,
) -> pd.DataFrame:
    """
    Reconstruct playoff series from a LeagueGameFinder-format playoff games CSV.
    Returns DataFrame with one row per (team, game): team_id, opponent_id, series
    metadata, home_court, won, rest_days, series tracking columns.
    """
    pg = _load_playoff_games(season, playoff_dir)
    if len(pg) == 0:
        return pd.DataFrame()

    # Rest days per team: days since their previous game in this playoff run
    pg_sorted = pg.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    pg_sorted["rest_days"] = (
        pg_sorted.groupby("TEAM_ID")["GAME_DATE"]
        .diff().dt.days
        .fillna(REST_DEFAULT)
        .clip(upper=REST_CAP)
        .astype(int)
    )
    rest_lookup = {
        (int(row["TEAM_ID"]), str(row["GAME_ID"])): int(row["rest_days"])
        for _, row in pg_sorted.iterrows()
    }

    # Build per-game records keyed by GAME_ID
    game_records = {}
    for _, row in pg.iterrows():
        gid = str(row["GAME_ID"])
        if gid not in game_records:
            game_records[gid] = {"game_date": row["GAME_DATE"], "teams": []}
        game_records[gid]["teams"].append({
            "TEAM_ID": int(row["TEAM_ID"]),
            "WL": str(row["WL"]),
            "MATCHUP": str(row.get("MATCHUP", "")),
        })

    # Group games into series by opposing team pair
    series_games: dict = {}
    for gid, info in game_records.items():
        team_ids = [t["TEAM_ID"] for t in info["teams"]]
        if len(team_ids) < 2:
            continue
        key = frozenset(team_ids[:2])
        series_games.setdefault(key, []).append((gid, info["game_date"]))

    for key in series_games:
        series_games[key].sort(key=lambda x: x[1])

    all_rows = []
    for series_key, games in series_games.items():
        team_list = sorted(series_key)
        team_a, team_b = team_list[0], team_list[1]
        series_id = f"{season}_{team_a}_{team_b}"

        wins_a, wins_b = 0, 0
        for game_num, (gid, gdate) in enumerate(games, start=1):
            team_data = {t["TEAM_ID"]: t for t in game_records[gid]["teams"]}

            for focal_id, opp_id in [(team_a, team_b), (team_b, team_a)]:
                if focal_id not in team_data:
                    continue
                td = team_data[focal_id]
                won = 1 if td["WL"] == "W" else 0
                home_court = 1 if " vs. " in td["MATCHUP"] else 0
                rest = rest_lookup.get((focal_id, gid), REST_DEFAULT)
                opp_rest = rest_lookup.get((opp_id, gid), REST_DEFAULT)
                series_wins = wins_a if focal_id == team_a else wins_b
                series_losses = wins_b if focal_id == team_a else wins_a

                all_rows.append({
                    "season": season,
                    "game_date": gdate,
                    "team_id": focal_id,
                    "opponent_id": opp_id,
                    "game_id": gid,
                    "series_id": series_id,
                    "series_game_num": game_num,
                    "home_court": home_court,
                    "won": won,
                    "rest_days": rest,
                    "opp_rest_days": opp_rest,
                    "rest_days_diff": rest - opp_rest,
                    "series_wins": series_wins,
                    "series_losses": series_losses,
                    "series_lead": series_wins - series_losses,
                    "is_bubble": 1 if season == BUBBLE_SEASON else 0,
                })

            # Update series score after each game
            for t in game_records[gid]["teams"]:
                if t["WL"] == "W":
                    if t["TEAM_ID"] == team_a:
                        wins_a += 1
                    else:
                        wins_b += 1

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def compute_playoff_experience(
    team_id: int,
    current_season: str,
    all_seasons: list,
    playoff_dir: str = None,
) -> int:
    """Count playoff appearances in the 3 seasons immediately prior to current_season."""
    if playoff_dir is None:
        playoff_dir = PLAYOFF_DATA_DIR
    idx = all_seasons.index(current_season) if current_season in all_seasons else len(all_seasons)
    prior_seasons = all_seasons[max(0, idx - 3):idx]
    count = 0
    for s in prior_seasons:
        path = os.path.join(playoff_dir, f"playoff_games_{s}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=["TEAM_ID"])
            if team_id in df["TEAM_ID"].values:
                count += 1
    return count


def _load_reg_season_stats(season: str, data_dir: str) -> pd.DataFrame:
    """Load and merge regular-season advanced + basic stats for a season."""
    adv_path = os.path.join(data_dir, "team_stats", f"{season}_Regular_Season_advanced.csv")
    base_path = os.path.join(data_dir, "team_stats", f"{season}_Regular_Season_base.csv")

    if not os.path.exists(adv_path) or not os.path.exists(base_path):
        return pd.DataFrame()

    adv = pd.read_csv(adv_path)
    base = pd.read_csv(base_path)

    if "FG3A" in base.columns and "FGA" in base.columns:
        base["three_pt_rate"] = base["FG3A"] / base["FGA"].replace(0, np.nan)
    else:
        base["three_pt_rate"] = 0.35

    base_slim = base[["TEAM_ID", "three_pt_rate"]].copy()
    merged = adv.merge(base_slim, on="TEAM_ID", how="left")
    merged["three_pt_rate"] = merged["three_pt_rate"].fillna(0.35)
    return merged


def _build_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all differential columns from joined focal + opponent stats."""
    df = df.copy()

    def _diff(focal_col, opp_col, default=0.0):
        a = df[focal_col].fillna(default) if focal_col in df.columns else default
        b = df[opp_col].fillna(default) if opp_col in df.columns else default
        return a - b

    df["off_rtg_diff"] = _diff("off_rtg", "opp_def_rtg")
    df["def_rtg_diff"] = _diff("def_rtg", "opp_off_rtg")
    df["net_rtg_diff"] = _diff("net_rtg", "opp_net_rtg")
    df["pace_diff"] = _diff("pace", "opp_pace")
    df["win_pct_diff"] = _diff("team_win_pct_reg", "opp_team_win_pct_reg")
    df["three_pt_rate_diff"] = _diff("three_pt_rate", "opp_three_pt_rate", 0.35)
    df["playoff_exp_diff"] = _diff("playoff_exp_3yr", "opp_playoff_exp_3yr")
    return df


def merge_team_stats_onto_games(
    games_df: pd.DataFrame,
    reg_season_stats: pd.DataFrame,
    season: str,
    all_seasons: list,
    playoff_dir: str = None,
) -> pd.DataFrame:
    """Join regular-season stats onto each game row and compute differential features."""
    if len(games_df) == 0 or len(reg_season_stats) == 0:
        return pd.DataFrame()

    stats = reg_season_stats.copy()
    stats = stats.rename(columns={
        "OFF_RATING": "off_rtg",
        "DEF_RATING": "def_rtg",
        "NET_RATING": "net_rtg",
        "PACE": "pace",
        "W_PCT": "team_win_pct_reg",
    })
    if "net_rtg" not in stats.columns and "off_rtg" in stats.columns:
        stats["net_rtg"] = stats["off_rtg"] - stats["def_rtg"]

    # Playoff experience for every team seen in this season's games
    all_team_ids = pd.concat([games_df["team_id"], games_df["opponent_id"]]).unique()
    exp_map = {
        int(tid): compute_playoff_experience(int(tid), season, all_seasons, playoff_dir)
        for tid in all_team_ids
    }
    stats["playoff_exp_3yr"] = stats["TEAM_ID"].map(exp_map).fillna(0).astype(int)

    stat_cols = [c for c in [
        "TEAM_ID", "off_rtg", "def_rtg", "net_rtg", "pace",
        "team_win_pct_reg", "three_pt_rate", "playoff_exp_3yr",
    ] if c in stats.columns]

    stats_focal = stats[stat_cols].rename(columns={"TEAM_ID": "team_id"})
    stats_opp = stats_focal.rename(columns={
        "team_id": "opponent_id",
        **{c: f"opp_{c}" for c in stats_focal.columns if c != "team_id"},
    })

    df = games_df.merge(stats_focal, on="team_id", how="left")
    df = df.merge(stats_opp, on="opponent_id", how="left")
    return _build_diff_features(df)


def build_training_dataframe(
    seasons: list = None,
    data_dir: str = "data/raw",
    output_path: str = "data/processed/training_data.csv",
    playoff_dir: str = None,
) -> pd.DataFrame:
    """
    Build complete training DataFrame from all cached season data.
    Uses playoff_games CSV files for series reconstruction and advanced stats
    CSVs for team ratings. Saves to output_path and returns the DataFrame.
    """
    if seasons is None:
        seasons = TRAINING_SEASONS
    # Only use seasons that have playoff game files
    seasons = [s for s in seasons if s in TRAINING_SEASONS]

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    all_frames = []
    for season in seasons:
        log.info(f"Building features for season: {season}")

        series_df = build_series_records(season, data_dir, playoff_dir)
        if len(series_df) == 0:
            log.warning(f"  No playoff series data for {season} — skipping")
            continue

        reg_stats = _load_reg_season_stats(season, data_dir)
        if len(reg_stats) == 0:
            log.warning(f"  No regular season stats for {season} — skipping")
            continue

        enriched = merge_team_stats_onto_games(series_df, reg_stats, season, seasons, playoff_dir)
        if len(enriched) > 0:
            all_frames.append(enriched)
            log.info(f"  {season}: {len(enriched)} game-team rows")

    if not all_frames:
        raise RuntimeError("No training data built. Check playoff_dir and data_dir paths.")

    training_df = pd.concat(all_frames, ignore_index=True)
    critical_cols = [c for c in ["net_rtg_diff", "off_rtg_diff", "def_rtg_diff", "won"]
                     if c in training_df.columns]
    training_df = training_df.dropna(subset=critical_cols)

    log.info(f"Training DataFrame: {len(training_df)} rows, {len(training_df.columns)} cols")
    log.info(f"Win rate: {training_df['won'].mean():.3f} (should be ~0.50)")
    training_df.to_csv(output_path, index=False)
    log.info(f"Saved training data to {output_path}")
    return training_df


def build_current_season_features(
    season: str = "2024-25",
    playoff_seedings: dict = None,
    data_dir: str = "data/raw",
    output_path: str = "data/processed/current_season.csv",
    all_seasons: list = None,
    playoff_dir: str = None,
) -> pd.DataFrame:
    """
    Build feature rows for all first-round playoff matchups of the current season.
    playoff_seedings: {"West": [(seed, team_id), ...], "East": [...]}
    """
    if all_seasons is None:
        all_seasons = TRAINING_SEASONS + [season]
    if playoff_dir is None:
        playoff_dir = PLAYOFF_DATA_DIR

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    reg_stats = _load_reg_season_stats(season, data_dir)
    if len(reg_stats) == 0:
        raise RuntimeError(
            f"No regular season stats found for {season}. "
            f"Expected files in {data_dir}/team_stats/{season}_Regular_Season_*.csv"
        )

    rows = []
    if not playoff_seedings:
        return pd.DataFrame()

    for conf, seeds in playoff_seedings.items():
        seed_map = {s: tid for s, tid in seeds}
        for higher_seed, lower_seed in [(1, 8), (2, 7), (3, 6), (4, 5)]:
            if higher_seed not in seed_map or lower_seed not in seed_map:
                continue
            team_a = seed_map[higher_seed]  # higher seed has home court
            team_b = seed_map[lower_seed]

            for focal_id, opp_id, home_court, focal_seed, opp_seed in [
                (team_a, team_b, 1, higher_seed, lower_seed),
                (team_b, team_a, 0, lower_seed, higher_seed),
            ]:
                exp_focal = compute_playoff_experience(focal_id, season, all_seasons, playoff_dir)
                exp_opp = compute_playoff_experience(opp_id, season, all_seasons, playoff_dir)
                rows.append({
                    "season": season,
                    "conference": conf,
                    "team_id": focal_id,
                    "opponent_id": opp_id,
                    "team_seed": focal_seed,
                    "opp_seed": opp_seed,
                    "series_id": f"{season}_{conf}_{higher_seed}v{lower_seed}",
                    "series_game_num": 1,
                    "home_court": home_court,
                    "rest_days": 10,
                    "opp_rest_days": 10,
                    "rest_days_diff": 0,
                    "series_wins": 0,
                    "series_losses": 0,
                    "series_lead": 0,
                    "is_bubble": 0,
                    "playoff_exp_3yr": exp_focal,
                    "opp_playoff_exp_3yr": exp_opp,
                    "playoff_exp_diff": exp_focal - exp_opp,
                })

    current_df = pd.DataFrame(rows)
    if len(current_df) == 0:
        return current_df

    stats = reg_stats.copy()
    stats = stats.rename(columns={
        "OFF_RATING": "off_rtg", "DEF_RATING": "def_rtg",
        "NET_RATING": "net_rtg", "PACE": "pace", "W_PCT": "team_win_pct_reg",
    })
    if "net_rtg" not in stats.columns and "off_rtg" in stats.columns:
        stats["net_rtg"] = stats["off_rtg"] - stats["def_rtg"]

    stat_cols = [c for c in [
        "TEAM_ID", "off_rtg", "def_rtg", "net_rtg", "pace",
        "team_win_pct_reg", "three_pt_rate",
    ] if c in stats.columns]
    stats_slim = stats[stat_cols].rename(columns={"TEAM_ID": "team_id"})
    stats_opp = stats_slim.rename(columns={
        "team_id": "opponent_id",
        **{c: f"opp_{c}" for c in stats_slim.columns if c != "team_id"},
    })

    current_df = current_df.merge(stats_slim, on="team_id", how="left")
    current_df = current_df.merge(stats_opp, on="opponent_id", how="left")
    current_df = _build_diff_features(current_df)

    current_df.to_csv(output_path, index=False)
    log.info(f"Saved current season features to {output_path} ({len(current_df)} rows)")
    return current_df
