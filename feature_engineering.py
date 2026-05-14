"""
feature_engineering.py — Build model-ready training and prediction DataFrames.

Each playoff game produces two rows (one per team perspective) with differential
features derived from regular-season stats. Playoff series structure, rest days,
and in-series / season momentum are derived from pre-fetched playoff_games CSV
files (LeagueGameFinder format). No additional API calls are needed.
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

TRAINING_SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Use local data/raw/ first (works on Render); fall back to sibling emv project for dev
_LOCAL_DIR = os.path.join(_THIS_DIR, "data", "raw")
_EMV_DIR   = os.path.normpath(os.path.join(_THIS_DIR, "..", "emv_proj1_Playoff_Predictor", "data", "raw"))

PLAYOFF_DATA_DIR = (
    _LOCAL_DIR
    if os.path.exists(os.path.join(_LOCAL_DIR, "playoff_games_2023-24.csv"))
    else _EMV_DIR
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


def _load_team_game_log(team_id: int, season: str, data_dir: str) -> pd.DataFrame:
    """Load regular season game log CSV sorted oldest-first. Empty DF if not found."""
    path = os.path.join(data_dir, "game_logs", f"{team_id}_{season}_Regular_Season.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values("GAME_DATE")


def _compute_team_efficiency_from_logs(
    team_id: int, season: str, data_dir: str, n_recent: int = 15
) -> dict:
    """TS%, TOV_rate, FT_rate (full season) and recent win% (last n_recent games)."""
    df = _load_team_game_log(team_id, season, data_dir)
    defaults = {"ts_pct": 0.57, "tov_rate": 0.13, "ft_rate": 0.26, "recent_win_pct": 0.5}
    if len(df) == 0 or not {"PTS", "FGA", "FTA", "TOV", "WL"}.issubset(df.columns):
        return defaults
    pts = df["PTS"].sum()
    fga = df["FGA"].sum()
    fta = df["FTA"].sum()
    tov = df["TOV"].sum()
    poss = max(fga + 0.44 * fta + tov, 1)
    return {
        "ts_pct":         float(pts / max(2 * (fga + 0.44 * fta), 1)),
        "tov_rate":       float(tov / poss),
        "ft_rate":        float(fta / max(fga, 1)),
        "recent_win_pct": float((df.tail(n_recent)["WL"] == "W").mean()),
    }


def _compute_h2h_win_pct(
    team_a_id: int, team_b_id: int, season: str, data_dir: str
) -> float:
    """team_a's regular-season win% vs team_b. Returns 0.5 if no H2H games found."""
    df_a = _load_team_game_log(team_a_id, season, data_dir)
    df_b = _load_team_game_log(team_b_id, season, data_dir)
    if len(df_a) == 0 or len(df_b) == 0:
        return 0.5
    shared_gids = set(df_b["Game_ID"].astype(str))
    h2h = df_a[df_a["Game_ID"].astype(str).isin(shared_gids)]
    if len(h2h) == 0:
        return 0.5
    return float((h2h["WL"] == "W").mean())


def _load_player_ratings(season: str, data_dir: str) -> dict:
    """Returns {team_id: avg NET_RATING of top-3 players by minutes}."""
    path = os.path.join(data_dir, "player_stats", f"{season}_Regular_Season_player_advanced.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if not {"TEAM_ID", "MIN", "NET_RATING"}.issubset(df.columns):
        return {}
    result = {}
    for team_id, grp in df.groupby("TEAM_ID"):
        top3 = grp.nlargest(3, "MIN")["NET_RATING"].dropna()
        if len(top3) > 0:
            result[int(team_id)] = float(top3.mean())
    return result


def _compute_playoff_team_stats(pg: pd.DataFrame) -> dict:
    """
    For each (team_id, game_id), compute avg playoff efficiency from all PRIOR games.
    Returns {} if required columns are missing or no team has >= 4 prior games.
    Only entries with prior_games_count >= 4 (completed at least one series) are stored.

    Keys: (team_id: int, game_id: str)
    Values: {playoff_off_rtg, playoff_def_rtg, playoff_net_rtg, playoff_pace, prior_games_count}
    """
    needed = {"TEAM_ID", "GAME_ID", "GAME_DATE", "PTS", "FGA", "FTA", "OREB", "TOV", "MIN", "PLUS_MINUS"}
    if not needed.issubset(set(pg.columns)):
        return {}

    df = pg[list(needed)].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    poss = (df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]).clip(lower=1)
    df["ortg"]   = df["PTS"] / poss * 100
    df["drtg"]   = (df["PTS"] - df["PLUS_MINUS"]) / poss * 100
    df["nrtg"]   = df["ortg"] - df["drtg"]
    df["pace_g"] = poss * 240.0 / df["MIN"].clip(lower=1)
    df["ts_g"]   = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"])).clip(lower=1)
    df["tov_g"]  = df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]).clip(lower=1)

    result = {}
    for team_id, grp in df.groupby("TEAM_ID"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        for col in ["ortg", "drtg", "nrtg", "pace_g", "ts_g", "tov_g"]:
            grp[f"pr_{col}"] = grp[col].shift(1).expanding().mean()

        for idx, row in grp.iterrows():
            if idx == 0 or pd.isna(row["pr_ortg"]):
                continue
            prior_count = int(idx)
            if prior_count < 4:
                continue
            result[(int(team_id), str(row["GAME_ID"]))] = {
                "playoff_off_rtg":   float(row["pr_ortg"]),
                "playoff_def_rtg":   float(row["pr_drtg"]),
                "playoff_net_rtg":   float(row["pr_nrtg"]),
                "playoff_pace":      float(row["pr_pace_g"]),
                "playoff_ts_pct":    float(row["pr_ts_g"]),
                "playoff_tov_rate":  float(row["pr_tov_g"]),
                "prior_games_count": prior_count,
            }
    return result


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two momentum features to a game-level DataFrame (one row per team per game):

    series_pts_diff      — focal team's avg PLUS_MINUS in PREVIOUS games of this
                           specific series. Captures within-series momentum.
                           0 for game 1 of any series.

    prior_playoff_pts_diff — focal team's avg PLUS_MINUS in ALL previous playoff
                             games this season (across any series). Captures overall
                             playoff form coming into the current game. 0 for the
                             team's very first playoff game of the season.
    """
    if "plus_minus" not in df.columns:
        df["series_pts_diff"] = 0.0
        df["prior_playoff_pts_diff"] = 0.0
        return df

    df = df.sort_values(["game_date", "game_id"]).copy()

    # series_pts_diff: expanding mean of PREVIOUS games in same (team, series) group
    df["series_pts_diff"] = (
        df.groupby(["team_id", "series_id"])["plus_minus"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.0)
    )

    # prior_playoff_pts_diff: expanding mean of ALL PREVIOUS playoff games this season
    df["prior_playoff_pts_diff"] = (
        df.groupby("team_id")["plus_minus"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.0)
    )

    return df


def build_series_records(
    season: str,
    data_dir: str = "data/raw",
    playoff_dir: str = None,
) -> pd.DataFrame:
    """
    Reconstruct playoff series from a LeagueGameFinder-format playoff games CSV.
    Returns DataFrame with one row per (team, game) including rest days,
    series context, and momentum features (series_pts_diff, prior_playoff_pts_diff).
    """
    pg = _load_playoff_games(season, playoff_dir)
    if len(pg) == 0:
        return pd.DataFrame()

    # Rest days per team
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

    # PLUS_MINUS lookup for momentum features
    pm_lookup = {}
    if "PLUS_MINUS" in pg.columns:
        for _, row in pg.iterrows():
            pm_lookup[(int(row["TEAM_ID"]), str(row["GAME_ID"]))] = float(row.get("PLUS_MINUS", 0))

    # Per-game playoff efficiency ratings (only populated for games with 4+ prior games)
    playoff_stats_lookup = _compute_playoff_team_stats(pg)

    # Build per-game records
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

    # Group games into series by team pair
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

                po_focal = playoff_stats_lookup.get((focal_id, gid))
                po_opp   = playoff_stats_lookup.get((opp_id,   gid))

                all_rows.append({
                    "season":              season,
                    "game_date":           gdate,
                    "team_id":             focal_id,
                    "opponent_id":         opp_id,
                    "game_id":             gid,
                    "series_id":           series_id,
                    "series_game_num":     game_num,
                    "home_court":          home_court,
                    "won":                 won,
                    "rest_days":           rest,
                    "opp_rest_days":       opp_rest,
                    "rest_days_diff":      rest - opp_rest,
                    "series_wins":         series_wins,
                    "series_losses":       series_losses,
                    "series_lead":         series_wins - series_losses,
                    "is_bubble":           1 if season == BUBBLE_SEASON else 0,
                    "plus_minus":          pm_lookup.get((focal_id, gid), 0.0),
                    # Playoff efficiency ratings (None for R1 / fewer than 4 prior games)
                    "playoff_off_rtg":     po_focal["playoff_off_rtg"]  if po_focal else None,
                    "playoff_def_rtg":     po_focal["playoff_def_rtg"]  if po_focal else None,
                    "playoff_net_rtg":     po_focal["playoff_net_rtg"]  if po_focal else None,
                    "playoff_pace":        po_focal["playoff_pace"]     if po_focal else None,
                    "playoff_ts_pct":      po_focal["playoff_ts_pct"]   if po_focal else None,
                    "playoff_tov_rate":    po_focal["playoff_tov_rate"] if po_focal else None,
                    "has_playoff_stats":   po_focal is not None,
                    "opp_playoff_off_rtg": po_opp["playoff_off_rtg"]    if po_opp else None,
                    "opp_playoff_def_rtg": po_opp["playoff_def_rtg"]    if po_opp else None,
                    "opp_playoff_net_rtg": po_opp["playoff_net_rtg"]    if po_opp else None,
                    "opp_playoff_pace":    po_opp["playoff_pace"]       if po_opp else None,
                    "opp_playoff_ts_pct":  po_opp["playoff_ts_pct"]     if po_opp else None,
                    "opp_playoff_tov_rate":po_opp["playoff_tov_rate"]   if po_opp else None,
                    "opp_has_playoff_stats": po_opp is not None,
                })

            for t in game_records[gid]["teams"]:
                if t["WL"] == "W":
                    if t["TEAM_ID"] == team_a:
                        wins_a += 1
                    else:
                        wins_b += 1

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = _add_momentum_features(df)
    return df


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
    adv_path  = os.path.join(data_dir, "team_stats", f"{season}_Regular_Season_advanced.csv")
    base_path = os.path.join(data_dir, "team_stats", f"{season}_Regular_Season_base.csv")

    if not os.path.exists(adv_path) or not os.path.exists(base_path):
        return pd.DataFrame()

    adv  = pd.read_csv(adv_path)
    base = pd.read_csv(base_path)

    if "FG3A" in base.columns and "FGA" in base.columns:
        base["three_pt_rate"] = base["FG3A"] / base["FGA"].replace(0, np.nan)
    else:
        base["three_pt_rate"] = 0.35

    # Efficiency stats from expanded base stats (PTS, FGA, FTA, TOV per game)
    if all(c in base.columns for c in ["PTS", "FGA", "FTA", "TOV"]):
        base["ts_pct"]   = base["PTS"] / (2 * (base["FGA"] + 0.44 * base["FTA"])).replace(0, np.nan)
        base["tov_rate"] = base["TOV"] / (base["FGA"] + 0.44 * base["FTA"] + base["TOV"]).replace(0, np.nan)
        base["ft_rate"]  = base["FTA"] / base["FGA"].replace(0, np.nan)
    else:
        base["ts_pct"]   = 0.57
        base["tov_rate"] = 0.13
        base["ft_rate"]  = 0.26

    keep = [c for c in ["TEAM_ID", "three_pt_rate", "ts_pct", "tov_rate", "ft_rate"] if c in base.columns]
    merged = adv.merge(base[keep], on="TEAM_ID", how="left")
    merged["three_pt_rate"] = merged["three_pt_rate"].fillna(0.35)
    merged["ts_pct"]        = merged["ts_pct"].fillna(0.57)
    merged["tov_rate"]      = merged["tov_rate"].fillna(0.13)
    merged["ft_rate"]       = merged["ft_rate"].fillna(0.26)
    return merged


def _build_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute differential columns from joined focal + opponent stats."""
    df = df.copy()

    def _diff(focal_col, opp_col, default=0.0):
        a = df[focal_col].fillna(default) if focal_col in df.columns else default
        b = df[opp_col].fillna(default) if opp_col in df.columns else default
        return a - b

    df["off_rtg_diff"]        = _diff("off_rtg",           "opp_def_rtg")
    df["def_rtg_diff"]        = _diff("def_rtg",           "opp_off_rtg")
    df["net_rtg_diff"]        = _diff("net_rtg",           "opp_net_rtg")
    df["pace_diff"]           = _diff("pace",               "opp_pace")
    df["win_pct_diff"]        = _diff("team_win_pct_reg",   "opp_team_win_pct_reg")
    df["three_pt_rate_diff"]  = _diff("three_pt_rate",      "opp_three_pt_rate",  0.35)
    df["playoff_exp_diff"]    = _diff("playoff_exp_3yr",    "opp_playoff_exp_3yr")
    df["ts_pct_diff"]         = _diff("ts_pct",             "opp_ts_pct",         0.57)
    df["tov_rate_diff"]       = _diff("tov_rate",           "opp_tov_rate",       0.13)
    df["ft_rate_diff"]        = _diff("ft_rate",            "opp_ft_rate",        0.26)
    df["recent_win_pct_diff"] = _diff("recent_win_pct",     "opp_recent_win_pct", 0.5)
    df["top3_net_rtg_diff"]   = _diff("top3_net_rtg",       "opp_top3_net_rtg",   0.0)
    # h2h_win_pct is already directional (focal team's perspective) — no diff needed
    if "h2h_win_pct" not in df.columns:
        df["h2h_win_pct"] = 0.5
    return df


def merge_team_stats_onto_games(
    games_df: pd.DataFrame,
    reg_season_stats: pd.DataFrame,
    season: str,
    all_seasons: list,
    playoff_dir: str = None,
    data_dir: str = "data/raw",
) -> pd.DataFrame:
    """Join regular-season stats onto each game row and compute differential features."""
    if len(games_df) == 0 or len(reg_season_stats) == 0:
        return pd.DataFrame()

    stats = reg_season_stats.copy()
    stats = stats.rename(columns={
        "OFF_RATING": "off_rtg",
        "DEF_RATING": "def_rtg",
        "NET_RATING": "net_rtg",
        "PACE":       "pace",
        "W_PCT":      "team_win_pct_reg",
    })
    if "net_rtg" not in stats.columns and "off_rtg" in stats.columns:
        stats["net_rtg"] = stats["off_rtg"] - stats["def_rtg"]

    all_team_ids = pd.concat([games_df["team_id"], games_df["opponent_id"]]).unique()
    exp_map = {
        int(tid): compute_playoff_experience(int(tid), season, all_seasons, playoff_dir)
        for tid in all_team_ids
    }
    stats["playoff_exp_3yr"] = stats["TEAM_ID"].map(exp_map).fillna(0).astype(int)

    stat_cols = [c for c in [
        "TEAM_ID", "off_rtg", "def_rtg", "net_rtg", "pace",
        "team_win_pct_reg", "three_pt_rate", "playoff_exp_3yr",
        "ts_pct", "tov_rate", "ft_rate",
    ] if c in stats.columns]

    stats_focal = stats[stat_cols].rename(columns={"TEAM_ID": "team_id"})
    stats_opp   = stats_focal.rename(columns={
        "team_id": "opponent_id",
        **{c: f"opp_{c}" for c in stats_focal.columns if c != "team_id"},
    })

    df = games_df.merge(stats_focal, on="team_id",   how="left")
    df = df.merge(stats_opp,         on="opponent_id", how="left")

    # Override reg season efficiency with playoff-computed stats for R2+ games
    for reg_col, po_col in [
        ("off_rtg",  "playoff_off_rtg"),  ("def_rtg",  "playoff_def_rtg"),
        ("net_rtg",  "playoff_net_rtg"),  ("pace",     "playoff_pace"),
        ("ts_pct",   "playoff_ts_pct"),   ("tov_rate", "playoff_tov_rate"),
    ]:
        if po_col in df.columns:
            mask = df.get("has_playoff_stats", pd.Series(False, index=df.index)).fillna(False)
            df.loc[mask, reg_col] = df.loc[mask, po_col]

    for reg_col, po_col in [
        ("opp_off_rtg",  "opp_playoff_off_rtg"),  ("opp_def_rtg",  "opp_playoff_def_rtg"),
        ("opp_net_rtg",  "opp_playoff_net_rtg"),  ("opp_pace",     "opp_playoff_pace"),
        ("opp_ts_pct",   "opp_playoff_ts_pct"),   ("opp_tov_rate", "opp_playoff_tov_rate"),
    ]:
        if po_col in df.columns:
            mask = df.get("opp_has_playoff_stats", pd.Series(False, index=df.index)).fillna(False)
            df.loc[mask, reg_col] = df.loc[mask, po_col]

    # ── Game-log-based features (recent form, H2H) ────────────────────────────
    all_team_ids = list(pd.concat([df["team_id"], df["opponent_id"]]).unique())

    eff_cache = {int(tid): _compute_team_efficiency_from_logs(int(tid), season, data_dir)
                 for tid in all_team_ids}

    def _eff(col, tid, default):
        return eff_cache.get(int(tid), {}).get(col, default)

    df["recent_win_pct"]     = df["team_id"].map(lambda x: _eff("recent_win_pct", x, 0.5))
    df["opp_recent_win_pct"] = df["opponent_id"].map(lambda x: _eff("recent_win_pct", x, 0.5))

    # Player star-power ratings
    player_ratings = _load_player_ratings(season, data_dir)
    df["top3_net_rtg"]     = df["team_id"].map(lambda x: player_ratings.get(int(x), 0.0))
    df["opp_top3_net_rtg"] = df["opponent_id"].map(lambda x: player_ratings.get(int(x), 0.0))

    # H2H win% (computed once per unique pair)
    h2h_cache: dict = {}
    def _h2h(focal_id, opp_id):
        key = (int(focal_id), int(opp_id))
        if key not in h2h_cache:
            h2h_cache[key] = _compute_h2h_win_pct(int(focal_id), int(opp_id), season, data_dir)
        return h2h_cache[key]

    df["h2h_win_pct"] = [_h2h(r["team_id"], r["opponent_id"]) for _, r in df.iterrows()]

    return _build_diff_features(df)


def build_training_dataframe(
    seasons: list = None,
    data_dir: str = "data/raw",
    output_path: str = "data/processed/training_data.csv",
    playoff_dir: str = None,
) -> pd.DataFrame:
    """
    Build complete training DataFrame from all cached season data.
    Saves to output_path and returns the DataFrame.
    """
    if seasons is None:
        seasons = TRAINING_SEASONS
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

        enriched = merge_team_stats_onto_games(series_df, reg_stats, season, seasons, playoff_dir, data_dir)
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
    """Build feature rows for first-round playoff matchups of the current season."""
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
            team_a = seed_map[higher_seed]
            team_b = seed_map[lower_seed]

            for focal_id, opp_id, home_court, focal_seed, opp_seed in [
                (team_a, team_b, 1, higher_seed, lower_seed),
                (team_b, team_a, 0, lower_seed, higher_seed),
            ]:
                exp_focal = compute_playoff_experience(focal_id, season, all_seasons, playoff_dir)
                exp_opp   = compute_playoff_experience(opp_id,   season, all_seasons, playoff_dir)
                rows.append({
                    "season":                 season,
                    "conference":             conf,
                    "team_id":                focal_id,
                    "opponent_id":            opp_id,
                    "team_seed":              focal_seed,
                    "opp_seed":               opp_seed,
                    "series_id":              f"{season}_{conf}_{higher_seed}v{lower_seed}",
                    "series_game_num":        1,
                    "home_court":             home_court,
                    "rest_days":              10,
                    "opp_rest_days":          10,
                    "rest_days_diff":         0,
                    "series_wins":            0,
                    "series_losses":          0,
                    "series_lead":            0,
                    "is_bubble":              0,
                    "playoff_exp_3yr":        exp_focal,
                    "opp_playoff_exp_3yr":    exp_opp,
                    "playoff_exp_diff":       exp_focal - exp_opp,
                    "series_pts_diff":        0.0,
                    "prior_playoff_pts_diff": 0.0,
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
    stats_opp  = stats_slim.rename(columns={
        "team_id": "opponent_id",
        **{c: f"opp_{c}" for c in stats_slim.columns if c != "team_id"},
    })

    current_df = current_df.merge(stats_slim, on="team_id",   how="left")
    current_df = current_df.merge(stats_opp,  on="opponent_id", how="left")

    # Game-log features for current season (recent form, H2H, player ratings)
    all_team_ids = list(pd.concat([current_df["team_id"], current_df["opponent_id"]]).unique())
    eff_cache = {int(tid): _compute_team_efficiency_from_logs(int(tid), season, data_dir)
                 for tid in all_team_ids}
    def _eff(col, tid, default):
        return eff_cache.get(int(tid), {}).get(col, default)

    current_df["recent_win_pct"]     = current_df["team_id"].map(lambda x: _eff("recent_win_pct", x, 0.5))
    current_df["opp_recent_win_pct"] = current_df["opponent_id"].map(lambda x: _eff("recent_win_pct", x, 0.5))

    player_ratings = _load_player_ratings(season, data_dir)
    current_df["top3_net_rtg"]     = current_df["team_id"].map(lambda x: player_ratings.get(int(x), 0.0))
    current_df["opp_top3_net_rtg"] = current_df["opponent_id"].map(lambda x: player_ratings.get(int(x), 0.0))

    h2h_cache: dict = {}
    def _h2h(focal_id, opp_id):
        key = (int(focal_id), int(opp_id))
        if key not in h2h_cache:
            h2h_cache[key] = _compute_h2h_win_pct(int(focal_id), int(opp_id), season, data_dir)
        return h2h_cache[key]
    current_df["h2h_win_pct"] = [_h2h(r["team_id"], r["opponent_id"]) for _, r in current_df.iterrows()]

    current_df = _build_diff_features(current_df)

    current_df.to_csv(output_path, index=False)
    log.info(f"Saved current season features to {output_path} ({len(current_df)} rows)")
    return current_df
