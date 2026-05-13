"""
data_fetcher.py — Fetch and cache NBA team stats and game logs via nba_api.

All API calls are cached to CSV on first fetch. Subsequent runs load from disk.
Rate limiting: 1s sleep before every call + exponential backoff retries.
"""

import os
import time
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"
]

SEASON_TYPES = ["Regular Season", "Playoffs"]


def _retry_api_call(api_callable, max_retries: int = 3, base_sleep: float = 1.0):
    """Call api_callable with sleep before each attempt and exponential backoff on failure."""
    last_exc = None
    for attempt in range(max_retries):
        sleep_time = base_sleep * (2 ** attempt)
        time.sleep(sleep_time)
        try:
            result = api_callable()
            return result
        except Exception as exc:
            last_exc = exc
            log.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {exc}")
    raise RuntimeError(f"API call failed after {max_retries} attempts: {last_exc}") from last_exc


def fetch_team_advanced_stats(
    season: str,
    season_type: str,
    cache_dir: str = "data/raw/team_stats",
    sleep_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch LeagueDashTeamStats with MeasureType=Advanced for a given season/type.
    Returns: TEAM_ID, TEAM_NAME, OFF_RATING, DEF_RATING, NET_RATING, PACE, W, L, W_PCT
    Caches to disk; loads from cache if present.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    safe_type = season_type.replace(" ", "_")
    cache_path = os.path.join(cache_dir, f"{season}_{safe_type}_advanced.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    from nba_api.stats.endpoints import leaguedashteamstats

    def _call():
        endpoint = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
            timeout=60,
        )
        df = endpoint.get_data_frames()[0]
        if len(df) < 20:
            raise ValueError(f"Only {len(df)} rows returned — likely empty response")
        return df

    df = _retry_api_call(_call, base_sleep=sleep_sec)
    cols = [c for c in ["TEAM_ID", "TEAM_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "W", "L", "W_PCT"] if c in df.columns]
    df = df[cols]
    df.to_csv(cache_path, index=False)
    log.info(f"Cached advanced stats: {cache_path}")
    return df


def fetch_team_basic_stats(
    season: str,
    season_type: str,
    cache_dir: str = "data/raw/team_stats",
    sleep_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch LeagueDashTeamStats with MeasureType=Base for FG3A and FGA (3-point rate).
    Caches to disk; loads from cache if present.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    safe_type = season_type.replace(" ", "_")
    cache_path = os.path.join(cache_dir, f"{season}_{safe_type}_base.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    from nba_api.stats.endpoints import leaguedashteamstats

    def _call():
        endpoint = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
            timeout=60,
        )
        df = endpoint.get_data_frames()[0]
        if len(df) < 20:
            raise ValueError(f"Only {len(df)} rows returned")
        return df

    df = _retry_api_call(_call, base_sleep=sleep_sec)
    cols = [c for c in ["TEAM_ID", "TEAM_NAME", "FG3A", "FGA"] if c in df.columns]
    df = df[cols]
    df.to_csv(cache_path, index=False)
    log.info(f"Cached basic stats: {cache_path}")
    return df


def fetch_game_log(
    team_id: int,
    season: str,
    season_type: str,
    cache_dir: str = "data/raw/game_logs",
    sleep_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch TeamGameLog for a single team/season/season_type.
    Returns: Game_ID, GAME_DATE, MATCHUP, WL + others.
    Caches to disk; loads from cache if present.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    safe_type = season_type.replace(" ", "_")
    cache_path = os.path.join(cache_dir, f"{team_id}_{season}_{safe_type}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df

    from nba_api.stats.endpoints import teamgamelog

    def _call():
        endpoint = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60,
        )
        return endpoint.get_data_frames()[0]

    df = _retry_api_call(_call, base_sleep=sleep_sec)
    if len(df) == 0:
        # No games for this team/season/type — cache empty frame so we don't retry
        df = pd.DataFrame(columns=["Game_ID", "GAME_DATE", "MATCHUP", "WL"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.to_csv(cache_path, index=False)
    return df


def fetch_all_team_ids(
    season: str = "2023-24",
    cache_dir: str = "data/raw",
    sleep_sec: float = 1.0,
) -> list:
    """
    Return list of all 30 NBA team IDs. Cached to team_ids.csv after first fetch.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(cache_dir, "team_ids.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        return df["TEAM_ID"].tolist()

    from nba_api.stats.endpoints import leaguedashteamstats

    def _call():
        endpoint = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
            timeout=60,
        )
        return endpoint.get_data_frames()[0]

    df = _retry_api_call(_call, base_sleep=sleep_sec)
    team_df = df[["TEAM_ID", "TEAM_NAME"]].copy()
    team_df.to_csv(cache_path, index=False)
    log.info(f"Cached {len(team_df)} team IDs to {cache_path}")
    return team_df["TEAM_ID"].tolist()


def fetch_all_seasons(
    seasons: list = None,
    sleep_sec: float = 0.6,
) -> None:
    """
    Master fetch function. Fetches all team stats and game logs for all seasons.
    Caches everything to disk. First run: ~25-40 min. Subsequent runs: instant.
    """
    if seasons is None:
        seasons = SEASONS

    log.info("=== Starting data fetch for all seasons ===")
    log.info(f"Seasons: {seasons}")

    # Step 1: fetch all team IDs
    log.info("Fetching team IDs...")
    team_ids = fetch_all_team_ids(season=seasons[-1], sleep_sec=sleep_sec)
    log.info(f"Found {len(team_ids)} teams")

    # Step 2: fetch team stats for every season — REGULAR SEASON ONLY.
    # Playoff team-level aggregate stats are not used as model features;
    # only regular-season stats are joined onto each playoff game row.
    for season in seasons:
        log.info(f"Fetching advanced stats: {season} Regular Season")
        try:
            fetch_team_advanced_stats(season, "Regular Season", sleep_sec=sleep_sec)
        except Exception as e:
            log.error(f"Failed advanced stats {season} Regular Season: {e}")

        log.info(f"Fetching basic stats: {season} Regular Season")
        try:
            fetch_team_basic_stats(season, "Regular Season", sleep_sec=sleep_sec)
        except Exception as e:
            log.error(f"Failed basic stats {season} Regular Season: {e}")

    # Step 3: fetch game logs for each team, season, and season type
    # Also fetch PlayIn for rest-day lookback (2020-21 onward)
    total_teams = len(team_ids)
    for i, team_id in enumerate(team_ids):
        log.info(f"Fetching game logs for team {team_id} ({i+1}/{total_teams})")
        for season in seasons:
            for stype in SEASON_TYPES:
                try:
                    fetch_game_log(team_id, season, stype, sleep_sec=sleep_sec)
                except Exception as e:
                    log.error(f"Failed game log {team_id} {season} {stype}: {e}")

            # Fetch play-in logs for rest-day computation (2020-21 onward)
            if season >= "2020-21":
                try:
                    fetch_game_log(team_id, season, "PlayIn", sleep_sec=sleep_sec)
                except Exception as e:
                    log.warning(f"No play-in log {team_id} {season}: {e}")

    log.info("=== Data fetch complete ===")


if __name__ == "__main__":
    fetch_all_seasons()
