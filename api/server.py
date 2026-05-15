"""
api/server.py — FastAPI backend for the NBA Playoff Predictor web app.

Run from the project root:
    uvicorn api.server:app --reload --port 8000
"""

import copy
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import MODEL_FEATURES
from feature_engineering import compute_playoff_experience

log = logging.getLogger("uvicorn.error")

ROOT        = Path(__file__).parent.parent
MODEL_PATH  = ROOT / "models" / "logistic_model.joblib"
DATA_DIR    = ROOT / "data" / "raw"
CACHE_FILE  = ROOT / "data" / "cache" / "playoff_games_2025-26.json"
CACHE_TTL   = 3600

CURRENT_SEASON         = "2025-26"
HOME_GAMES_HIGHER_SEED = {1, 2, 5, 7}

ALL_SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

# Abbreviation lookup for ESPN injury API mapping
_TEAM_ABBR: dict[int, str] = {
    1610612760: "OKC", 1610612759: "SAS", 1610612743: "DEN", 1610612747: "LAL",
    1610612745: "HOU", 1610612750: "MIN", 1610612756: "PHX", 1610612757: "POR",
    1610612765: "DET", 1610612752: "NYK", 1610612739: "CLE", 1610612738: "BOS",
    1610612755: "PHI", 1610612737: "ATL", 1610612753: "ORL", 1610612761: "TOR",
}

_model = joblib.load(MODEL_PATH)

def _load_team_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    adv  = pd.read_csv(DATA_DIR / "team_stats" / f"{season}_Regular_Season_advanced.csv")
    base = pd.read_csv(DATA_DIR / "team_stats" / f"{season}_Regular_Season_base.csv")
    base["three_pt_rate"] = base["FG3A"] / base["FGA"].replace(0, np.nan)
    if all(c in base.columns for c in ["PTS", "FGA", "FTA", "TOV"]):
        base["ts_pct"]   = base["PTS"] / (2 * (base["FGA"] + 0.44 * base["FTA"])).replace(0, np.nan)
        base["tov_rate"] = base["TOV"] / (base["FGA"] + 0.44 * base["FTA"] + base["TOV"]).replace(0, np.nan)
        base["ft_rate"]  = base["FTA"] / base["FGA"].replace(0, np.nan)
    keep = [c for c in ["TEAM_ID", "three_pt_rate", "ts_pct", "tov_rate", "ft_rate"] if c in base.columns]
    return adv.merge(base[keep], on="TEAM_ID")

_team_stats = _load_team_stats()


def _load_player_ratings_for_season(season: str = CURRENT_SEASON) -> dict:
    """Returns {team_id: avg NET_RATING of top-3 players by minutes}."""
    path = DATA_DIR / "player_stats" / f"{season}_Regular_Season_player_advanced.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    result = {}
    for team_id, grp in df.groupby("TEAM_ID"):
        top3 = grp.nlargest(3, "MIN")["NET_RATING"].dropna()
        if len(top3) > 0:
            result[int(team_id)] = float(top3.mean())
    return result

_player_ratings = _load_player_ratings_for_season()


def _load_player_ratings_by_name(season: str = CURRENT_SEASON) -> dict:
    """Returns {player_name_lower: net_rating} for injury weighting."""
    path = DATA_DIR / "player_stats" / f"{season}_Regular_Season_player_advanced.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "NET_RATING" not in df.columns or "PLAYER_NAME" not in df.columns:
        return {}
    return {
        str(row["PLAYER_NAME"]).lower().strip(): float(row["NET_RATING"])
        for _, row in df.iterrows()
        if not pd.isna(row["NET_RATING"])
    }

_player_ratings_by_name = _load_player_ratings_by_name()

# ── 2025-26 bracket (R1 results hardcoded; R2+ enriched from live data) ────────
# Bracket halves (group field):
#   group 0 → 1-seed half  (1v8, 4v5 → Semis game 0)
#   group 1 → 2-seed half  (2v7, 3v6 → Semis game 1)
#
# Team IDs:
#   OKC=1610612760  SAS=1610612759  DEN=1610612743  LAL=1610612747
#   HOU=1610612745  MIN=1610612750  PHX=1610612756  POR=1610612757
#   DET=1610612765  NYK=1610612752  CLE=1610612739  BOS=1610612738
#   PHI=1610612755  ATL=1610612737  ORL=1610612753  TOR=1610612761

BRACKET_STRUCTURE = {
    "West": {
        "r1": [
            # group 0 — upper half
            {"group": 0, "seed_a": 1, "id_a": 1610612760, "name_a": "Thunder",      "abbr_a": "OKC",
             "seed_b": 8, "id_b": 1610612756, "name_b": "Suns",          "abbr_b": "PHX",
             "wins_a": 4, "wins_b": 0, "completed": True, "winner_id": 1610612760},
            {"group": 0, "seed_a": 4, "id_a": 1610612747, "name_a": "Lakers",       "abbr_a": "LAL",
             "seed_b": 5, "id_b": 1610612745, "name_b": "Rockets",       "abbr_b": "HOU",
             "wins_a": 4, "wins_b": 2, "completed": True, "winner_id": 1610612747},
            # group 1 — lower half
            {"group": 1, "seed_a": 2, "id_a": 1610612759, "name_a": "Spurs",        "abbr_a": "SAS",
             "seed_b": 7, "id_b": 1610612757, "name_b": "Trail Blazers", "abbr_b": "POR",
             "wins_a": 4, "wins_b": 1, "completed": True, "winner_id": 1610612759},
            {"group": 1, "seed_a": 3, "id_a": 1610612743, "name_a": "Nuggets",      "abbr_a": "DEN",
             "seed_b": 6, "id_b": 1610612750, "name_b": "Timberwolves",  "abbr_b": "MIN",
             "wins_a": 2, "wins_b": 4, "completed": True, "winner_id": 1610612750},
        ],
        "r2": [
            # group 0: OKC (won 1v8) vs LAL (won 4v5)
            {"group": 0, "seed_a": 1, "id_a": 1610612760, "name_a": "Thunder",      "abbr_a": "OKC",
             "seed_b": 4, "id_b": 1610612747, "name_b": "Lakers",        "abbr_b": "LAL",
             "wins_a": 0, "wins_b": 0, "completed": False},
            # group 1: SAS (won 2v7) vs MIN (upset 3v6)
            {"group": 1, "seed_a": 2, "id_a": 1610612759, "name_a": "Spurs",        "abbr_a": "SAS",
             "seed_b": 6, "id_b": 1610612750, "name_b": "Timberwolves",  "abbr_b": "MIN",
             "wins_a": 0, "wins_b": 0, "completed": False},
        ],
        "r3": [
            # West Conference Finals — participants TBD from R2 winners
            {"seed_a": None, "id_a": None, "name_a": "TBD", "abbr_a": "—",
             "seed_b": None, "id_b": None, "name_b": "TBD", "abbr_b": "—",
             "wins_a": 0, "wins_b": 0, "completed": False},
        ],
    },
    "East": {
        "r1": [
            # group 0 — upper half: 1-seed + 4-seed
            {"group": 0, "seed_a": 1, "id_a": 1610612765, "name_a": "Pistons",      "abbr_a": "DET",
             "seed_b": 8, "id_b": 1610612753, "name_b": "Magic",         "abbr_b": "ORL",
             "wins_a": 4, "wins_b": 3, "completed": True, "winner_id": 1610612765},
            {"group": 0, "seed_a": 4, "id_a": 1610612739, "name_a": "Cavaliers",    "abbr_a": "CLE",
             "seed_b": 5, "id_b": 1610612761, "name_b": "Raptors",       "abbr_b": "TOR",
             "wins_a": 4, "wins_b": 3, "completed": True, "winner_id": 1610612739},
            # group 1 — lower half: 2-seed + 3-seed
            {"group": 1, "seed_a": 2, "id_a": 1610612752, "name_a": "Knicks",       "abbr_a": "NYK",
             "seed_b": 7, "id_b": 1610612737, "name_b": "Hawks",         "abbr_b": "ATL",
             "wins_a": 4, "wins_b": 1, "completed": True, "winner_id": 1610612752},
            {"group": 1, "seed_a": 3, "id_a": 1610612738, "name_a": "Celtics",      "abbr_a": "BOS",
             "seed_b": 6, "id_b": 1610612755, "name_b": "76ers",         "abbr_b": "PHI",
             "wins_a": 3, "wins_b": 4, "completed": True, "winner_id": 1610612755},
        ],
        "r2": [
            # group 0: DET (won 1v8) vs CLE (won 4v5)
            {"group": 0, "seed_a": 1, "id_a": 1610612765, "name_a": "Pistons",      "abbr_a": "DET",
             "seed_b": 4, "id_b": 1610612739, "name_b": "Cavaliers",     "abbr_b": "CLE",
             "wins_a": 0, "wins_b": 0, "completed": False},
            # group 1: NYK (won 2v7) vs PHI (upset 3v6)
            {"group": 1, "seed_a": 2, "id_a": 1610612752, "name_a": "Knicks",       "abbr_a": "NYK",
             "seed_b": 6, "id_b": 1610612755, "name_b": "76ers",         "abbr_b": "PHI",
             "wins_a": 0, "wins_b": 0, "completed": False},
        ],
        "r3": [
            # East Conference Finals — participants TBD from R2 winners
            {"seed_a": None, "id_a": None, "name_a": "TBD", "abbr_a": "—",
             "seed_b": None, "id_b": None, "name_b": "TBD", "abbr_b": "—",
             "wins_a": 0, "wins_b": 0, "completed": False},
        ],
    },
    "finals": {
        "seed_a": None, "id_a": None, "name_a": "TBD", "abbr_a": "—",
        "seed_b": None, "id_b": None, "name_b": "TBD", "abbr_b": "—",
        "wins_a": 0, "wins_b": 0, "completed": False,
    },
}


# ── Live data fetch & enrichment ───────────────────────────────────────────────

def _fetch_live_series_scores() -> dict:
    """Fetch 2025-26 playoff game results (single API call, cached 1 hr)."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        if time.time() - cached.get("timestamp", 0) < CACHE_TTL:
            return _parse_series_scores(pd.DataFrame(cached["games"]))

    try:
        from nba_api.stats.endpoints import leaguegamefinder
        time.sleep(0.6)
        df = leaguegamefinder.LeagueGameFinder(
            season_nullable=CURRENT_SEASON,
            season_type_nullable="Playoffs",
            timeout=30,
        ).get_data_frames()[0]
        if len(df) == 0:
            raise ValueError("Empty response")

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
        with open(CACHE_FILE, "w") as f:
            json.dump({"timestamp": time.time(), "games": df.to_dict("records")}, f)
        return _parse_series_scores(df)

    except Exception as e:
        log.warning(f"Live fetch failed ({e})")
        if CACHE_FILE.exists():
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            return _parse_series_scores(pd.DataFrame(cached["games"]))
        return {}


def _parse_series_scores(df: pd.DataFrame) -> dict:
    """Returns {'{id_lo}_{id_hi}': {wins_a, wins_b, last_date}} for each series."""
    if len(df) == 0:
        return {}
    game_records: dict = {}
    for _, row in df.iterrows():
        gid = str(row["GAME_ID"])
        if gid not in game_records:
            game_records[gid] = {"date": row.get("GAME_DATE", ""), "teams": []}
        game_records[gid]["teams"].append({"id": int(row["TEAM_ID"]), "wl": str(row.get("WL", ""))})

    series: dict = {}
    for gid, info in game_records.items():
        teams = [t for t in info["teams"] if t["id"] > 0]
        if len(teams) < 2:
            continue
        id_a, id_b = sorted(t["id"] for t in teams[:2])
        key = f"{id_a}_{id_b}"
        if key not in series:
            series[key] = {"id_a": id_a, "id_b": id_b, "wins_a": 0, "wins_b": 0, "last_date": ""}
        for t in teams:
            if t["wl"] == "W":
                if t["id"] == id_a:
                    series[key]["wins_a"] += 1
                else:
                    series[key]["wins_b"] += 1
        series[key]["last_date"] = info["date"]
    return series


def _get_last_game_dates() -> dict:
    """Returns {team_id: pd.Timestamp} of each team's most recent playoff game."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        df = pd.DataFrame(cached.get("games", []))
        if len(df) == 0:
            return {}
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return {int(tid): ts for tid, ts in df.groupby("TEAM_ID")["GAME_DATE"].max().items()}
    except Exception:
        return {}


def _compute_series_momentum(team_a_id: int, team_b_id: int) -> tuple:
    """
    Returns (series_pts_diff_a, series_pts_diff_b) from the live game cache.
    Each value is that team's average PLUS_MINUS in previous games of this series.
    """
    if not CACHE_FILE.exists():
        return 0.0, 0.0
    try:
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        games = cached.get("games", [])
        if not games:
            return 0.0, 0.0

        from collections import defaultdict
        game_teams: dict = defaultdict(set)
        game_pm: dict = {}
        for row in games:
            gid = str(row["GAME_ID"])
            tid = int(row["TEAM_ID"])
            game_teams[gid].add(tid)
            game_pm[(tid, gid)] = float(row.get("PLUS_MINUS", 0))

        series_gids = sorted(
            gid for gid, teams in game_teams.items()
            if team_a_id in teams and team_b_id in teams
        )
        if not series_gids:
            return 0.0, 0.0

        pm_a = [game_pm.get((team_a_id, gid), 0.0) for gid in series_gids]
        pm_b = [game_pm.get((team_b_id, gid), 0.0) for gid in series_gids]
        return float(np.mean(pm_a)), float(np.mean(pm_b))
    except Exception as e:
        log.warning(f"Series momentum failed: {e}")
        return 0.0, 0.0


def _compute_prior_playoff_pts(team_id: int) -> float:
    """Average PLUS_MINUS for a team across all their games in the current playoffs."""
    if not CACHE_FILE.exists():
        return 0.0
    try:
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        values = [
            float(g.get("PLUS_MINUS", 0))
            for g in cached.get("games", [])
            if int(g["TEAM_ID"]) == team_id
        ]
        return float(np.mean(values)) if values else 0.0
    except Exception:
        return 0.0


def _compute_playoff_team_ratings(team_id: int) -> dict | None:
    """
    Avg efficiency ratings from the team's playoff games this season.
    Returns None if fewer than 4 games played (not enough data for R2+ override).
    """
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        team_games = [g for g in cached.get("games", []) if int(g["TEAM_ID"]) == team_id]
        if len(team_games) < 4:
            return None

        off_rtgs, def_rtgs, paces = [], [], []
        for g in team_games:
            pts  = float(g.get("PTS",  0) or 0)
            fga  = float(g.get("FGA",  0) or 0)
            fta  = float(g.get("FTA",  0) or 0)
            oreb = float(g.get("OREB", 0) or 0)
            tov  = float(g.get("TOV",  0) or 0)
            mins = float(g.get("MIN", 240) or 240)
            pm   = float(g.get("PLUS_MINUS", 0) or 0)

            poss    = max(fga - oreb + tov + 0.44 * fta, 1.0)
            off_rtg = pts / poss * 100
            def_rtg = (pts - pm) / poss * 100
            pace    = poss * 240.0 / max(mins, 1.0)
            off_rtgs.append(off_rtg)
            def_rtgs.append(def_rtg)
            paces.append(pace)

        avg_off = float(np.mean(off_rtgs))
        avg_def = float(np.mean(def_rtgs))
        return {"off_rtg": avg_off, "def_rtg": avg_def,
                "net_rtg": avg_off - avg_def, "pace": float(np.mean(paces))}
    except Exception as e:
        log.warning(f"Playoff ratings failed (team {team_id}): {e}")
        return None


def _injury_player_weight(display_name: str) -> float:
    """Weight an injured player by their NET_RATING relative to league average.
    Unknown players default to 1.0. Range clamped to [0.3, 3.0].
    """
    net_rtg = _player_ratings_by_name.get(display_name.lower().strip())
    if net_rtg is None:
        return 1.0
    return float(np.clip(1.0 + net_rtg / 15.0, 0.3, 3.0))


def _fetch_injury_burden(abbr: str) -> float:
    """
    Returns a [0, 1] injury burden for a team via the ESPN public injury API.
    Players are weighted by their NET_RATING so losing a star hurts more than
    losing a bench player. Gracefully returns 0.0 on any failure.
    """
    if not abbr:
        return 0.0
    try:
        import urllib.request
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())

        burden = 0.0
        for team_entry in data.get("injuries", []):
            if team_entry.get("team", {}).get("abbreviation", "").upper() == abbr.upper():
                for item in team_entry.get("items", []):
                    status = item.get("status", "").lower()
                    name   = item.get("athlete", {}).get("displayName", "")
                    weight = _injury_player_weight(name)
                    if "out" in status:
                        burden += 1.0 * weight
                    elif "doubtful" in status:
                        burden += 0.75 * weight
                    elif "questionable" in status:
                        burden += 0.25 * weight
                break

        return min(burden / 5.0, 1.0)
    except Exception as e:
        log.warning(f"Injury fetch failed ({abbr}): {e}")
        return 0.0


# ── Regular season game log cache (for recent form + h2h) ─────────────────────
# Regular season is over during playoffs, so 24h TTL is sufficient.

_game_log_cache: dict[int, tuple] = {}  # team_id → (df, timestamp)
_GAME_LOG_TTL = 86400  # 24 hours


def _fetch_team_season_log(team_id: int) -> pd.DataFrame:
    """Fetch current regular season game log for a team. Cached in-memory for 24h."""
    cached = _game_log_cache.get(team_id)
    if cached and (time.time() - cached[1]) < _GAME_LOG_TTL:
        return cached[0]
    try:
        from nba_api.stats.endpoints import teamgamelog
        time.sleep(0.6)
        df = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=CURRENT_SEASON,
            season_type_all_star="Regular Season",
            timeout=15,
        ).get_data_frames()[0]
        _game_log_cache[team_id] = (df, time.time())
        return df
    except Exception as e:
        log.warning(f"Game log fetch failed (team {team_id}): {e}")
        return pd.DataFrame()


def _recent_win_pct(team_id: int, n: int = 10) -> float:
    """Win% of last n regular season games. Returns 0.5 on failure."""
    df = _fetch_team_season_log(team_id)
    if len(df) == 0 or "WL" not in df.columns:
        return 0.5
    return float((df.head(n)["WL"] == "W").mean())


def _h2h_win_pct(team_a_id: int, team_b_id: int) -> float:
    """Team A's win% vs Team B in the current regular season. Returns 0.5 if no games."""
    abbr_b = _TEAM_ABBR.get(team_b_id, "")
    if not abbr_b:
        return 0.5
    df = _fetch_team_season_log(team_a_id)
    if len(df) == 0 or "MATCHUP" not in df.columns:
        return 0.5
    h2h = df[df["MATCHUP"].str.contains(abbr_b, na=False)]
    if len(h2h) == 0:
        return 0.5
    return float((h2h["WL"] == "W").mean())


def _live_key(id_a: int, id_b: int) -> str:
    return f"{min(id_a, id_b)}_{max(id_a, id_b)}"


def _apply_live(s: dict, live: dict) -> None:
    """Mutate series slot s with live win counts if available."""
    if s.get("id_a") is None or s.get("id_b") is None:
        return
    key = _live_key(s["id_a"], s["id_b"])
    if key not in live:
        return
    lv = live[key]
    if s["id_a"] < s["id_b"]:
        s["wins_a"], s["wins_b"] = lv["wins_a"], lv["wins_b"]
    else:
        s["wins_a"], s["wins_b"] = lv["wins_b"], lv["wins_a"]
    if max(s["wins_a"], s["wins_b"]) >= 4:
        s["completed"]  = True
        s["winner_id"]  = s["id_a"] if s["wins_a"] > s["wins_b"] else s["id_b"]
        s["winner_name"] = s["name_a"] if s["wins_a"] > s["wins_b"] else s["name_b"]


def _winner_of(series: dict) -> dict | None:
    """Return {id, name, seed, abbr} for the winner of a completed series, or None."""
    if not series.get("completed"):
        return None
    if series.get("wins_a", 0) > series.get("wins_b", 0):
        return {"id": series["id_a"], "name": series["name_a"],
                "seed": series["seed_a"], "abbr": series["abbr_a"]}
    return {"id": series["id_b"], "name": series["name_b"],
            "seed": series["seed_b"], "abbr": series["abbr_b"]}


def _fill_tbd(slot: dict, team: dict | None, side: str) -> None:
    """Fill a TBD side ('a' or 'b') of a series slot once the team is known."""
    if team is None:
        return
    slot[f"id_{side}"]   = team["id"]
    slot[f"name_{side}"] = team["name"]
    slot[f"seed_{side}"] = team.get("seed")
    slot[f"abbr_{side}"] = team.get("abbr", "")


def _build_bracket() -> dict:
    structure = copy.deepcopy(BRACKET_STRUCTURE)
    live = _fetch_live_series_scores()

    for conf in ("West", "East"):
        # Enrich R2
        for s in structure[conf]["r2"]:
            s.setdefault("wins_a", 0)
            s.setdefault("wins_b", 0)
            s.setdefault("completed", False)
            s.setdefault("winner_id", None)
            _apply_live(s, live)

        # Derive R3 (Conference Finals) participants from R2 winners
        r2 = structure[conf]["r2"]
        cf = structure[conf]["r3"][0]
        cf.setdefault("wins_a", 0)
        cf.setdefault("wins_b", 0)
        cf.setdefault("completed", False)

        w0 = _winner_of(r2[0])  # group 0 winner → CF side a
        w1 = _winner_of(r2[1])  # group 1 winner → CF side b
        _fill_tbd(cf, w0, "a")
        _fill_tbd(cf, w1, "b")
        _apply_live(cf, live)

    # Derive Finals participants from Conference Finals winners
    finals = structure["finals"]
    finals.setdefault("wins_a", 0)
    finals.setdefault("wins_b", 0)
    finals.setdefault("completed", False)

    wcf_winner = _winner_of(structure["West"]["r3"][0])
    ecf_winner = _winner_of(structure["East"]["r3"][0])
    _fill_tbd(finals, wcf_winner, "a")
    _fill_tbd(finals, ecf_winner, "b")
    _apply_live(finals, live)

    return structure


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="NBA Playoff Predictor")

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/bracket")
def get_bracket():
    return _build_bracket()


_REFRESH_SECRET = os.environ.get("REFRESH_SECRET", "")

@app.post("/api/refresh")
def refresh_bracket(x_refresh_token: Optional[str] = Header(default=None)):
    if _REFRESH_SECRET and x_refresh_token != _REFRESH_SECRET:
        raise HTTPException(403, "Invalid or missing X-Refresh-Token header")
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    return _build_bracket()


class PredictRequest(BaseModel):
    team_a_id: int
    team_b_id: int
    seed_a: int
    seed_b: int
    wins_a: int
    wins_b: int
    game_num: Optional[int] = None


class PredictResponse(BaseModel):
    team_a_id: int
    team_b_id: int
    team_a_name: str
    team_b_name: str
    prob_a: float
    prob_b: float
    predicted_winner_id: int
    predicted_winner_name: str
    game_num: int
    home_team: str
    features: dict


def _get_stats(team_id: int) -> pd.Series:
    row = _team_stats[_team_stats["TEAM_ID"] == team_id]
    if len(row) == 0:
        raise HTTPException(404, f"Team {team_id} not in stats")
    return row.iloc[0]


def _name(team_id: int) -> str:
    row = _team_stats[_team_stats["TEAM_ID"] == team_id]
    return str(row.iloc[0]["TEAM_NAME"]) if len(row) else str(team_id)


@app.post("/api/predict", response_model=PredictResponse)
def predict_game(req: PredictRequest):
    sa = _get_stats(req.team_a_id)
    sb = _get_stats(req.team_b_id)

    game_num = req.game_num or (req.wins_a + req.wins_b + 1)
    higher_id = req.team_a_id if req.seed_a < req.seed_b else req.team_b_id
    home_a = (higher_id == req.team_a_id) == (game_num in HOME_GAMES_HIGHER_SEED)

    # ── Rest days (from live game cache) ──────────────────────────────────────
    today = pd.Timestamp.today().normalize()
    last_dates = _get_last_game_dates()
    date_a = last_dates.get(req.team_a_id)
    date_b = last_dates.get(req.team_b_id)
    rest_days_diff = int((today - date_a).days - (today - date_b).days) if date_a and date_b else 0

    # ── Playoff experience (prior 3 seasons) ──────────────────────────────────
    exp_a = compute_playoff_experience(req.team_a_id, CURRENT_SEASON, ALL_SEASONS, str(DATA_DIR))
    exp_b = compute_playoff_experience(req.team_b_id, CURRENT_SEASON, ALL_SEASONS, str(DATA_DIR))

    # ── In-series & season momentum (from live game cache) ───────────────────
    series_pm_a, series_pm_b = _compute_series_momentum(req.team_a_id, req.team_b_id)
    prior_pm_a = _compute_prior_playoff_pts(req.team_a_id)
    prior_pm_b = _compute_prior_playoff_pts(req.team_b_id)

    # ── Playoff efficiency ratings (replace reg season for R2+ teams) ─────────
    po_rtg_a = _compute_playoff_team_ratings(req.team_a_id)
    po_rtg_b = _compute_playoff_team_ratings(req.team_b_id)

    def g(s, col, d=0.0):
        return float(s[col]) if col in s.index and not pd.isna(s[col]) else d

    off_a = po_rtg_a["off_rtg"] if po_rtg_a else g(sa, "OFF_RATING")
    def_a = po_rtg_a["def_rtg"] if po_rtg_a else g(sa, "DEF_RATING")
    net_a = po_rtg_a["net_rtg"] if po_rtg_a else g(sa, "NET_RATING")
    pac_a = po_rtg_a["pace"]    if po_rtg_a else g(sa, "PACE")

    off_b = po_rtg_b["off_rtg"] if po_rtg_b else g(sb, "OFF_RATING")
    def_b = po_rtg_b["def_rtg"] if po_rtg_b else g(sb, "DEF_RATING")
    net_b = po_rtg_b["net_rtg"] if po_rtg_b else g(sb, "NET_RATING")
    pac_b = po_rtg_b["pace"]    if po_rtg_b else g(sb, "PACE")

    ts_a  = g(sa, "ts_pct",   0.57);  ts_b  = g(sb, "ts_pct",   0.57)
    tov_a = g(sa, "tov_rate", 0.13);  tov_b = g(sb, "tov_rate", 0.13)
    ft_a  = g(sa, "ft_rate",  0.26);  ft_b  = g(sb, "ft_rate",  0.26)
    top3_a = _player_ratings.get(req.team_a_id, 0.0)
    top3_b = _player_ratings.get(req.team_b_id, 0.0)

    features = {
        "net_rtg_diff":            net_a - net_b,
        "off_rtg_diff":            off_a - def_b,
        "def_rtg_diff":            def_a - off_b,
        "pace_diff":               pac_a - pac_b,
        "rest_days_diff":          rest_days_diff,
        "home_court":              int(home_a),
        "win_pct_diff":            g(sa, "W_PCT") - g(sb, "W_PCT"),
        "series_game_num":         game_num,
        "series_lead":             req.wins_a - req.wins_b,
        "three_pt_rate_diff":      g(sa, "three_pt_rate", .35) - g(sb, "three_pt_rate", .35),
        "playoff_exp_diff":        float(exp_a - exp_b),
        "is_bubble":               0,
        "series_pts_diff":         round(series_pm_a - series_pm_b, 2),
        "prior_playoff_pts_diff":  round(prior_pm_a  - prior_pm_b,  2),
        "ts_pct_diff":             round(ts_a  - ts_b,  4),
        "tov_rate_diff":           round(tov_a - tov_b, 4),
        "ft_rate_diff":            round(ft_a  - ft_b,  4),
        "recent_win_pct_diff":     round(_recent_win_pct(req.team_a_id) - _recent_win_pct(req.team_b_id), 4),
        "h2h_win_pct":             round(_h2h_win_pct(req.team_a_id, req.team_b_id), 4),
        "top3_net_rtg_diff":       round(top3_a - top3_b, 4),
    }

    prob_a = float(_model.predict_proba(pd.DataFrame([features])[MODEL_FEATURES])[0, 1])

    # ── Injury adjustment (post-model, ESPN API) ──────────────────────────────
    abbr_a = _TEAM_ABBR.get(req.team_a_id, "")
    abbr_b = _TEAM_ABBR.get(req.team_b_id, "")
    inj_a  = _fetch_injury_burden(abbr_a)
    inj_b  = _fetch_injury_burden(abbr_b)
    INJURY_SCALE = 0.12
    prob_a = float(np.clip(prob_a + (inj_b - inj_a) * INJURY_SCALE, 0.05, 0.95))
    prob_b = 1.0 - prob_a

    winner_id = req.team_a_id if prob_a >= 0.5 else req.team_b_id
    home_team = _name(req.team_a_id) if home_a else _name(req.team_b_id)

    return PredictResponse(
        team_a_id=req.team_a_id,        team_b_id=req.team_b_id,
        team_a_name=_name(req.team_a_id), team_b_name=_name(req.team_b_id),
        prob_a=round(prob_a, 4),         prob_b=round(prob_b, 4),
        predicted_winner_id=winner_id,
        predicted_winner_name=_name(winner_id),
        game_num=game_num,               home_team=home_team,
        features={**features, "injury_burden_a": round(inj_a, 3), "injury_burden_b": round(inj_b, 3)},
    )


# ── Static file serving (production: FastAPI serves the React build) ─────────

STATIC_DIR = ROOT / "web" / "dist"

if STATIC_DIR.exists():
    _assets = STATIC_DIR / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    _STATIC_ROOT = STATIC_DIR.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        candidate = (STATIC_DIR / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(_STATIC_ROOT):
            return FileResponse(str(candidate))
        return FileResponse(str(STATIC_DIR / "index.html"))
