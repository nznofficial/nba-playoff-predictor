"""
api/server.py — FastAPI backend for the NBA Playoff Predictor web app.

Run from the project root:
    uvicorn api.server:app --reload --port 8000
"""

import copy
import json
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import MODEL_FEATURES

log = logging.getLogger("uvicorn.error")

ROOT        = Path(__file__).parent.parent
MODEL_PATH  = ROOT / "models" / "logistic_model.joblib"
DATA_DIR    = ROOT / "data" / "raw"
CACHE_FILE  = ROOT / "data" / "cache" / "playoff_games_2025-26.json"
CACHE_TTL   = 3600

CURRENT_SEASON         = "2025-26"
HOME_GAMES_HIGHER_SEED = {1, 2, 5, 7}

_model = joblib.load(MODEL_PATH)

def _load_team_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    adv  = pd.read_csv(DATA_DIR / "team_stats" / f"{season}_Regular_Season_advanced.csv")
    base = pd.read_csv(DATA_DIR / "team_stats" / f"{season}_Regular_Season_base.csv")
    base["three_pt_rate"] = base["FG3A"] / base["FGA"].replace(0, np.nan)
    return adv.merge(base[["TEAM_ID", "three_pt_rate"]], on="TEAM_ID")

_team_stats = _load_team_stats()

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/bracket")
def get_bracket():
    return _build_bracket()


@app.post("/api/refresh")
def refresh_bracket():
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

    def g(s, col, d=0.0):
        return float(s[col]) if col in s.index and not pd.isna(s[col]) else d

    features = {
        "net_rtg_diff":       g(sa, "NET_RATING")    - g(sb, "NET_RATING"),
        "off_rtg_diff":       g(sa, "OFF_RATING")    - g(sb, "DEF_RATING"),
        "def_rtg_diff":       g(sa, "DEF_RATING")    - g(sb, "OFF_RATING"),
        "pace_diff":          g(sa, "PACE")           - g(sb, "PACE"),
        "rest_days_diff":     0,
        "home_court":         int(home_a),
        "win_pct_diff":       g(sa, "W_PCT")          - g(sb, "W_PCT"),
        "series_game_num":    game_num,
        "series_lead":        req.wins_a - req.wins_b,
        "three_pt_rate_diff": g(sa, "three_pt_rate", .35) - g(sb, "three_pt_rate", .35),
        "playoff_exp_diff":   0.0,
        "is_bubble":          0,
    }

    prob_a = float(_model.predict_proba(pd.DataFrame([features])[MODEL_FEATURES])[0, 1])
    winner_id = req.team_a_id if prob_a >= 0.5 else req.team_b_id
    home_team = _name(req.team_a_id) if home_a else _name(req.team_b_id)

    return PredictResponse(
        team_a_id=req.team_a_id,   team_b_id=req.team_b_id,
        team_a_name=_name(req.team_a_id), team_b_name=_name(req.team_b_id),
        prob_a=round(prob_a, 4),   prob_b=round(1 - prob_a, 4),
        predicted_winner_id=winner_id,
        predicted_winner_name=_name(winner_id),
        game_num=game_num,         home_team=home_team,
        features=features,
    )


# ── Static file serving (production: FastAPI serves the React build) ─────────

STATIC_DIR = ROOT / "web" / "dist"

if STATIC_DIR.exists():
    _assets = STATIC_DIR / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        candidate = STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(STATIC_DIR / "index.html"))
