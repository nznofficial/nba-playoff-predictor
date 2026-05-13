"""
predictor.py — Simulate NBA playoff bracket using the trained logistic regression model.

Predicts game-by-game outcomes for each series, propagating winners through the bracket.
Home court: higher seed (lower seed number) has games 1, 2, 5, 7 at home.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from model import MODEL_FEATURES

log = logging.getLogger(__name__)

# NBA home court format for best-of-7: which game numbers are at the higher seed's home
HOME_GAMES_HIGHER_SEED = {1, 2, 5, 7}


def _build_game_feature_row(
    team_a_stats: pd.Series,
    team_b_stats: pd.Series,
    game_num: int,
    series_wins_a: int,
    series_wins_b: int,
    home_team: str,  # "a" or "b"
    is_finals: bool = False,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame of features from team A's perspective.
    team_a_stats and team_b_stats are rows from current_season features DataFrame.
    """
    rest_a = min(int(team_a_stats.get("rest_days", 5)), 10)
    rest_b = min(int(team_b_stats.get("rest_days", 5)), 10)

    def _get(series, key, default=0.0):
        return float(series.get(key, default)) if key in series.index else default

    row = {
        "net_rtg_diff": _get(team_a_stats, "net_rtg") - _get(team_b_stats, "net_rtg"),
        "off_rtg_diff": _get(team_a_stats, "off_rtg") - _get(team_b_stats, "def_rtg"),
        "def_rtg_diff": _get(team_a_stats, "def_rtg") - _get(team_b_stats, "off_rtg"),
        "pace_diff": _get(team_a_stats, "pace") - _get(team_b_stats, "pace"),
        "rest_days_diff": rest_a - rest_b,
        "home_court": 1 if home_team == "a" else 0,
        "win_pct_diff": _get(team_a_stats, "team_win_pct_reg") - _get(team_b_stats, "team_win_pct_reg"),
        "series_game_num": game_num,
        "series_lead": series_wins_a - series_wins_b,
        "three_pt_rate_diff": _get(team_a_stats, "three_pt_rate", 0.35) - _get(team_b_stats, "three_pt_rate", 0.35),
        "playoff_exp_diff": _get(team_a_stats, "playoff_exp_3yr") - _get(team_b_stats, "playoff_exp_3yr"),
        "is_bubble": 0,
    }

    # Fill any missing MODEL_FEATURES with 0
    for feat in MODEL_FEATURES:
        if feat not in row:
            row[feat] = 0.0

    return pd.DataFrame([row])[MODEL_FEATURES]


def predict_game(
    team_a_stats: pd.Series,
    team_b_stats: pd.Series,
    game_num: int,
    series_wins_a: int,
    series_wins_b: int,
    home_team: str,
    model: Pipeline,
) -> dict:
    """
    Predict outcome of a single playoff game.
    Returns win probabilities and predicted winner from team A's perspective.
    """
    feat_row = _build_game_feature_row(
        team_a_stats, team_b_stats, game_num, series_wins_a, series_wins_b, home_team
    )
    prob_a_wins = float(model.predict_proba(feat_row)[0, 1])
    prob_b_wins = 1.0 - prob_a_wins
    predicted_winner = "a" if prob_a_wins >= 0.5 else "b"

    return {
        "game_num": game_num,
        "win_prob_a": prob_a_wins,
        "win_prob_b": prob_b_wins,
        "predicted_winner": predicted_winner,
        "home_team": home_team,
    }


def simulate_series(
    team_a_id: int,
    team_b_id: int,
    team_a_seed: int,
    team_b_seed: int,
    current_features: pd.DataFrame,
    model: Pipeline,
    series_format: int = 7,
) -> dict:
    """
    Simulate a best-of-7 playoff series game by game.
    Higher seed (lower seed number) has home court advantage.
    Returns winner_id, series_result string, game_by_game details.
    """
    wins_needed = (series_format // 2) + 1  # 4 for best-of-7

    # Get team stats rows from current_features
    def _get_team_stats(team_id):
        rows = current_features[current_features["team_id"] == team_id]
        if len(rows) == 0:
            log.warning(f"Team {team_id} not found in current features — using zeros")
            return pd.Series(dtype=float)
        return rows.iloc[0]

    stats_a = _get_team_stats(team_a_id)
    stats_b = _get_team_stats(team_b_id)

    higher_seed = team_a_id if team_a_seed < team_b_seed else team_b_id
    wins_a = 0
    wins_b = 0
    game_by_game = []
    game_num = 1

    while wins_a < wins_needed and wins_b < wins_needed:
        # Determine home team for this game
        if game_num in HOME_GAMES_HIGHER_SEED:
            home_team = "a" if higher_seed == team_a_id else "b"
        else:
            home_team = "b" if higher_seed == team_a_id else "a"

        result = predict_game(
            stats_a, stats_b, game_num, wins_a, wins_b, home_team, model
        )
        game_by_game.append(result)

        if result["predicted_winner"] == "a":
            wins_a += 1
        else:
            wins_b += 1

        game_num += 1

    winner_id = team_a_id if wins_a >= wins_needed else team_b_id
    loser_id = team_b_id if winner_id == team_a_id else team_a_id
    winner_wins = wins_a if winner_id == team_a_id else wins_b
    loser_wins = wins_b if winner_id == team_a_id else wins_a

    # Average win probability for the winner
    winner_probs = [
        g["win_prob_a"] if winner_id == team_a_id else g["win_prob_b"]
        for g in game_by_game
    ]
    avg_win_prob = float(np.mean(winner_probs)) if winner_probs else 0.5

    return {
        "team_a_id": team_a_id,
        "team_b_id": team_b_id,
        "winner_id": winner_id,
        "loser_id": loser_id,
        "series_result": f"{winner_wins}-{loser_wins}",
        "winner_wins": winner_wins,
        "loser_wins": loser_wins,
        "total_games": len(game_by_game),
        "game_by_game": game_by_game,
        "winner_win_prob_avg": avg_win_prob,
    }


def simulate_conference(
    conference: str,
    seedings: list,
    current_features: pd.DataFrame,
    model: Pipeline,
) -> dict:
    """
    Simulate all 3 rounds of a conference bracket.
    seedings: [(seed, team_id), ...] — 8 entries for first round.
    Standard NBA bracket: 1v8, 2v7, 3v6, 4v5.
    Returns nested dict with round results and conference_champion_id.
    """
    seed_map = {s: tid for s, tid in seedings}

    def _run_round(matchups_seeds):
        round_results = {}
        next_round_seeds = {}
        for (seed_a, seed_b) in matchups_seeds:
            tid_a = seed_map[seed_a]
            tid_b = seed_map[seed_b]
            result = simulate_series(tid_a, tid_b, seed_a, seed_b, current_features, model)
            key = f"{seed_a}v{seed_b}"
            round_results[key] = result
            # Winner keeps the better (lower number) seed for bracket purposes
            winner_seed = seed_a if result["winner_id"] == tid_a else seed_b
            next_round_seeds[winner_seed] = result["winner_id"]
            # Update seed_map for next round
            seed_map[winner_seed] = result["winner_id"]
            log.info(
                f"  {conference} {key}: Team {result['winner_id']} wins "
                f"({result['series_result']}) avg prob={result['winner_win_prob_avg']:.3f}"
            )
        return round_results, next_round_seeds

    # Round 1: 1v8, 2v7, 3v6, 4v5
    log.info(f"\n{conference} Conference — Round 1")
    r1_results, r1_winners = _run_round([(1, 8), (2, 7), (3, 6), (4, 5)])

    # Round 2: winners of (1/8) vs (4/5), winners of (2/7) vs (3/6)
    # In the NBA bracket: top half = (1v8 winner vs 4v5 winner), bottom half = (2v7 winner vs 3v6 winner)
    r1_winner_seeds = sorted(r1_winners.keys())
    top_half = r1_winner_seeds[:2]
    bot_half = r1_winner_seeds[2:]

    log.info(f"\n{conference} Conference — Round 2 (Semifinals)")
    r2_matchups = [(min(top_half), max(top_half)), (min(bot_half), max(bot_half))]
    r2_results, r2_winners = _run_round(r2_matchups)

    # Conference Final
    r2_winner_seeds = sorted(r2_winners.keys())
    log.info(f"\n{conference} Conference — Finals")
    cf_matchups = [(min(r2_winner_seeds), max(r2_winner_seeds))]
    cf_results, cf_winners = _run_round(cf_matchups)

    champion_seed = min(cf_winners.keys())
    champion_id = cf_winners[champion_seed]

    return {
        "conference": conference,
        "round1": r1_results,
        "round2": r2_results,
        "conference_final": cf_results,
        "conference_champion_id": champion_id,
        "seedings": seedings,
        "seed_map_original": {s: tid for s, tid in seedings},
    }


def simulate_finals(
    west_champion_id: int,
    east_champion_id: int,
    current_features: pd.DataFrame,
    model: Pipeline,
    west_seedings: list = None,
    east_seedings: list = None,
) -> dict:
    """
    Simulate the NBA Finals.
    Home court: team with better regular-season record (higher win_pct_reg).
    """
    def _get_win_pct(team_id):
        rows = current_features[current_features["team_id"] == team_id]
        if len(rows) == 0:
            return 0.5
        return float(rows.iloc[0].get("team_win_pct_reg", 0.5))

    west_wpct = _get_win_pct(west_champion_id)
    east_wpct = _get_win_pct(east_champion_id)

    # Assign seeds: home court = seed 1
    if west_wpct >= east_wpct:
        west_seed, east_seed = 1, 2
    else:
        west_seed, east_seed = 2, 1

    log.info(f"\nNBA Finals: West {west_champion_id} (seed {west_seed}) vs East {east_champion_id} (seed {east_seed})")

    result = simulate_series(
        west_champion_id, east_champion_id,
        west_seed, east_seed,
        current_features, model,
    )
    return result


def run_full_bracket(
    west_seedings: list,
    east_seedings: list,
    current_features: pd.DataFrame,
    model: Pipeline,
    team_names: dict = None,
) -> dict:
    """
    Simulate the complete NBA Playoffs bracket.
    west_seedings / east_seedings: [(seed, team_id), ...] each with 8 entries.
    Returns full results dict including champion.
    """
    log.info("=" * 60)
    log.info("SIMULATING NBA PLAYOFFS")
    log.info("=" * 60)

    west = simulate_conference("West", west_seedings, current_features, model)
    east = simulate_conference("East", east_seedings, current_features, model)

    finals = simulate_finals(
        west["conference_champion_id"],
        east["conference_champion_id"],
        current_features,
        model,
        west_seedings,
        east_seedings,
    )

    champion_id = finals["winner_id"]
    champion_name = ""
    if team_names:
        champion_name = team_names.get(champion_id, str(champion_id))

    log.info(f"\nNBA CHAMPION: {champion_name or champion_id}")
    log.info(f"Finals result: {finals['series_result']}")

    return {
        "west": west,
        "east": east,
        "finals": finals,
        "champion_id": champion_id,
        "champion_name": champion_name,
        "west_champion_id": west["conference_champion_id"],
        "east_champion_id": east["conference_champion_id"],
    }
