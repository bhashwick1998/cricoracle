"""
Debutants and Empirical Bayes Shrinkage Engine for CricOracle
Handles uncapped players, foreign condition deltas, and hierarchical role priors.
"""

from typing import Dict, Any

# League Translation Factors (multipliers applied to domestic/franchise metrics)
LEAGUE_TRANSLATION_FACTORS = {
    "tier_1_ipl": {"bat_sr": 0.92, "bat_avg": 0.88, "econ": 1.05, "confidence": 0.85},
    "tier_2_bbl_psl": {"bat_sr": 0.85, "bat_avg": 0.80, "econ": 1.10, "confidence": 0.70},
    "domestic_list_a": {"bat_sr": 0.75, "bat_avg": 0.72, "econ": 1.18, "confidence": 0.50},
    "none": {"bat_sr": 0.65, "bat_avg": 0.60, "econ": 1.25, "confidence": 0.35},
}

# Hierarchical Role Baselines (mu_0 for uncapped players)
ROLE_BASELINES = {
    "top_order_batter": {
        "t20_batting_sr": 128.0,
        "t20_batting_avg": 26.5,
        "powerplay_batting_rating": 56.0,
        "middle_batting_rating": 52.0,
        "death_batting_rating": 45.0,
        "powerplay_bowling_rating": 20.0,
        "middle_bowling_rating": 20.0,
        "death_bowling_rating": 15.0,
        "t20_economy": 9.5,
        "t20_bowling_sr": 28.0,
    },
    "middle_order_batter": {
        "t20_batting_sr": 132.0,
        "t20_batting_avg": 25.0,
        "powerplay_batting_rating": 48.0,
        "middle_batting_rating": 55.0,
        "death_batting_rating": 52.0,
        "powerplay_bowling_rating": 20.0,
        "middle_bowling_rating": 20.0,
        "death_bowling_rating": 15.0,
        "t20_economy": 9.2,
        "t20_bowling_sr": 26.0,
    },
    "finisher": {
        "t20_batting_sr": 145.0,
        "t20_batting_avg": 24.0,
        "powerplay_batting_rating": 40.0,
        "middle_batting_rating": 50.0,
        "death_batting_rating": 65.0,
        "powerplay_bowling_rating": 20.0,
        "middle_bowling_rating": 25.0,
        "death_bowling_rating": 20.0,
        "t20_economy": 9.8,
        "t20_bowling_sr": 24.0,
    },
    "wicket_keeper": {
        "t20_batting_sr": 130.0,
        "t20_batting_avg": 25.5,
        "powerplay_batting_rating": 50.0,
        "middle_batting_rating": 52.0,
        "death_batting_rating": 52.0,
        "powerplay_bowling_rating": 10.0,
        "middle_bowling_rating": 10.0,
        "death_bowling_rating": 10.0,
        "t20_economy": 10.0,
        "t20_bowling_sr": 30.0,
    },
    "all_rounder": {
        "t20_batting_sr": 133.0,
        "t20_batting_avg": 23.0,
        "powerplay_batting_rating": 48.0,
        "middle_batting_rating": 52.0,
        "death_batting_rating": 56.0,
        "powerplay_bowling_rating": 50.0,
        "middle_bowling_rating": 52.0,
        "death_bowling_rating": 50.0,
        "t20_economy": 8.4,
        "t20_bowling_sr": 19.5,
    },
    "frontline_pacer": {
        "t20_batting_sr": 105.0,
        "t20_batting_avg": 11.0,
        "powerplay_batting_rating": 20.0,
        "middle_batting_rating": 20.0,
        "death_batting_rating": 25.0,
        "powerplay_bowling_rating": 58.0,
        "middle_bowling_rating": 52.0,
        "death_bowling_rating": 58.0,
        "t20_economy": 8.1,
        "t20_bowling_sr": 16.5,
    },
    "frontline_spinner": {
        "t20_batting_sr": 95.0,
        "t20_batting_avg": 9.0,
        "powerplay_batting_rating": 15.0,
        "middle_batting_rating": 20.0,
        "death_batting_rating": 20.0,
        "powerplay_bowling_rating": 48.0,
        "middle_bowling_rating": 62.0,
        "death_bowling_rating": 45.0,
        "t20_economy": 7.6,
        "t20_bowling_sr": 17.0,
    },
}

N0_THRESHOLD_BALLS = 120  # Minimum sample size before prior shrinkage fades

def compute_empirical_bayes_shrinkage(
    sample_val: float,
    sample_size: int,
    prior_mean: float,
    prior_sample_weight: int = N0_THRESHOLD_BALLS
) -> float:
    """
    Computes Empirical Bayes shrinkage:
    theta_hat = (N / (N + N_0)) * sample_val + (N_0 / (N + N_0)) * prior_mean
    """
    weight_sample = sample_size / (sample_size + prior_sample_weight)
    weight_prior = prior_sample_weight / (sample_size + prior_sample_weight)
    return (weight_sample * sample_val) + (weight_prior * prior_mean)

def compute_debutant_prior(
    role: str,
    bowling_style: str,
    league_tier: str = "none",
    observed_sr: float = None,
    observed_avg: float = None,
    observed_econ: float = None
) -> Dict[str, Any]:
    """
    Constructs a shrunk prior profile for a debutant or low-sample international player.
    """
    baseline = ROLE_BASELINES.get(role, ROLE_BASELINES["all_rounder"]).copy()
    tf = LEAGUE_TRANSLATION_FACTORS.get(league_tier, LEAGUE_TRANSLATION_FACTORS["none"])

    # If observed franchise/domestic numbers are provided, apply translation factor
    if observed_sr is not None:
        baseline["t20_batting_sr"] = observed_sr * tf["bat_sr"]
    if observed_avg is not None:
        baseline["t20_batting_avg"] = observed_avg * tf["bat_avg"]
    if observed_econ is not None:
        baseline["t20_economy"] = observed_econ * tf["econ"]

    # Archetype adjustments for bowling style
    if bowling_style in ["SLA", "LBG", "OB"]:
        baseline["middle_bowling_rating"] = min(75.0, baseline.get("middle_bowling_rating", 50.0) * 1.08)
    elif bowling_style in ["RF", "RMF", "LAWS"]:
        baseline["powerplay_bowling_rating"] = min(75.0, baseline.get("powerplay_bowling_rating", 50.0) * 1.06)
        baseline["death_bowling_rating"] = min(75.0, baseline.get("death_bowling_rating", 50.0) * 1.08)

    baseline["uncertainty_variance"] = 1.0 - (tf["confidence"] * 0.5)
    baseline["is_debutant"] = 1
    return baseline
