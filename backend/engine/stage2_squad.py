"""
Stage 2: SQUAD_ANNOUNCED Engine (Gear 2A)
Selects the Probable 11v11 from 15-man squads using player pitch adaptability on the ground's
format-specific pitch behavior ('flat', 'green', 'seam', 'bounce', 'spin heavy'),
computes bottom-up player ratings with Foreign Condition Deltas & Debutant Shrinkage,
and blends 70% Projected XI + 30% Stage 1 Macro prediction.
"""

from typing import List, Dict, Any, Tuple
import math
from .debut_priors import compute_empirical_bayes_shrinkage, compute_debutant_prior
from .cricsheet_pitch_intelligence import (
    get_ground_pitch_archetype,
    compute_player_5way_pitch_adaptability
)

def calculate_player_pitch_fit(player: Dict[str, Any], venue: Dict[str, Any], match_format: str = "T20I") -> float:
    """
    Computes a 0-100 score of how well this player fits the ground's format-specific pitch archetype
    ('flat', 'green', 'seam', 'bounce', 'spin heavy').
    """
    fmt = match_format.upper() if match_format else "T20I"
    pitch_archetype = get_ground_pitch_archetype(venue.get("id", ""), fmt)
    adaptability = compute_player_5way_pitch_adaptability(player, match_format=fmt)
    return adaptability["pitch_adaptability"].get(pitch_archetype, 70.0)

def select_probable_xi(
    squad: List[Dict[str, Any]],
    venue: Dict[str, Any],
    match_format: str = "T20I",
    pitch_archetype: str = "flat"
) -> List[Dict[str, Any]]:
    """
    Heuristic algorithm to select the optimal 11 from a 15-man squad:
    - At least 1 Wicketkeeper
    - Top order and middle order depth
    - Pitch-fit bowling attack tailored to pitch archetype (Spin-heavy vs Seam/Green vs Bounce)
    - Bowling quota: at least 5 viable bowlers
    """
    if len(squad) <= 11:
        return squad

    # Score each player by composite heuristic
    scored_players = []
    for p in squad:
        adaptability = compute_player_5way_pitch_adaptability(p, match_format=match_format)
        fit_score = adaptability["pitch_adaptability"].get(pitch_archetype, 70.0)
        sel_weight = float(p.get("recent_selection_weight", 0.8))
        composite_score = (sel_weight * 50.0) + (fit_score * 0.50)
        scored_players.append((composite_score, fit_score, p))

    # Sort descending by composite score
    scored_players.sort(key=lambda x: x[0], reverse=True)

    selected: List[Dict[str, Any]] = []
    has_wk = False
    spinners_count = 0
    pacers_count = 0

    # 1. Guarantee 1 Wicketkeeper first
    for _, fit, p in scored_players:
        if p.get("role") == "wicket_keeper":
            p["pitch_fit_rating"] = fit
            selected.append(p)
            has_wk = True
            break

    # 2. Add specialist batters & all-rounders
    for _, fit, p in scored_players:
        if any(x.get("id") == p.get("id") or x.get("name") == p.get("name") for x in selected):
            continue
        role = p.get("role", "")
        if role in ["top_order_batter", "middle_order_batter", "finisher", "all_rounder"]:
            batters_in_selected = len([x for x in selected if "batter" in x.get("role", "") or x.get("role") == "wicket_keeper"])
            if batters_in_selected < 6:
                p["pitch_fit_rating"] = fit
                selected.append(p)

    # 3. Add bowlers according to the 5 canonical pitch archetypes
    canonical_arch = pitch_archetype.lower()
    if canonical_arch in ["dusty", "spin heavy", "spin"]:
        target_spinners = 3
        target_pacers = 2
    elif canonical_arch in ["green", "bouncy", "bounce"]:
        target_spinners = 1
        target_pacers = 4
    else:  # balanced or flat
        target_spinners = 2
        target_pacers = 3

    for _, fit, p in scored_players:
        if any(x.get("id") == p.get("id") or x.get("name") == p.get("name") for x in selected):
            continue
        role = p.get("role", "")
        style = str(p.get("bowling_style", "")).upper()

        is_spinner = "spinner" in role or style in ["SLA", "LBG", "OB", "LAWS"]
        is_pacer = "pacer" in role or style in ["RF", "RMF", "LF", "LMF", "FAST"]

        if is_spinner and spinners_count < target_spinners:
            p["pitch_fit_rating"] = fit
            selected.append(p)
            spinners_count += 1
        elif is_pacer and pacers_count < target_pacers:
            p["pitch_fit_rating"] = fit
            selected.append(p)
            pacers_count += 1

    # Fill remaining slots up to 11 with highest-rated remaining players
    for _, fit, p in scored_players:
        if len(selected) >= 11:
            break
        if not any(x.get("id") == p.get("id") or x.get("name") == p.get("name") for x in selected):
            p["pitch_fit_rating"] = fit
            selected.append(p)

    return selected[:11]

def evaluate_lineup_power(
    lineup: List[Dict[str, Any]],
    venue: Dict[str, Any],
    pitch_archetype: str = "flat",
    match_format: str = "T20I"
) -> Tuple[float, float, float]:
    """
    Aggregates lineup batting and bowling ratings weighted by player pitch adaptability.
    Format-calibrated across TEST (defensive resilience & bowling strike rate),
    ODI (strike rotation & economy), and T20I (boundary power & death execution).
    """
    venue_region = venue.get("region_cluster", "sena").lower()
    fmt = match_format.upper() if match_format else "T20I"

    total_batting = 0.0
    total_bowling = 0.0
    total_cond_delta = 0.0

    for p in lineup:
        balls_faced = p.get("intl_balls_faced", 500)
        balls_bowled = p.get("intl_balls_bowled", 300)

        adaptability = compute_player_5way_pitch_adaptability(p, match_format=fmt)
        fit_score = adaptability["pitch_adaptability"].get(pitch_archetype, 70.0)
        fit_multiplier = fit_score / 75.0  # Normalized around 1.0

        cond_ratio = p.get("foreign_condition_ratio_subcontinent", 1.0) if venue_region == "subcontinent" else p.get("foreign_condition_ratio_sena", 1.0)
        shrunk_cond_ratio = compute_empirical_bayes_shrinkage(cond_ratio, balls_faced, 1.0, prior_sample_weight=150)
        total_cond_delta += (shrunk_cond_ratio - 1.0)

        if fmt == "TEST":
            base_avg = float(p.get("test_batting_avg") or p.get("batting_avg") or (p.get("t20_batting_avg", 28.0) * 1.35))
            if balls_faced < 120 or p.get("is_debutant", 0) == 1:
                prior = compute_debutant_prior(p.get("role", "all_rounder"), p.get("bowling_style", "NONE"), p.get("league_experience_tier", "none"))
                base_avg = compute_empirical_bayes_shrinkage(base_avg, balls_faced, prior.get("test_batting_avg", 32.0))
            balls_resilience = min(1.2, float(balls_faced) / 800.0)
            bat_rating = ((base_avg / 42.0) * 0.75 + balls_resilience * 0.25) * shrunk_cond_ratio * fit_multiplier * 50.0

            if p.get("role") in ["frontline_pacer", "frontline_spinner", "all_rounder", "bowling_all_rounder", "bowler", "pacer", "spinner"]:
                raw_b_avg = p.get("test_bowling_avg") or p.get("bowling_avg")
                base_bowl_avg = float(raw_b_avg) if (raw_b_avg is not None and float(raw_b_avg) > 0) else 28.0
                raw_b_sr = p.get("test_bowling_sr")
                base_bowl_sr = float(raw_b_sr) if (raw_b_sr is not None and float(raw_b_sr) > 0) else 52.0
                if balls_bowled < 120 or p.get("is_debutant", 0) == 1:
                    prior = compute_debutant_prior(p.get("role", "all_rounder"), p.get("bowling_style", "NONE"), p.get("league_experience_tier", "none"))
                    base_bowl_avg = compute_empirical_bayes_shrinkage(base_bowl_avg, balls_bowled, 30.0)
                bowl_rating = ((28.0 / max(18.0, base_bowl_avg)) * 0.60 + (52.0 / max(35.0, base_bowl_sr)) * 0.40) * fit_multiplier * 50.0
                total_bowling += bowl_rating

        elif fmt == "ODI":
            base_avg = float(p.get("odi_batting_avg") or p.get("batting_avg") or p.get("t20_batting_avg", 32.0))
            base_sr = float(p.get("odi_batting_sr") or (p.get("t20_batting_sr", 120.0) * 0.72))
            if balls_faced < 120 or p.get("is_debutant", 0) == 1:
                prior = compute_debutant_prior(p.get("role", "all_rounder"), p.get("bowling_style", "NONE"), p.get("league_experience_tier", "none"))
                base_avg = compute_empirical_bayes_shrinkage(base_avg, balls_faced, prior.get("odi_batting_avg", 30.0))
                base_sr = compute_empirical_bayes_shrinkage(base_sr, balls_faced, 85.0)
            bat_rating = ((base_avg / 38.0) * 0.50 + (base_sr / 90.0) * 0.50) * shrunk_cond_ratio * fit_multiplier * 50.0

            if p.get("role") in ["frontline_pacer", "frontline_spinner", "all_rounder", "bowling_all_rounder", "bowler", "pacer", "spinner"]:
                raw_odi_econ = p.get("odi_economy")
                raw_t20_econ = p.get("t20_economy")
                base_econ = float(raw_odi_econ) if (raw_odi_econ is not None and float(raw_odi_econ) > 0) else (float(raw_t20_econ) * 0.72 if (raw_t20_econ is not None and float(raw_t20_econ) > 0) else 5.4)
                raw_odi_sr = p.get("odi_bowling_sr")
                base_bowl_sr = float(raw_odi_sr) if (raw_odi_sr is not None and float(raw_odi_sr) > 0) else 34.0
                if balls_bowled < 120 or p.get("is_debutant", 0) == 1:
                    prior = compute_debutant_prior(p.get("role", "all_rounder"), p.get("bowling_style", "NONE"), p.get("league_experience_tier", "none"))
                    base_econ = compute_empirical_bayes_shrinkage(base_econ, balls_bowled, 5.4)
                bowl_rating = ((5.2 / max(3.8, base_econ)) * 0.50 + (34.0 / max(22.0, base_bowl_sr)) * 0.50) * fit_multiplier * 50.0
                total_bowling += bowl_rating

        else: # T20I
            base_sr = float(p.get("t20_batting_sr", 130.0))
            base_avg = float(p.get("t20_batting_avg", 28.0))
            if balls_faced < 120 or p.get("is_debutant", 0) == 1:
                prior = compute_debutant_prior(p.get("role", "all_rounder"), p.get("bowling_style", "NONE"), p.get("league_experience_tier", "none"))
                base_sr = compute_empirical_bayes_shrinkage(base_sr, balls_faced, prior["t20_batting_sr"])
                base_avg = compute_empirical_bayes_shrinkage(base_avg, balls_faced, prior["t20_batting_avg"])
            bat_rating = ((base_sr / 135.0) * 0.65 + (base_avg / 30.0) * 0.35) * shrunk_cond_ratio * fit_multiplier * 50.0

            if p.get("role") in ["frontline_pacer", "frontline_spinner", "all_rounder", "bowling_all_rounder", "bowler", "pacer", "spinner"]:
                raw_t20_econ = p.get("t20_economy")
                base_econ = float(raw_t20_econ) if (raw_t20_econ is not None and float(raw_t20_econ) > 0) else 8.2
                raw_t20_sr = p.get("t20_bowling_sr")
                base_bowl_sr = float(raw_t20_sr) if (raw_t20_sr is not None and float(raw_t20_sr) > 0) else 18.0
                if balls_bowled < 120 or p.get("is_debutant", 0) == 1:
                    prior = compute_debutant_prior(p.get("role", "all_rounder"), p.get("bowling_style", "NONE"), p.get("league_experience_tier", "none"))
                    base_econ = compute_empirical_bayes_shrinkage(base_econ, balls_bowled, prior["t20_economy"])
                    base_bowl_sr = compute_empirical_bayes_shrinkage(base_bowl_sr, balls_bowled, prior["t20_bowling_sr"])
                bowl_rating = ((8.0 / max(5.0, base_econ)) * 0.55 + (18.0 / max(10.0, base_bowl_sr)) * 0.45) * fit_multiplier * 50.0
                total_bowling += bowl_rating

        total_batting += bat_rating

    return total_batting / 11.0, total_bowling / 5.0, total_cond_delta / 11.0

def calculate_stage2_squad(
    squad_a: List[Dict[str, Any]],
    squad_b: List[Dict[str, Any]],
    venue: Dict[str, Any],
    p1_macro_prob_a: float,
    match_format: str = "T20I"
) -> Dict[str, Any]:
    """
    Computes Stage 2 Squad-Adjusted Prediction:
    - Ground format pitch archetype is identified ('flat', 'green', 'seam', 'bounce', 'spin heavy')
    - Player 5-way pitch adaptability scores are used to select Probable XI
    - Evaluates XI strength differential on the ground's pitch behavior
    - Blends 70% Projected XI + 30% Stage 1 Macro
    """
    fmt = match_format.upper() if match_format else "T20I"
    pitch_archetype = get_ground_pitch_archetype(venue.get("id", ""), fmt)

    proj_xi_a = select_probable_xi(squad_a, venue, fmt, pitch_archetype)
    proj_xi_b = select_probable_xi(squad_b, venue, fmt, pitch_archetype)

    bat_a, bowl_a, cond_a = evaluate_lineup_power(proj_xi_a, venue, pitch_archetype, match_format=fmt)
    bat_b, bowl_b, cond_b = evaluate_lineup_power(proj_xi_b, venue, pitch_archetype, match_format=fmt)

    delta_bat = bat_a - bat_b
    delta_bowl = bowl_a - bowl_b
    delta_cond = cond_a - cond_b

    # Sixth-bowler safety quotient in white-ball cricket
    viable_a = len([p for p in proj_xi_a if p.get("role") in ["frontline_pacer", "frontline_spinner", "all_rounder", "bowling_all_rounder", "bowler", "pacer", "spinner"]])
    viable_b = len([p for p in proj_xi_b if p.get("role") in ["frontline_pacer", "frontline_spinner", "all_rounder", "bowling_all_rounder", "bowler", "pacer", "spinner"]])
    safety_delta = (min(6, viable_a) - min(6, viable_b)) * 0.12 if fmt in ["ODI", "T20I"] else 0.0

    # Mean squad pitch adaptability scores
    mean_adapt_a = sum(p.get("pitch_fit_rating", 70.0) for p in proj_xi_a) / max(1, len(proj_xi_a))
    mean_adapt_b = sum(p.get("pitch_fit_rating", 70.0) for p in proj_xi_b) / max(1, len(proj_xi_b))
    delta_adaptability = (mean_adapt_a - mean_adapt_b) / 10.0

    # Bottom-up logistic model for 11v11
    xi_logit = (delta_bat / 15.0) * 0.50 + (delta_bowl / 15.0) * 0.30 + delta_cond * 0.25 + delta_adaptability * 0.35 + safety_delta
    p_xi_a = 1.0 / (1.0 + math.exp(-xi_logit)) * 100.0

    if fmt == "TEST":
        p_draw = round(venue.get("test_draw_pct", 0.11) * 100.0, 1)
        decisive_pool = 100.0 - p_draw
        p1_decisive_share = min(99.5, max(0.5, (p1_macro_prob_a / max(1.0, decisive_pool)) * 100.0))
        p_blended_a = (0.65 * p_xi_a) + (0.35 * p1_decisive_share)
        p_team_a = round((p_blended_a / 100.0) * decisive_pool, 1)
        p_team_b = round(decisive_pool - p_team_a, 1)
    else:
        p_draw = 0.0
        p_blended_a = (0.70 * p_xi_a) + (0.30 * p1_macro_prob_a)
        p_team_a = round(p_blended_a, 1)
        p_team_b = round(100.0 - p_team_a, 1)

    delta_shift = round(p_team_a - p1_macro_prob_a, 1)

    notable_inclusions = [p["name"] for p in proj_xi_a if p.get("pitch_fit_rating", 50) > 85][:2]

    return {
        "stage": "SQUAD_ANNOUNCED",
        "badge": "SQUAD-ADJUSTED PREDICTION • PROBABLE XI",
        "gear": "Gear 2A",
        "p_team_a": p_team_a,
        "p_team_b": p_team_b,
        "p_draw": p_draw,
        "pitch_archetype": pitch_archetype,
        "delta_squad_shift": delta_shift,
        "probable_xi_a_ids": [p.get("id") or p.get("player_id") for p in proj_xi_a],
        "probable_xi_b_ids": [p.get("id") or p.get("player_id") for p in proj_xi_b],
        "details": {
            "pitch_archetype": pitch_archetype,
            "mean_xi_a_adaptability": round(mean_adapt_a, 1),
            "mean_xi_b_adaptability": round(mean_adapt_b, 1),
            "batting_diff": round(delta_bat, 2),
            "bowling_diff": round(delta_bowl, 2),
            "foreign_condition_diff": round(delta_cond, 2),
            "notable_inclusions": notable_inclusions,
            "stage1_anchor": p1_macro_prob_a
        }
    }
