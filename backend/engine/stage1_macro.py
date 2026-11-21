"""
Stage 1: UPCOMING_NO_SQUAD Macro Condition-Adjusted Engine
Calculates win probability prior before squads are announced using:
- Format-specific Ground Pitch Categorization ('flat', 'green', 'seam', 'bounce', 'spin heavy')
- Country Pitch Adaptability Scores (for each format and pitch archetype)
- Condition-Specific Team Elo (Host vs SENA/Subcontinent Away rating)
- Pitch-to-Team Fit Differential (Spin vs Pace ratio weighted by venue)
- Phase Net Ratings (Powerplay, Middle, Death)
- Cricsheet Venue Chase/Bat-First Bias and Day/Night Dew adjustment
"""

import math
from typing import Dict, Any
from .cricsheet_pitch_intelligence import (
    get_ground_pitch_archetype,
    get_country_pitch_adaptability,
    get_country_pitch_adaptability_details,
    compute_format_specific_prediction_variables,
    get_international_h2h,
    get_team_rolling_momentum,
    compute_atmospheric_aerodynamic_variables,
    CRICSHEET_TOSS_FACTORS
)

def calculate_stage1_macro(
    team_a: Dict[str, Any],
    team_b: Dict[str, Any],
    venue: Dict[str, Any],
    match_format: str = "T20I",
    is_day_night: bool = True
) -> Dict[str, Any]:
    """
    Computes Stage 1 Macro Prediction (when no squads are known yet).
    Uses country pitch adaptability on the ground's average pitch behavior in the given format.
    """
    fmt = match_format.upper() if match_format else "T20I"
    venue_id = venue.get("id", "")
    venue_region = venue.get("region_cluster", "sena").lower()
    venue_country = venue.get("country", "").lower()

    # 1. Ground Pitch Archetype for the given format
    pitch_archetype = get_ground_pitch_archetype(venue_id, fmt)

    # 2. Country Pitch Adaptability Scores (65% Weight on Recent 1-Year International Matches)
    team_a_code = team_a.get("short_name") or team_a.get("id") or "IND"
    team_b_code = team_b.get("short_name") or team_b.get("id") or "NZ"

    country_a_details = get_country_pitch_adaptability_details(team_a_code, fmt, pitch_archetype, recency_weight=0.65)
    country_b_details = get_country_pitch_adaptability_details(team_b_code, fmt, pitch_archetype, recency_weight=0.65)
    country_a_score = country_a_details["composite_score"]
    country_b_score = country_b_details["composite_score"]
    delta_country_adaptability = (country_a_score - country_b_score) / 10.0

    # 3. Condition-Specific Elo Selection
    team_a_is_host = (team_a.get("name", "").lower() in venue_country) or (venue_country in team_a.get("name", "").lower())
    team_b_is_host = (team_b.get("name", "").lower() in venue_country) or (venue_country in team_b.get("name", "").lower())

    if team_a_is_host:
        elo_a = team_a.get("elo_home", 1600.0)
    elif venue_region == "subcontinent":
        elo_a = team_a.get("elo_away_subcontinent", 1550.0)
    else:
        elo_a = team_a.get("elo_away_sena", 1550.0)

    if team_b_is_host:
        elo_b = team_b.get("elo_home", 1600.0)
    elif venue_region == "subcontinent":
        elo_b = team_b.get("elo_away_subcontinent", 1550.0)
    else:
        elo_b = team_b.get("elo_away_sena", 1550.0)

    delta_elo = elo_a - elo_b

    # 4. Team-to-Pitch Spin/Pace Ratio Fit
    spin_ratio = venue.get("spin_wicket_ratio", 0.40)
    pace_ratio = 1.0 - spin_ratio

    team_a_spin = team_a.get("spin_index", 50.0)
    team_b_spin = team_b.get("spin_index", 50.0)
    team_a_pace = team_a.get("pace_index", 50.0)
    team_b_pace = team_b.get("pace_index", 50.0)

    delta_pitch = (team_a_spin - team_b_spin) * spin_ratio + (team_a_pace - team_b_pace) * pace_ratio

    # 5. Format-Specific Phase & Ball-Age Dynamics
    format_vars = compute_format_specific_prediction_variables(fmt, team_a, team_b, [], [], venue)

    if fmt == "TEST":
        # In Test cricket, there are NO Powerplays or Death overs.
        # The 3 strategic phases are governed by Ball Age and Multi-Day Session Dynamics:
        # Phase 1: New Ball Attack & Top-Order Resilience (Overs 1-20): 35%
        #          (Pace/seam movement, lateral deviation in channel, top-order survival)
        # Phase 2: Old-Ball Attrition & Spin/Reverse Grind (Overs 21-80): 45%
        #          (Softened ball, reverse swing, spin grip, middle-order partnership endurance)
        # Phase 3: Second New Ball Threat (Overs 80+) & Tail-End Resistance (Wickets 8-11): 20%
        #          (80-over replacement hard ball burst vs lower-order tail survival)
        test_phase_weights = {"new_ball": 0.35, "old_ball_attrition": 0.45, "second_new_ball_tail": 0.20}

        # Authentic Test Phase Net Ratings
        new_ball_a = float(team_a.get("pace_index", 50.0)) * 0.65 + float(team_a.get("powerplay_rating", 50.0)) * 0.35
        new_ball_b = float(team_b.get("pace_index", 50.0)) * 0.65 + float(team_b.get("powerplay_rating", 50.0)) * 0.35

        old_ball_a = float(team_a.get("spin_index", 50.0)) * 0.55 + float(team_a.get("middle_overs_rating", 50.0)) * 0.45
        old_ball_b = float(team_b.get("spin_index", 50.0)) * 0.55 + float(team_b.get("middle_overs_rating", 50.0)) * 0.45

        test_vars = format_vars.get("test_specific", {})
        tail_delta = float(test_vars.get("tail_runs_delta", 0.0))
        tail_a = 50.0 + (tail_delta * 1.2)
        tail_b = 50.0 - (tail_delta * 1.2)

        team_a_net_phase = (
            new_ball_a * test_phase_weights["new_ball"]
            + old_ball_a * test_phase_weights["old_ball_attrition"]
            + tail_a * test_phase_weights["second_new_ball_tail"]
        )
        team_b_net_phase = (
            new_ball_b * test_phase_weights["new_ball"]
            + old_ball_b * test_phase_weights["old_ball_attrition"]
            + tail_b * test_phase_weights["second_new_ball_tail"]
        )
    elif fmt == "ODI":
        # Official ICC White-Ball Fielding Phases:
        # PP1 (Overs 1-10): 20%, Middle (Overs 11-40): 55%, Death (Overs 41-50): 25%
        phase_weights = {"pp": 0.20, "mid": 0.55, "death": 0.25}
        team_a_net_phase = (
            team_a.get("powerplay_rating", 50.0) * phase_weights["pp"]
            + team_a.get("middle_overs_rating", 50.0) * phase_weights["mid"]
            + team_a.get("death_overs_rating", 50.0) * phase_weights["death"]
        )
        team_b_net_phase = (
            team_b.get("powerplay_rating", 50.0) * phase_weights["pp"]
            + team_b.get("middle_overs_rating", 50.0) * phase_weights["mid"]
            + team_b.get("death_overs_rating", 50.0) * phase_weights["death"]
        )
    else:
        # Official ICC T20I Fielding Phases:
        # Powerplay (Overs 1-6): 30%, Middle (Overs 7-15): 45%, Death (Overs 16-20): 25%
        phase_weights = {"pp": 0.30, "mid": 0.45, "death": 0.25}
        team_a_net_phase = (
            team_a.get("powerplay_rating", 50.0) * phase_weights["pp"]
            + team_a.get("middle_overs_rating", 50.0) * phase_weights["mid"]
            + team_a.get("death_overs_rating", 50.0) * phase_weights["death"]
        )
        team_b_net_phase = (
            team_b.get("powerplay_rating", 50.0) * phase_weights["pp"]
            + team_b.get("middle_overs_rating", 50.0) * phase_weights["mid"]
            + team_b.get("death_overs_rating", 50.0) * phase_weights["death"]
        )

    delta_phase = team_a_net_phase - team_b_net_phase

    # 5B. Recent Momentum and 3-Year Bilateral H2H Dominance
    form_a_data = get_team_rolling_momentum(team_a_code, fmt)
    form_b_data = get_team_rolling_momentum(team_b_code, fmt)
    h2h_data = get_international_h2h(team_a_code, team_b_code, fmt)

    recent_form_a = float(team_a.get("recent_l5_win_pct") if "recent_l5_win_pct" in team_a and team_a.get("recent_l5_win_pct") != 0.60 else form_a_data["recent_l5_win_pct"])
    recent_form_b = float(team_b.get("recent_l5_win_pct") if "recent_l5_win_pct" in team_b and team_b.get("recent_l5_win_pct") != 0.50 else form_b_data["recent_l5_win_pct"])
    momentum_delta = (recent_form_a - recent_form_b) * 100.0

    h2h_delta = (h2h_data["last_3yr_win_pct_a"] - h2h_data["last_3yr_win_pct_b"]) * 100.0

    # Atmospheric Aerodynamic Modifiers
    atmos = compute_atmospheric_aerodynamic_variables(venue, weather=venue.get("weather"), is_day_night=is_day_night)
    pace_swing_edge = 0.0
    if atmos.get("aerodynamic_swing_multiplier", 1.0) > 1.05:
        pace_swing_edge = (float(team_a.get("pace_index", 50.0)) - float(team_b.get("pace_index", 50.0))) * 0.005

    # 6. Calibrated Logistic Regression combining Elo, 1-Year Country Pitch Adaptability, Momentum, Phase, H2H, and Aerodynamics
    logit = (
        0.0
        + 1.05 * (delta_elo / 400.0)
        + 0.45 * (delta_country_adaptability / 2.0)
        + 0.20 * (delta_pitch / 25.0)
        + 0.25 * (delta_phase / 20.0)
        + 0.18 * (momentum_delta / 25.0)
        + 0.22 * (h2h_delta / 25.0)
        + pace_swing_edge
    )

    # 7. Cricsheet Toss & Venue Chase Bias Integration
    toss_data = CRICSHEET_TOSS_FACTORS.get(pitch_archetype, {}).get(fmt, {"bat_first_win_pct": 0.50, "field_first_win_pct": 0.50})
    chase_win_pct = toss_data.get("field_first_win_pct", venue.get("chase_win_pct", 0.52))
    dew_factor = venue.get("dew_factor", 0.30) if is_day_night else 0.05
    chase_advantage_adj = (chase_win_pct - 0.50) * (1.0 + dew_factor)

    # Integrate over unknown toss: P(A wins) = 0.5 * P(A wins | bats 1st) + 0.5 * P(A wins | chases)
    p_a_batting_first = 1.0 / (1.0 + math.exp(-(logit - chase_advantage_adj)))
    p_a_chasing = 1.0 / (1.0 + math.exp(-(logit + chase_advantage_adj)))
    p_a_blended = 0.5 * p_a_batting_first + 0.5 * p_a_chasing

    if fmt == "TEST":
        p_draw = round(venue.get("test_draw_pct", 0.11) * 100.0, 1)
        decisive_pool = 100.0 - p_draw
        p_team_a = round(p_a_blended * decisive_pool, 1)
        p_team_b = round(decisive_pool - p_team_a, 1)
    else:
        p_draw = 0.0
        p_team_a = round(float(p_a_blended * 100.0), 1)
        p_team_b = round(100.0 - p_team_a, 1)

    return {
        "stage": "UPCOMING_NO_SQUAD",
        "badge": "MACRO PREDICTION • SQUADS TBA",
        "gear": "Gear 1",
        "p_team_a": p_team_a,
        "p_team_b": p_team_b,
        "p_draw": p_draw,
        "pitch_archetype": pitch_archetype,
        "country_a_adaptability": country_a_score,
        "country_b_adaptability": country_b_score,
        "country_a_adaptability_details": country_a_details,
        "country_b_adaptability_details": country_b_details,
        "format_specific_variables": format_vars,
        "details": {
            "format": fmt,
            "pitch_archetype": pitch_archetype,
            "country_a_code": team_a_code,
            "country_b_code": team_b_code,
            "country_a_score": country_a_score,
            "country_b_score": country_b_score,
            "country_a_recent_1yr": country_a_details["recent_1yr_score"],
            "country_b_recent_1yr": country_b_details["recent_1yr_score"],
            "country_recency_weight": 0.65,
            "momentum_delta_pct": round(momentum_delta, 1),
            "h2h_delta_pct": round(h2h_delta, 1),
            "h2h_bilateral": h2h_data,
            "team_a_form": form_a_data,
            "team_b_form": form_b_data,
            "atmospheric_aerodynamics": atmos,
            "delta_country_adaptability": round(delta_country_adaptability, 2),
            "delta_elo": round(delta_elo, 1),
            "elo_a_used": round(elo_a, 1),
            "elo_b_used": round(elo_b, 1),
            "delta_pitch_fit": round(delta_pitch, 2),
            "delta_phase_rating": round(delta_phase, 2),
            "venue_spin_ratio": spin_ratio,
            "cricsheet_chase_win_pct": chase_win_pct,
            "dew_factor": dew_factor,
        }
    }
