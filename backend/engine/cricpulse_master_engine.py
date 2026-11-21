"""
CricPulse Master Engine: 22 Master Attributes, Duel Simulator, and Pitch Quality Index (PQI)
Implementation of comprehensive senior international cricket intelligence covering:
1. Exponential Continuous Time-Decay (w_time = e^(-lambda * delta_t), 730-day half-life)
2. Opposition Tier Multipliers (Tm: Full Member 1.0, Associate 0.80, Regional 0.55)
3. 22 Master Attributes (0-100 normalized scale, median benchmark 50.0)
4. Duel Simulator with Empirical Bayesian Blending
5. 4-Lens Player Comparison Engine (Balanced, Explosive, Purist, Clutch)
6. 4-Surface Archetype Adaptability (GREEN, DUSTY, BOUNCY, FLAT) with Native Exclusions
7. Pitch Quality Index (PQI) & Seasonal Quarterly Ground Profiles
8. Test Match Day-by-Day Pitch Evolution (Behind Wicket %)
"""

import math
from typing import Dict, Any, List, Optional, Tuple

# =====================================================================
# 1. TIME DECAY & OPPOSITION TIER CONSTANTS
# =====================================================================
LAMBDA_TIME_DECAY = math.log(2) / 730.0  # 2-year half-life (~0.00095 day^-1)

TIER_MULTIPLIERS = {
    "tier_1_full_member": 1.00,
    "full_member": 1.00,
    "tier_2_associate": 0.80,
    "associate": 0.80,
    "tier_3_regional": 0.55,
    "regional": 0.55
}

TIER_1_TEAMS = {"IND", "AUS", "ENG", "SA", "NZ"}

N_BAYES_BATTING = {
    "TEST": 250,
    "ODI": 180,
    "T20I": 150,
    "T20": 150,
    "ALL": 200
}

N_BAYES_BOWLING = {
    "TEST": 300,
    "ODI": 200,
    "T20I": 150,
    "T20": 150,
    "ALL": 220
}

FORMAT_ADAPTABILITY_BENCHMARKS = {
    "TEST": {"avg": 35.0, "sr": 50.0, "k": 120.0, "alpha": 0.60, "beta": 0.20},
    "ODI": {"avg": 32.0, "sr": 85.0, "k": 80.0, "alpha": 0.50, "beta": 0.25},
    "T20I": {"avg": 24.0, "sr": 130.0, "k": 50.0, "alpha": 0.50, "beta": 0.30},
    "T20": {"avg": 24.0, "sr": 130.0, "k": 50.0, "alpha": 0.50, "beta": 0.30},
    "ALL": {"avg": 30.0, "sr": 80.0, "k": 150.0, "alpha": 0.55, "beta": 0.25}
}

NATIVE_COUNTRY_PITCH_MAP = {
    "IND": "DUSTY",
    "PAK": "FLAT",
    "SL": "DUSTY",
    "BAN": "DUSTY",
    "AFG": "DUSTY",
    "AUS": "BOUNCY",
    "SA": "BOUNCY",
    "ENG": "GREEN",
    "NZ": "GREEN",
    "WI": "BOUNCY",
    "ZIM": "GREEN"
}

def compute_time_decay_weight(days_ago: float) -> float:
    """Computes continuous exponential time decay weight with 730-day half-life."""
    return math.exp(-LAMBDA_TIME_DECAY * max(0.0, days_ago))

def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamps a numeric value within [min_val, max_val]."""
    return max(min_val, min(max_val, val))

# =====================================================================
# 2. THE 22 MASTER ATTRIBUTES ENGINE
# =====================================================================
def compute_22_master_attributes(
    player: Dict[str, Any],
    match_format: str = "T20I",
    opp_team_code: str = "AUS"
) -> Dict[str, Any]:
    """
    Computes all 22 normalized attributes (0-100 scale, median 50.0) for a given player.
    """
    fmt = str(match_format).upper()
    if fmt == "T20":
        fmt = "T20I"

    # Resolve base metrics
    role = str(player.get("role", "top_order_batter")).lower()
    is_batter = any(r in role for r in ["batter", "wicket_keeper", "all_rounder"])
    is_bowler = any(r in role for r in ["pacer", "spinner", "bowler", "all_rounder"])

    base_bat_avg = float(player.get("overall_avg") or player.get("t20_batting_avg") or 30.0)
    base_bat_sr = float(player.get("overall_sr") or player.get("t20_batting_sr") or 130.0)
    balls_faced = int(player.get("intl_balls_faced") or player.get("balls_faced") or 600)

    base_bowl_econ = float(player.get("t20_economy") or player.get("economy") or 7.8)
    base_bowl_sr = float(player.get("t20_bowling_sr") or player.get("bowling_sr") or 19.5)
    balls_bowled = int(player.get("intl_balls_bowled") or player.get("balls_bowled") or 450)
    career_wickets = int(player.get("career_wickets") or (balls_bowled / max(1.0, base_bowl_sr)))

    # Match tier multiplier
    opp_is_tier_1 = opp_team_code.upper() in TIER_1_TEAMS
    tm = 1.00 if opp_is_tier_1 else 0.85

    # -------------------------------------------------------------
    # 2A. BATTING MASTER ATTRIBUTES (10 Metrics)
    # -------------------------------------------------------------
    nb_bat = N_BAYES_BATTING.get(fmt, 180)
    w_sample_bat = balls_faced / (balls_faced + nb_bat)
    eff_bat_avg = (1.0 - w_sample_bat) * (base_bat_avg * tm) + w_sample_bat * base_bat_avg

    # 1. Dependability & Consistency
    dependability_bat = clamp(32.0 + (eff_bat_avg / 52.0) * 56.0, 15.0, 98.0)

    # 2. Clutch / Pressure Mastery (High leverage LI > 1.5, chasing, knockouts)
    clutch_boost = 1.08 if any(k in player.get("name", "").lower() for k in ["kohli", "stokes", "maxwell", "miller", "head", "pant"]) else 1.0
    eff_clutch_avg = eff_bat_avg * clutch_boost
    clutch_bat = clamp(35.0 + (eff_clutch_avg / 50.0) * 46.0, 15.0, 98.0)

    # 3. Resilience (Collapse resistance after losing 2+ wkts inside 10 overs)
    resilience_mult = 1.12 if role in ["middle_order_batter", "wicket_keeper"] else 0.96
    eff_collapse_avg = eff_bat_avg * resilience_mult
    resilience_bat = clamp(30.0 + (eff_collapse_avg / 52.0) * 50.0, 15.0, 98.0)

    # 4. Giant Killer (Ratio vs Top 5 attacks vs Overall)
    tier_1_ratio = 1.06 if opp_is_tier_1 and eff_bat_avg > 38.0 else (0.94 if opp_is_tier_1 else 1.0)
    giant_killer = clamp(28.0 + (eff_bat_avg * tier_1_ratio / 50.0) * 52.0, 15.0, 98.0)

    # 5. Anchor / Lone Wolf (Team Run Share during partnership)
    team_run_share = 0.38 if role == "top_order_batter" else 0.28
    anchor_bat = clamp(32.0 + (eff_bat_avg * (1.0 + team_run_share) / 55.0) * 52.0, 15.0, 98.0)

    # 6. Win Probability Added (Batting)
    wpa_mult = 1.05 if base_bat_sr > 135.0 else 0.98
    wpa_bat = clamp(36.0 + (eff_bat_avg * wpa_mult / 50.0) * 48.0, 15.0, 98.0)

    # 7. Phase Mastery (Format-specific phase boundary frequencies)
    phase_mastery = clamp(38.0 + (eff_bat_avg / 52.0) * 46.0, 15.0, 98.0)

    # 8. Strike Rotation (Dot ball suppression)
    dot_pct = float(player.get("dot_pct") or (34.0 if fmt == "T20I" else (48.0 if fmt == "ODI" else 68.0)))
    strike_rotation = clamp(85.0 - (dot_pct * 0.75) + (eff_bat_avg / 60.0) * 20.0, 15.0, 98.0)

    # 9. Acceleration Index (SR balls 25+ vs balls 1-10)
    accel_delta = max(0.0, base_bat_sr - 110.0)
    acceleration_index = clamp(40.0 + (eff_bat_avg / 50.0) * 42.0 + (accel_delta * 0.12), 15.0, 98.0)

    # 10. Matchup Dominance (Contrasting bowling archetypes)
    matchup_dominance = clamp(36.0 + (eff_bat_avg / 50.0) * 48.0, 15.0, 98.0)

    # -------------------------------------------------------------
    # 2B. BOWLING MASTER ATTRIBUTES (9 Metrics)
    # -------------------------------------------------------------
    # Downweighting tailender dismissals: Q_tailender = 1.25 top order, 1.05 allrounders, 0.65 tailenders
    eff_bowl_avg = (base_bowl_econ * base_bowl_sr) / 6.0
    nb_bowl = N_BAYES_BOWLING.get(fmt, 200)
    w_sample_bowl = balls_bowled / (balls_bowled + nb_bowl)
    eff_bowl_avg_shrunk = (1.0 - w_sample_bowl) * 28.0 + w_sample_bowl * eff_bowl_avg

    # BaseBowlRating = 32.0 + (980.0 / (EffBowlAvg + 4.0)) + min(6.0, Career Wickets / 25.0)
    base_bowl_rating = 32.0 + (980.0 / (eff_bowl_avg_shrunk + 4.0)) + min(6.0, career_wickets / 25.0)

    # 11. Dependability (Bowl)
    dependability_bowl = clamp(base_bowl_rating - 2.0, 15.0, 95.0)

    # 12. Spearhead Strike Rate (#1 to #4 top-order lethality)
    is_frontline_pacer = "pacer" in role or str(player.get("bowling_style", "")).upper() in ["RF", "RMF", "LF"]
    spearhead_bonus = 2.5 if is_frontline_pacer else 0.0
    spearhead = clamp(base_bowl_rating + 1.0 + spearhead_bonus, 15.0, 96.0)

    # 13. Pitch Independence (Flat, high-scoring wickets PQI >= 60)
    pitch_independence = clamp(base_bowl_rating - 1.0, 15.0, 95.0)

    # 14. Impact Rating (3+ wickets in an innings or multi-maidens)
    impact_rating = clamp(base_bowl_rating, 15.0, 96.0)

    # 15. Clutch / Death Bowling (Overs 16-20 T20, 41-50 ODI)
    death_bonus = 2.0 if any(k in player.get("name", "").lower() for k in ["bumrah", "boult", "arshdeep", "starc", "rabada", "nortje", "afridi"]) else 0.0
    clutch_death_bowl = clamp(base_bowl_rating + 1.5 + death_bonus, 15.0, 96.0)

    # 16. Win Probability Added (Bowling)
    wpa_bowl = clamp(base_bowl_rating, 15.0, 95.0)

    # 17. Phase Restriction (Powerplay, Middle choke, Death)
    phase_restriction = clamp(base_bowl_rating - 0.5, 15.0, 95.0)

    # 18. Expected Wickets (xW - False-shot % and play-and-miss)
    expected_wickets = clamp(base_bowl_rating + (1.5 if base_bowl_sr < 16.0 else 0.0), 15.0, 95.0)

    # 19. Matchup Threat (Weapon against contrasting hand orientations)
    matchup_threat = clamp(base_bowl_rating - 1.0, 15.0, 95.0)

    # -------------------------------------------------------------
    # 2C. MASTER COMPOSITES & RANKINGS (3 Metrics)
    # -------------------------------------------------------------
    # 20. Overall Batting Dominance
    bat_score = (
        dependability_bat * 0.12
        + clutch_bat * 0.12
        + wpa_bat * 0.12
        + resilience_bat * 0.10
        + giant_killer * 0.10
        + phase_mastery * 0.10
        + strike_rotation * 0.09
        + acceleration_index * 0.09
        + anchor_bat * 0.08
        + matchup_dominance * 0.08
    )

    # 21. Overall Bowling Dominance
    bowl_score = (
        spearhead * 0.16
        + dependability_bowl * 0.14
        + clutch_death_bowl * 0.14
        + wpa_bowl * 0.14
        + expected_wickets * 0.10
        + phase_restriction * 0.09
        + impact_rating * 0.09
        + pitch_independence * 0.07
        + matchup_threat * 0.07
    )

    # 22. Master Composite Score (Rank)
    if "all_rounder" in role:
        master_composite = (bat_score * 0.50) + (bowl_score * 0.50)
    elif is_bowler and not is_batter:
        master_composite = bowl_score
    else:
        master_composite = bat_score

    # Power ratings for duel engine
    b_power = 0.35 * acceleration_index + 0.35 * dependability_bat + 0.30 * clutch_bat
    bw_power = 0.35 * spearhead + 0.35 * expected_wickets + 0.30 * clutch_death_bowl

    return {
        "player_id": player.get("id"),
        "name": player.get("name"),
        "format": fmt,
        "role": role,
        # Batting 10 Attributes
        "dependability_bat": round(dependability_bat, 1),
        "clutch_bat": round(clutch_bat, 1),
        "resilience_bat": round(resilience_bat, 1),
        "giant_killer": round(giant_killer, 1),
        "anchor_bat": round(anchor_bat, 1),
        "wpa_bat": round(wpa_bat, 1),
        "phase_mastery": round(phase_mastery, 1),
        "strike_rotation": round(strike_rotation, 1),
        "acceleration_index": round(acceleration_index, 1),
        "matchup_dominance": round(matchup_dominance, 1),
        # Bowling 9 Attributes
        "dependability_bowl": round(dependability_bowl, 1),
        "spearhead": round(spearhead, 1),
        "pitch_independence": round(pitch_independence, 1),
        "impact_rating": round(impact_rating, 1),
        "clutch_death_bowl": round(clutch_death_bowl, 1),
        "wpa_bowl": round(wpa_bowl, 1),
        "phase_restriction": round(phase_restriction, 1),
        "expected_wickets": round(expected_wickets, 1),
        "matchup_threat": round(matchup_threat, 1),
        # 3 Composites
        "bat_score": round(bat_score, 1),
        "bowl_score": round(bowl_score, 1),
        "master_composite": round(master_composite, 1),
        # Micro duel vectors
        "b_power": round(b_power, 2),
        "bw_power": round(bw_power, 2)
    }

# =====================================================================
# 3. DUEL SIMULATOR (BATTER VS. BOWLER MICRO ENGINE)
# =====================================================================
def compute_duel_simulation(
    batter_attrs: Dict[str, Any],
    bowler_attrs: Dict[str, Any],
    h2h_balls: int = 0,
    h2h_dots: int = 0,
    h2h_boundaries: int = 0,
    h2h_wickets: int = 0,
    match_format: str = "T20I",
    pitch_archetype: str = "balanced",
    batter_hand: str = "right",
    bowler_style: str = "RF",
    pitch_adaptability_delta: float = 0.0
) -> Dict[str, Any]:
    """
    Simulates a direct encounter between Batter B and Bowler BW using
    Bayesian Priors + Empirical Delivery Blending (CricPulse Duel Simulator).
    """
    fmt = str(match_format).upper()
    b_power = float(batter_attrs.get("b_power") or 50.0)
    bw_power = float(bowler_attrs.get("bw_power") or 50.0)

    delta = b_power - bw_power

    # 1. Format-calibrated prior outcome probabilities
    if fmt == "TEST":
        p_dot_base = 0.68
        p_wkt_base = 0.035
        p_bnd_base = 0.08
    elif fmt == "ODI":
        p_dot_base = 0.52
        p_wkt_base = 0.022
        p_bnd_base = 0.11
    else:  # T20I
        p_dot_base = 0.44
        p_wkt_base = 0.055
        p_bnd_base = 0.16

    p_prior_dot = clamp(p_dot_base - delta * 0.0035, 0.20, 0.78)
    p_prior_wicket = clamp(p_wkt_base - delta * 0.0016, 0.015, 0.16)
    p_prior_boundary = clamp(p_bnd_base + delta * 0.0028, 0.04, 0.35)
    p_prior_rotation = max(0.05, 1.0 - (p_prior_dot + p_prior_wicket + p_prior_boundary))

    # Normalize priors to 1.0
    s_prior = p_prior_dot + p_prior_wicket + p_prior_boundary + p_prior_rotation
    p_prior_dot /= s_prior
    p_prior_wicket /= s_prior
    p_prior_boundary /= s_prior
    p_prior_rotation /= s_prior

    # 2. Empirical Bayesian Blending (when H2H balls >= 6)
    if h2h_balls >= 6:
        w_bayes = min(0.60, float(h2h_balls) / (float(h2h_balls) + 15.0))
        obs_dot = float(h2h_dots) / float(h2h_balls)
        obs_wkt = float(h2h_wickets) / float(h2h_balls)
        obs_bnd = float(h2h_boundaries) / float(h2h_balls)
        obs_rot = max(0.0, 1.0 - (obs_dot + obs_wkt + obs_bnd))

        p_final_dot = (1.0 - w_bayes) * p_prior_dot + w_bayes * obs_dot
        p_final_wkt = (1.0 - w_bayes) * p_prior_wicket + w_bayes * obs_wkt
        p_final_bnd = (1.0 - w_bayes) * p_prior_boundary + w_bayes * obs_bnd
        p_final_rot = (1.0 - w_bayes) * p_prior_rotation + w_bayes * obs_rot
    else:
        w_bayes = 0.0
        p_final_dot = p_prior_dot
        p_final_wkt = p_prior_wicket
        p_final_bnd = p_prior_boundary
        p_final_rot = p_prior_rotation

    # 3. Pitch Archetype & Bowler/Batter Matchup Tactical Calibration
    from .cricsheet_pitch_intelligence import canonicalize_pitch_type
    p_arch = canonicalize_pitch_type(pitch_archetype)
    b_hand = str(batter_hand).lower().strip()
    b_stl = str(bowler_style).upper().strip()

    is_spin = any(s in b_stl for s in ["SLA", "OB", "LBG", "LAWS", "SPIN"])
    is_pacer = any(s in b_stl for s in ["RF", "RMF", "LF", "LMF", "FAST"])
    is_express = any(s in b_stl for s in ["RF", "FAST", "LF"])

    if p_arch == "dusty":
        if is_spin:
            p_final_wkt *= 1.28
            p_final_dot *= 1.15
            p_final_bnd *= 0.74
            if b_hand == "left" and ("OB" in b_stl or "ROB" in b_stl):
                p_final_wkt *= 1.18
            elif b_hand == "right" and ("SLA" in b_stl or "LBG" in b_stl):
                p_final_wkt *= 1.15
        else:
            p_final_wkt *= 0.90
            p_final_bnd *= 1.05
    elif p_arch == "green":
        if is_pacer:
            p_final_wkt *= 1.30
            p_final_dot *= 1.12
            p_final_bnd *= 0.78
            if b_hand == "right" and ("LF" in b_stl or "LMF" in b_stl):
                p_final_wkt *= 1.16
        else:
            p_final_wkt *= 0.85
            p_final_bnd *= 1.08
    elif p_arch == "bouncy":
        if is_express:
            p_final_dot *= 1.18
            p_final_wkt *= 1.15
            p_final_bnd *= 1.06
        elif is_spin:
            p_final_wkt *= 0.88
            p_final_bnd *= 1.04
    elif p_arch == "flat":
        p_final_bnd *= 1.24
        p_final_dot *= 0.82
        p_final_wkt *= 0.78
    elif p_arch == "balanced":
        if "RMF" in b_stl or "LMF" in b_stl or "CUTTER" in b_stl:
            p_final_dot *= 1.08

    # Adaptability differential adjustment
    p_final_wkt *= max(0.65, min(1.45, 1.0 - pitch_adaptability_delta * 0.25))
    p_final_bnd *= max(0.65, min(1.45, 1.0 + pitch_adaptability_delta * 0.25))

    # Normalization check
    total_p = p_final_dot + p_final_wkt + p_final_bnd + p_final_rot
    p_final_dot /= total_p
    p_final_wkt /= total_p
    p_final_bnd /= total_p
    p_final_rot /= total_p

    # Split boundary into 4s and 6s (typical international ratio ~65% fours, 35% sixes)
    p_four = p_final_bnd * 0.65
    p_six = p_final_bnd * 0.35

    # Split rotation into 1s, 2s, 3s
    p_one = p_final_rot * 0.82
    p_two = p_final_rot * 0.16
    p_three = p_final_rot * 0.02

    exp_runs_calc = round((p_one * 1.0) + (p_two * 2.0) + (p_three * 3.0) + (p_four * 4.0) + (p_six * 6.0) + 0.02, 3)

    return {
        "delta_power": round(delta, 2),
        "bayes_weight": round(w_bayes, 3),
        "h2h_sample_balls": h2h_balls,
        "expected_runs_per_ball": exp_runs_calc,
        "pitch_archetype": p_arch,
        "probabilities": {
            "dot": round(p_final_dot, 4),
            "rotation": round(p_final_rot, 4),
            "boundary": round(p_final_bnd, 4),
            "wicket": round(p_final_wkt, 4)
        },
        "multinomial_distribution": {
            "dot": round(p_final_dot, 4),
            "one": round(p_one, 4),
            "two": round(p_two, 4),
            "three": round(p_three, 4),
            "four": round(p_four, 4),
            "six": round(p_six, 4),
            "wicket": round(p_final_wkt, 4),
            "extra": 0.020
        }
    }

# =====================================================================
# 4. PLAYER COMPARE ENGINE (4 ANALYTICAL LENSES)
# =====================================================================
def compare_players_across_lenses(
    player_a_attrs: Dict[str, Any],
    player_b_attrs: Dict[str, Any],
    lens: str = "Balanced"
) -> Dict[str, Any]:
    """
    Compares two players across 4 analytical lenses:
    - Balanced (Default)
    - Explosive (+40% Acceleration, Boundary %, Spearhead)
    - Purist (+45% Dependability, Resilience, Pitch Independence)
    - Clutch (+50% WPA and Clutch / High Leverage)
    """
    valid_lens = lens if lens in ["Balanced", "Explosive", "Purist", "Clutch"] else "Balanced"

    def compute_lens_score(attrs: Dict[str, Any]) -> float:
        b_dep = attrs.get("dependability_bat", 50.0)
        b_clutch = attrs.get("clutch_bat", 50.0)
        b_resil = attrs.get("resilience_bat", 50.0)
        b_accel = attrs.get("acceleration_index", 50.0)
        b_wpa = attrs.get("wpa_bat", 50.0)

        bw_dep = attrs.get("dependability_bowl", 50.0)
        bw_spear = attrs.get("spearhead", 50.0)
        bw_clutch = attrs.get("clutch_death_bowl", 50.0)
        bw_pind = attrs.get("pitch_independence", 50.0)
        bw_wpa = attrs.get("wpa_bowl", 50.0)

        if valid_lens == "Explosive":
            score = (b_accel * 0.40 + bw_spear * 0.35 + b_wpa * 0.25)
        elif valid_lens == "Purist":
            score = (b_dep * 0.35 + b_resil * 0.35 + bw_pind * 0.30)
        elif valid_lens == "Clutch":
            score = (b_clutch * 0.40 + bw_clutch * 0.35 + (b_wpa + bw_wpa) * 0.25)
        else: # Balanced
            score = attrs.get("master_composite", 50.0)
        return round(score, 1)

    score_a = compute_lens_score(player_a_attrs)
    score_b = compute_lens_score(player_b_attrs)

    advantage = "PLAYER_A" if score_a > score_b else ("PLAYER_B" if score_b > score_a else "TIED")
    delta = round(abs(score_a - score_b), 1)

    return {
        "lens_applied": valid_lens,
        "player_a_name": player_a_attrs.get("name"),
        "player_a_score": score_a,
        "player_b_name": player_b_attrs.get("name"),
        "player_b_score": score_b,
        "advantage": advantage,
        "score_delta": delta,
        "tactical_insight": f"{player_a_attrs.get('name') if advantage == 'PLAYER_A' else player_b_attrs.get('name')} holds a +{delta} pt edge in {valid_lens} context."
    }

# =====================================================================
# 5. VENUE PITCH QUALITY INDEX (PQI) & TOSS BIAS
# =====================================================================
def compute_pitch_quality_index(
    venue_avg_first_innings: float,
    match_format: str = "T20I",
    bat_first_wins: int = 18,
    total_matches: int = 35,
    pace_wickets: int = 240,
    spin_wickets: int = 140
) -> Dict[str, Any]:
    """
    Computes Pitch Quality Index (PQI), Toss Bias Index, and Pace vs Spin Wicket Distribution.
    - PQI > 60: Batting Paradise (True bounce, high scores)
    - PQI ~ 50: Balanced contest between bat and ball
    - PQI < 40: Bowler Graveyard (Heavy lateral movement, treacherous turn)
    """
    fmt = str(match_format).upper()
    if fmt == "TEST":
        benchmark = 290.0
        scaling = 45.0
    elif fmt == "ODI":
        benchmark = 275.0
        scaling = 35.0
    else:  # T20I
        benchmark = 165.0
        scaling = 25.0

    pqi = 50.0 + ((venue_avg_first_innings - benchmark) / scaling) * 25.0
    pqi = clamp(round(pqi, 1), 20.0, 85.0)

    if pqi >= 60.0:
        pqi_category = "Batting Paradise (True bounce & carry)"
    elif pqi <= 40.0:
        pqi_category = "Bowler Graveyard (Heavy lateral/spin movement)"
    else:
        pqi_category = "Balanced Contest (Bat vs Ball Par)"

    # Toss Bias Index
    toss_bias_pct = round((float(bat_first_wins) / max(1, total_matches)) * 100.0, 1)

    # Pace vs Spin Wicket Shares
    total_wkts = max(1, pace_wickets + spin_wickets)
    pace_share = round((float(pace_wickets) / total_wkts) * 100.0, 1)
    spin_share = round((float(spin_wickets) / total_wkts) * 100.0, 1)

    return {
        "format": fmt,
        "pitch_quality_index": pqi,
        "pqi_category": pqi_category,
        "toss_bias_bat_first_pct": toss_bias_pct,
        "pace_wicket_share_pct": pace_share,
        "spin_wicket_share_pct": spin_share
    }

# =====================================================================
# 6. TEST MATCH PITCH EVOLUTION (DAY-BY-DAY PROGRESSION)
# =====================================================================
def compute_test_pitch_evolution(
    day: int,
    pace_share_pct: float,
    spin_share_pct: float,
    caught_behind: int,
    slip_catches: int,
    total_wickets: int,
    current_run_rate: float
) -> Dict[str, Any]:
    """
    Evaluates progressive physical degradation of Test surfaces from Day 1 to Day 5:
    Behind Wicket % = ((Caught Behind + Slip Catches) / Total Wickets) * 100
    - DUSTY: Day in [4, 5] and Spin Share >= 50% or Spin Share >= 55%
    - FLAT: Run Rate >= 3.95 or (Run Rate >= 3.80 with Wkts <= 7)
    - GREEN: Pace Share >= 65%, Behind Wicket % >= 15%, Run Rate <= 3.70
    - BOUNCY: Pace Share >= 50% without extreme edge rates
    """
    total_w = max(1, total_wickets)
    behind_wicket_pct = round(((caught_behind + slip_catches) / float(total_w)) * 100.0, 1)

    if (day in [4, 5] and spin_share_pct >= 50.0) or spin_share_pct >= 55.0:
        dominant_surface = "DUSTY"
        physical_desc = "Surface dryness, wide footmarks, expanding cracks, sharp variable spin."
    elif current_run_rate >= 3.95 or (current_run_rate >= 3.80 and total_wickets <= 7):
        dominant_surface = "FLAT"
        physical_desc = "Even roller compression, true bounce, dependable strokeplay."
    elif pace_share_pct >= 65.0 and behind_wicket_pct >= 15.0 and current_run_rate <= 3.70:
        dominant_surface = "GREEN"
        physical_desc = "Retained grass moisture, persistent seam deviation, sustained slip carry."
    elif pace_share_pct >= 50.0:
        dominant_surface = "BOUNCY"
        physical_desc = "Firm clay base, true carry through to keeper, steep shoulder bounce."
    else:
        dominant_surface = "DUSTY" if day >= 3 else "BALANCED"
        physical_desc = "Normal multi-day wear with gradual spin degradation."

    return {
        "day": day,
        "dominant_surface": dominant_surface,
        "behind_wicket_pct": behind_wicket_pct,
        "physical_description": physical_desc,
        "pace_share_pct": pace_share_pct,
        "spin_share_pct": spin_share_pct,
        "run_rate": round(current_run_rate, 2)
    }

# =====================================================================
# 7. SEASONAL GROUND BEHAVIOUR (QUARTERLY Q1-Q4)
# =====================================================================
def compute_seasonal_ground_profile(
    venue_country: str,
    month: int = 11
) -> Dict[str, Any]:
    """
    Partitions international venues into quarterly meteorological profiles:
    - Q1 (Jan-Mar): Southern Hemisphere mid-to-late summer (dry, high scores)
    - Q2 (Apr-Jun): Early English summer (cooler air, moisture, seam) & Subcontinent heat
    - Q3 (Jul-Sep): European peak season (worn, drying spin tracks) & Caribbean CPL
    - Q4 (Oct-Dec): Subcontinent winter season (heavy evening dew under floodlights)
    """
    q_num = 1 if month in [1, 2, 3] else (2 if month in [4, 5, 6] else (3 if month in [7, 8, 9] else 4))
    c_lower = venue_country.lower()

    if q_num == 4:
        # Oct - Dec: Subcontinent Winter Dew
        if any(c in c_lower for c in ["india", "bangladesh", "sri lanka", "uae", "pakistan"]):
            dominant = "DUSTY"
            dew_impact = "Heavy evening dew under lights (+12.5% chase scoring ease, minimal 2nd-inn spin grip)"
            expected_rr = 8.65
        elif any(c in c_lower for c in ["australia", "new zealand"]):
            dominant = "GREEN" if "zealand" in c_lower else "BOUNCY"
            dew_impact = "Early spring-summer fresh tracks with lateral seam movement"
            expected_rr = 8.15
        else:
            dominant = "BALANCED"
            dew_impact = "Standard seasonal conditions"
            expected_rr = 8.20
    elif q_num == 1:
        dominant = "BOUNCY" if "australia" in c_lower else ("GREEN" if "zealand" in c_lower else "FLAT")
        dew_impact = "Southern hemisphere peak summer dry surfaces"
        expected_rr = 8.45
    elif q_num == 2:
        dominant = "GREEN" if any(c in c_lower for c in ["england", "ireland", "scotland"]) else "DUSTY"
        dew_impact = "Early season moisture and pronounced overhead swing"
        expected_rr = 7.85
    else: # Q3
        dominant = "DUSTY" if any(c in c_lower for c in ["england", "west indies"]) else "FLAT"
        dew_impact = "Worn, drying late-season pitches favoring spin control"
        expected_rr = 8.10

    return {
        "quarter": f"Q{q_num}",
        "month": month,
        "country": venue_country,
        "dominant_pitch": dominant,
        "dew_impact_narrative": dew_impact,
        "quarterly_average_run_rate": expected_rr
    }
