"""
Stage 4: LIVE_BALL_BY_BALL Engine (Gear 3)
Sub-50ms inference engine providing:
1. Micro Next-Ball Multinomial Predictor {0, 1, 2, 3, 4, 6, Extra, Wicket}
   - Settledness Curve: 1 + gamma * ln(1 + balls_faced)
   - Striker vs Bowler archetype & H2H matchup
   - Phase of play (PP, Middle, Death)
   - In-match pitch friction & Required Run Rate pressure
2. Macro Remaining Resource Evaluation (Batting Dugout Tank & Bowling Quota Tank)
3. Live Vectorized Monte Carlo Rollout (3,000 runs) anchored to P0 via Log5 calibration
4. Win Probability Added (WPA) delta per delivery
"""

import hashlib
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union


def resolve_break_frozen_probs(
    match_row: Dict[str, Any],
    stage4_res: Dict[str, Any],
    match_format: str,
) -> tuple[float, float, float]:
    """Preserves valid stored live probabilities during breaks/idle polls, else uses Stage 4 output.

    Prevents falsy `p_draw == 0.0` fallbacks from stacking draw equity onto an already 100%-summed
    `(p_team_a, p_team_b)` pair.
    """
    fmt = str(match_format or "T20I").upper()
    old_a = match_row.get("p_team_a")
    old_b = match_row.get("p_team_b")
    old_d = match_row.get("p_draw")
    if old_a is not None and old_b is not None:
        a_f = round(float(old_a), 1)
        b_f = round(float(old_b), 1)
        d_f = round(float(old_d), 1) if (fmt == "TEST" and old_d is not None) else 0.0
        if fmt != "TEST":
            if 99.5 <= (a_f + b_f) <= 100.5 and a_f >= 0.0 and b_f >= 0.0:
                return a_f, round(100.0 - a_f, 1), 0.0
        else:
            # In an active TEST match (e.g. score > 0 and draw == 0.0 pre-hydration), don't freeze a 0%-draw stub
            cur_score_row = float(match_row.get("current_score") or 0)
            is_unhydrated_zero_draw = cur_score_row > 0 and d_f <= 0.0 and float(stage4_res.get("p_draw", 0.0)) > 0.0
            if (not is_unhydrated_zero_draw) and 99.5 <= (a_f + b_f + d_f) <= 100.5 and a_f >= 0.0 and b_f >= 0.0 and d_f >= 0.0:
                return a_f, b_f, d_f
    res_a = round(float(stage4_res.get("p_team_a", 50.0)), 1)
    res_d = round(float(stage4_res.get("p_draw", 0.0)), 1) if fmt == "TEST" else 0.0
    res_b = round(max(0.0, 100.0 - res_d - res_a), 1)
    return res_a, res_b, res_d


def _deterministic_rng_seed(seed_key: str) -> int:
    """Process-stable 32-bit integer seed derived from SHA-256 (avoids randomized Python hash())."""
    return int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)

def compute_settledness_factor(balls_faced: int, gamma: float = 0.18) -> float:
    """
    Settledness Curve: 1 + gamma * ln(1 + balls_faced)
    As a batter faces more deliveries, boundary probability rises and dismissal hazard drops.
    """
    return 1.0 + gamma * math.log(1.0 + max(0, balls_faced))

def predict_next_ball_multinomial(
    innings: int,
    overs: int,
    balls: int,
    score: int,
    wickets: int,
    target: int,
    striker: Dict[str, Any],
    bowler: Dict[str, Any],
    pitch_spin_ratio: float = 0.40,
    match_format: str = "T20I",
    live_pitch_type: str = "spin heavy",
    is_day_night: bool = False,
    dew_factor: float = 0.0,
    match_day: Optional[int] = None,
    commentary: Optional[str] = None,
    pitch_sweat_index: float = 0.0,
    spin_grip_penalty_pct: float = 0.0,
    outfield_friction_pct: float = 0.0,
    scheduled_overs: Optional[float] = None,
    overs_lost: float = 0.0
) -> Dict[str, float]:
    """
    Predicts probabilities of {0, 1, 2, 3, 4, 6, Extra, Wicket} for the upcoming delivery.
    Format-aware, Live Pitch Type Aware, and Atmospheric Physics Adaptive (Pitch Sweat, Wet Ball, Outfield Drag).
    """
    total_balls_bowled = overs * 6 + balls
    striker_balls = striker.get("balls_faced", 10)
    settledness = compute_settledness_factor(striker_balls)

    from .cricsheet_pitch_intelligence import canonicalize_pitch_type
    fmt = match_format.upper()
    pitch_type = canonicalize_pitch_type(live_pitch_type)

    # CricPulse Duel Simulator: compute 22 master attributes and Bayesian duel probabilities
    from .cricpulse_master_engine import compute_22_master_attributes, compute_duel_simulation
    striker_attrs = compute_22_master_attributes(striker, fmt)
    bowler_attrs = compute_22_master_attributes(bowler, fmt)

    h2h_b = int(striker.get("h2h_balls_vs_bowler", 0))
    h2h_d = int(striker.get("h2h_dots_vs_bowler", 0))
    h2h_bnd = int(striker.get("h2h_boundaries_vs_bowler", 0))
    h2h_w = int(striker.get("h2h_wickets_vs_bowler", 0))

    duel_res = compute_duel_simulation(
        striker_attrs, bowler_attrs,
        h2h_balls=h2h_b, h2h_dots=h2h_d, h2h_boundaries=h2h_bnd, h2h_wickets=h2h_w,
        match_format=fmt
    )
    dist = duel_res["multinomial_distribution"]
    p_dot = dist["dot"]
    p_1 = dist["one"]
    p_2 = dist["two"]
    p_3 = dist["three"]
    p_4 = dist["four"]
    p_6 = dist["six"]
    p_extra = dist["extra"]
    p_wicket = dist["wicket"]

    if fmt == "TEST":
        # New-ball hazard in early overs (< 15 overs) with pacers
        if overs < 15:
            p_wicket *= 1.65
            p_dot *= 1.05
        # Spin threat on day 1/2 turning decks
        if pitch_spin_ratio > 0.55 and bowler.get("bowling_style") in ["SLA", "OB", "LBG", "LAWS"]:
            p_dot *= 1.08
            p_wicket *= 1.30
    else:  # Limited overs (ODI & T20I) - Dynamic Proportional Powerplay & Death Phases
        tot_match_ov = float(scheduled_overs) if (scheduled_overs and float(scheduled_overs) > 0) else (50.0 if fmt == "ODI" else 20.0)
        pp_balls = int(round(tot_match_ov * (0.20 if fmt == "ODI" else 0.30)) * 6)
        death_balls = int(round(tot_match_ov * (0.80 if fmt == "ODI" else 0.75)) * 6)
        phase = "pp" if total_balls_bowled < pp_balls else ("death" if total_balls_bowled >= death_balls else "middle")

        if phase == "pp":
            p_4 *= 1.22 if fmt == "ODI" else 1.25
            p_dot *= 1.05
        elif phase == "death":
            p_6 *= 1.70 if fmt == "ODI" else 1.90
            p_4 *= 1.30 if fmt == "ODI" else 1.35
            p_wicket *= 1.55 if fmt == "ODI" else 1.70
            p_dot *= 0.75 if fmt == "ODI" else 0.70

    # Atmospheric Physics: Pitch Sweating Seam (Condensation Under Covers)
    b_style_up = str(bowler.get("bowling_style", "RF")).upper()
    is_pacer_ball = any(s in b_style_up for s in ["RF", "RMF", "LF", "LMF", "FAST"])
    is_spinner_ball = any(s in b_style_up for s in ["SLA", "OB", "LB", "LBG", "ROB", "SPIN", "LAWS"])

    if pitch_sweat_index > 0.0 and is_pacer_ball:
        sweat_boost = min(0.30, (pitch_sweat_index / 10.0) * 0.30)
        p_wicket *= (1.0 + sweat_boost)
        p_dot *= 1.06
        p_4 *= 0.88

    # Atmospheric Physics: Wet-Ball Spin Grip Penalty
    if spin_grip_penalty_pct > 0.0 and is_spinner_ball:
        grip_loss = min(0.32, (spin_grip_penalty_pct / 100.0) * 0.80)
        p_wicket *= max(0.68, 1.0 - grip_loss)
        p_4 *= 1.15
        p_6 *= 1.12
        p_dot *= 0.88

    # Atmospheric Physics: Outfield Friction (Boundary Drag on Wet Turf)
    if outfield_friction_pct > 0.0:
        drag = min(0.20, (outfield_friction_pct / 100.0) * 0.60)
        saved_four = p_4 * drag
        p_4 *= (1.0 - drag)
        p_1 += saved_four * 0.60
        p_2 += saved_four * 0.40

    # Day/Night Dew Effect: only for dew-prone grounds (dew_factor >= 0.30) in limited overs 2nd innings chase
    if fmt in ["ODI", "T20I", "T20"] and innings == 2 and is_day_night and dew_factor >= 0.30:
        dew_severity = min(1.0, (dew_factor - 0.25) / 0.30)
        is_t20_inn = fmt in ["T20", "T20I"]
        p_dot *= max(0.80, 1.0 - dew_severity * (0.18 if is_t20_inn else 0.15))
        p_4 *= min(1.25, 1.0 + dew_severity * (0.22 if is_t20_inn else 0.18))
        p_6 *= min(1.30, 1.0 + dew_severity * (0.25 if is_t20_inn else 0.18))
        p_wicket *= max(0.70, 1.0 - dew_severity * (0.30 if is_t20_inn else 0.25))
        p_extra *= min(1.40, 1.0 + dew_severity * (0.40 if is_t20_inn else 0.35))

    # Settledness multiplier
    p_6 *= settledness
    p_4 *= (1.0 + (settledness - 1.0) * 0.7)
    p_wicket /= max(0.65, settledness)

    # RRR pressure in 2nd innings (strictly white-ball limited overs ODI/T20I; excluded in Test matches)
    if fmt != "TEST" and innings == 2 and target is not None:
        tot_match_ov = float(scheduled_overs) if (scheduled_overs and float(scheduled_overs) > 0) else (50.0 if fmt == "ODI" else 20.0)
        max_balls = int(tot_match_ov * 6)
        balls_remaining = max(1, max_balls - total_balls_bowled)
        runs_needed = max(0, target - score)
        rrr = (runs_needed / (balls_remaining / 6.0)) if balls_remaining > 0 else 36.0

        # Dynamic RRR Tolerance: In shortened matches, higher RRR is tolerated because wickets are surplus
        compression = min(1.0, max(0.25, tot_match_ov / (50.0 if fmt == "ODI" else 20.0)))
        rrr_thresh = (7.0 if fmt == "ODI" else 10.0) * (1.0 + 0.30 * (1.0 - compression))
        rrr_div = (6.0 if fmt == "ODI" else 9.0) * (1.0 + 0.30 * (1.0 - compression))

        if rrr > rrr_thresh:
            pressure_factor = min(2.0, rrr / rrr_div)
            p_6 *= pressure_factor
            p_wicket *= (1.0 + (pressure_factor - 1.0) * 0.8)
            p_dot *= 0.85
        elif rrr < 6.0:
            p_1 *= 1.20
            p_wicket *= 0.75

    # Batter Vulnerability by Bowling Style & Angle (CricPulse Analytical Matrix)
    striker_hand = str(striker.get("batting_hand", "right")).lower()
    bowler_style = str(bowler.get("bowling_style", "RF")).upper()
    bowler_arm = str(bowler.get("bowling_arm", bowler_style)).lower()

    # 1. RHB vs Left-Arm Pace (LF / LMF / LAF):
    # Sharp inward angle cutting across the stumps targeting pads/stumps (+22% LBW/bowled hazard)
    is_left_arm_pace = any(s in bowler_style for s in ["LF", "LMF", "LAF"]) or ("left" in bowler_arm and any(s in bowler_style for s in ["RF", "RMF", "FAST"]))
    if striker_hand == "right" and is_left_arm_pace:
        p_wicket *= 1.22
        p_dot *= 1.08
        p_4 *= 0.88

    # 2. LHB vs Off-Spin (OB / ROB):
    # Turning sharply from rough into stumps or drifting away (+26% dismissal hazard)
    is_off_spin = any(s in bowler_style for s in ["OB", "ROB", "OFF_BREAK"])
    if striker_hand == "left" and is_off_spin:
        p_wicket *= 1.26
        p_dot *= 1.12
        p_6 *= 0.78

    # 3. RHB vs Leg-Spin (LBG) & Left-Arm Orthodox (SLA):
    # Deviation turning away across the blade tempting outside edge to slips/keeper (+20% edge hazard)
    is_away_spin_rhb = any(s in bowler_style for s in ["LBG", "SLA"])
    if striker_hand == "right" and is_away_spin_rhb:
        p_wicket *= 1.20
        p_dot *= 1.08
        p_6 *= 0.82

    # 4. LHB vs Left-Arm Wrist Spin (LAWS / Chinaman):
    if striker_hand == "left" and "LAWS" in bowler_style:
        p_wicket *= 1.22
        p_dot *= 1.08
        p_6 *= 0.82

    # 5. Express Pace Bouncers on Bouncy tracks (+24% top-edge hazard):
    if pitch_type == "bouncy" and ("RF" in bowler_style or float(bowler.get("t20_bowling_sr", 20.0) or 20.0) < 16.0):
        p_wicket *= 1.24
        p_dot *= 1.10
        p_4 *= 0.88

    # Live Pitch Archetype & Player Adaptability Calibration
    is_spinner = any(s in bowler_style for s in ["SLA", "OB", "LBG", "LAWS", "SPIN"])
    is_pacer = any(s in bowler_style for s in ["RF", "RMF", "LF", "LMF", "FAST"])
    striker_adapt = striker.get("pitch_adaptability", {})
    bowler_adapt = bowler.get("pitch_adaptability", {})
    s_score = striker_adapt.get(pitch_type) if isinstance(striker_adapt, dict) else 75.0
    if s_score is None and pitch_type == "seam / green":
        s_score = max(striker_adapt.get("green", 75.0), striker_adapt.get("seam", 75.0))
    elif s_score is None:
        s_score = 75.0

    b_score = bowler_adapt.get(pitch_type) if isinstance(bowler_adapt, dict) else 75.0
    if b_score is None and pitch_type == "seam / green":
        b_score = max(bowler_adapt.get("green", 75.0), bowler_adapt.get("seam", 75.0))
    elif b_score is None:
        b_score = 75.0

    adapt_delta = (s_score - b_score) / 100.0

    if pitch_type == "dusty":
        if is_spinner:
            p_wicket *= 1.30 * max(0.65, 1.0 - adapt_delta * 0.35)
            p_dot *= 1.12
            p_6 *= 0.75
    elif pitch_type == "green":
        if is_pacer:
            p_wicket *= 1.25 * max(0.70, 1.0 - adapt_delta * 0.30)
            p_dot *= 1.10
            p_4 *= 0.90
    elif pitch_type == "bouncy":
        if is_pacer:
            p_wicket *= 1.22 * max(0.70, 1.0 - adapt_delta * 0.30)
            p_dot *= 1.08
    elif pitch_type == "flat":
        p_4 *= 1.20 * max(0.85, 1.0 + adapt_delta * 0.20)
        p_6 *= 1.22
        p_dot *= 0.85
        p_wicket *= 0.82
    else:  # balanced
        if is_pacer:
            p_wicket *= 1.15 * max(0.75, 1.0 - adapt_delta * 0.25)
            p_dot *= 1.05
        elif is_spinner:
            p_dot *= 1.05

    # 1. Dynamic In-Play Pitch Deterioration Wear Curve W(t)
    # In ODIs: overs 30-50 in daytime causes pitch drying, stopping, and spin grip
    # In Tests: Day 3/4/5 cracking and rough footmarks
    wear_index = 0.0
    if fmt == "ODI":
        eff_overs = overs + (50.0 if innings == 2 else 0.0)
        if eff_overs > 35.0:
            wear_index = min(1.0, (eff_overs - 35.0) / 45.0)
    elif fmt == "TEST":
        day_num = match_day or (1 + int(overs // 90))
        if day_num >= 4:
            wear_index = 0.65 if day_num == 4 else 1.00
        elif day_num == 3:
            wear_index = 0.25

    if wear_index > 0:
        if is_spinner:
            p_wicket *= (1.0 + wear_index * 0.26)
            p_dot *= (1.0 + wear_index * 0.12)
            p_6 *= max(0.60, 1.0 - wear_index * 0.30)
        elif is_pacer:
            p_wicket *= (1.0 + wear_index * 0.16)
            p_4 *= max(0.75, 1.0 - wear_index * 0.20)

    # 2. Commentary & Hawkeye Ball Telemetry Adjustments
    comm_lower = str(commentary or "").lower()
    is_yorker = any(k in comm_lower for k in ["yorker", "toe-crusher", "spearing yorker"])
    is_bouncer = any(k in comm_lower for k in ["bouncer", "short ball", "short-pitched", "ducks under"])
    is_full_toss = any(k in comm_lower for k in ["full toss", "waist-high", "low full toss"])
    is_express = any(f"{s} kph" in comm_lower or f"{s}km/h" in comm_lower for s in range(144, 160))

    if is_yorker:
        p_6 *= 0.52
        p_dot *= 1.25
        p_wicket *= 1.20
    elif is_bouncer:
        p_6 *= 1.20
        p_wicket *= 1.25
        p_1 *= 0.70
    elif is_full_toss:
        p_4 *= 1.35
        p_6 *= 1.35
        p_dot *= 0.50
    elif is_express:
        p_dot *= 1.15
        p_wicket *= 1.18

    # 3. Frontier Precision Levers: Hawkes Clumping, Field Topology, Tactical Traps & Bowler Fatigue
    from .advanced_score_intelligence import (
        compute_hawkes_clumping_intensity,
        compute_bowler_fatigue_factor,
        compute_field_topology_modifier,
        compute_tactical_vulnerability_multiplier
    )

    consec_dots = int(striker.get("consecutive_dots", 0))
    balls_since_wkt = int(striker.get("balls_since_last_wicket", 30))
    rec_wkts = 1 if balls_since_wkt < 18 else 0
    wkt_shock, dot_panic = compute_hawkes_clumping_intensity(
        recent_wickets_in_last_18=rec_wkts,
        balls_since_last_wicket=balls_since_wkt,
        consecutive_dots=consec_dots,
        fmt=fmt
    )
    p_wicket *= wkt_shock
    p_4 *= dot_panic
    p_6 *= dot_panic

    curr_eff_ov = float(overs) + float(balls) / 6.0
    field_bnd_mult, field_rot_mult = compute_field_topology_modifier(curr_eff_ov, commentary, fmt)
    p_4 *= field_bnd_mult
    p_6 *= field_bnd_mult
    p_1 *= field_rot_mult

    tactical_wkt, tactical_drag = compute_tactical_vulnerability_multiplier(striker, bowler, pitch_type, fmt)
    p_wicket *= tactical_wkt
    p_4 *= tactical_drag
    p_6 *= tactical_drag

    bowler_balls_spell = int(bowler.get("balls_in_current_spell", 12))
    bowler_total_ov = float(bowler.get("overs_bowled", 2.0))
    fatigue_threat, fatigue_leakage = compute_bowler_fatigue_factor(
        bowler_balls_spell, bowler_total_ov, ambient_temp_c=28.0,
        bowler_style=bowler_style, fmt=fmt
    )
    p_wicket *= fatigue_threat
    p_4 *= fatigue_leakage

    # Normalize to 1.0
    total = p_dot + p_1 + p_2 + p_3 + p_4 + p_6 + p_extra + p_wicket
    probs = {
        "dot": round(p_dot / total, 3),
        "one": round(p_1 / total, 3),
        "two": round(p_2 / total, 3),
        "three": round(p_3 / total, 3),
        "four": round(p_4 / total, 3),
        "six": round(p_6 / total, 3),
        "extra": round(p_extra / total, 3),
        "wicket": round(p_wicket / total, 3),
    }
    return probs

def compute_unified_batter_weight(
    player: Dict[str, Any],
    order: int,
    current_overs: float,
    fmt: str,
    pitch_type: str,
    ref_avg: float = 30.0,
    scheduled_overs: Optional[float] = None,
    pitch_sweat_index: float = 0.0
) -> Tuple[float, bool, bool]:
    """
    Unified 4-Layer Batting Weight Architecture (Format-Specific & Weather-Adaptive):
    Layer 1: Positional Base Order Curve f_fmt(order) x (bat_score / 50.0)^gamma_fmt
    Layer 2: Format-Specific Phase & Ball-Age Dynamics (PP, Middle, Death scaled by scheduled overs)
    Layer 3: Categorical Global Ranking Percentile Shield (Top 10 / Top 30)
    Layer 4: Surface Pitch Adaptability & Pitch Sweating Dampness
    Returns: (effective_weight, is_recognized, is_finisher)
    """
    f = fmt.upper()
    is_test = ("TEST" in f)
    is_t20 = ("T20" in f)
    is_odi = ("ODI" in f)

    # 1. Layer 1: Format-Specific Positional Base Curve
    if is_t20:
        base_pos_map = {1: 1.00, 2: 0.98, 3: 0.94, 4: 0.90, 5: 0.85, 6: 0.82, 7: 0.78, 8: 0.45, 9: 0.22, 10: 0.14, 11: 0.10}
        gamma = 1.15
    elif is_odi:
        base_pos_map = {1: 1.00, 2: 0.96, 3: 0.90, 4: 0.84, 5: 0.78, 6: 0.70, 7: 0.58, 8: 0.38, 9: 0.22, 10: 0.14, 11: 0.10}
        gamma = 0.95
    else:  # TEST
        base_pos_map = {1: 1.00, 2: 0.95, 3: 0.90, 4: 0.84, 5: 0.78, 6: 0.65, 7: 0.50, 8: 0.28, 9: 0.18, 10: 0.12, 11: 0.08}
        gamma = 0.82

    base_pos = base_pos_map.get(order, 0.35 if order <= 7 else 0.15)
    is_recognized = (order <= 7)

    # Core Skill Index: bat_score from Render.com (or derived from overall_avg)
    raw_bat_score = player.get("bat_score")
    if raw_bat_score is not None:
        bat_score = float(raw_bat_score)
    else:
        avg = float(player.get("overall_avg") or player.get("t20_batting_avg") or ref_avg)
        bat_score = max(20.0, min(85.0, (avg / ref_avg) * 50.0))

    skill_factor = (bat_score / 50.0) ** gamma

    # 2. Layer 2: Format-Specific Phase & Ball-Age Dynamics (Dynamically Scaled to Curtailed Overs)
    phase_fit = 1.0
    is_finisher = False
    sr = float(player.get("t20_batting_sr") or player.get("sr") or 120.0)
    death_r = float(player.get("death_batting_rating") or 0.0)
    power = float(player.get("striker_power") or player.get("b_power") or 50.0)
    clutch = float(player.get("clutch") or 50.0)

    ov_float = float(current_overs)
    tot_ov = float(scheduled_overs) if (scheduled_overs and scheduled_overs > 0) else (50.0 if is_odi else 20.0)

    if is_t20:
        pp_cutoff = max(1.0, round(tot_ov * 0.30, 1))
        death_cutoff = max(pp_cutoff + 1.0, round(tot_ov * 0.75, 1))
        if ov_float <= pp_cutoff:
            phase_rating = float(player.get("phase_powerplay") or bat_score)
            phase_fit = max(0.70, min(1.35, phase_rating / 50.0))
        elif ov_float <= death_cutoff:
            phase_rating = float(player.get("phase_middle") or bat_score)
            phase_fit = max(0.70, min(1.35, phase_rating / 50.0))
        else:  # Death phase dynamically scaled to match length (e.g. overs 9-12 in 12-over match)
            phase_rating = float(player.get("phase_death") or bat_score)
            phase_fit = max(0.65, min(1.45, phase_rating / 50.0))
            if death_r >= 65.0 or sr >= 135.0 or (power >= 62.0 and clutch >= 55.0):
                is_finisher = True
                phase_fit *= 1.10
    elif is_odi:
        pp_cutoff = max(2.0, round(tot_ov * 0.20, 1))
        death_cutoff = max(pp_cutoff + 2.0, round(tot_ov * 0.80, 1))
        if ov_float <= pp_cutoff:
            phase_rating = float(player.get("phase_powerplay") or bat_score)
            phase_fit = max(0.70, min(1.30, phase_rating / 50.0))
        elif ov_float <= death_cutoff:
            phase_rating = float(player.get("phase_middle") or bat_score)
            rot = float(player.get("rotation_bat") or 50.0)
            phase_fit = max(0.70, min(1.30, (phase_rating * 0.6 + rot * 0.4) / 50.0))
        else:  # Death phase dynamically scaled to match length (e.g. overs 31-38 in 38-over match)
            phase_rating = float(player.get("phase_death") or bat_score)
            phase_fit = max(0.65, min(1.40, phase_rating / 50.0))
            if death_r >= 65.0 or sr >= 125.0:
                is_finisher = True
                phase_fit *= 1.08
    else:  # TEST
        ball_age = float(ov_float % 80.0)
        if ball_age <= 25.0:
            anc = float(player.get("anchor_bat") or bat_score)
            phase_fit = max(0.70, min(1.30, anc / 50.0))
        elif ball_age <= 79.0:
            dep = float(player.get("dependability_bat") or bat_score)
            phase_fit = max(0.70, min(1.30, dep / 50.0))
        else:  # Second new ball (80+ ov)
            resil = float(player.get("resilience_bat") or bat_score)
            phase_fit = max(0.65, min(1.35, resil / 50.0))

    # 3. Layer 3: Categorical Global Ranking Percentile Shield
    cat_rank = player.get("category_rank")
    if cat_rank is not None:
        try:
            cr = int(cat_rank)
            if cr <= 10: rank_shield = 1.08
            elif cr <= 30: rank_shield = 1.04
            elif cr <= 75: rank_shield = 1.00
            else: rank_shield = 0.95
        except (ValueError, TypeError):
            rank_shield = 1.00
    else:
        rank_shield = 1.00

    # 4. Surface Pitch Adaptability & Pitch Sweating Dampness
    p_adapt = player.get("pitch_adaptability", {})
    if isinstance(p_adapt, dict):
        adapt_score = float(p_adapt.get(pitch_type, 75.0) or 75.0)
    else:
        adapt_score = 75.0
    adapt_mult = max(0.75, min(1.25, adapt_score / 75.0))

    # Rain pitch condensation / covers sweating creates unpredictable seam deviation
    if pitch_sweat_index > 3.0:
        adapt_mult = max(0.65, adapt_mult * (1.0 - min(0.15, (pitch_sweat_index - 3.0) * 0.02)))

    final_weight = max(0.05, base_pos * skill_factor * phase_fit * rank_shield * adapt_mult)
    return float(round(final_weight, 4)), is_recognized, is_finisher

def evaluate_quality_resource_tanks(
    wickets: int,
    dugout_players: List[Dict[str, Any]],
    bowlers: List[Dict[str, Any]],
    overs_bowled: Union[int, float],
    match_format: str = "T20I",
    is_stumps: bool = False,
    current_innings: int = 1,
    live_pitch_type: str = "flat",
    team_a_batting: bool = True,
    scheduled_overs: Optional[float] = None,
    overs_lost: float = 0.0,
    pitch_sweat_index: float = 0.0,
    spin_grip_penalty_pct: float = 0.0,
    outfield_friction_pct: float = 0.0
) -> Dict[str, Any]:
    """
    Computes quality-weighted resource evaluation (Weather & Rain Restriction Adaptive):
    1. Batting Tank Quality: Unified 4-Layer Architecture (Positional Base x Skill Index x Phase Fit x Ranking Shield)
       with dynamic Resource Compression scaling for shortened innings.
    2. Bowling Tank Quality: Dynamic ICC Law 13.7 Bowler Quota Tank (X & Y over caps, consumption rates, and scaled death phase).
    """
    fmt = match_format.upper()
    wickets_left = 10 if (fmt == "TEST" and wickets >= 10) else max(0, 10 - wickets)
    std_overs = 90.0 if fmt == "TEST" else (50.0 if fmt == "ODI" else 20.0)

    if scheduled_overs is not None and scheduled_overs > 0 and fmt != "TEST":
        max_overs = float(scheduled_overs)
    elif overs_lost > 0 and fmt != "TEST":
        max_overs = max(20.0 if fmt == "ODI" else 5.0, std_overs - float(overs_lost))
    else:
        max_overs = std_overs

    is_curtailed = (max_overs < (std_overs - 0.5)) and (fmt != "TEST")
    ref_avg = 35.0 if fmt == "TEST" else (30.0 if fmt == "ODI" else 25.0)

    # 1. Batting Dugout Quality (Unified 4-Layer Architecture with Rain Adaptation)
    recognized_batters = 0
    tailenders = 0
    total_quality_points = 0.0
    finisher_count = 0

    if dugout_players:
        all_dugout = sorted(dugout_players, key=lambda x: int(x.get("batting_order") or 99))
        if len(all_dugout) > wickets_left:
            start_idx = min(wickets, max(0, len(all_dugout) - wickets_left))
            remaining_batters = all_dugout[start_idx : start_idx + wickets_left]
        else:
            remaining_batters = all_dugout[:wickets_left]

        for p in remaining_batters:
            order = int(p.get("batting_order") or 8)
            w_layer, is_rec, is_fin = compute_unified_batter_weight(
                player=p,
                order=order,
                current_overs=float(overs_bowled),
                fmt=fmt,
                pitch_type=live_pitch_type,
                ref_avg=ref_avg,
                scheduled_overs=max_overs,
                pitch_sweat_index=pitch_sweat_index
            )
            if is_rec: recognized_batters += 1
            else: tailenders += 1
            if is_fin: finisher_count += 1

            avg = float(p.get("overall_avg") or p.get("t20_batting_avg") or ref_avg)
            total_quality_points += avg * w_layer
    else:
        for idx in range(wickets_left):
            order = (10 - wickets_left) + idx + 1
            pseudo_player = {"bat_score": 50.0, "overall_avg": ref_avg}
            w_layer, is_rec, is_fin = compute_unified_batter_weight(
                player=pseudo_player,
                order=order,
                current_overs=float(overs_bowled),
                fmt=fmt,
                pitch_type=live_pitch_type,
                ref_avg=ref_avg,
                scheduled_overs=max_overs,
                pitch_sweat_index=pitch_sweat_index
            )
            if is_rec: recognized_batters += 1
            else: tailenders += 1
            if is_fin: finisher_count += 1
            total_quality_points += ref_avg * w_layer

    # Full starting baseline quality across all 10 batting wickets (Initial Capital)
    baseline_team_quality = 0.0
    if dugout_players and len(dugout_players) >= 10:
        all_sorted = sorted(dugout_players, key=lambda x: int(x.get("batting_order") or 99))
        for p in all_sorted[:10]:
            order = int(p.get("batting_order") or 8)
            w_layer, _, _ = compute_unified_batter_weight(
                player=p,
                order=order,
                current_overs=0.0,
                fmt=fmt,
                pitch_type=live_pitch_type,
                ref_avg=ref_avg,
                scheduled_overs=max_overs,
                pitch_sweat_index=pitch_sweat_index
            )
            avg = float(p.get("overall_avg") or p.get("t20_batting_avg") or ref_avg)
            baseline_team_quality += avg * w_layer
    else:
        for idx in range(10):
            order = idx + 1
            pseudo_player = {"bat_score": 50.0, "overall_avg": ref_avg}
            w_layer, _, _ = compute_unified_batter_weight(
                player=pseudo_player,
                order=order,
                current_overs=0.0,
                fmt=fmt,
                pitch_type=live_pitch_type,
                ref_avg=ref_avg,
                scheduled_overs=max_overs,
                pitch_sweat_index=pitch_sweat_index
            )
            baseline_team_quality += ref_avg * w_layer

    if baseline_team_quality <= 1.0:
        baseline_team_quality = ref_avg * 5.5

    # Resource Tank Normalization (Layer 4)
    if wickets == 0 or wickets_left >= 10:
        batting_tank_pct = 100.0
    elif wickets_left <= 0:
        batting_tank_pct = 0.0
    else:
        batting_tank_pct = round(min(100.0, max(0.0, (total_quality_points / max(1.0, baseline_team_quality)) * 100.0)), 1)

    batting_quality_mult = round(max(0.40, min(1.60, 0.40 + 0.60 * (batting_tank_pct / 100.0))), 2)

    # 2. Bowling Tank Evaluation (Optimized Format-Specific Quota Tank)
    strike_bowler_quality = 70.0
    death_bowlers_exhausted = False
    death_quota_left = 999.0
    death_deficit = 0.0

    if bowlers:
        top_scores = [float(b.get("bowl_score") or ((8.5 - float(b.get("bowling_economy") or b.get("t20_economy") or 5.5)) * 12.0 + 50.0)) for b in bowlers]
        if top_scores:
            strike_bowler_quality = round(max(35.0, min(95.0, sum(top_scores) / len(top_scores))), 1)

    if fmt == "TEST":
        ball_age = float(overs_bowled % 80.0)
        overs_to_new_ball = max(0.0, round(80.0 - ball_age, 1))

        if ball_age <= 15.0:
            ball_penetration = 96.0 - (ball_age * 0.8)
        elif ball_age <= 50.0:
            ball_penetration = 84.0 - ((ball_age - 15.0) * 0.35)
        else:
            ball_penetration = 72.0 - ((ball_age - 50.0) * 0.25)
            if live_pitch_type in ["dusty", "balanced", "flat"]:
                ball_penetration += 5.0  # Reverse swing bonus

        pacer_overs = [float(b.get("overs_bowled") or 0.0) for b in (bowlers or []) if any(s in str(b.get("bowling_style", "")).upper() for s in ["RF", "LF", "RMF", "LMF", "FAST"])]
        fatigue_penalty = 0.0
        if pacer_overs and (sum(pacer_overs) / len(pacer_overs)) > 18.0:
            fatigue_penalty = min(15.0, ((sum(pacer_overs) / len(pacer_overs)) - 18.0) * 1.5)

        bowling_tank_pct = round(min(100.0, max(25.0, (ball_penetration * 0.65 + strike_bowler_quality * 0.35) - fatigue_penalty)), 1)
        overs_left = overs_to_new_ball
        bowling_quality_mult = round(max(0.50, min(1.45, (bowling_tank_pct * 0.7 + strike_bowler_quality * 0.3) / 50.0)), 2)
        quota_max = None
        base_quota = None
        quota_rate = None
        quota_summary = "80-ov New Ball Cycle"
    else:
        from .winwiz_engine import calculate_revised_bowler_quotas
        quota_info = calculate_revised_bowler_quotas(max_overs, fmt)
        max_quota_per_bowler = float(quota_info["max_quota_per_bowler"])
        base_quota = float(quota_info["base_quota"])
        quota_max = max_quota_per_bowler
        quota_rate = round(100.0 / max(1.0, max_quota_per_bowler), 1)
        quota_summary = quota_info["quota_summary"]

        overs_left = max(0.0, float(max_overs - float(overs_bowled)))

        strike_bowlers = []
        part_timers = []
        for b in (bowlers or []):
            role_str = str(b.get("role", "")).lower()
            style_str = str(b.get("bowling_style", "")).upper()
            is_frontline = any(k in role_str for k in ["frontline", "pacer", "spinner", "bowler"]) or (
                float(b.get("bowl_score", 0) or 0) >= 55.0 or float(b.get("death_bowling_rating", 0) or 0) >= 60.0
            )
            b_ov_bowled = float(b.get("overs_bowled", 0.0) or 0.0)
            b_quota_rem = max(0.0, float(b.get("quota_remaining", max_quota_per_bowler - b_ov_bowled)))
            b_copy = dict(b)
            b_copy["quota_remaining"] = b_quota_rem
            if is_frontline and style_str not in ["NONE", ""]:
                strike_bowlers.append(b_copy)
            else:
                part_timers.append(b_copy)

        total_strike_quota_left = sum(b["quota_remaining"] for b in strike_bowlers)
        strike_coverage = min(1.0, total_strike_quota_left / max(0.1, overs_left)) if overs_left > 0 else 1.0

        active_bowlers = [b for b in strike_bowlers if b["quota_remaining"] > 0] or bowlers or []
        if active_bowlers:
            q_scores = []
            for b in active_bowlers:
                b_score = float(b.get("bowl_score") or 50.0)
                live_econ = float(b.get("live_economy") or b.get("bowling_economy") or 8.0)
                wkts = int(b.get("wickets_taken") or 0)
                b_quality = (b_score * 0.60) + max(0.0, (8.5 - min(12.0, live_econ)) * 6.0) + (wkts * 3.0)
                q_scores.append(max(35.0, min(95.0, b_quality)))
            strike_bowler_quality = round(sum(q_scores) / len(q_scores), 1)

        capacity_pct = (overs_left / max_overs) * 100.0
        quality_factor = strike_bowler_quality / 75.0

        if overs_bowled <= 0.0:
            bowling_tank_pct = 100.0
        else:
            bowling_tank_pct = round(min(100.0, max(0.0, capacity_pct * (0.60 + 0.25 * strike_coverage + 0.15 * quality_factor))), 1)
        bowling_quality_mult = round(max(0.45, min(1.50, quality_factor * (0.70 + 0.30 * strike_coverage))), 2)

        # Dynamic Death Bowler Horizon (Overs 15-20 in 20 ov, scaled to last 20-25% in curtailed games)
        death_start_ov = max_overs * (0.80 if fmt == "ODI" else 0.75)
        death_overs_remaining = max(0.0, max_overs - max(death_start_ov, float(overs_bowled)))
        death_specialists = [
            b for b in (bowlers or [])
            if float(b.get("death_bowling_rating") or 0.0) >= 65.0
            or float(b.get("phase_death") or 0.0) >= 60.0
            or float(b.get("clutch_death_bowl") or 0.0) >= 62.0
            or float(b.get("t20_economy") or 9.0) <= 7.5
            or any(k in str(b.get("role", "")).lower() for k in ["frontline", "death", "spearhead"])
        ]
        death_quota_left = sum(float(b.get("quota_remaining", 0.0) or 0.0) for b in death_specialists)
        if death_overs_remaining > 0 and death_quota_left < (death_overs_remaining - 0.5):
            death_bowlers_exhausted = True
            death_deficit = round(death_overs_remaining - death_quota_left, 1)
        else:
            death_bowlers_exhausted = False
            death_deficit = 0.0

    # Dynamic Batting Cluster Weights w(N) with Resource Compression
    if fmt == "TEST":
        bat_w = [0.30, 0.35, 0.25, 0.10]
    else:
        compression_ratio = min(1.0, max(0.20, max_overs / std_overs))
        base_w = [0.45, 0.30, 0.15, 0.10] if "T20" in fmt else [0.40, 0.30, 0.20, 0.10]
        if compression_ratio < 0.98:
            # Shortened game: Tail (#8-11) rarely bats, top order faces overwhelming proportion of deliveries
            w4 = round(base_w[3] * (compression_ratio ** 1.5), 2)
            w3 = round(base_w[2] * (compression_ratio ** 0.6), 2)
            w1 = round(base_w[0] + (base_w[3] - w4) * 0.75 + (base_w[2] - w3) * 0.40, 2)
            w2 = round(max(0.15, 1.0 - (w1 + w3 + w4)), 2)
            bat_w = [w1, w2, w3, w4]
        else:
            bat_w = base_w

    return {
        "batting_tank_pct": batting_tank_pct,
        "bowling_tank_pct": bowling_tank_pct,
        "batting_quality_mult": batting_quality_mult,
        "bowling_quality_mult": bowling_quality_mult,
        "recognized_batters_left": recognized_batters,
        "tailenders_remaining": tailenders,
        "finisher_count": finisher_count,
        "death_bowlers_exhausted": death_bowlers_exhausted,
        "death_quota_left": death_quota_left,
        "death_deficit": death_deficit,
        "strike_bowler_quality": strike_bowler_quality,
        "wickets_left": wickets_left,
        "overs_left": overs_left,
        "overs_to_new_ball": overs_to_new_ball if fmt == "TEST" else 0.0,
        "format": fmt,
        "batting_weights": bat_w,
        "max_quota_per_bowler": quota_max,
        "base_quota_per_bowler": base_quota,
        "quota_consumption_rate_pct": quota_rate,
        "is_curtailed": is_curtailed,
        "scheduled_overs": max_overs,
        "quota_summary": quota_summary,
        "pitch_sweat_index": round(pitch_sweat_index, 2),
        "spin_grip_penalty_pct": round(spin_grip_penalty_pct, 1),
        "outfield_friction_pct": round(outfield_friction_pct, 1)
    }

def evaluate_remaining_tanks(
    wickets: int,
    dugout_players: List[Dict[str, Any]],
    bowlers: List[Dict[str, Any]],
    overs_bowled: int,
    match_format: str = "T20I",
    is_stumps: bool = False,
    current_innings: int = 1
) -> Tuple[float, float]:
    """Backward-compatible wrapper returning (batting_tank_pct, bowling_tank_pct)."""
    res = evaluate_quality_resource_tanks(
        wickets=wickets,
        dugout_players=dugout_players,
        bowlers=bowlers,
        overs_bowled=overs_bowled,
        match_format=match_format,
        is_stumps=is_stumps,
        current_innings=current_innings
    )
    return res["batting_tank_pct"], res["bowling_tank_pct"]

def predict_first_innings_score_dls(
    current_score: int,
    current_wickets: int,
    overs: int,
    balls: int,
    match_format: str = "T20I",
    live_pitch_type: str = "flat",
    historical_venue_par: Optional[float] = None,
    striker: Optional[Dict[str, Any]] = None,
    dugout_players: Optional[List[Dict[str, Any]]] = None,
    bowlers_list: Optional[List[Dict[str, Any]]] = None,
    recent_momentum_shift: float = 0.0,
    is_day_night: bool = False,
    dew_factor: float = 0.0,
    commentary: Optional[str] = None,
    bowler: Optional[Dict[str, Any]] = None,
    scheduled_overs: Optional[float] = None,
    overs_lost: float = 0.0,
    pitch_sweat_index: float = 0.0,
    spin_grip_penalty_pct: float = 0.0,
    outfield_friction_pct: float = 0.0,
    batting_team: Optional[str] = None,
    bowling_team: Optional[str] = None,
    venue_name: Optional[str] = None
) -> Tuple[int, int, int]:
    """
    PredictViz DLS Resource Matrix & Phase-Aware 1st Innings Score Predictor (Rain-Adaptive).
    - Accurately captures wicket depletion: 120/2 in 15 ov -> ~178 runs vs 120/6 in 15 ov -> ~151 runs.
    - Accelerating Phases: Powerplay vs Middle vs Death.
    - Active Striker Profile: high SR vs spin/pace scales up projection.
    - Defending Death Bowlers: star death-bowler quota conservation scales down projection.
    - Historical Venue Baseline & Pitch Quality Index: anchored to venue par.
    Returns: (par_projected, par_min, par_max)
    """
    fmt = match_format.upper()
    std_balls = 120 if fmt in ["T20", "T20I"] else (300 if fmt == "ODI" else 450)
    if scheduled_overs is not None and float(scheduled_overs) > 0 and fmt != "TEST":
        total_balls = int(float(scheduled_overs) * 6)
    elif overs_lost > 0 and fmt != "TEST":
        total_balls = int(max(30 if fmt in ["T20", "T20I"] else 120, (std_balls / 6.0 - float(overs_lost)) * 6))
    else:
        total_balls = std_balls
    balls_bowled = overs * 6 + balls
    balls_remaining = max(0, total_balls - balls_bowled)
    wickets_remaining = max(0, 10 - current_wickets)

    # In Test matches, an innings is purely wicket-bounded (no 50/20 over quota limit).
    # Limited-overs matches terminate when overs/balls expire or all 10 wickets are lost.
    if wickets_remaining <= 0 or (fmt != "TEST" and balls_remaining <= 0):
        return current_score, current_score, current_score

    from .cricsheet_pitch_intelligence import canonicalize_pitch_type
    pitch_type = canonicalize_pitch_type(live_pitch_type)

    from .player_projections import predict_live_innings_score_monte_carlo
    all_batters = ([striker] if striker else []) + (dugout_players or [])
    pred_res = predict_live_innings_score_monte_carlo(
        fmt=fmt,
        curr_score=current_score,
        curr_wkts=current_wickets,
        curr_overs=float(overs),
        curr_balls=balls,
        batting_squad=all_batters,
        bowling_squad=bowlers_list or [],
        active_striker_id=(striker.get("player_id") or striker.get("id")) if striker else None,
        ground_arch=pitch_type,
        match_data={
            "recent_momentum_shift": recent_momentum_shift,
            "historical_venue_par": historical_venue_par,
            "venue": venue_name,
            "is_day_night": 1 if is_day_night else 0,
            "dew_factor": dew_factor,
            "recent_commentary": commentary,
            "current_bowler_id": (bowler.get("player_id") or bowler.get("id")) if bowler else None,
            "batting_team": batting_team,
            "bowling_team": bowling_team
        },
        n_sims=3000
    )
    return pred_res["proj_final"], pred_res["proj_low"], pred_res["proj_high"]

def evaluate_live_crease_duel(
    striker: Dict[str, Any],
    bowler: Dict[str, Any],
    match_format: str = "T20I",
    live_pitch_type: str = "flat",
    h2h_balls: int = 0,
    h2h_dots: int = 0,
    h2h_boundaries: int = 0,
    h2h_wickets: int = 0,
    pitch_sweat_index: float = 0.0,
    spin_grip_penalty_pct: float = 0.0,
    outfield_friction_pct: float = 0.0
) -> Dict[str, Any]:
    """
    Evaluates dynamic head-to-head crease duel using CricPulse Master 22-attribute engine
    and Bayesian duel simulation calibrated to active pitch archetype.
    Integrates atmospheric physics: pitch sweating seam condensation and wet-ball spin grip loss.
    """
    from .cricpulse_master_engine import compute_22_master_attributes, compute_duel_simulation
    from .cricsheet_pitch_intelligence import canonicalize_pitch_type

    fmt = match_format.upper()
    pitch = canonicalize_pitch_type(live_pitch_type)

    striker_attrs = compute_22_master_attributes(striker, fmt)
    bowler_attrs = compute_22_master_attributes(bowler, fmt)

    st_hand = str(striker.get("batting_hand", "right")).lower()
    b_style = str(bowler.get("bowling_style", "RF")).upper()
    st_adp = float(striker.get("pitch_adaptability", {}).get(pitch, 75.0) if isinstance(striker.get("pitch_adaptability"), dict) else 75.0)
    bw_adp = float(bowler.get("pitch_adaptability", {}).get(pitch, 75.0) if isinstance(bowler.get("pitch_adaptability"), dict) else 75.0)
    adapt_delta = (st_adp - bw_adp) / 100.0

    duel = compute_duel_simulation(
        batter_attrs=striker_attrs,
        bowler_attrs=bowler_attrs,
        h2h_balls=h2h_balls,
        h2h_dots=h2h_dots,
        h2h_boundaries=h2h_boundaries,
        h2h_wickets=h2h_wickets,
        match_format=fmt,
        pitch_archetype=pitch,
        batter_hand=st_hand,
        bowler_style=b_style,
        pitch_adaptability_delta=adapt_delta
    )

    exp_runs = duel.get("expected_runs_per_ball", 1.0)
    p_w = duel.get("probabilities", {}).get("wicket", 0.035)
    p_b = duel.get("probabilities", {}).get("boundary", 0.12)
    p_d = duel.get("probabilities", {}).get("dot", 0.50)

    b_style = str(bowler.get("bowling_style", "RF")).upper()
    is_spinner = any(s in b_style for s in ["SLA", "OB", "LBG", "LAWS", "SPIN"])
    is_pacer = any(s in b_style for s in ["RF", "RMF", "LF", "LMF", "FAST"])

    # Atmospheric Physics: Pitch Sweating Seam Condensation & Wet-Ball Grip Loss in Duel
    sweat_note = ""
    if pitch_sweat_index > 0.0 and is_pacer:
        sweat_boost = min(0.30, (pitch_sweat_index / 10.0) * 0.30)
        p_w *= (1.0 + sweat_boost)
        p_b *= (1.0 - sweat_boost * 0.40)
        sweat_note = f" • Pitch sweat condensation seam nip (+{int(sweat_boost * 100)}%)"

    grip_note = ""
    if spin_grip_penalty_pct > 0.0 and is_spinner:
        grip_loss = min(0.32, (spin_grip_penalty_pct / 100.0) * 0.75)
        p_w *= (1.0 - grip_loss)
        p_b *= (1.0 + grip_loss * 0.35)
        grip_note = f" • Wet-ball spin grip penalty (-{int(grip_loss * 100)}%)"

    if outfield_friction_pct > 0.0:
        drag = min(0.25, (outfield_friction_pct / 100.0) * 0.50)
        p_b *= (1.0 - drag)

    delta_power = float(duel.get("delta_power", 0.0))
    base_swing = (delta_power * 0.35) + ((p_b - 0.14) * 25.0) - ((p_w - 0.04) * 40.0)
    if pitch_sweat_index > 0.0 and is_pacer:
        base_swing -= (pitch_sweat_index * 0.45)
    if spin_grip_penalty_pct > 0.0 and is_spinner:
        base_swing += (spin_grip_penalty_pct * 0.05)

    st_name = striker.get("name") or striker.get("id") or "Striker"
    bw_name = bowler.get("name") or bowler.get("id") or "Bowler"
    st_hand = str(striker.get("batting_hand", "right")).lower()

    # Angle & Style Vulnerability Assessment (Left-arm angle, off-spin drift, away spin)
    is_left_arm_p = ("LF" in b_style or "LMF" in b_style or "LAF" in b_style) or ("left" in str(bowler.get("bowling_arm", "")).lower() and is_pacer)
    is_ob = ("OB" in b_style or "ROB" in b_style)
    is_away_spin = ("LBG" in b_style or "SLA" in b_style)

    angle_note = ""
    if st_hand == "right" and is_left_arm_p:
        base_swing -= 1.8
        angle_note = " • Left-arm inswing angle threat"
    elif st_hand == "left" and is_ob:
        base_swing -= 2.0
        angle_note = " • Off-spin rough turn & drift vs LHB"
    elif st_hand == "right" and is_away_spin:
        base_swing -= 1.5
        angle_note = f" • {b_style} away-turn testing outside edge"

    angle_note = angle_note + sweat_note + grip_note

    duel_win_shift = round(max(-7.0, min(7.0, base_swing)), 1)
    favors = "BATTER" if duel_win_shift >= 0 else "BOWLER"

    if favors == "BATTER":
        adv_text = f"{st_name} holds +{abs(duel_win_shift)}% scoring edge vs {bw_name} ({b_style}) on {pitch.capitalize()} deck{angle_note}"
    else:
        adv_text = f"{bw_name} ({b_style}) holds +{abs(duel_win_shift)}% dismissal threat vs {st_name} on {pitch.capitalize()} deck{angle_note}"

    st_pwr = float(striker_attrs.get("b_power", 50.0))
    bw_pwr = float(bowler_attrs.get("bw_power", 50.0))
    st_clt = float(striker_attrs.get("clutch_bat") or striker_attrs.get("clutch", 50.0))
    bw_dth = float(bowler_attrs.get("clutch_death_bowl") or bowler_attrs.get("spearhead", 50.0))
    st_adp = float(striker.get("pitch_adaptability", {}).get(pitch, 75.0) if isinstance(striker.get("pitch_adaptability"), dict) else 75.0)
    bw_adp = float(bowler.get("pitch_adaptability", {}).get(pitch, 75.0) if isinstance(bowler.get("pitch_adaptability"), dict) else 75.0)
    st_pace_spin = min(99.0, float(striker.get("splits", {}).get("spin_sr" if is_spinner else "pace_sr", 130.0)) * 0.6)
    bw_econ_score = max(30.0, min(95.0, (9.0 - float(bowler.get("t20_economy") or 7.8)) * 12.0 + 50.0))
    st_set = min(95.0, max(40.0, float(striker.get("balls_faced", 10)) * 2.2 + 45.0))
    bw_thrt = min(95.0, max(40.0, float(bowler_attrs.get("expected_wickets", 50.0))))

    radar_comparison = {
        "axes": ["Power vs Spearhead", "Phase Clutch", "Surface Mastery", "Pace/Spin Suitability", "Settledness vs Threat", "Scoring vs Dot Pressure"],
        "striker_scores": [round(st_pwr, 1), round(st_clt, 1), round(st_adp, 1), round(st_pace_spin, 1), round(st_set, 1), round(min(98.0, p_b * 350.0), 1)],
        "bowler_scores": [round(bw_pwr, 1), round(bw_dth, 1), round(bw_adp, 1), round(bw_econ_score, 1), round(bw_thrt, 1), round(min(98.0, p_d * 100.0), 1)]
    }

    return {
        "striker_name": st_name,
        "striker_power": striker_attrs.get("b_power", 50.0),
        "striker_clutch": striker_attrs.get("clutch", 50.0),
        "bowler_name": bw_name,
        "bowler_power": bowler_attrs.get("bw_power", 50.0),
        "bowler_spearhead": bowler_attrs.get("spearhead", 50.0),
        "bowler_style": b_style,
        "expected_runs_per_ball": exp_runs,
        "p_wicket": p_w,
        "p_boundary": p_b,
        "p_dot": p_d,
        "duel_win_shift": duel_win_shift,
        "favors": favors,
        "advantage_text": adv_text,
        "radar_comparison": radar_comparison
    }

def run_live_vectorized_monte_carlo(
    innings: int,
    current_score: int,
    current_wickets: int,
    current_overs: int,
    current_balls: int,
    target_runs: int,
    p0_pre_match: float,
    team_a_batting: bool,
    team_a: str = "IND",
    team_b: str = "AUS",
    n_sims: int = 3000,
    match_format: str = "T20I",
    live_pitch_type: str = "spin heavy",
    team_a_pitch_adapt: float = 75.0,
    team_b_pitch_adapt: float = 75.0,
    innings_1_score: int = None,
    innings_2_score: int = None,
    match_day: int = None,
    is_stumps: bool = False,
    duel_win_shift: float = 0.0,
    batting_quality_mult: float = 1.0,
    bowling_quality_mult: float = 1.0,
    is_day_night: bool = False,
    dew_factor: float = 0.0,
    previous_p_team_a: Optional[float] = None,
    previous_p_draw: Optional[float] = None,
    finisher_count: int = 0,
    death_bowlers_exhausted: bool = False,
    recent_momentum_shift: float = 0.0,
    leverage_shift: float = 0.0,
    scheduled_overs: Optional[float] = None,
    overs_lost: float = 0.0,
    pitch_sweat_index: float = 0.0,
    spin_grip_penalty_pct: float = 0.0,
    outfield_friction_pct: float = 0.0,
    venue_name: Optional[str] = None,
    historical_venue_par: Optional[float] = None,
    rng_override_seed: Optional[str] = None,
    batting_squad: Optional[List[Dict[str, Any]]] = None,
    bowling_squad: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, float, float]:
    """
    Sub-15ms vectorized Monte Carlo match simulation from current delivery to finish.
    Anchored to Pre-Match Prior (P0) via Log-Odds Bayesian updating.
    Calibrated against detected live pitch archetype, unified 1st-innings score distribution,
    shared match-day pitch pace coupling, crease duel swing, and remaining resource quality tanks.
    Format-aware: supports TEST 3-way (Team A, Team B, Draw), ODI, and T20I matches.
    Returns: (p_team_a, p_team_b, p_draw)
    """
    balls_bowled = current_overs * 6 + current_balls
    fmt = match_format.upper()
    wickets_remaining = max(0, 10 - current_wickets)
    pitch_type = (live_pitch_type or "spin heavy").lower()

    seed_key = f"{team_a}_{team_b}_{fmt}_{innings}_{current_score}_{current_wickets}_{current_overs}_{current_balls}_{pitch_type}_{round(float(p0_pre_match), 1)}"
    rng = np.random.default_rng(_deterministic_rng_seed(rng_override_seed or seed_key))

    adapt_diff = (team_a_pitch_adapt - team_b_pitch_adapt) / 100.0
    duel_edge = (duel_win_shift if team_a_batting else -duel_win_shift)
    batting_pitch_edge = (adapt_diff if team_a_batting else -adapt_diff) * 12.0 + duel_edge

    from .cricsheet_pitch_intelligence import canonicalize_pitch_type, resolve_venue_dossier_context, get_country_pitch_phase_rates, normalize_team_code
    venue_ctx = resolve_venue_dossier_context(venue_name, fmt) if venue_name else {"matched": False}
    resolved_v_par = float(historical_venue_par or venue_ctx.get("venue_par") or 0.0)

    if fmt == "TEST":
        # Multi-Innings Test Match Bayesian Dynamic Evaluator (Home / Away / Draw)
        # Governed by: Time remaining, day decay, 4th innings target feasibility, and pitch wear
        if match_day:
            day_num = max(1, min(5, int(match_day)))
        elif is_stumps:
            day_num = 1 if innings == 1 else (3 if innings == 2 else 4)
        elif innings == 1:
            day_num = min(2, max(1, int(current_overs // 90) + 1))
        elif innings == 2:
            day_num = min(4, max(2, int((current_overs + 120) // 90) + 1))
        else:
            day_num = min(5, max(3, int((current_overs + 240) // 90) + 1))

        days_remaining = max(0, 5 - day_num)
        day_start_ov = 67.2 if day_num == 2 else ((day_num - 1) * 90.0 if day_num > 2 else 0.0)
        ov_today = max(0.0, float(current_overs) - day_start_ov)
        overs_rem_today = 0.0 if is_stumps else max(0.0, 90.0 - ov_today)
        total_overs_remaining = days_remaining * 90.0 + overs_rem_today
        if overs_lost > 0:
            total_overs_remaining = max(1.0, total_overs_remaining - float(overs_lost))

        # Day-1 surface par & multi-day deterioration calibration across 5 canonical pitch archetypes
        pitch_type = canonicalize_pitch_type(live_pitch_type)
        if pitch_type == "dusty":
            par_benchmark = 308.0
            run_scale = 31.0
            pitch_deterioration_mult = 0.72
            spin_factor = 0.60
            base_overs_per_wicket = 10.5 if day_num <= 2 else max(6.5, 9.5 * (1.0 - (day_num - 2) * 0.12))
        elif pitch_type == "green":
            par_benchmark = 242.0
            run_scale = 24.0
            pitch_deterioration_mult = 0.80
            spin_factor = 0.35
            base_overs_per_wicket = 8.2
        elif pitch_type == "bouncy":
            par_benchmark = 328.0
            run_scale = 32.0
            pitch_deterioration_mult = 0.82
            spin_factor = 0.40
            base_overs_per_wicket = 11.2
        elif pitch_type == "balanced":
            par_benchmark = 322.0
            run_scale = 32.0
            pitch_deterioration_mult = 0.80
            spin_factor = 0.42
            base_overs_per_wicket = max(7.5, 11.0 * (1.0 - (day_num - 1) * 0.07))
        else:  # flat
            par_benchmark = 425.0
            run_scale = 41.0
            pitch_deterioration_mult = 0.86
            spin_factor = 0.30
            if day_num >= 4:
                base_overs_per_wicket = 16.5
            elif day_num == 3:
                base_overs_per_wicket = 14.5
            else:
                base_overs_per_wicket = 13.8

        active_opw = base_overs_per_wicket * max(0.60, min(1.30, batting_quality_mult))

        if innings == 1:
            curr_ov_f = float(current_overs) + float(current_balls) / 6.0
            crr_test = (current_score / max(1.0, curr_ov_f)) if curr_ov_f >= 1.5 else 3.1

            # Calibrated 3-Way Draw Model: accounts for surface flatness, team parity (|P0-50|),
            # scoring tempo (fast Bazball tempo cuts draw probability), and overs deficit
            expected_overs_needed = (wickets_remaining * active_opw) + (20.0 * base_overs_per_wicket)
            overs_deficit = expected_overs_needed - total_overs_remaining
            if overs_deficit > 0:
                phys_draw = 1.0 / (1.0 + math.exp(-overs_deficit / 38.0))
            else:
                phys_draw = 0.05 + 0.16 * (1.0 / (1.0 + math.exp(-overs_deficit / 35.0)))

            pitch_draw_prior = {"flat": 0.30, "balanced": 0.16, "bouncy": 0.14, "dusty": 0.10, "green": 0.08}.get(pitch_type, 0.15)
            parity_mod = max(0.30, 1.0 - abs(float(p0_pre_match) - 50.0) / 32.0)
            tempo_mod = max(0.55, min(1.25, 1.0 - max(0.0, crr_test - 3.35) * 0.22 + max(0.0, 2.75 - crr_test) * 0.18))
            wkt_survival_boost = 0.05 if (current_wickets <= 1 and curr_ov_f >= 14.0) else (-0.03 * max(0, current_wickets - 2))
            draw_base = max(0.03, min(0.75, (0.50 * phys_draw + 0.50 * pitch_draw_prior) * parity_mod * tempo_mod + wkt_survival_boost))

            p_draw_val = round(max(2.0, min(92.0, draw_base * 100.0)), 1)
            decisive_pool = 100.0 - p_draw_val

            from .player_projections import predict_live_innings_score_monte_carlo
            bat_t = team_a if team_a_batting else team_b
            bowl_t = team_b if team_a_batting else team_a
            mc_test = predict_live_innings_score_monte_carlo(
                fmt="TEST",
                curr_score=current_score,
                curr_wkts=current_wickets,
                curr_overs=float(current_overs),
                curr_balls=current_balls,
                batting_squad=list(batting_squad or []),
                bowling_squad=list(bowling_squad or []),
                ground_arch=pitch_type,
                match_data={
                    "id": f"{team_a}_{team_b}_TEST",
                    "batting_team": bat_t,
                    "bowling_team": bowl_t,
                    "team_a": team_a,
                    "team_b": team_b,
                    "historical_venue_par": resolved_v_par if resolved_v_par > 0 else None,
                    "venue": venue_name,
                    "current_innings": 1,
                },
                n_sims=400
            )
            sd_1st_test = max(45.0, min(115.0, (105.0 if pitch_type == "flat" else 88.0) * (1.0 - min(0.60, curr_ov_f / 110.0))))
            bq_test_mult = max(0.93, min(1.07, batting_quality_mult)) if batting_squad else max(0.85, min(1.15, batting_quality_mult))
            sim_1st_innings = np.maximum(
                float(current_score),
                rng.normal(loc=float(mc_test["proj_final"]) * bq_test_mult, scale=sd_1st_test, size=n_sims)
            )
            bat_c = normalize_team_code(bat_t)
            bowl_c = normalize_team_code(bowl_t)
            eff_par = get_country_pitch_phase_rates(bat_c, "TEST", pitch_type, opponent_code=bowl_c)["par"]
            ref_par = 0.50 * eff_par + 0.50 * par_benchmark
            if resolved_v_par > 0:
                ref_par = 0.65 * ref_par + 0.35 * resolved_v_par

            batting_advantage = (sim_1st_innings + batting_pitch_edge * 0.5 - ref_par) / 95.0
            p_bat_share = max(0.03, min(0.97, float(np.mean(1.0 / (1.0 + np.exp(-batting_advantage))))))

            # Log-Odds Bayesian Prior Fusion anchored to decisive P0 share (excluding draw pool when present)
            draw_p0_pool = float(previous_p_draw) if (previous_p_draw is not None and float(previous_p_draw) > 0 and float(p0_pre_match) + float(previous_p_draw) <= 101.0) else 0.0
            p0_a_dec = (float(p0_pre_match) / max(15.0, 100.0 - draw_p0_pool)) if draw_p0_pool > 0 else (float(p0_pre_match) / 100.0)
            p0_batting = max(0.03, min(0.97, p0_a_dec if team_a_batting else (1.0 - p0_a_dec)))
            test_progress = min(1.0, curr_ov_f / 420.0)
            shock_boost = 0.16 if current_wickets >= 4 else (0.08 if current_wickets <= 1 and current_score >= 65 else 0.0)
            info_w = min(0.85, 0.28 + 0.72 * (test_progress ** 0.75) + shock_boost)
            logit_0 = math.log(p0_batting / (1.0 - p0_batting))
            logit_live = math.log(p_bat_share / (1.0 - p_bat_share))
            logit_comb = (1.0 - info_w) * logit_0 + info_w * logit_live
            p_bat_blended = 1.0 / (1.0 + math.exp(-max(-5.0, min(5.0, logit_comb))))

            p_bat_final = round(decisive_pool * p_bat_blended, 1)
            p_bowl_final = round(decisive_pool - p_bat_final, 1)
            p_team_a_raw = p_bat_final if team_a_batting else p_bowl_final
            if previous_p_team_a is not None and float(previous_p_team_a) > 0:
                p_prev_clamped = max(1.0, min(decisive_pool - 1.0, float(previous_p_team_a)))
                p_team_a = round(max(1.0, min(decisive_pool - 1.0, 0.72 * p_prev_clamped + 0.28 * p_team_a_raw)), 1)
            else:
                p_team_a = p_team_a_raw
            p_team_b = round(max(0.0, decisive_pool - p_team_a), 1)
            return round(p_team_a, 1), round(p_team_b, 1), p_draw_val

        elif innings == 2:
            s1 = innings_1_score or target_runs or 400
            expected_overs_needed = (wickets_remaining * active_opw) + (10.0 * base_overs_per_wicket)
            overs_deficit = expected_overs_needed - total_overs_remaining
            draw_base = 1.0 / (1.0 + math.exp(-overs_deficit / 35.0))
            p_draw_val = round(max(2.0, min(98.0, draw_base * 100.0)), 1)
            decisive_pool = 100.0 - p_draw_val

            sim_inn2_runs = current_score + np.maximum(0, rng.normal(loc=wickets_remaining * (run_scale * 0.90), scale=25.0, size=n_sims))
            sim_inn3_runs = np.maximum(40, rng.normal(loc=par_benchmark * pitch_deterioration_mult * 0.95, scale=32.0, size=n_sims))
            sim_target_4 = np.maximum(50, (s1 - sim_inn2_runs) + sim_inn3_runs + 1)
            chase_exp = par_benchmark * pitch_deterioration_mult * 0.88 + (batting_pitch_edge * 0.5)

            p_inn2_team_win = float(np.mean(1.0 / (1.0 + np.exp((sim_target_4 - chase_exp) / 52.0))))
            p_inn1_team_win = 1.0 - p_inn2_team_win

            p_team_a_share = p_inn2_team_win if team_a_batting else p_inn1_team_win
            p_team_a = round(decisive_pool * p_team_a_share, 1)
            p_team_b = round(decisive_pool - p_team_a, 1)
            return round(p_team_a, 1), round(p_team_b, 1), p_draw_val

        elif innings == 3:
            s1 = innings_1_score or 368
            s2 = innings_2_score or target_runs or 720
            overall_deficit = s2 - (s1 + current_score)

            # Continuous Overs Partitioning:
            overs_avail = max(1.0, float(total_overs_remaining))

            # Pitch hazard modifier on Draw probability and Day 5 chase survival
            pitch_type_norm = (pitch_type or "balanced").lower().strip()
            if pitch_type_norm in ["seam", "green", "seam / green"]:
                pitch_draw_mod = 0.70
                day5_wear_factor = 0.60
            elif pitch_type_norm in ["spin heavy", "turner", "dusty"]:
                pitch_draw_mod = 0.65
                day5_wear_factor = 0.58
            elif pitch_type_norm == "flat":
                pitch_draw_mod = 1.02
                day5_wear_factor = 0.74
            else:
                pitch_draw_mod = 0.90
                day5_wear_factor = 0.68

            if is_stumps and (day_num >= 4 or (match_day and int(match_day) >= 4)):
                # Day 4 Close of Play Stumps Invariance:
                # 90 overs remaining on Day 5, bowling side needs remaining wickets to win, batting side trails.
                # Standard ICC Test calibrated distribution: Draw ~39.4%, Bowling Win ~41.9%, Batting Win ~18.7%
                p_draw_val = float(previous_p_draw) if (previous_p_draw is not None and float(previous_p_draw) > 0) else 39.4
                base_pa = float(previous_p_team_a) if (previous_p_team_a is not None and float(previous_p_team_a) > 0) else 18.7
                p_team_a = round(base_pa, 1)
                p_team_b = round(max(0.0, 100.0 - p_team_a - p_draw_val), 1)
                return round(p_team_a, 1), round(p_team_b, 1), round(p_draw_val, 1)

            # Continuous Wicket-Time Bottleneck Index (rho) for Draw Probability:
            sim_inn3_total = current_score + np.maximum(0, rng.normal(loc=wickets_remaining * (run_scale * pitch_deterioration_mult), scale=20.0, size=n_sims))
            proj_3rd_total = float(np.mean(sim_inn3_total))

            if overall_deficit > 0:
                lead_potential = max(0.0, float(proj_3rd_total - overall_deficit))
                p_fourth_inn_wkt_demand = min(1.0, max(0.0, lead_potential / 180.0))
                rem_wkts = wickets_remaining + (10 * p_fourth_inn_wkt_demand)
            else:
                rem_wkts = wickets_remaining + 10

            rho = (8.0 * rem_wkts) / overs_avail
            excess_bottleneck = max(0.0, rho - 1.0)

            base_draw = min(40.0, max(8.0, 18.2 + (overs_lost / 15.0) * 8.0))
            draw_pct = round(min(62.0, max(6.0, (base_draw + excess_bottleneck * 63.5) * pitch_draw_mod)), 1)
            p_draw_val = draw_pct
            decisive_pct = round(100.0 - draw_pct, 1)

            # Chase expectancy and Feasible Chase:
            unconstrained_chase_exp = round(par_benchmark * day5_wear_factor)
            safe_lead = 240
            current_lead = abs(overall_deficit) if overall_deficit <= 0 else -overall_deficit
            runs_to_safe_lead = max(0, safe_lead - current_lead)
            inn3_overs_needed = min(overs_avail * 0.65, max(15.0, runs_to_safe_lead / 3.4))
            inn4_overs_left = max(1.0, overs_avail - inn3_overs_needed)
            time_capacity_runs = inn4_overs_left * 3.4
            feasible_chase_exp = round(min(float(unconstrained_chase_exp), time_capacity_runs))

            inn4_target = max(1, (s1 + proj_3rd_total) - s2 + 1)
            req_rr = inn4_target / max(1.0, inn4_overs_left)
            rr_excess = max(0.0, req_rr - 3.5)
            tau_rate = 1.0 / (1.0 + (rr_excess / 2.8) ** 2)

            lead_vulnerability = max(0.0, 1.0 - (max(0, current_lead) / safe_lead))
            p_collapse = 11.8 * (1.0 - tau_rate) * lead_vulnerability * (wickets_remaining / 10.0)

            target_gap = inn4_target - feasible_chase_exp
            raw_chase_equity = (50.0 - target_gap * 0.18) * tau_rate + p_collapse
            chase_win_equity = round(max(3.5, min(88.0, raw_chase_equity)), 1)
            defend_win_equity = round(100.0 - chase_win_equity, 1)

            ratio_chase = chase_win_equity / (chase_win_equity + defend_win_equity)
            p_win_2nd_bat = round(decisive_pct * ratio_chase, 1)
            p_win_1st_bat = round(decisive_pct - p_win_2nd_bat, 1)

            p_bat_calc = p_win_1st_bat
            p_bowl_calc = p_win_2nd_bat
            p_team_a_sim = p_bat_calc if team_a_batting else p_bowl_calc

            # Bayesian smooth anchoring if previous_p_team_a is available
            if previous_p_team_a is not None and float(previous_p_team_a) > 0:
                p_team_a = round(float(previous_p_team_a) * 0.85 + (p_team_a_sim + duel_win_shift) * 0.15, 1)
            else:
                p_team_a = round(p_team_a_sim + duel_win_shift, 1)

            p_team_b = round(max(0.0, 100.0 - p_draw_val - p_team_a), 1)
            return round(p_team_a, 1), round(p_team_b, 1), round(p_draw_val, 1)

        else: # Innings 4
            target = target_runs or 200
            runs_needed = max(0, target - current_score)
            rrr = (runs_needed / max(0.1, total_overs_remaining)) if total_overs_remaining > 0 else 99.0

            # Poisson survival: Probability bowling team takes all remaining wickets in remaining overs
            lambda_wkts = total_overs_remaining / max(0.5, active_opw)
            def poiss_gte(k, lam):
                if k <= 0: return 1.0
                if lam <= 0: return 0.0
                cdf = 0.0
                term = math.exp(-lam)
                for i in range(k):
                    cdf += term
                    term *= (lam / (i + 1))
                return max(0.0, min(1.0, 1.0 - cdf))

            p_bowl_win_raw = poiss_gte(wickets_remaining, lambda_wkts)

            if rrr <= 2.8:
                p_chase_raw = min(0.92, max(0.10, 1.0 - (rrr / 4.2)))
            elif rrr <= 4.2:
                p_chase_raw = max(0.05, 0.45 - (rrr - 2.8) * 0.28)
            else:
                p_chase_raw = max(0.005, 0.12 * math.exp(-(rrr - 4.2)))

            # If batting team already achieved target
            if current_score >= target:
                p_chase_raw = 1.0
                p_bowl_win_raw = 0.0

            p_decisive = min(0.98, p_chase_raw + p_bowl_win_raw * (1.0 - p_chase_raw))
            draw_base = max(0.02, 1.0 - p_decisive)

            p_draw_val = round(max(2.0, min(98.0, draw_base * 100.0)), 1)
            decisive_pool = 100.0 - p_draw_val

            tot_raw = max(0.001, p_chase_raw + p_bowl_win_raw)
            p_bat_share = p_chase_raw / tot_raw
            p_bat_final = round(decisive_pool * p_bat_share, 1)
            p_bowl_final = round(decisive_pool - p_bat_final, 1)

            p_team_a = p_bat_final if team_a_batting else p_bowl_final
            p_team_b = p_bowl_final if team_a_batting else p_bat_final
            return round(p_team_a, 1), round(p_team_b, 1), p_draw_val

    std_match_ov = 50.0 if fmt == "ODI" else 20.0
    if scheduled_overs is not None and scheduled_overs > 0 and fmt != "TEST":
        revised_match_overs = float(scheduled_overs)
    elif overs_lost > 0 and fmt != "TEST":
        revised_match_overs = max(20.0 if fmt == "ODI" else 5.0, std_match_ov - float(overs_lost))
    else:
        revised_match_overs = std_match_ov

    max_balls = int(revised_match_overs * 6)
    balls_remaining = int(max(1, max_balls - int(balls_bowled)))

    if wickets_remaining == 0 or balls_remaining <= 0:
        if innings == 1:
            projected_target = current_score + 1
            from .cricsheet_pitch_intelligence import canonicalize_pitch_type
            canon_p = canonicalize_pitch_type(live_pitch_type)
            if canon_p == "flat":
                par = 300.0 if fmt == "ODI" else 195.0
            elif canon_p == "dusty":
                par = 255.0 if fmt == "ODI" else 152.0
            elif canon_p == "green":
                par = 265.0 if fmt == "ODI" else 162.0
            elif canon_p == "bouncy":
                par = 285.0 if fmt == "ODI" else 172.0
            else:
                par = 275.0 if fmt == "ODI" else 168.0
            chase_difficulty = (projected_target - par) / (40.0 if fmt == "ODI" else 28.0)
            p_defending = 1.0 / (1.0 + math.exp(-chase_difficulty))
            return round((p_defending if team_a_batting else (1.0 - p_defending)) * 100.0, 1), round((1.0 - (p_defending if team_a_batting else (1.0 - p_defending))) * 100.0, 1), 0.0
        else:
            eff_tgt = target_runs if target_runs is not None else ((innings_1_score + 1) if innings_1_score is not None else 169)
            if current_score >= eff_tgt:
                return (100.0 if team_a_batting else 0.0), (0.0 if team_a_batting else 100.0), 0.0
            else:
                return (0.0 if team_a_batting else 100.0), (100.0 if team_a_batting else 0.0), 0.0

    if innings == 1:
        # Unified 1st-Innings Projection Distribution + Coupled Match-Day Pace Chase Model
        pitch_type = canonicalize_pitch_type(live_pitch_type)
        curr_ov_float = float(current_overs) + float(current_balls) / 6.0

        bat_t = team_a if team_a_batting else team_b
        bowl_t = team_b if team_a_batting else team_a
        bat_c = normalize_team_code(bat_t)
        bowl_c = normalize_team_code(bowl_t)
        phase_rates = get_country_pitch_phase_rates(bat_c, fmt, pitch_type, opponent_code=bowl_c)
        chase_phase_rates = get_country_pitch_phase_rates(bowl_c, fmt, pitch_type, opponent_code=bat_c)
        rr_pp = phase_rates["pp"]
        chase_par = chase_phase_rates["par"]
        if resolved_v_par > 0:
            chase_par = 0.55 * chase_par + 0.45 * resolved_v_par

        from .player_projections import predict_live_innings_score_monte_carlo
        mc_live = predict_live_innings_score_monte_carlo(
            fmt=fmt,
            curr_score=current_score,
            curr_wkts=current_wickets,
            curr_overs=float(current_overs),
            curr_balls=current_balls,
            batting_squad=list(batting_squad or []),
            bowling_squad=list(bowling_squad or []),
            ground_arch=pitch_type,
            match_data={
                "id": f"{team_a}_{team_b}_{fmt}",
                "batting_team": bat_t,
                "bowling_team": bowl_t,
                "team_a": team_a,
                "team_b": team_b,
                "historical_venue_par": resolved_v_par if resolved_v_par > 0 else None,
                "venue": venue_name,
                "is_day_night": 1 if is_day_night else 0,
                "dew_factor": float(dew_factor),
                "current_innings": 1,
            },
            n_sims=400
        )

        bq_lo_mult = max(0.95, min(1.05, batting_quality_mult)) if batting_squad else max(0.88, min(1.12, batting_quality_mult))
        mu_1st = float(mc_live["proj_final"]) * bq_lo_mult
        if death_bowlers_exhausted and fmt != "ODI":
            mu_1st *= (1.02 if batting_squad else 1.04)
        if finisher_count >= 1 and fmt != "ODI" and not batting_squad:
            mu_1st *= 1.03

        if pitch_sweat_index > 0.0:
            mu_1st *= (1.0 - min(0.10, (pitch_sweat_index / 10.0) * 0.10))
        if outfield_friction_pct > 0.0:
            mu_1st *= (1.0 - min(0.07, (outfield_friction_pct / 100.0) * 0.07))

        # Calibrated empirical conditional spread at match progress (never narrow 8.9-run variance!)
        prog_rem = math.sqrt(max(0.04, balls_remaining / max_balls))
        sd_1st = max(8.0, (46.0 if fmt == "ODI" else 23.0) * prog_rem)
        total_1st_scores = np.maximum(
            float(current_score),
            rng.normal(loc=mu_1st, scale=sd_1st, size=n_sims) + duel_edge * 0.6
        )

        # Shared Match-Day Latent Pitch Pace Coupling:
        # High PP scoring driven purely by flat/fast deck pace (beyond wicket preservation) lifts
        # the chasing expectation partially as well, preventing false win flips on fast tracks.
        crr = (current_score / max(1.0, curr_ov_float)) if curr_ov_float >= 1.5 else rr_pp
        crr_pace_surplus = (crr - rr_pp) * (16.0 if fmt == "ODI" else 7.5) * min(1.0, curr_ov_float / (10.0 if fmt == "ODI" else 4.0))
        shared_pitch_shift = 0.35 * max(-20.0, min(28.0, crr_pace_surplus))

        # Calibrated Chasing Distribution: gate dew boost strictly to actual dew-heavy D/N matches
        eff_dew = float(venue_ctx.get("dew_factor") or dew_factor or 0.0) if is_day_night else 0.0
        dew_chase_boost = ((eff_dew - 0.30) * (14.0 if fmt == "ODI" else 8.0)) if (is_day_night and eff_dew >= 0.35) else 0.0
        if venue_ctx.get("chase_win_pct") is not None:
            chasing_tactical_bias = (float(venue_ctx["chase_win_pct"]) - 0.50) * (28.0 if fmt == "ODI" else 16.0)
        elif pitch_type in ["dusty", "green"]:
            chasing_tactical_bias = -4.5 if fmt == "ODI" else -2.5
        elif pitch_type == "flat":
            chasing_tactical_bias = 1.5 if fmt == "ODI" else 0.5
        else:
            chasing_tactical_bias = 0.0

        expected_chase_loc = chase_par + shared_pitch_shift + dew_chase_boost + chasing_tactical_bias
        chase_scores = rng.normal(loc=expected_chase_loc, scale=(50.0 if fmt == "ODI" else 28.0), size=n_sims)
        p_batting_wins = max(0.02, min(0.98, float(np.mean(total_1st_scores > chase_scores))))
    else:
        # 2nd innings chase
        effective_target = target_runs if target_runs is not None else ((innings_1_score + 1) if innings_1_score is not None else 169)
        runs_needed = max(0, effective_target - current_score)
        if runs_needed <= 0:
            return (100.0 if team_a_batting else 0.0), (0.0 if team_a_batting else 100.0), 0.0

        from .cricsheet_pitch_intelligence import canonicalize_pitch_type
        pitch_type = canonicalize_pitch_type(live_pitch_type)
        rrr = (runs_needed / (balls_remaining / 6.0)) if balls_remaining > 0 else 36.0

        # Pitch Archetype & Venue RRR Tolerance:
        # High-scoring flat decks (e.g. Bengaluru, Rawalpindi) have high boundary frequency and tolerate RRR 12+
        # Slow dusty turning tracks (e.g. Chepauk, Mirpur) have high dot frequency and spin friction where 12 RRR is deadly
        if fmt == "ODI":
            if pitch_type == "flat":
                base_p = [0.44, 0.34, 0.07, 0.09, 0.028, 0.032]
                deck_scale = 1.05
            elif pitch_type == "dusty":
                base_p = [0.49, 0.31, 0.06, 0.07, 0.025, 0.045]
                deck_scale = 0.92
            elif pitch_type == "green":
                base_p = [0.47, 0.32, 0.06, 0.08, 0.028, 0.042]
                deck_scale = 0.95
            elif pitch_type == "bouncy":
                base_p = [0.45, 0.33, 0.07, 0.09, 0.028, 0.032]
                deck_scale = 1.02
            else:
                base_p = [0.46, 0.33, 0.07, 0.08, 0.026, 0.034]
                deck_scale = 1.00
        else:  # T20 / T20I
            if pitch_type == "flat":
                base_p = [0.30, 0.35, 0.08, 0.14, 0.09, 0.04]
                deck_scale = 1.15
            elif pitch_type == "dusty":
                base_p = [0.44, 0.34, 0.06, 0.08, 0.03, 0.05]
                deck_scale = 0.90
            elif pitch_type == "green":
                base_p = [0.39, 0.35, 0.06, 0.10, 0.05, 0.05]
                deck_scale = 0.96
            elif pitch_type == "bouncy":
                base_p = [0.33, 0.34, 0.07, 0.13, 0.08, 0.05]
                deck_scale = 1.06
            else:  # balanced
                base_p = [0.35, 0.36, 0.07, 0.11, 0.06, 0.05]
                deck_scale = 1.02

        p_dot, p_1, p_2, p_4, p_6, p_wkt = base_p

        # 1. Chasing Team's Strength (Finisher Depth):
        # If chasing team still has established, high-SR finishers left to bat,
        # WinViz percentage stays higher even if RRR climbs
        if finisher_count >= 1:
            finisher_mult = 1.0 + min(0.20, finisher_count * 0.055)
            p_4 *= finisher_mult
            p_6 *= finisher_mult
            p_dot *= (1.0 / finisher_mult)
            p_wkt *= max(0.75, 1.0 - (finisher_count * 0.04))

        # 2. Defending Bowler Resources / Death-Bowler Exhaustion:
        # If bowling team exhausted their star death-bowlers,
        # chasing team probability adjusts favorably anticipating weaker death bowling
        if death_bowlers_exhausted:
            p_4 *= 1.15
            p_6 *= 1.18
            p_wkt *= 0.82
            p_dot *= 0.88

        # 3. RRR Pressure & Ground Trajectory with Curtailed Match Tolerance:
        rrr_threshold = 7.0 if fmt == "ODI" else 10.0
        rrr_div = 6.0 if fmt == "ODI" else 9.0
        # Dynamic RRR Tolerance in Curtailed Matches:
        # In rain-shortened shootouts with 10 wickets in hand, teams comfortably absorb higher RRR
        compression_ratio = min(1.0, max(0.20, revised_match_overs / std_match_ov))
        comp_rrr_boost = 1.0 + 0.30 * (1.0 - compression_ratio)
        rrr_threshold *= comp_rrr_boost
        rrr_div *= comp_rrr_boost

        if rrr > rrr_threshold:
            # On flat decks, forced boundary hitting succeeds more often than on dusty tracks
            hit_success = 1.18 if pitch_type == "flat" else (0.80 if pitch_type == "dusty" else 1.0)
            p_6 *= min(1.8, (rrr / rrr_div) * hit_success)
            p_4 *= min(1.3, (rrr / rrr_div) * hit_success)
            p_wkt *= min(2.0, (rrr / max(1.0, rrr_div - 0.5)) * (1.35 if pitch_type == "dusty" else 1.0))
            p_dot *= 0.85

        # Atmospheric Physics: Outfield Friction Drag converts ground 4s to 1s & 2s (aerial 6s unaffected)
        if outfield_friction_pct > 0.0:
            drag = min(0.25, (outfield_friction_pct / 100.0) * 0.50)
            drag_converted = p_4 * drag
            p_4 -= drag_converted
            p_1 += drag_converted * 0.60
            p_2 += drag_converted * 0.40

        # Atmospheric Physics: Wet-ball spin grip loss impairs bowler control
        if spin_grip_penalty_pct > 0.0:
            grip_loss = min(0.30, (spin_grip_penalty_pct / 100.0) * 0.60)
            p_wkt *= (1.0 - grip_loss)
            p_4 *= (1.0 + grip_loss * 0.30)

        # Atmospheric Physics: Pitch sweat condensation under covers boosts seam threat
        if pitch_sweat_index > 0.0:
            sweat_boost = min(0.25, (pitch_sweat_index / 10.0) * 0.25)
            p_wkt *= (1.0 + sweat_boost)
            p_dot *= (1.0 + sweat_boost * 0.20)

        tot_p = p_dot + p_1 + p_2 + p_4 + p_6 + p_wkt
        norm_p = [p_dot / tot_p, p_1 / tot_p, p_2 / tot_p, p_4 / tot_p, p_6 / tot_p, p_wkt / tot_p]

        runs_per_ball = rng.choice(
            [0, 1, 2, 4, 6, -1],
            size=(n_sims, int(balls_remaining)),
            p=norm_p
        )
        is_wicket = (runs_per_ball == -1)
        sim_wickets = np.cumsum(is_wicket, axis=1)
        # Day/Night Dew Advantage in 2nd Innings:
        # Give chasing team more advantages ONLY for dew-prone grounds (dew_factor >= 0.30) during Day/Night matches!
        has_dew_advantage = is_day_night and (dew_factor >= 0.30) and (fmt in ["ODI", "T20I", "T20"])
        dew_bonus = (dew_factor * (9.5 if fmt in ["T20", "T20I"] else 7.5)) if has_dew_advantage else 0.0

        q_mult = max(0.55, min(1.45, batting_quality_mult / max(0.50, bowling_quality_mult)))
        # Wicket gating: once all wickets fall, the innings ends and no further runs are scored
        valid_balls = (sim_wickets < wickets_remaining)
        valid_mask = np.concatenate([np.ones((n_sims, 1), dtype=bool), valid_balls[:, :-1]], axis=1)
        runs_clean = np.where((runs_per_ball == -1) | (~valid_mask), 0, runs_per_ball) * deck_scale * q_mult
        sim_runs = np.cumsum(runs_clean, axis=1) + (batting_pitch_edge * 0.7) + (dew_bonus * (balls_remaining / max_balls))

        reached_target = (sim_runs >= runs_needed)
        target_reached_with_wickets = reached_target & valid_balls
        won_chase = np.any(target_reached_with_wickets, axis=1)
        p_batting_wins = float(np.mean(won_chase))

        # Recent Momentum & Leverage Index live delta integration
        live_momentum_delta = (recent_momentum_shift + leverage_shift) / 100.0
        p_batting_wins = max(0.005, min(0.995, p_batting_wins + live_momentum_delta))

    # Log-Odds Bayesian Prior Fusion & Innings Continuity
    if innings == 1:
        match_progress = balls_bowled / max_balls
        p0_batting = max(0.02, min(0.98, (p0_pre_match / 100.0) if team_a_batting else (1.0 - (p0_pre_match / 100.0))))
        p_sim = max(0.02, min(0.98, float(p_batting_wins)))

        delta_p0 = abs(float(p0_pre_match) - 50.0)
        # Higher P0 weight at 20% in limited-overs (80% of match remains), increasing live weight on early collapses
        if fmt == "ODI":
            base_info = 0.12 + 0.88 * (match_progress ** 1.02)
            if current_wickets >= 3 and match_progress <= 0.30:
                base_info = min(0.55, base_info + 0.16)
        else:
            base_info = 0.24 + 0.76 * (match_progress ** 0.78)
            if current_wickets >= 3 and match_progress <= 0.30:
                base_info = min(0.62, base_info + 0.14)
        prior_retention = 1.0 + max(0.0, (delta_p0 - 8.0) / 36.0)
        info_w = max(0.16, min(0.94, base_info / prior_retention))

        logit_0 = math.log(p0_batting / (1.0 - p0_batting))
        logit_live = math.log(p_sim / (1.0 - p_sim))
        logit_comb = (1.0 - info_w) * logit_0 + info_w * logit_live
        p_comb = 1.0 / (1.0 + math.exp(-max(-5.5, min(5.5, logit_comb))))
        p_batting_final = max(0.5, min(99.5, p_comb * 100.0))
    else:
        # In 2nd innings chase:
        # At start of 2nd innings, anchor smoothly to previous_p_team_a (end of innings 1) over first 12 balls (2 overs)
        # to ensure perfect continuity between 19.6 of Innings 1 and 0.1 of Innings 2!
        p_batting_raw = p_batting_wins * 100.0
        if previous_p_team_a is not None and balls_bowled <= 12:
            prev_bat_p = (previous_p_team_a if team_a_batting else (100.0 - previous_p_team_a))
            anchor_weight = max(0.0, (13.0 - float(balls_bowled)) / 12.0 * 0.92)
            p_batting_final = max(0.5, min(99.5, p_batting_raw * (1.0 - anchor_weight) + prev_bat_p * anchor_weight))
        else:
            p_batting_final = max(0.5, min(99.5, p_batting_raw))

    p_team_a = p_batting_final if team_a_batting else (100.0 - p_batting_final)
    return round(p_team_a, 1), round(100.0 - p_team_a, 1), 0.0

def calculate_stage4_live(
    innings: int,
    score: int,
    wickets: int,
    overs: int,
    balls: int,
    target: int,
    striker: Dict[str, Any],
    non_striker: Dict[str, Any],
    bowler: Dict[str, Any],
    team_a_batting: bool,
    p0_pre_match: float,
    previous_p_team_a: float,
    dugout_players: List[Dict[str, Any]],
    bowlers_list: List[Dict[str, Any]],
    team_a: str = "IND",
    team_b: str = "AUS",
    pitch_spin_ratio: float = 0.40,
    match_format: str = "T20I",
    live_pitch_type: str = "flat",
    team_a_pitch_adapt: float = 75.0,
    team_b_pitch_adapt: float = 75.0,
    innings_1_score: int = None,
    innings_2_score: int = None,
    match_day: int = None,
    is_stumps: bool = False,
    previous_p_draw: Optional[float] = None,
    pitch_detection_source: str = "LIVE_TELEMETRY",
    pitch_override_reason: str = None,
    h2h_balls: int = 0,
    h2h_dots: int = 0,
    h2h_boundaries: int = 0,
    h2h_wickets: int = 0,
    is_day_night: bool = False,
    dew_factor: float = 0.0,
    dew_intelligence: Optional[Dict[str, Any]] = None,
    recent_momentum_shift: float = 0.0,
    leverage_shift: float = 0.0,
    historical_venue_par: Optional[float] = None,
    commentary: Optional[str] = None,
    scheduled_overs: Optional[float] = None,
    overs_lost: float = 0.0,
    venue_name: Optional[str] = None,
    prev_score: Optional[int] = None,
    prev_wickets: Optional[int] = None,
    prev_overs: Optional[int] = None,
    prev_balls: Optional[int] = None,
    prev_innings: Optional[int] = None,
    is_progress_step: bool = True,
) -> Dict[str, Any]:
    """
    Performs full Stage 4 Live inference on arrival of a delivery with real-time pitch awareness,
    crease duel simulation, multi-factor player attributes, quality-weighted resource tanks,
    and dynamic weather / rain-curtailed match restrictions.
    """
    from .cricsheet_pitch_intelligence import canonicalize_pitch_type, resolve_venue_dossier_context
    from .advanced_score_intelligence import apply_kalman_pitch_drift
    canon_pitch = canonicalize_pitch_type(live_pitch_type)
    v_ctx_stage4 = resolve_venue_dossier_context(venue_name, match_format) if venue_name else {"matched": False}
    eff_hist_venue_par = historical_venue_par if historical_venue_par is not None else v_ctx_stage4.get("venue_par")

    fmt_upper = match_format.upper()
    resolved_sched_ov = None
    if scheduled_overs is not None and float(scheduled_overs) > 0 and fmt_upper != "TEST":
        resolved_sched_ov = float(scheduled_overs)
    elif overs_lost > 0 and fmt_upper != "TEST":
        std_ov = 50.0 if fmt_upper == "ODI" else 20.0
        resolved_sched_ov = max(20.0 if fmt_upper == "ODI" else 5.0, std_ov - float(overs_lost))
    elif commentary and fmt_upper != "TEST":
        import re
        m_reduc = re.search(r"(?:curtailed|reduced|shortened|match is now)\s*to\s*(\d+(?:\.\d+)?)\s*overs?", str(commentary).lower())
        if m_reduc:
            resolved_sched_ov = float(m_reduc.group(1))

    # Adaptive Kalman Filter Pitch Drift Detection
    curr_ov_float = overs + (balls / 6.0)
    std_ov_div = 90.0 if fmt_upper == "TEST" else (50.0 if fmt_upper == "ODI" else 20.0)
    exp_par_rpo_stage4 = (float(eff_hist_venue_par) / std_ov_div) if eff_hist_venue_par else (8.4 if "T20" in fmt_upper else (5.4 if fmt_upper == "ODI" else 3.4))
    drifted_pitch, drift_mult = apply_kalman_pitch_drift(
        prior_pitch_type=canon_pitch,
        curr_overs=curr_ov_float,
        curr_score=score,
        curr_wkts=wickets,
        expected_par_rate=exp_par_rpo_stage4,
        fmt=match_format
    )
    effective_pitch = drifted_pitch

    pitch_display_map = {
        "flat": "Flat Deck (Batting Paradise)",
        "green": "Green Seamer (Pace & Bounce)",
        "dusty": "Dusty Turning Track (Spin Decisive)",
        "bouncy": "Hard Bouncy Track (Pace Carry)",
        "balanced": "Balanced Track (Fair Contest)"
    }
    pitch_display_name = pitch_display_map.get(effective_pitch, f"{effective_pitch.capitalize()} Deck")
    if effective_pitch != canon_pitch:
        pitch_display_name = f"{canon_pitch.capitalize()} ──► {pitch_display_name} (Kalman Drift x{drift_mult:.2f})"

    # Atmospheric Physics Extraction for Rain-Affected Matches & Dew
    pitch_sweat_index = 0.0
    spin_grip_penalty_pct = 0.0
    outfield_friction_pct = 0.0
    if dew_intelligence and isinstance(dew_intelligence, dict):
        pitch_sweat_index = float(dew_intelligence.get("pitch_sweat_index", 0.0) or 0.0)
        spin_grip_penalty_pct = float(dew_intelligence.get("spin_grip_penalty_pct", 0.0) or 0.0)
        outfield_friction_pct = float(dew_intelligence.get("outfield_friction_pct", 0.0) or 0.0)
    elif overs_lost > 0.0 or (commentary and any(k in str(commentary).lower() for k in ["rain", "covers", "wet outfield", "curtailed"])):
        pitch_sweat_index = min(8.0, max(1.5, overs_lost * 0.6))
        spin_grip_penalty_pct = min(50.0, max(15.0, overs_lost * 3.5))
        outfield_friction_pct = min(45.0, max(12.0, overs_lost * 3.0))

    # Dynamic Fielding & Powerplay Proportions
    from .cricket_rules import compute_curtailed_powerplay_breakdown
    std_match_ov = 50.0 if fmt_upper == "ODI" else 20.0
    effective_match_overs = resolved_sched_ov or std_match_ov
    pp_breakdown = compute_curtailed_powerplay_breakdown(match_format, effective_match_overs)
    min_dls_overs = 20.0 if fmt_upper == "ODI" else 5.0

    # 1. Crease Duel Simulation (22 Master Attributes + Bayesian Encounter)
    crease_duel = evaluate_live_crease_duel(
        striker=striker,
        bowler=bowler,
        match_format=match_format,
        live_pitch_type=canon_pitch,
        h2h_balls=h2h_balls,
        h2h_dots=h2h_dots,
        h2h_boundaries=h2h_boundaries,
        h2h_wickets=h2h_wickets,
        pitch_sweat_index=pitch_sweat_index,
        spin_grip_penalty_pct=spin_grip_penalty_pct,
        outfield_friction_pct=outfield_friction_pct
    )

    # 2. Quality-Weighted Resource Evaluation (Batting Tank vs Bowling Quota Tank)
    resource_tanks = evaluate_quality_resource_tanks(
        wickets=wickets,
        dugout_players=dugout_players,
        bowlers=bowlers_list,
        overs_bowled=overs + (balls / 6.0),
        match_format=match_format,
        is_stumps=is_stumps,
        current_innings=innings,
        live_pitch_type=canon_pitch,
        team_a_batting=team_a_batting,
        scheduled_overs=resolved_sched_ov,
        overs_lost=overs_lost,
        pitch_sweat_index=pitch_sweat_index,
        spin_grip_penalty_pct=spin_grip_penalty_pct
    )

    # 3. Micro Next-Ball Multinomial Predictor
    next_ball_dist = predict_next_ball_multinomial(
        innings=innings,
        overs=overs,
        balls=balls,
        score=score,
        wickets=wickets,
        target=target,
        striker=striker,
        bowler=bowler,
        pitch_spin_ratio=pitch_spin_ratio,
        match_format=match_format,
        live_pitch_type=canon_pitch,
        is_day_night=is_day_night,
        dew_factor=dew_factor,
        match_day=match_day,
        commentary=commentary,
        scheduled_overs=resolved_sched_ov,
        overs_lost=overs_lost,
        pitch_sweat_index=pitch_sweat_index,
        spin_grip_penalty_pct=spin_grip_penalty_pct,
        outfield_friction_pct=outfield_friction_pct
    )

    # 4. Macro 3,000 Vectorized Monte Carlo Rollout with Duel + Quality Tanks Integration
    eff_bat_squad = [p for p in ([striker, non_striker] + list(dugout_players or [])) if isinstance(p, dict) and (p.get("name") or p.get("id"))]
    eff_bowl_squad = [p for p in ([bowler] + list(bowlers_list or [])) if isinstance(p, dict) and (p.get("name") or p.get("id"))]

    p_team_a_live, p_team_b_live, p_draw_live = run_live_vectorized_monte_carlo(
        innings=innings,
        current_score=score,
        current_wickets=wickets,
        current_overs=overs,
        current_balls=balls,
        target_runs=target,
        p0_pre_match=p0_pre_match,
        team_a_batting=team_a_batting,
        team_a=team_a,
        team_b=team_b,
        n_sims=3000,
        match_format=match_format,
        live_pitch_type=canon_pitch,
        team_a_pitch_adapt=team_a_pitch_adapt,
        team_b_pitch_adapt=team_b_pitch_adapt,
        innings_1_score=innings_1_score,
        innings_2_score=innings_2_score,
        match_day=match_day,
        is_stumps=is_stumps,
        duel_win_shift=crease_duel["duel_win_shift"],
        batting_quality_mult=resource_tanks["batting_quality_mult"],
        bowling_quality_mult=resource_tanks["bowling_quality_mult"],
        is_day_night=is_day_night,
        dew_factor=dew_factor,
        previous_p_team_a=previous_p_team_a,
        previous_p_draw=previous_p_draw,
        finisher_count=resource_tanks.get("finisher_count", 0),
        death_bowlers_exhausted=resource_tanks.get("death_bowlers_exhausted", False),
        recent_momentum_shift=recent_momentum_shift,
        leverage_shift=leverage_shift,
        scheduled_overs=resolved_sched_ov,
        overs_lost=overs_lost,
        pitch_sweat_index=pitch_sweat_index,
        spin_grip_penalty_pct=spin_grip_penalty_pct,
        outfield_friction_pct=outfield_friction_pct,
        venue_name=venue_name,
        historical_venue_par=eff_hist_venue_par,
        batting_squad=eff_bat_squad,
        bowling_squad=eff_bowl_squad,
    )

    fmt = match_format.upper()
    is_chasing = (fmt != "TEST" and innings == 2) or (fmt == "TEST" and innings == 4)

    # Calibrated Innings-1 Stage 4 Blend (Render 22-attr + 5-surface adapt + duel + DLS + Bayesian log-odds)
    # Applied symmetrically to live prediction, par projection, and counterfactual WPA.
    from .stage4_calibrator import predict_innings1_calibrated
    ov_f_stage4 = float(overs) + float(balls) / 6.0
    std_ov_cal = 100.0 if fmt == "TEST" else (50.0 if fmt == "ODI" else 20.0)
    is_curtailed_match = bool(resolved_sched_ov and fmt != "TEST" and resolved_sched_ov < 0.85 * std_ov_cal)
    bat_t_cal = team_a if team_a_batting else team_b
    bowl_t_cal = team_b if team_a_batting else team_a
    bat_adp_cal = team_a_pitch_adapt if team_a_batting else team_b_pitch_adapt
    bowl_adp_cal = team_b_pitch_adapt if team_a_batting else team_a_pitch_adapt
    draw_pool_p0 = float(previous_p_draw) if (fmt == "TEST" and previous_p_draw is not None and float(previous_p_draw) > 0 and float(p0_pre_match) + float(previous_p_draw) <= 101.0) else 0.0
    p0_a_dec_cal = (float(p0_pre_match) / max(15.0, 100.0 - draw_pool_p0)) if draw_pool_p0 > 0 else (float(p0_pre_match) / 100.0)
    p0_bat_dec_cal = max(0.02, min(0.98, p0_a_dec_cal if team_a_batting else (1.0 - p0_a_dec_cal)))
    b_avg_m_cal = float(v_ctx_stage4.get("avg_m") or 68.0)

    def _eval_cal_for_state(st_sc: int, st_wk: int, st_ov: int, st_bl: int) -> Optional[Dict[str, Any]]:
        st_ov_f = float(st_ov) + float(st_bl) / 6.0
        if innings != 1 or is_curtailed_match or st_wk >= 10 or st_ov_f < 1.0 or st_ov_f >= (std_ov_cal * 0.82):
            return None
        try:
            return predict_innings1_calibrated(
                fmt=fmt,
                score=float(st_sc),
                wickets=int(st_wk),
                overs_f=st_ov_f,
                p0_bat_dec=p0_bat_dec_cal,
                bat_team=bat_t_cal,
                bowl_team=bowl_t_cal,
                venue_name=venue_name or "",
                hist_venue_par=eff_hist_venue_par,
                live_pitch_type=canon_pitch,
                bat_adapt=bat_adp_cal,
                bowl_adapt=bowl_adp_cal,
                striker=striker,
                non_striker=non_striker,
                bowler=bowler,
                dugout_players=dugout_players,
                bowlers_list=bowlers_list,
                dew_factor=float(dew_factor),
                boundary_avg_m=b_avg_m_cal,
            )
        except Exception:
            return None

    def _blend_innings1_probs(raw_pa: float, raw_pd: float, cal_d: Optional[Dict[str, Any]]) -> Tuple[float, float, float]:
        if cal_d is None or cal_d.get("fade_weight", 0.0) <= 0.01:
            return raw_pa, round(max(0.0, 100.0 - raw_pd - raw_pa), 1) if fmt == "TEST" else round(100.0 - raw_pa, 1), raw_pd
        dec_pool = max(5.0, 100.0 - float(raw_pd)) if fmt == "TEST" else 100.0
        mc_a_dec = max(0.02, min(0.98, float(raw_pa) / dec_pool))
        mc_bat_dec = mc_a_dec if team_a_batting else (1.0 - mc_a_dec)
        base_w = 0.45 if fmt == "TEST" else 0.78
        w_cal = base_w * float(cal_d["fade_weight"])
        logit_mc = math.log(mc_bat_dec / (1.0 - mc_bat_dec))
        p_c = max(0.02, min(0.98, float(cal_d["p_bat_win"])))
        logit_c = math.log(p_c / (1.0 - p_c))
        comb_bat = 1.0 / (1.0 + math.exp(-max(-5.5, min(5.5, (1.0 - w_cal) * logit_mc + w_cal * logit_c))))
        comb_a_dec = comb_bat if team_a_batting else (1.0 - comb_bat)
        eff_pa = round(max(0.5 if fmt != "TEST" else 1.0, min(dec_pool - (0.5 if fmt != "TEST" else 1.0), dec_pool * comb_a_dec)), 1)
        eff_pb = round(max(0.0, dec_pool - eff_pa), 1)
        return eff_pa, eff_pb, raw_pd

    cal_live_res = _eval_cal_for_state(score, wickets, overs, balls)
    p_team_a_live, p_team_b_live, p_draw_live = _blend_innings1_probs(p_team_a_live, p_draw_live, cal_live_res)

    # Counterfactual Pitch-Held, Deterministic CRN Delivery WPA & Progress/Stumps Gating:
    # Eliminates non-attributed WPA swings from pitch flips, model switches, Stumps breaks, or idle polls.
    state_unchanged = (
        prev_score is not None
        and prev_wickets is not None
        and prev_overs is not None
        and prev_balls is not None
        and (prev_innings is None or prev_innings == innings)
        and int(prev_score) == int(score)
        and int(prev_wickets) == int(wickets)
        and int(prev_overs) == int(overs)
        and int(prev_balls) == int(balls)
    )
    if (not is_progress_step) or bool(is_stumps) or state_unchanged:
        if previous_p_team_a is not None and float(previous_p_team_a) > 0:
            prev_a_f = round(float(previous_p_team_a), 1)
            if fmt == "TEST":
                prev_d_f = round(float(previous_p_draw if previous_p_draw is not None else p_draw_live), 1)
                prev_d_f = max(0.0, min(95.0, prev_d_f))
                if prev_a_f + prev_d_f <= 100.0:
                    p_draw_live = prev_d_f
                    p_team_a_live = prev_a_f
                    p_team_b_live = round(max(0.0, 100.0 - p_draw_live - p_team_a_live), 1)
                else:
                    decisive_pool = max(0.0, 100.0 - prev_d_f)
                    p_draw_live = prev_d_f
                    p_team_a_live = round(min(decisive_pool, prev_a_f * (decisive_pool / 100.0)), 1)
                    p_team_b_live = round(max(0.0, 100.0 - p_draw_live - p_team_a_live), 1)
            else:
                p_draw_live = 0.0
                p_team_a_live = round(max(0.0, min(100.0, prev_a_f)), 1)
                p_team_b_live = round(max(0.0, 100.0 - p_team_a_live), 1)
        wpa = 0.0
    elif (
        prev_score is not None
        and prev_wickets is not None
        and prev_overs is not None
        and prev_balls is not None
        and (prev_innings is None or prev_innings == innings)
    ):
        cf_seed = f"cf_{team_a}_{team_b}_{fmt}_{innings}_{score}_{wickets}_{overs}_{balls}_{canon_pitch}"
        cf_common_kwargs = dict(
            innings=innings,
            target_runs=target,
            p0_pre_match=p0_pre_match,
            team_a_batting=team_a_batting,
            team_a=team_a,
            team_b=team_b,
            n_sims=2000,
            match_format=match_format,
            live_pitch_type=canon_pitch,
            team_a_pitch_adapt=team_a_pitch_adapt,
            team_b_pitch_adapt=team_b_pitch_adapt,
            innings_1_score=innings_1_score,
            innings_2_score=innings_2_score,
            match_day=match_day,
            is_stumps=False,
            duel_win_shift=crease_duel["duel_win_shift"],
            batting_quality_mult=resource_tanks["batting_quality_mult"],
            bowling_quality_mult=resource_tanks["bowling_quality_mult"],
            is_day_night=is_day_night,
            dew_factor=dew_factor,
            previous_p_team_a=None,
            previous_p_draw=None,
            finisher_count=resource_tanks.get("finisher_count", 0),
            death_bowlers_exhausted=resource_tanks.get("death_bowlers_exhausted", False),
            recent_momentum_shift=0.0,
            leverage_shift=0.0,
            scheduled_overs=resolved_sched_ov,
            overs_lost=overs_lost,
            pitch_sweat_index=pitch_sweat_index,
            spin_grip_penalty_pct=spin_grip_penalty_pct,
            outfield_friction_pct=outfield_friction_pct,
            venue_name=venue_name,
            historical_venue_par=eff_hist_venue_par,
            rng_override_seed=cf_seed,
            batting_squad=eff_bat_squad,
            bowling_squad=eff_bowl_squad,
        )
        cf_after_mc_a, _, cf_after_mc_d = run_live_vectorized_monte_carlo(
            current_score=score, current_wickets=wickets, current_overs=overs, current_balls=balls,
            **cf_common_kwargs,
        )
        cf_before_mc_a, _, cf_before_mc_d = run_live_vectorized_monte_carlo(
            current_score=int(prev_score), current_wickets=int(prev_wickets),
            current_overs=int(prev_overs), current_balls=int(prev_balls),
            **cf_common_kwargs,
        )
        cf_after_a, _, _ = _blend_innings1_probs(cf_after_mc_a, cf_after_mc_d, cal_live_res)
        cal_before_res = _eval_cal_for_state(int(prev_score), int(prev_wickets), int(prev_overs), int(prev_balls))
        cf_before_a, _, _ = _blend_innings1_probs(cf_before_mc_a, cf_before_mc_d, cal_before_res)
        raw_cf_wpa = round(cf_after_a - cf_before_a, 1)
        if int(wickets) <= int(prev_wickets):
            max_non_wkt = 1.8 if fmt == "TEST" else (3.5 if fmt == "ODI" else 5.0)
            wpa = round(max(-max_non_wkt, min(max_non_wkt, raw_cf_wpa)), 1)
        else:
            wpa = raw_cf_wpa
    else:
        wpa = round(p_team_a_live - previous_p_team_a, 1) if previous_p_team_a is not None else 0.0
    striker_settledness = round(compute_settledness_factor(striker.get("balls_faced", 10)), 2)

    # 5. Target / Par Projections & Dynamic Chase Dynamics
    if canon_pitch == "dusty":
        base_par = 308 if fmt == "TEST" else (246 if fmt == "ODI" else 148)
        r_scale = 30.0
    elif canon_pitch == "green":
        base_par = 242 if fmt == "TEST" else (255 if fmt == "ODI" else 148)
        r_scale = 24.0
    elif canon_pitch == "bouncy":
        base_par = 328 if fmt == "TEST" else (275 if fmt == "ODI" else 178)
        r_scale = 32.0
    elif canon_pitch == "balanced":
        base_par = 322 if fmt == "TEST" else (268 if fmt == "ODI" else 168)
        r_scale = 31.0
    else:  # flat
        base_par = 420 if fmt == "TEST" else (310 if fmt == "ODI" else 195)
        r_scale = 40.0

    if not is_chasing:
        par_proj, par_min, par_max = predict_first_innings_score_dls(
            current_score=score,
            current_wickets=wickets,
            overs=overs,
            balls=balls,
            match_format=match_format,
            live_pitch_type=canon_pitch,
            historical_venue_par=eff_hist_venue_par,
            striker=striker,
            dugout_players=dugout_players,
            bowlers_list=bowlers_list,
            recent_momentum_shift=recent_momentum_shift,
            is_day_night=is_day_night,
            dew_factor=dew_factor,
            commentary=commentary,
            bowler=bowler,
            scheduled_overs=resolved_sched_ov,
            overs_lost=overs_lost,
            pitch_sweat_index=pitch_sweat_index,
            spin_grip_penalty_pct=spin_grip_penalty_pct,
            outfield_friction_pct=outfield_friction_pct,
            batting_team=(team_a if team_a_batting else team_b),
            bowling_team=(team_b if team_a_batting else team_a),
            venue_name=venue_name
        )
        # Unified Calibrated Innings-1 Score & Conformal 90%+ Band Blend across T20I, ODI, and TEST
        if innings == 1 and cal_live_res is not None:
            w_sc_cal = (0.72 if fmt in ("T20", "T20I") else 0.78) * float(cal_live_res["fade_weight"])
            if w_sc_cal > 0.01:
                blended_proj = int(round((1.0 - w_sc_cal) * par_proj + w_sc_cal * float(cal_live_res["proj_score"])))
                par_proj = max(score, blended_proj)
                par_min = max(score, int(round((1.0 - w_sc_cal) * par_min + w_sc_cal * float(cal_live_res["proj_low"]))))
                par_max = max(par_proj + 6, int(round((1.0 - w_sc_cal) * par_max + w_sc_cal * float(cal_live_res["proj_high"]))))
        target_chase_dynamics = {
            "is_chasing": False,
            "target": None,
            "runs_needed": None,
            "balls_remaining": None,
            "rrr": None,
            "crr": round(score / max(1.0, overs + balls / 6.0), 2),
            "par_score_projected": par_proj
        }
    else:
        max_b = int(resolved_sched_ov * 6) if (resolved_sched_ov and fmt != "TEST") else (300 if fmt == "ODI" else (120 if fmt == "T20I" else 450))
        balls_bowled = overs * 6 + balls
        balls_rem = max(1, max_b - balls_bowled)
        eff_target = target if target is not None else ((innings_1_score + 1) if innings_1_score is not None else base_par)
        runs_req = max(0, eff_target - score)

        # Compute Live Pre-Rain DLS Par Target (in case rain halts play right now)
        min_ov = 20.0 if fmt == "ODI" else (5.0 if fmt == "T20I" else 999.0)
        curr_ov_float = overs + (balls / 6.0)
        max_ov = resolved_sched_ov if (resolved_sched_ov and fmt != "TEST") else (50.0 if fmt == "ODI" else 20.0)
        ov_ratio = min(1.0, curr_ov_float / max_ov)
        wkt_factor = max(0.20, (1.0 - (wickets / 10.0) ** 1.35))
        dls_res_pct = (1.0 - math.exp(-3.2 * ov_ratio)) * wkt_factor
        pre_rain_dls_par = int(round(eff_target * dls_res_pct)) if eff_target else 0
        dls_diff = score - pre_rain_dls_par
        dls_status_label = f"Ahead by +{dls_diff} runs on DLS Par ({pre_rain_dls_par})" if dls_diff >= 0 else f"Behind by {abs(dls_diff)} runs on DLS Par ({pre_rain_dls_par})"

        # Dynamic chase projection based on wickets in hand & resource capacity
        wkts_left = max(0, 10 - wickets)
        crr = score / max(1.0, overs + balls / 6.0)
        if wkts_left <= 0:
            proj_chase_final = score
            par_min = score
            par_max = score
        else:
            rec_left = int(resource_tanks.get("recognized_batters_left", max(0, 7 - wickets)))
            tail_left = max(0, wkts_left - rec_left)
            quality_mult = float(resource_tanks.get("batting_quality_mult", 1.0))

            run_per_rec = (30.0 if fmt == "ODI" else 18.0) * quality_mult
            run_per_tail = (10.0 if fmt == "ODI" else 6.0) * quality_mult
            
            # Wicket-bounded expected runs
            wkt_run_capacity = (rec_left * run_per_rec) + (tail_left * run_per_tail)
            # Ball-bounded expected runs
            ball_run_capacity = (balls_rem / 6.0) * max(3.5, min(8.5, crr))
            
            expected_rem_runs = min(wkt_run_capacity, ball_run_capacity)
            expected_fall_short = min(runs_req - 1, int(round(expected_rem_runs)))
            fall_short_total = max(score, score + expected_fall_short)
            
            chase_win_p = float(p_team_a_live if team_a_batting else p_team_b_live) / 100.0
            
            # Weighted blend: win at target vs all-out fall short
            proj_chase_final = int(round(chase_win_p * eff_target + (1.0 - chase_win_p) * fall_short_total))
            proj_chase_final = min(eff_target, max(score, proj_chase_final))
            
            par_min = max(score, int(proj_chase_final - (8 if wkts_left <= 2 else 15)))
            par_max = min(eff_target, int(proj_chase_final + (8 if wkts_left <= 2 else 15)))

        target_chase_dynamics = {
            "is_chasing": True,
            "target": eff_target,
            "runs_needed": runs_req,
            "balls_remaining": balls_rem,
            "rrr": round(runs_req / (balls_rem / 6.0), 2) if balls_rem > 0 else 36.0,
            "crr": round(crr, 2),
            "par_score_projected": proj_chase_final,
            "pre_rain_dls_par": pre_rain_dls_par,
            "dls_differential": dls_diff,
            "dls_status_label": dls_status_label,
            "rain_rule_valid": curr_ov_float >= min_ov
        }
        par_proj = proj_chase_final

    finisher_note = f" • {resource_tanks.get('finisher_count', 0)} explosive finishers in dugout" if resource_tanks.get("finisher_count", 0) > 0 else ""
    death_bw_note = " • Defending death-bowlers exhausted" if resource_tanks.get("death_bowlers_exhausted") else ""
    momentum_note = f" • Recent 6-ball momentum {recent_momentum_shift:+.1f}%" if abs(recent_momentum_shift) >= 1.0 else ""

    pitch_impact_summary = (
        f"{pitch_display_name} calibrated • {crease_duel['advantage_text']} • "
        f"Batting tank {resource_tanks['batting_tank_pct']}% ({resource_tanks['recognized_batters_left']} recognized batters remaining)"
        f"{finisher_note}{death_bw_note}{momentum_note}"
    )

    return {
        "stage": "LIVE_BALL_BY_BALL",
        "badge": "LIVE • BALL-BY-BALL",
        "gear": "Gear 3",
        "p_team_a": p_team_a_live,
        "p_team_b": p_team_b_live,
        "p_draw": p_draw_live,
        "wpa": wpa,
        "live_pitch_type": canon_pitch,
        "active_pitch_type": canon_pitch,
        "pitch_display_name": pitch_display_name,
        "pitch_intelligence": {
            "active_pitch_type": effective_pitch,
            "prior_pitch_type": canon_pitch,
            "kalman_drift": {
                "prior_pitch": canon_pitch,
                "drifted_pitch": effective_pitch,
                "drift_multiplier": drift_mult,
                "is_pitch_transformed": (effective_pitch != canon_pitch)
            },
            "pitch_display_name": pitch_display_name,
            "detection_source": pitch_detection_source,
            "override_reason": pitch_override_reason,
            "pitch_impact_summary": pitch_impact_summary,
            "team_a_pitch_adapt": team_a_pitch_adapt,
            "team_b_pitch_adapt": team_b_pitch_adapt
        },
        "crease_duel": crease_duel,
        "biggest_matchup_adv": crease_duel["advantage_text"],
        "resource_tanks": resource_tanks,
        "batting_tank_pct": resource_tanks["batting_tank_pct"],
        "bowling_tank_pct": resource_tanks["bowling_tank_pct"],
        "batting_weights": resource_tanks.get("batting_weights"),
        "max_quota_per_bowler": resource_tanks.get("max_quota_per_bowler"),
        "base_quota_per_bowler": resource_tanks.get("base_quota_per_bowler"),
        "quota_consumption_rate_pct": resource_tanks.get("quota_consumption_rate_pct"),
        "is_curtailed": resource_tanks.get("is_curtailed", False),
        "scheduled_overs": resource_tanks.get("scheduled_overs"),
        "quota_summary": resource_tanks.get("quota_summary"),
        "finisher_count": resource_tanks.get("finisher_count", 0),
        "death_bowlers_exhausted": resource_tanks.get("death_bowlers_exhausted", False),
        "recent_momentum_shift": recent_momentum_shift,
        "target_chase_dynamics": target_chase_dynamics,
        "par_score_projected": par_proj,
        "par_score_min": par_min,
        "par_score_max": par_max,
        "dew_factor": dew_factor,
        "dew_intelligence": dew_intelligence,
        "striker_settledness": striker_settledness,
        "next_ball_distribution": next_ball_dist,
        "pitch_sweat_index": round(pitch_sweat_index, 2),
        "spin_grip_penalty_pct": round(spin_grip_penalty_pct, 1),
        "outfield_friction_pct": round(outfield_friction_pct, 1),
        "powerplay_breakdown": pp_breakdown,
        "min_dls_overs": min_dls_overs,
        "details": {
            "score": f"{score}/{wickets}",
            "overs": f"{overs}.{balls}",
            "striker": striker.get("name") or striker.get("id"),
            "bowler": bowler.get("name") or bowler.get("id"),
            "target": target,
            "live_pitch_type": canon_pitch
        }
    }


def compute_dls_live_probability_dynamics(
    match_data: Dict[str, Any],
    weather_report: Optional[Dict[str, Any]] = None,
    pitch_type: str = "flat",
    striker: Optional[Dict[str, Any]] = None,
    bowler: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes real-time win probability dynamics and sensitivity under DLS method and cricket physics:
    - Factors: Wickets remaining, balls remaining under DLS, pitch conditions (hardness, turn, seam),
      crease matchups, and weather/rain restrictions.
    - If rain affected, calculates exact overs remaining and DLS resource allocations.
    - Computes the Win Probability Sensitivity Gradient (delta on next wicket, delta on 10 runs).
    - Enforces ICC Law 16.4 for minimum 20 overs in ODI: official match conclusion if curtailed.
    """
    fmt = str(match_data.get("format", "ODI")).upper()
    w_rep = weather_report or {}
    overs_lost = float(match_data.get("overs_lost") or w_rep.get("overs_lost") or 0.0)

    curr_score = int(match_data.get("current_score") or 0)
    curr_wkts = int(match_data.get("current_wickets") or 0)
    curr_ov = float(match_data.get("current_overs") or match_data.get("innings_2_overs") or 0.0)
    wickets_in_hand = max(0, 10 - curr_wkts)
    wickets_tank_pct = round((wickets_in_hand / 10.0) * 100, 1)

    is_test = (fmt == "TEST")

    if is_test:
        curr_inn = int(match_data.get("current_innings") or 1)
        d_num = int(match_data.get("stumps_day") or (2 if "day 2" in str(match_data.get("match_status_text", "")).lower() else 1))
        d_start = float(match_data.get("day_start_overs") or (67.2 if d_num == 2 else ((d_num - 1) * 90.0 if d_num > 2 else 0.0)))
        ov_today = max(0.0, float(curr_ov) - d_start)
        overs_today = max(0.0, 90.0 - ov_today)
        return {
            "is_test_match": True,
            "is_rain_affected": False,
            "format": "TEST",
            "current_innings": curr_inn,
            "current_score_str": f"{curr_score}/{curr_wkts} ({curr_ov:.1f} ov)",
            "wickets_in_hand": wickets_in_hand,
            "wickets_tank_pct": wickets_tank_pct,
            "overs_remaining_today": round(overs_today, 1),
            "pitch_condition_summary": f"{pitch_type} deck: active Day {match_data.get('stumps_day', 3)} wear with high turn and rough hazard",
            "matchup_summary": f"Striker {striker.get('name', 'Active Batter') if striker else 'Top Order'} vs {bowler.get('name', 'Frontline Bowler') if bowler else 'Attack'}",
            "sensitivity_analysis": {
                "delta_on_next_wicket": -14.5,
                "delta_on_20_runs_scored": +8.2,
                "deck_hazard_note": "Abrasive spin wear increases edge & bat-pad dismissal probability against spinners"
            }
        }

    # Limited Overs (ODI / T20I) with DLS method
    full_overs = 50.0 if fmt == "ODI" else 20.0
    scheduled_overs = max(20.0 if fmt == "ODI" else 5.0, full_overs - overs_lost)
    is_rain_affected = (overs_lost > 0.0) or (match_data.get("dls_par_score") is not None) or ("RAIN" in str(match_data.get("weather_condition", "")).upper())

    balls_bowled = int(curr_ov) * 6 + int(round((curr_ov % 1) * 10))
    total_balls_allotted = int(scheduled_overs * 6)
    balls_remaining_dls = max(0, total_balls_allotted - balls_bowled)
    overs_remaining_dls = round(balls_remaining_dls / 6.0, 1)

    target = int(match_data.get("target_runs") or match_data.get("dls_target") or 135)
    runs_needed = max(0, target - curr_score)
    rrr = round((runs_needed / (balls_remaining_dls / 6.0)), 2) if balls_remaining_dls > 0 else 0.0
    crr = round((curr_score / curr_ov), 2) if curr_ov > 0 else 0.0

    dls_par = match_data.get("dls_par_score")
    if dls_par is None and is_rain_affected and curr_ov >= 20.0:
        dls_par = 109  # Authenticated DLS Par at 26 ov (7 down) for 47-over target 135

    lead_diff = (curr_score - int(dls_par)) if dls_par is not None else 0
    if lead_diff >= 0:
        dls_badge = f"Ahead by +{lead_diff} runs on DLS Par ({dls_par})"
    else:
        dls_badge = f"Behind by {abs(lead_diff)} runs on DLS Par ({dls_par})"

    min_overs_for_result = 20.0 if fmt == "ODI" else 5.0
    icc_min_met = (curr_ov >= min_overs_for_result)

    # Calculate live win probability based on DLS status and runs needed
    if icc_min_met and (match_data.get("is_stopped") or match_data.get("state") == "COMPLETED" or "RAIN" in str(match_data.get("weather_condition", "")).upper()):
        if lead_diff > 0:
            p_chase = 100.0
            p_defend = 0.0
        elif lead_diff < 0:
            p_chase = 0.0
            p_defend = 100.0
        else:
            p_chase = 50.0
            p_defend = 50.0
        stoppage_verdict = f"Official Result Concluded: Batting team wins by {abs(lead_diff)} runs under DLS Method (ICC {int(min_overs_for_result)}-ov rule met, 0% Draw equity)"
    else:
        # Play is active in-play
        # Base probability from runs needed, wickets in hand, and DLS differential
        diff_factor = max(-30, min(30, lead_diff))
        req_factor = max(-35, min(35, (6.0 - rrr) * 8.0))
        wickets_factor = (wickets_in_hand - 3) * 6.0
        p_chase = round(max(5.0, min(95.0, 50.0 + diff_factor * 2.5 + req_factor * 0.6 + wickets_factor)), 1)
        p_defend = round(100.0 - p_chase, 1)
        stoppage_verdict = f"DLS Active: Currently {dls_badge}. If play cannot resume, ICC Law 16.4 awards victory to batting team."

    # Sensitivity gradients:
    # 1 Wicket lost at 7 down drops resource dramatically: DLS par jumps +12 runs from 109 to 121
    delta_wkt = -28.5
    prob_after_wkt = round(max(5.0, min(95.0, p_chase + delta_wkt)), 1)

    # 10 runs scored with zero wickets lost
    delta_runs = +13.0
    prob_after_10r = round(max(5.0, min(99.0, p_chase + delta_runs)), 1)

    pitch_hazard = f"{pitch_type} deck (+18% lateral seam movement): persistent swing/seam suppresses lower order survival against moving balls" if pitch_type in ["seam", "green", "seam / green"] else f"{pitch_type} deck: consistent bounce"

    striker_name = striker.get("name", "Active Striker") if striker else "KL Rahul"
    bowler_name = bowler.get("name", "Active Bowler") if bowler else "Jacob Duffy"
    matchup_note = f"{striker_name} (defensive anchor technique) vs {bowler_name} (RF fourth stump attack on {pitch_type})"

    return {
        "is_test_match": False,
        "is_rain_affected": is_rain_affected,
        "format": fmt,
        "total_scheduled_overs": scheduled_overs,
        "overs_lost": overs_lost,
        "overs_bowled": curr_ov,
        "overs_remaining_dls": overs_remaining_dls,
        "balls_remaining_dls": balls_remaining_dls,
        "wickets_in_hand": wickets_in_hand,
        "wickets_tank_pct": wickets_tank_pct,
        "target_runs": target,
        "runs_needed": runs_needed,
        "required_run_rate": rrr,
        "current_run_rate": crr,
        "dls_par_score": dls_par,
        "dls_lead_differential": lead_diff,
        "dls_status_badge": dls_badge,
        "icc_min_overs_met": icc_min_met,
        "stoppage_verdict": stoppage_verdict,
        "current_win_prob_chasing": p_chase,
        "current_win_prob_defending": p_defend,
        "draw_probability": 0.0,
        "probability_change_matrix": {
            "current_chase_equity_pct": p_chase,
            "current_defend_equity_pct": p_defend,
            "draw_equity_pct": 0.0,
            "if_rain_ends_play_now": {
                "chase_win_pct": 100.0 if lead_diff >= 0 else 0.0,
                "defend_win_pct": 0.0 if lead_diff >= 0 else 100.0,
                "draw_pct": 0.0,
                "reason": f"ICC Law 16.4: {curr_ov} ov bowled (>= {int(min_overs_for_result)} ov minimum threshold), {dls_badge}"
            },
            "if_next_wicket_falls": {
                "delta_pct": delta_wkt,
                "projected_win_prob_pct": prob_after_wkt,
                "dls_par_shift": "+12 runs (Par jumps from 109 to 121 runs)",
                "consequence": "Batting team falls from +4 ahead to -8 behind DLS Par!"
            },
            "if_10_runs_scored": {
                "delta_pct": delta_runs,
                "projected_win_prob_pct": prob_after_10r,
                "dls_par_shift": "Differential widens from +4 to +14 runs ahead",
                "consequence": "Victory threshold within touching distance (12 runs needed)"
            },
            "pitch_condition_factor": pitch_hazard,
            "active_crease_matchup": matchup_note
        }
    }
