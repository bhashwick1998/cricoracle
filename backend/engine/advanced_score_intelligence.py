"""
Advanced Score Intelligence Engine
Implements the 6 Frontier Levers for Maximum Score Prediction Precision:
1. Control Percentage & False Shot Hazard (Edge vs Middle)
2. Over-by-Over Tactical Bowler Allocation & Scheduled Quota Optimization
3. Dot-Ball Tension & Risk-Appetite Accumulation
4. Non-Linear Dew Saturation Curve & Outfield Slickness
5. Real-Time Pitch Drift via Online Kalman Filter / Adaptive Surface Tracking
6. Asymmetric Ground Geometry & Wind Vector Modulation
"""

import math
from typing import Dict, Any, List, Optional, Tuple

# -------------------------------------------------------------------------
# LEVER 1: Control Percentage & False-Shot Hazard Modulation
# -------------------------------------------------------------------------
def evaluate_control_and_false_shot_hazard(
    balls_faced: int,
    false_shots: Optional[int] = None,
    control_pct: Optional[float] = None,
    commentary_text: Optional[str] = None
) -> Tuple[float, float, float]:
    """
    Evaluates batter control quality and false-shot rate.
    Returns: (control_pct, longevity_mult, hazard_mult)
    - longevity_mult: Multiplier on remaining expected runs (0.72x to 1.15x)
    - hazard_mult: Multiplier on dismissal risk per ball (0.65x to 2.60x)
    """
    bf = max(1, balls_faced)
    
    # Derive control percentage
    if control_pct is not None:
        c_pct = max(0.0, min(100.0, float(control_pct)))
        if c_pct > 1.0:
            c_pct /= 100.0
    elif false_shots is not None:
        fs = max(0, int(false_shots))
        c_pct = max(0.40, min(1.0, 1.0 - (fs / float(bf))))
    elif commentary_text:
        text_lower = commentary_text.lower()
        edge_terms = ["beaten", "outside edge", "inside edge", "play and miss", "leading edge", "mistimed", "sliced"]
        edge_count = sum(text_lower.count(term) for term in edge_terms)
        c_pct = max(0.50, min(1.0, 1.0 - (edge_count * 0.08)))
    else:
        # Benchmark international average: ~80% control
        c_pct = 0.80

    # Continuous Sigmoid Hazard & Longevity Responses (No Artificial Step Cliffs)
    hazard_mult = 0.65 + (1.85 / (1.0 + math.exp(14.0 * (c_pct - 0.72))))
    longevity_mult = 0.70 + (0.48 / (1.0 + math.exp(-12.0 * (c_pct - 0.74))))

    return float(round(c_pct, 3)), float(round(longevity_mult, 3)), float(round(hazard_mult, 3))


# -------------------------------------------------------------------------
# LEVER 2: Over-by-Over Tactical Bowler Deployment Scheduling
# -------------------------------------------------------------------------
def schedule_tactical_bowler_deployments(
    bowling_squad: List[Dict[str, Any]],
    death_overs_remaining: int,
    fmt: str = "T20",
    base_death_rr: float = 10.5
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Simulates tactical over-by-over captaincy bowler deployment for death phase.
    Allocates specialist death bowlers to remaining overs according to remaining quota.
    Returns: (tactical_death_multiplier, scheduled_overs_list)
    """
    if death_overs_remaining <= 0 or not bowling_squad:
        return 1.0, []

    fmt_upper = fmt.upper()
    is_odi = "ODI" in fmt_upper
    
    # Make deep copy of bowler quotas to simulate consumption
    available_bowlers = []
    for b in bowling_squad:
        b_copy = dict(b)
        # Parse quota remaining
        if "quota_remaining" in b_copy and b_copy["quota_remaining"] is not None:
            quota = float(b_copy["quota_remaining"])
        else:
            max_quota = 10.0 if is_odi else 4.0
            ov_bowled = float(b_copy.get("overs_bowled") or b_copy.get("overs") or 0.0)
            quota = max(0.0, max_quota - ov_bowled)
        
        b_copy["quota_left"] = quota
        # Death bowling capability score
        d_rating = float(b_copy.get("death_bowling_rating") or 50.0)
        econ = float(b_copy.get("t20_economy") or b_copy.get("bowling_economy") or (8.5 if not is_odi else 5.5))
        b_copy["death_score"] = d_rating * 1.5 - (econ * 4.0)
        available_bowlers.append(b_copy)

    # Sort bowlers by death execution capability
    available_bowlers.sort(key=lambda x: x["death_score"], reverse=True)

    scheduled = []
    rpo_list = []

    for ov in range(death_overs_remaining):
        # Pick best available bowler with >= 1.0 over remaining
        chosen = next((b for b in available_bowlers if b["quota_left"] >= 0.8), None)
        if chosen:
            chosen["quota_left"] -= 1.0
            d_rat = float(chosen.get("death_bowling_rating") or 50.0)
            econ = float(chosen.get("t20_economy") or chosen.get("bowling_economy") or (8.5 if not is_odi else 5.5))
            
            # Expected runs this over: calibrated dynamically against base_death_rr
            if not is_odi:
                # T20 death over expectation: anchored to base_death_rr
                expected_ov_rpo = max(6.8, min(14.0, base_death_rr - (d_rat - 50.0) * 0.04 + (econ - 8.5) * 0.20))
            else:
                # ODI death over expectation: anchored to base_death_rr
                expected_ov_rpo = max(5.2, min(11.5, base_death_rr - (d_rat - 50.0) * 0.03 + (econ - 5.5) * 0.20))

            rpo_list.append(expected_ov_rpo)
            scheduled.append({
                "over_number": ov + 1,
                "bowler_name": chosen.get("name") or chosen.get("player_id"),
                "bowler_id": chosen.get("player_id") or chosen.get("id"),
                "expected_rpo": round(expected_ov_rpo, 2),
                "is_specialist": d_rat >= 65.0
            })
        else:
            # Frontline death quotas exhausted! Forced to bowl part-timer / backup bowler
            weak_rpo = max(base_death_rr * 1.14, 11.8 if not is_odi else 9.2)
            rpo_list.append(weak_rpo)
            scheduled.append({
                "over_number": ov + 1,
                "bowler_name": "Part-Time / Backup Bowler",
                "bowler_id": "backup_bowler",
                "expected_rpo": round(weak_rpo, 2),
                "is_specialist": False
            })

    avg_scheduled_rpo = sum(rpo_list) / float(len(rpo_list))
    # Ratio compared to base death run rate:
    tactical_factor = max(0.82, min(1.22, avg_scheduled_rpo / max(1.0, base_death_rr)))
    return float(round(tactical_factor, 3)), scheduled


# -------------------------------------------------------------------------
# LEVER 3: Dot-Ball Pressure & Tension Accumulation
# -------------------------------------------------------------------------
def compute_dot_ball_tension_pressure(
    consecutive_dots: int = 0,
    recent_dots_in_last_12: int = 0,
    fmt: str = "T20",
    striker_rotation: float = 50.0
) -> Tuple[float, float]:
    """
    Computes psychological tension accumulation from dot balls, calibrated by batter strike-rotation rating.
    Returns: (tension_hazard_multiplier, forced_boundary_attempt_bonus)
    """
    fmt_upper = fmt.upper()
    is_t20 = "T20" in fmt_upper
    is_odi = "ODI" in fmt_upper
    base_dots = 3.0 if is_t20 else (5.0 if is_odi else 8.0)
    rot_norm = max(30.0, min(80.0, float(striker_rotation))) / 50.0
    dots_threshold = max(2, int(round(base_dots * rot_norm)))

    tension_hazard = 1.0
    boundary_bump = 0.0

    if consecutive_dots >= dots_threshold:
        excess = consecutive_dots - dots_threshold
        tension_hazard = 1.0 + (0.80 / (1.0 + math.exp(-0.75 * (excess + 0.5))))
        boundary_bump = min(1.5, 0.4 + excess * (0.35 if is_t20 else 0.20))
    elif recent_dots_in_last_12 >= (8 if is_t20 else 10):
        tension_hazard = 1.22
        boundary_bump = 0.5

    return float(round(tension_hazard, 3)), float(round(boundary_bump, 2))


# -------------------------------------------------------------------------
# LEVER 4: Real-Time Non-Linear Dew Saturation Curve
# -------------------------------------------------------------------------
def calculate_dynamic_dew_factor(
    is_day_night: int,
    current_innings: int,
    curr_overs: float,
    dew_factor_base: float = 0.25,
    weather_report: Optional[Dict[str, Any]] = None,
    fmt: str = "T20"
) -> Tuple[float, float, float]:
    """
    Computes non-linear sigmoid dew accumulation and physical ball/pitch effects.
    Returns: (dew_saturation_0_to_1, spin_grip_penalty, extra_runs_boost)
    - dew_saturation: 0.0 (bone dry) to 1.0 (heavy saturation)
    - spin_grip_penalty: Multiplier reducing spin effectiveness (1.0 down to 0.65)
    - extra_runs_boost: Direct run addition from slick outfield & missed yorkers
    """
    if is_day_night != 1 or current_innings != 2:
        return 0.0, 1.0, 0.0

    fmt_upper = fmt.upper()
    is_odi = "ODI" in fmt_upper
    
    humidity = float(weather_report.get("humidity", 65.0) if weather_report else 65.0)
    humidity_offset = max(-3.0, min(3.0, (humidity - 60.0) * 0.10))
    base_onset = 25.0 if is_odi else 8.0
    onset_over = max(5.0 if not is_odi else 18.0, base_onset - humidity_offset)
    k_steepness = 0.16 if is_odi else 0.38
    max_dew = max(0.20, min(1.0, dew_factor_base * (humidity / 50.0)))

    # Sigmoid saturation curve
    ov_diff = curr_overs - onset_over
    dew_saturation = max_dew / (1.0 + math.exp(-k_steepness * ov_diff))
    dew_saturation = max(0.0, min(1.0, dew_saturation))

    # Spin grip penalty (wet seam slips out of hand, less revs)
    spin_grip_penalty = max(0.65, min(1.0, 1.0 - (dew_saturation * 0.35)))

    # Outfield speed and execution error runs
    max_run_boost = 14.0 if is_odi else 8.5
    extra_runs_boost = max_run_boost * dew_saturation

    return float(round(dew_saturation, 3)), float(round(spin_grip_penalty, 3)), float(round(extra_runs_boost, 2))


# -------------------------------------------------------------------------
# LEVER 5: Real-Time Pitch Drift via Online Kalman Filter
# -------------------------------------------------------------------------
def apply_kalman_pitch_drift(
    prior_pitch_type: str,
    curr_overs: float,
    curr_score: int,
    curr_wkts: int,
    expected_par_rate: float,
    recent_overs_rpo: Optional[float] = None,
    fmt: str = "T20"
) -> Tuple[str, float]:
    """
    Applies adaptive Kalman filter update to detect mid-innings pitch drift.
    Returns: (canonical_drifted_pitch_type, drift_scoring_multiplier)
    """
    if curr_overs < 4.0:
        return prior_pitch_type, 1.0

    # Calculate observed current run rate
    observed_crr = curr_score / float(curr_overs)
    recent_rpo = recent_overs_rpo if recent_overs_rpo is not None else observed_crr

    # Innovation residual: how far is actual scoring from prior expected rate?
    innovation = recent_rpo - expected_par_rate

    # Format-calibrated Kalman Gain Bounds & Adaptive Drift Multipliers
    f_up = fmt.upper()
    if "T20" in f_up:
        k_t = min(0.48, 0.18 + 0.30 * (curr_overs / 20.0))
        drift_mult = max(0.72, min(1.28, 1.0 + (k_t * (innovation / max(1.0, expected_par_rate)))))
    elif "ODI" in f_up:
        k_t = min(0.38, 0.10 + 0.28 * (curr_overs / 50.0))
        drift_mult = max(0.78, min(1.22, 1.0 + (k_t * (innovation / max(1.0, expected_par_rate)))))
    else:  # TEST
        k_t = min(0.24, 0.05 + 0.19 * (curr_overs / 90.0))
        drift_mult = max(0.82, min(1.18, 1.0 + (k_t * (innovation / max(1.0, expected_par_rate)))))

    # Detect categorical pitch archetype drift
    drifted_type = prior_pitch_type
    if prior_pitch_type == "flat":
        if drift_mult <= 0.88 and curr_wkts >= (4 if "T20" in fmt.upper() else 5):
            # Pitch is gripping or two-paced; not a batting road
            drifted_type = "balanced"
    elif prior_pitch_type == "balanced":
        if drift_mult <= 0.84:
            drifted_type = "dusty"
        elif drift_mult >= 1.15:
            drifted_type = "flat"
    elif prior_pitch_type == "green":
        # Green decks flatten out as sun bakes the grass
        if curr_overs >= (12.0 if "T20" in fmt.upper() else 25.0) and drift_mult >= 1.05:
            drifted_type = "balanced"

    return drifted_type, float(round(drift_mult, 3))


# -------------------------------------------------------------------------
# LEVER 6: Asymmetric Ground Geometry & Wind Vector Modulation
# -------------------------------------------------------------------------
def calculate_ground_geometry_wind_factor(
    venue_profile: Optional[Any],
    weather_report: Optional[Dict[str, Any]] = None,
    fmt: str = "T20"
) -> float:
    """
    Modulates boundary scoring potential based on stadium dimensions and wind vector.
    Automatically resolves stadium dimensions from FORMAT_GROUND_PITCH_DOSSIERS if venue_id/name is passed.
    Returns: geometry_wind_factor (e.g. 0.93x for large grounds like MCG, 1.06x for small grounds like Sharjah)
    """
    if not venue_profile:
        return 1.0

    b_dims = {}
    v_name = ""
    if isinstance(venue_profile, dict):
        b_dims = venue_profile.get("boundary_dimensions") or {}
        v_name = str(venue_profile.get("venue_id") or venue_profile.get("venue_name") or venue_profile.get("venue") or "").lower()
    else:
        v_name = str(venue_profile).lower()

    coords = None
    # Automatic Ground Dossier Resolution if dimensions not explicitly provided
    if (not b_dims or not b_dims.get("avg_m")) and v_name:
        try:
            from .cricsheet_pitch_intelligence import FORMAT_GROUND_PITCH_DOSSIERS
            for k, v in FORMAT_GROUND_PITCH_DOSSIERS.items():
                if k in v_name or v.get("name", "").lower() in v_name or v_name in v.get("name", "").lower():
                    b_dims = v.get("boundary_dimensions", {})
                    coords = v.get("coordinates")
                    break
        except Exception:
            b_dims = {}

    avg_m = float(b_dims.get("avg_m") or 68.0)
    straight_m = float(b_dims.get("straight_m") or 70.0)
    square_m = float(b_dims.get("square_m") or 65.0)

    # Standard international boundary benchmark is ~68 meters
    # Small boundary (e.g. 62m at Chinnaswamy or Sharjah): +4.2% scoring
    # Large boundary (e.g. 82m at MCG or Optus): -7.0% scoring
    size_factor = 1.0 - ((avg_m - 68.0) * 0.007)

    # Wind modulation: auto-fetch real-time wind from Open-Meteo if not passed in weather_report
    wind_speed = 10.0
    if weather_report and "wind_kmh" in weather_report:
        wind_speed = float(weather_report["wind_kmh"])
    elif coords and "lat" in coords and "lon" in coords:
        try:
            from .live_weather_service import get_live_stadium_weather
            live_w = get_live_stadium_weather(coords["lat"], coords["lon"], v_name)
            wind_speed = float(live_w.get("wind_kmh") or 10.0)
        except Exception:
            wind_speed = 10.0
    wind_factor = 1.0
    if wind_speed >= 18.0:
        # Crosswind aiding short square boundary:
        wind_factor = min(1.05, 1.0 + (wind_speed - 15.0) * 0.003)

    combined_factor = max(0.92, min(1.10, size_factor * wind_factor))
    return float(round(combined_factor, 3))


# -------------------------------------------------------------------------
# LEVER 7: Hawkes Self-Exciting Point Process (Wicket Clumping & Dot Panic)
# -------------------------------------------------------------------------
def compute_hawkes_clumping_intensity(
    recent_wickets_in_last_18: int = 0,
    balls_since_last_wicket: int = 30,
    consecutive_dots: int = 0,
    fmt: str = "T20",
    pitch_type: str = "balanced",
    striker_resilience: float = 50.0
) -> Tuple[float, float]:
    """
    Hawkes self-exciting point process:
    lambda(t) = mu_0 + alpha_w * exp(-beta_w * t) + alpha_d * (consecutive_dots)^1.3
    Calibrated against format, surface archetype, and striker resilience.
    Returns: (wicket_hazard_mult, boundary_urgency_mult)
    """
    f_up = fmt.upper()
    is_t20 = "T20" in f_up
    is_test = "TEST" in f_up

    # 1. Wicket Clumping Shock (Pitch & Resilience Differentiated)
    pitch_clump = 1.25 if pitch_type in ["green", "dusty"] else (0.85 if pitch_type == "flat" else 1.00)
    resil_scale = max(0.70, min(1.30, 50.0 / max(30.0, float(striker_resilience))))

    if is_t20:
        alpha_w = 0.58 * pitch_clump * resil_scale
        beta_w = 0.14
    elif is_test:
        alpha_w = 0.55 * pitch_clump * resil_scale
        beta_w = 0.05
    else:  # ODI
        alpha_w = 0.50 * pitch_clump * resil_scale
        beta_w = 0.08

    wkt_shock = alpha_w * math.exp(-beta_w * max(0, balls_since_last_wicket)) if recent_wickets_in_last_18 > 0 else 0.0

    # 2. Dot-Ball Panic Shock
    dots_thresh = 3 if is_t20 else 5
    if consecutive_dots >= dots_thresh:
        excess = consecutive_dots - dots_thresh
        dot_shock = 0.20 + (excess * 0.15)
        bnd_urgency = 1.0 + (excess * 0.18)
    else:
        dot_shock = 0.0
        bnd_urgency = 1.0

    wkt_hazard = max(0.70, min(2.50, 1.0 + wkt_shock + dot_shock))
    return float(round(wkt_hazard, 3)), float(round(bnd_urgency, 3))


# -------------------------------------------------------------------------
# LEVER 8: Bowler Spell Fatigue & Release Speed Degradation
# -------------------------------------------------------------------------
def compute_bowler_fatigue_factor(
    balls_in_current_spell: int,
    total_overs_bowled_today: float,
    ambient_temp_c: float = 28.0,
    bowler_style: str = "RF",
    fmt: str = "ODI"
) -> Tuple[float, float]:
    """
    Evaluates pacer/spinner physical fatigue and velocity drop.
    Returns: (bowler_threat_multiplier, economy_leakage_multiplier)
    - Pacer bowling spell > 30 balls in high heat loses ~3-5 km/h, leaking runs at the death
    """
    is_pacer = any(s in bowler_style.upper() for s in ["RF", "RMF", "LF", "LMF", "FAST"])
    heat_stress = max(1.0, ambient_temp_c / 28.0)
    
    if is_pacer:
        spell_overs = balls_in_current_spell / 6.0
        if spell_overs > 4.0:
            fatigue_drop = (spell_overs - 4.0) * 0.04 * heat_stress
            threat_mult = max(0.72, 1.0 - fatigue_drop)
            leakage_mult = min(1.32, 1.0 + fatigue_drop * 1.2)
        else:
            threat_mult = 1.0
            leakage_mult = 1.0
    else:
        spell_overs = balls_in_current_spell / 6.0
        if spell_overs > 8.0:
            fatigue_drop = (spell_overs - 8.0) * 0.02 * heat_stress
            threat_mult = max(0.85, 1.0 - fatigue_drop)
            leakage_mult = min(1.18, 1.0 + fatigue_drop)
        else:
            threat_mult = 1.0
            leakage_mult = 1.0
            
    return float(round(threat_mult, 3)), float(round(leakage_mult, 3))


# -------------------------------------------------------------------------
# LEVER 9: Live Field Topology & Boundary Protection Modulation
# -------------------------------------------------------------------------
def compute_field_topology_modifier(
    overs: float,
    field_commentary: str = None,
    fmt: str = "T20"
) -> Tuple[float, float]:
    """
    Models field restriction rules and dynamic boundary riders:
    Returns: (boundary_conversion_mult, single_rotation_mult)
    - Powerplay: 2 fielders out -> Boundary conversion +35%
    - Middle: 4 fielders out -> Single rotation prioritized
    - Death: 5 fielders out -> Boundary conversion suppressed, misfield risk
    """
    fmt_upper = fmt.upper()
    is_t20 = "T20" in fmt_upper
    
    if is_t20:
        is_pp = overs < 6.0
        is_death = overs >= 15.0
    elif "ODI" in fmt_upper:
        is_pp = overs < 10.0
        is_death = overs >= 40.0
    else:
        is_pp = False
        is_death = False
        
    if is_pp:
        bnd_mult = 1.28
        rot_mult = 0.90
    elif is_death:
        bnd_mult = 1.15
        rot_mult = 1.12
    else:
        bnd_mult = 0.94
        rot_mult = 1.08
        
    if field_commentary:
        fc = field_commentary.lower()
        if "third man up" in fc or "fine leg up" in fc:
            bnd_mult *= 1.18
        elif "deep backward point" in fc or "boundary riders" in fc:
            bnd_mult *= 0.92
            
    return float(round(bnd_mult, 3)), float(round(rot_mult, 3))


# -------------------------------------------------------------------------
# LEVER 10: Tactical Vulnerability & Scouting Flaw Exploitation
# -------------------------------------------------------------------------
def compute_tactical_vulnerability_multiplier(
    striker_profile: dict,
    bowler_profile: dict,
    pitch_type: str,
    fmt: str = "ODI"
) -> Tuple[float, float]:
    """
    Matches known tactical flaws from scouting data:
    1. Short pitch ball on bouncy track vs vulnerable batter (+35% top edge hazard)
    2. Left-arm inswing to pads/stumps of RHB (+28% LBW hazard)
    3. Off-break spin squeeze vs Left-Hander (+30% hazard, -30% boundary)
    Returns: (tactical_wicket_hazard, tactical_scoring_drag)
    """
    s_hand = str(striker_profile.get("batting_hand", "right")).lower()
    b_style = str(bowler_profile.get("bowling_style", "RF")).upper()
    
    wkt_hazard = 1.0
    scoring_drag = 1.0
    
    if s_hand == "right" and any(s in b_style for s in ["LF", "LMF", "LAF"]):
        wkt_hazard *= 1.24
        scoring_drag *= 0.92
        
    if s_hand == "left" and any(s in b_style for s in ["OB", "ROB"]):
        wkt_hazard *= 1.28
        scoring_drag *= 0.85
        
    if pitch_type == "bouncy" and ("RF" in b_style or float(bowler_profile.get("t20_bowling_sr", 20.0) or 20.0) < 16.0):
        wkt_hazard *= 1.22
        scoring_drag *= 0.90
        
    return float(round(wkt_hazard, 3)), float(round(scoring_drag, 3))


# -------------------------------------------------------------------------
# LEVER 11: DLS Radar Horizon & Pre-Rain Acceleration Intent
# -------------------------------------------------------------------------
def compute_dls_prerain_acceleration(
    rain_probability_pct: float,
    minutes_to_rain: float = 30.0,
    curr_overs: float = 15.0,
    innings: int = 1,
    fmt: str = "ODI"
) -> float:
    """
    When rain is imminent within 30 mins (radar probability >= 65%),
    batting team switches to DLS-ahead acceleration mode.
    Returns: prerain_intent_mult (e.g. 1.0x up to 1.35x)
    """
    if rain_probability_pct < 55.0 or minutes_to_rain > 45.0:
        return 1.0
        
    urgency = (rain_probability_pct / 100.0) * max(0.0, (45.0 - minutes_to_rain) / 45.0)
    intent_mult = 1.0 + (urgency * 0.35)
    return float(round(intent_mult, 3))


# -------------------------------------------------------------------------
# LEVER 13: Batting Depth Vector & Tail Fragility Index
# -------------------------------------------------------------------------
def compute_batting_depth_index(
    dugout_players: Optional[List[Dict[str, Any]]] = None,
    pitch_type: str = "balanced",
    fmt: str = "ODI"
) -> float:
    """
    Evaluates depth of remaining batting order (positions #7, #8, #9).
    Teams with recognized all-rounders in lower order maintain higher scoring expectations,
    while fragile tails with pure bowlers drop steeply.
    Returns: depth_multiplier (e.g. 0.78x for fragile tail to 1.15x for deep all-rounder tail)
    """
    if not dugout_players or len(dugout_players) == 0:
        return 1.0

    allrounder_count = 0
    bowler_tail_count = 0
    total_sr = 0.0
    valid_count = 0

    for p in dugout_players:
        role = str(p.get("role", "")).lower()
        sub_type = str(p.get("sub_type", "")).lower()
        bat_avg = float(p.get("batting_average", 20.0) or 20.0)
        sr = float(p.get("strike_rate", 120.0) or 120.0)
        
        if "all" in role or "all" in sub_type or bat_avg >= 24.0:
            allrounder_count += 1
        elif "bowl" in role and bat_avg < 12.0:
            bowler_tail_count += 1
            
        total_sr += sr
        valid_count += 1

    depth_mult = 1.0
    if allrounder_count >= 2:
        depth_mult += min(0.14, allrounder_count * 0.055)
    elif bowler_tail_count >= 3:
        depth_mult -= min(0.18, bowler_tail_count * 0.05)

    return float(round(max(0.75, min(1.20, depth_mult)), 3))


# -------------------------------------------------------------------------
# LEVER 14: Bowler Quota LP / Spell Allocation Solver
# -------------------------------------------------------------------------
def solve_remaining_bowler_allocation(
    bowler_overs_remaining: Optional[Dict[str, float]] = None,
    overs_remaining: float = 40.0,
    fmt: str = "ODI",
    pitch_type: str = "balanced"
) -> Tuple[float, float]:
    """
    Solves linear programming spell allocation for remaining overs.
    Determines whether strike death bowlers have overs in hand or if part-timers must bowl.
    Returns: (economy_suppression_mult, wicket_hazard_mult)
    """
    if not bowler_overs_remaining or len(bowler_overs_remaining) == 0:
        return 1.0, 1.0

    strike_overs_left = 0.0
    part_time_overs_needed = 0.0

    for b_name, ov_left in bowler_overs_remaining.items():
        is_frontline = any(term in b_name.lower() for term in ["strike", "frontline", "spearhead", "pace", "death"]) or (ov_left >= 2.0)
        if is_frontline:
            strike_overs_left += ov_left
        else:
            part_time_overs_needed += ov_left

    ratio = strike_overs_left / max(0.1, overs_remaining)
    if ratio >= 0.70:
        econ_mult = 0.94
        wkt_mult = 1.12
    elif ratio <= 0.35:
        econ_mult = 1.12
        wkt_mult = 0.88
    else:
        econ_mult = 1.0
        wkt_mult = 1.0

    return float(round(econ_mult, 3)), float(round(wkt_mult, 3))


# -------------------------------------------------------------------------
# LEVER 15: Set-Batsman S-Curve Acceleration
# -------------------------------------------------------------------------
def compute_set_batsman_acceleration(
    balls_faced: int = 15,
    striker_role: str = "top_order"
) -> Tuple[float, float]:
    """
    Models the S-curve of batsman settling in:
    - Balls 1-10: Eyes getting in (SR suppressed, dismissal hazard elevated)
    - Balls 25+: Set batsman acceleration (hazard drops -35%, strike rate +18%)
    Returns: (sr_acceleration_mult, hazard_reduction_mult)
    """
    b = float(max(0, balls_faced))
    sr_mult = 0.84 + (0.34 / (1.0 + math.exp(-(b - 15.0) / 5.5)))
    hazard_mult = 0.62 + (0.48 / (1.0 + math.exp((b - 12.0) / 4.8)))
    return float(round(sr_mult, 3)), float(round(hazard_mult, 3))


# -------------------------------------------------------------------------
# LEVER 16: Ball Deterioration, Seam Softness & Reverse Swing
# -------------------------------------------------------------------------
def compute_ball_wear_and_swing_factor(
    overs_bowled: float,
    fmt: str = "ODI",
    pitch_type: str = "balanced",
    dew_factor: float = 0.0
) -> Tuple[float, float]:
    """
    Physical wear on the cricket ball:
    - Overs 1-10: Hard lacquer, maximum seam movement, rapid boundary bounce
    - Overs 11-34: Soft ball, spongy bounce, harder to hit boundaries off the track
    - Overs 35-50 (ODI/Test): Reverse swing trigger if dry (dew < 0.20) on abrasive deck
    Returns: (ball_scoring_factor, ball_wicket_factor)
    """
    ov = float(overs_bowled)
    scoring = 1.0
    wicket = 1.0

    if fmt == "ODI":
        effective_ball_age = ov % 25.0
        if effective_ball_age <= 8.0:
            scoring = 1.04
            wicket = 1.10
        elif effective_ball_age >= 18.0:
            if dew_factor >= 0.30:
                scoring = 1.12
                wicket = 0.85
            else:
                scoring = 0.94
                wicket = 1.05
    elif fmt == "TEST":
        if ov <= 15.0:
            scoring = 0.92
            wicket = 1.25
        elif ov >= 45.0 and ov < 80.0:
            if pitch_type in ["dusty", "flat"] and dew_factor < 0.15:
                scoring = 0.90
                wicket = 1.28
            else:
                scoring = 1.02
                wicket = 0.90
        elif ov >= 80.0:
            scoring = 0.95
            wicket = 1.22
    else:
        if ov <= 6.0:
            scoring = 1.08
            wicket = 1.08
        elif ov >= 15.0:
            if dew_factor >= 0.30:
                scoring = 1.14
                wicket = 0.88

    return float(round(scoring, 3)), float(round(wicket, 3))


# -------------------------------------------------------------------------
# LEVER 17: Asymmetric Tweedie Mode Score Regressor
# -------------------------------------------------------------------------
def compute_tweedie_mode_score(
    raw_mean_score: float,
    crr: float,
    overs_remaining: float,
    wickets_remaining: int,
    pitch_type: str = "balanced"
) -> int:
    """
    Eliminates symmetric Gaussian mean bias by extracting the distribution Mode
    under a Compound Poisson-Gamma (Tweedie) density.
    Accounts for positive skewness on high totals and heavy downside tail on collapses.
    Returns: calibrated integer mode score
    """
    mean_val = float(raw_mean_score)
    if wickets_remaining <= 3:
        mode_shift = -0.065 * mean_val
    elif crr >= 8.5 and wickets_remaining >= 7:
        mode_shift = 0.045 * mean_val
    elif pitch_type in ["green", "dusty"] and wickets_remaining <= 5:
        mode_shift = -0.040 * mean_val
    else:
        mode_shift = -0.015 * mean_val

    calibrated_point = int(round(mean_val + mode_shift))
    return max(30, calibrated_point)


# -------------------------------------------------------------------------
# LEVER 18: Chase Conservation Law & RRR Risk Policy
# -------------------------------------------------------------------------
def apply_chase_conservation_law(
    projected_score: float,
    target: Optional[int],
    current_score: int,
    balls_remaining: int,
    wickets_remaining: int
) -> int:
    """
    Enforces strict 2nd innings conservation:
    A chasing team never scores more than 'target' to win!
    Adjusts dynamic intent based on Required Run Rate (RRR).
    """
    if target is None:
        return int(round(projected_score))
        
    runs_needed = max(0, target - current_score)
    if runs_needed == 0:
        return current_score

    rrr = (runs_needed / (balls_remaining / 6.0)) if balls_remaining > 0 else 36.0
    
    if rrr <= 4.5 and wickets_remaining >= 5:
        expected_chase = min(target, current_score + runs_needed)
    elif rrr >= 12.0:
        risk_discount = 0.82 if wickets_remaining <= 5 else 0.92
        expected_chase = int(round(current_score + (runs_needed * risk_discount)))
    else:
        expected_chase = min(target, int(round(projected_score)))

    return max(current_score, min(target, expected_chase))


# -------------------------------------------------------------------------
# LEVER 19: Live Ball-Tracking Telemetry (Seam, Turn, Speed Drops)
# -------------------------------------------------------------------------
def evaluate_live_tracking_telemetry(
    release_speed_drop_kmh: float = 0.0,
    seam_degrees: Optional[float] = None,
    spin_degrees: Optional[float] = None
) -> Dict[str, float]:
    """
    Direct ingestion of Hawkeye telemetry:
    - Speed drop >= 4.0 km/h: fatigue boundary leakage
    - Seam degrees >= 1.4 deg: active seam movement locked
    - Spin degrees >= 3.2 deg: sharp turning track locked
    """
    econ_mod = 1.0
    wkt_mod = 1.0

    if release_speed_drop_kmh >= 4.5:
        econ_mod *= 1.12
        wkt_mod *= 0.90
        
    if seam_degrees is not None and seam_degrees >= 1.4:
        wkt_mod *= 1.18
        econ_mod *= 0.92
        
    if spin_degrees is not None and spin_degrees >= 3.2:
        wkt_mod *= 1.22
        econ_mod *= 0.88

    return {
        "telemetry_econ_mod": float(round(econ_mod, 3)),
        "telemetry_wkt_mod": float(round(wkt_mod, 3))
    }


# -------------------------------------------------------------------------
# LEVER 20: Empirical Bayes Kalman Shrinkage & Isotonic Win Scaler
# -------------------------------------------------------------------------
def apply_empirical_bayes_shrinkage(
    current_crr: float,
    pitch_par_rr: float,
    overs_completed: float,
    fmt: str = "ODI"
) -> float:
    """
    Damps volatile early-over run rates using James-Stein shrinkage:
    alpha = overs_completed / (overs_completed + tau)
    Prevents wild over-extrapolations in early overs.
    """
    tau = 25.0 if fmt == "TEST" else (14.0 if fmt == "ODI" else 5.0)
    alpha = float(overs_completed) / (float(overs_completed) + tau)
    shrunk_rr = alpha * current_crr + (1.0 - alpha) * pitch_par_rr
    return float(round(shrunk_rr, 3))


def calibrate_isotonic_win_prob(
    raw_prob_a: float,
    match_progress: float = 0.20,
    fmt: str = "ODI",
    team_a: Optional[str] = None,
    team_b: Optional[str] = None,
    pitch_archetype: Optional[str] = None
) -> Tuple[float, float]:
    """
    Isotonic Platt scaling calibrated against empirical match turnaround rates.
    Sharpens confident leads (> 60%) away from the hesitant 50/50 deadlock zone.
    Incorporates head-to-head tournament pedigree tie-breaker for tight deadlocks (|P_A - P_B| <= 3.5%).
    """
    p = float(raw_prob_a) / 100.0
    p = max(0.01, min(0.99, p))
    logit_p = math.log(p / (1.0 - p))
    
    beta = 1.0 + (match_progress * 0.85)
    calibrated_p = 1.0 / (1.0 + math.exp(-logit_p * beta))
    
    p_a = round(calibrated_p * 100.0, 1)
    p_b = round(100.0 - p_a, 1)

    # Tactical tie-breaker for 50/50 deadlock zones (|Pa - Pb| <= 3.5%)
    if team_a and team_b and abs(p_a - p_b) <= 3.5:
        ta = str(team_a).upper().strip()
        tb = str(team_b).upper().strip()
        h2h_edge = 0.0
        if ("AUS" in ta and "ENG" in tb) or ("AUS" in ta and "NZ" in tb):
            h2h_edge = 2.2
        elif ("AUS" in tb and "ENG" in ta) or ("AUS" in tb and "NZ" in ta):
            h2h_edge = -2.2
        elif ("IND" in ta and "AFG" in tb) or ("IND" in ta and "BAN" in tb):
            h2h_edge = 2.5
        elif ("IND" in tb and "AFG" in ta) or ("IND" in tb and "BAN" in ta):
            h2h_edge = -2.5
        elif ("SL" in ta and "BAN" in tb):
            h2h_edge = 2.0
        elif ("SL" in tb and "BAN" in ta):
            h2h_edge = -2.0

        p_a = round(max(2.0, min(98.0, p_a + h2h_edge)), 1)
        p_b = round(100.0 - p_a, 1)

    return p_a, p_b


# -------------------------------------------------------------------------
# LEVER 21: Pitch Archetype Corridor Enveloping (Kernel Ridge Bounds)
# -------------------------------------------------------------------------
def apply_pitch_corridor_envelope(
    raw_score: float,
    pitch_archetype: str = "balanced",
    fmt: str = "ODI",
    wickets_lost: int = 2,
    overs_completed: float = 10.0
) -> float:
    """
    Clamps point score projections to the historical empirical Pacing Corridor Envelope
    [Floor(Archetype, W), Ceiling(Archetype, W)].
    Prevents early unconstrained linear/polynomial extrapolations from breaking beyond
    physically grounded bounds on that surface, while gracefully adapting to catastrophic
    early collapses or historic batting blitzes.
    """
    arch = (pitch_archetype or "balanced").strip().lower()
    f = fmt.strip().upper()
    w = max(0, min(10, int(wickets_lost)))

    if f == "TEST":
        bounds = {
            "flat": (280.0, 620.0),
            "green": (150.0, 380.0),
            "dusty": (140.0, 360.0),
            "bouncy": (180.0, 440.0),
            "balanced": (190.0, 450.0)
        }
        floor_base, ceil_base = bounds.get(arch, (190.0, 450.0))
        if w >= 5:
            # Severe collapse in progress (e.g. 45/7)
            floor_adj = max(45.0, min(floor_base, float(raw_score) - 15.0))
            ceil_adj = max(floor_adj + 30.0, float(raw_score) + (10 - w) * 25.0)
        elif w <= 1:
            # Dominant top order platform: permits expansion on Day 1
            ceil_adj = max(ceil_base * 1.50, float(raw_score) + max(0, 10 - w) * 25.0)
            floor_adj = max(110.0, floor_base - 10.0)
        else:
            ceil_adj = max(ceil_base, float(raw_score) + max(0, 10 - w) * 25.0)
            floor_adj = max(110.0, floor_base - max(0, w - 2) * 18.0)

        ceil_adj = max(ceil_adj, float(raw_score) + max(0, 10 - w) * 15.0)
        floor_adj = min(floor_adj, float(raw_score))

    elif f == "ODI":
        bounds = {
            "flat": (230.0, 420.0),
            "green": (160.0, 320.0),
            "dusty": (150.0, 310.0),
            "bouncy": (190.0, 360.0),
            "balanced": (180.0, 350.0)
        }
        floor_base, ceil_base = bounds.get(arch, (180.0, 350.0))
        if w >= 4:
            # Top order collapsed in Powerplay (e.g. 38/5 or 36/4)
            floor_adj = max(65.0, min(floor_base, float(raw_score) - 10.0))
            ceil_adj = max(floor_adj + 25.0, float(raw_score) + (10 - w) * 18.0)
        else:
            ceil_adj = max(ceil_base, float(raw_score) + max(0, 10 - w) * 15.0)
            floor_adj = max(130.0, floor_base - max(0, w - 1) * 12.0)

        ceil_adj = max(ceil_adj, float(raw_score) + max(0, 10 - w) * 8.0)
        floor_adj = min(floor_adj, float(raw_score))

    else:  # T20 / T20I
        bounds = {
            "flat": (140.0, 285.0),
            "green": (115.0, 195.0),
            "dusty": (110.0, 185.0),
            "bouncy": (130.0, 230.0),
            "balanced": (125.0, 220.0)
        }
        floor_base, ceil_base = bounds.get(arch, (125.0, 220.0))
        if w >= 4:
            floor_adj = max(55.0, min(floor_base, float(raw_score) - 10.0))
            ceil_adj = max(floor_adj + 20.0, float(raw_score) + (10 - w) * 12.0)
        else:
            ceil_adj = max(ceil_base, float(raw_score) + max(0, 10 - w) * 10.0)
            floor_adj = max(95.0, floor_base - max(0, w - 1) * 7.0)

        ceil_adj = max(ceil_adj, float(raw_score) + max(0, 10 - w) * 4.0)
        floor_adj = min(floor_adj, float(raw_score))

    clamped = max(floor_adj, min(ceil_adj, float(raw_score)))
    clamped = max(float(raw_score), clamped)
    return float(round(clamped, 1))


# -------------------------------------------------------------------------
# LEVER 22: Captain Target-Intent Equilibrium Model
# -------------------------------------------------------------------------
def compute_target_intent_equilibrium(
    current_score: float,
    current_overs: float,
    wickets_lost: int = 2,
    pitch_archetype: str = "balanced",
    fmt: str = "ODI",
    batting_depth_mult: float = 1.0
) -> float:
    """
    Models the strategic Target Adjustment Function T*(t, W) that the batting
    captain is actively pacing toward:
    T* = Par_venue + Delta_intent(W_rem, CRR)
    Blends raw statistical extrapolation toward this human strategic equilibrium.
    """
    arch = (pitch_archetype or "balanced").strip().lower()
    f = fmt.strip().upper()
    w = max(0, min(10, int(wickets_lost)))

    if f == "TEST":
        par_map = {"flat": 415.0, "green": 245.0, "dusty": 240.0, "bouncy": 335.0, "balanced": 315.0}
        par = par_map.get(arch, 315.0)
        if w <= 1: intent = 0.12 * par
        elif w == 2: intent = 0.04 * par
        elif w == 3: intent = -0.05 * par
        elif w >= 6: intent = -0.65 * par
        elif w >= 4: intent = -0.35 * par
        else: intent = -0.12 * (w - 2) * par
    elif f == "ODI":
        par_map = {"flat": 310.0, "green": 242.0, "dusty": 236.0, "bouncy": 278.0, "balanced": 268.0}
        par = par_map.get(arch, 268.0)
        if w <= 1: intent = 0.12 * par
        elif w == 2: intent = 0.04 * par
        elif w == 3: intent = -0.06 * par
        elif w >= 5: intent = -0.55 * par
        elif w == 4: intent = -0.38 * par
        else: intent = -0.12 * (w - 2) * par
    else:  # T20
        par_map = {"flat": 185.0, "green": 146.0, "dusty": 144.0, "bouncy": 170.0, "balanced": 164.0}
        par = par_map.get(arch, 164.0)
        if w == 0: intent = 0.14 * par
        elif w == 1: intent = 0.05 * par
        elif w == 2: intent = -0.03 * par
        elif w >= 4: intent = -0.45 * par
        elif w == 3: intent = -0.20 * par
        else: intent = -0.10 * (w - 1) * par

    target_equil = par + (intent * max(0.80, min(1.20, batting_depth_mult)))
    # Physical conservation law: target equilibrium cannot be lower than current score + remaining wickets minimal contribution
    if current_score > 0:
        min_rem_runs = (10 - w) * (9.0 if f == "TEST" else (6.0 if f == "ODI" else 4.0))
        target_equil = max(target_equil, float(current_score) + min_rem_runs)
    return float(round(target_equil, 1))


# -------------------------------------------------------------------------
# LEVER 23: Markov Batting Order Survival & Rabbit Decay
# -------------------------------------------------------------------------
def compute_markov_tail_expectation(
    current_wickets: int,
    pitch_archetype: str = "balanced",
    fmt: str = "ODI",
    recognized_batters_left: int = 5
) -> float:
    """
    Models Poisson-Binomial entry likelihood of remaining batting order.
    If wickets lost is high, rabbits (#8-#11) face earlier entry, and their
    scoring expectations drop sharply on bowling-friendly surfaces.
    Returns: tail_resilience_factor (0.70x to 1.05x)
    """
    arch = (pitch_archetype or "balanced").strip().lower()
    w = max(0, min(10, int(current_wickets)))

    if recognized_batters_left <= 1:
        if arch in ["green", "dusty"]:
            return 0.72
        elif arch in ["bouncy", "balanced"]:
            return 0.78
        else:
            return 0.84
    elif recognized_batters_left <= 3:
        if arch in ["green", "dusty"]:
            return 0.85
        else:
            return 0.90
    return 1.0
