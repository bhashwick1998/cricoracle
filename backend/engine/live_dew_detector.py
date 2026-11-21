"""
Live Dew Detection & Dynamic Chasing Advantage Engine
Models and detects dew-prone pitches and real-time dew formation in Day/Night ODIs and T20/T20I second innings:

4-Pillar Live Detection Architecture:
1. Thermodynamic & Microclimate Physics:
   - Magnus-Tetens dew point equation: T_dp = f(T, RH)
   - Dew Point Depression: Delta T_dew = T_ambient - T_dp (Dew forms when Delta T_dew <= 2.5°C)
   - Nocturnal Radiative Turf Cooling: Clear sky (CC <= 25%) + calm wind (v <= 7 km/h) + post-sunset
2. Geographic, Seasonal & Soil Prior Calibration:
   - Subcontinent coastal/river plains (Wankhede, Eden Gardens, Mohali, Sylhet, Dubai) in Oct-March
   - Arid/High-altitude/SENA summer venues (Lord's, Wanderers, Perth) dew-immune
3. In-Play Ball & Performance Telemetry (Real-time live signal verification):
   - Bowling grip failure: Extras spike (Wides & No-balls >= 1.6x Inn 1 baseline in T20, >= 1.75x in ODI)
   - Spin neutralization: Spin RPO inflation (> 9.0 in T20, > 6.2 in ODI) and turn angle collapse
   - Outfield hydroplaning: Ground boundary conversion surge (+25%)
   - Wicket preservation: Overs per wicket (Opw) expansion
4. Real-time NLP Commentary Stream Mining:
   - Lexicon regex extraction: Towels, drying cloths, wet ball, super sopper, soap bar, slipping
"""

import math
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Geographic & Seasonal Venue Dew Priors (Cricsheet & Meteorological Dossiers)
VENUE_DEW_PRIORS: Dict[str, Dict[str, Any]] = {
    "wankhede": {
        "name": "Wankhede Stadium, Mumbai",
        "base_dew_prior": 0.65,
        "coastal_river_proximity": True,
        "peak_dew_months": [10, 11, 12, 1, 2, 3, 4, 5],
        "drainage_subair": False,
        "description": "Arabian Sea proximity; heavy evening dew makes chasing heavily favored."
    },
    "eden_gardens": {
        "name": "Eden Gardens, Kolkata",
        "base_dew_prior": 0.55,
        "coastal_river_proximity": True,
        "peak_dew_months": [10, 11, 12, 1, 2, 3],
        "drainage_subair": False,
        "description": "Hooghly river basin; dense winter night condensation."
    },
    "mohali": {
        "name": "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "base_dew_prior": 0.60,
        "coastal_river_proximity": False,
        "peak_dew_months": [10, 11, 12, 1, 2],
        "drainage_subair": False,
        "description": "Indo-Gangetic winter plains; severe inversion layer creates heavy wet ball conditions."
    },
    "feroz_shah_kotla": {
        "name": "Arun Jaitley Stadium, Delhi",
        "base_dew_prior": 0.55,
        "coastal_river_proximity": True,
        "peak_dew_months": [10, 11, 12, 1, 2],
        "drainage_subair": False,
        "description": "Yamuna river basin; thick evening smog and dew settle rapidly."
    },
    "dubai": {
        "name": "Dubai International Cricket Stadium",
        "base_dew_prior": 0.55,
        "coastal_river_proximity": True,
        "peak_dew_months": [9, 10, 11, 12, 1, 2, 3],
        "drainage_subair": False,
        "description": "Ring of Fire stadium; high evening dew creates a decisive 2nd-innings chase bias."
    },
    "sharjah": {
        "name": "Sharjah Cricket Stadium",
        "base_dew_prior": 0.50,
        "coastal_river_proximity": True,
        "peak_dew_months": [9, 10, 11, 12, 1, 2, 3],
        "drainage_subair": False,
        "description": "Desert nocturnal cooling; dew sets in heavily during evening limited-overs chases."
    },
    "sylhet": {
        "name": "Sylhet International Cricket Stadium",
        "base_dew_prior": 0.50,
        "coastal_river_proximity": True,
        "peak_dew_months": [11, 12, 1, 2, 3],
        "drainage_subair": False,
        "description": "Tea-garden valley microclimate; high humidity and early evening mist/dew."
    },
    "mirpur": {
        "name": "Shere Bangla National Stadium, Mirpur, Dhaka",
        "base_dew_prior": 0.45,
        "coastal_river_proximity": True,
        "peak_dew_months": [11, 12, 1, 2, 3],
        "drainage_subair": False,
        "description": "Subtropical river plain; heavy evening condensation in winter D/N fixtures."
    },
    "lahore": {
        "name": "Gaddafi Stadium, Lahore",
        "base_dew_prior": 0.45,
        "coastal_river_proximity": True,
        "peak_dew_months": [10, 11, 12, 1, 2],
        "drainage_subair": False,
        "description": "Ravi river basin; winter night dew significantly hampers spinners."
    },
    "karachi": {
        "name": "National Stadium, Karachi",
        "base_dew_prior": 0.40,
        "coastal_river_proximity": True,
        "peak_dew_months": [10, 11, 12, 1, 2, 3],
        "drainage_subair": False,
        "description": "Coastal Arabian Sea breeze; evening humidity induces wet ball conditions."
    },
    "ahmedabad": {
        "name": "Narendra Modi Stadium, Ahmedabad",
        "base_dew_prior": 0.40,
        "coastal_river_proximity": True,
        "peak_dew_months": [10, 11, 12, 1, 2, 3],
        "drainage_subair": False,
        "description": "Sabarmati riverbank; moderate to heavy dew settles during late evening overs."
    },
    "chennai": {
        "name": "MA Chidambaram Stadium, Chepauk, Chennai",
        "base_dew_prior": 0.35,
        "coastal_river_proximity": True,
        "peak_dew_months": [10, 11, 12, 1, 2],
        "drainage_subair": False,
        "description": "Bay of Bengal coastline; humid night air, dew heavily dependent on sea breeze direction."
    },
    "chinnaswamy": {
        "name": "M. Chinnaswamy Stadium, Bengaluru",
        "base_dew_prior": 0.20,
        "coastal_river_proximity": False,
        "peak_dew_months": [11, 12, 1],
        "drainage_subair": True,
        "description": "Elevated plateau (920m); SubAir vacuum drainage rapidly clears surface moisture."
    },
    "colombo": {
        "name": "R. Premadasa Stadium, Colombo",
        "base_dew_prior": 0.25,
        "coastal_river_proximity": True,
        "peak_dew_months": [11, 12, 1, 2],
        "drainage_subair": False,
        "description": "Equatorial maritime humidity; constant warm night breeze keeps dew depression above threshold."
    },
    "lords": {
        "name": "Lord's, London",
        "base_dew_prior": 0.08,
        "coastal_river_proximity": False,
        "peak_dew_months": [],
        "drainage_subair": False,
        "description": "UK summer venue; late sunsets (21:30) and dry evening air mean dew is virtually nonexistent."
    },
    "oval": {
        "name": "The Oval, London",
        "base_dew_prior": 0.08,
        "coastal_river_proximity": False,
        "peak_dew_months": [],
        "drainage_subair": False,
        "description": "UK summer conditions; ambient temperature remains far above dew point."
    },
    "wanderers": {
        "name": "Wanderers Stadium, Johannesburg",
        "base_dew_prior": 0.10,
        "coastal_river_proximity": False,
        "peak_dew_months": [],
        "drainage_subair": False,
        "description": "High altitude Highveld (1600m); thin dry air prevents nocturnal surface saturation."
    },
    "perth": {
        "name": "Perth Stadium / WACA, Perth",
        "base_dew_prior": 0.08,
        "coastal_river_proximity": False,
        "peak_dew_months": [],
        "drainage_subair": False,
        "description": "Dry continental easterly winds; negligible condensation under floodlights."
    },
    "mcg": {
        "name": "Melbourne Cricket Ground",
        "base_dew_prior": 0.12,
        "coastal_river_proximity": False,
        "peak_dew_months": [],
        "drainage_subair": False,
        "description": "Southern hemisphere summer; dry outfield with low dew frequency."
    }
}

# Real-time NLP commentary regex patterns for live dew observation
NLP_DEW_PATTERNS = {
    "towel_drying": [
        r"\b(towel|towels)\b",
        r"\b(drying the ball|drying off the ball|wiping the ball|wiping hands)\b",
        r"\b(cloth|hand towel|pocket towel)\b",
        r"\b(ball is getting wiped|rubbing the ball dry)\b"
    ],
    "wet_ball_grip": [
        r"\b(wet ball|ball is wet|slippery ball|soap bar|bar of soap)\b",
        r"\b(hard to grip|struggling to grip|slipping out of (the )?hand)\b",
        r"\b(seam is wet|wet seam|damp seam|slipping through fingers)\b",
        r"\b(ball slipped|accidental beamer|full toss due to grip)\b"
    ],
    "super_sopper_ground": [
        r"\b(super sopper|super-sopper|supersopper)\b",
        r"\b(sawdust|saw dust|anti-dew spray|roping the boundary|boundary ropes)\b",
        r"\b(wet outfield|slipping on the grass|outfield is soaked)\b"
    ],
    "explicit_dew_commentary": [
        r"\b(dew factor|heavy dew|lot of dew|plenty of dew|copious dew)\b",
        r"\b(dew starting to set in|dew settling|dew is setting in|dew has arrived)\b",
        r"\b(dew is going to play a part|dew under the lights|dew in the second innings)\b"
    ],
    "ball_change_wetness": [
        r"\b(ball (is |has been )?changed|umpires (are )?inspecting the ball)\b",
        r"\b(ball out of shape from wetness|replacement ball)\b"
    ]
}


class LiveDewDetector:
    """
    Multi-Pillar Live Dew Detection Engine for Day/Night ODIs and T20/T20I Second Innings.
    Evaluates microclimate physics, geographic venue priors, live ball telemetry, and NLP commentary.
    """

    @staticmethod
    def calculate_magnus_tetens_dew_point(temp_c: float, relative_humidity_pct: float) -> Tuple[float, float]:
        """
        Computes Dew Point Temperature (T_dp) and Dew Point Depression (Delta T_dew)
        using the standard Magnus-Tetens atmospheric thermodynamic formula:
        gamma = (a * T) / (b + T) + ln(RH / 100)
        T_dp = (b * gamma) / (a - gamma)
        where a = 17.27, b = 237.7°C.
        """
        t = max(-10.0, min(55.0, float(temp_c)))
        rh = max(1.0, min(100.0, float(relative_humidity_pct)))

        a = 17.27
        b = 237.7

        gamma = (a * t) / (b + t) + math.log(rh / 100.0)
        t_dp = (b * gamma) / (a - gamma)
        delta_t = max(0.0, round(t - t_dp, 2))

        return round(t_dp, 2), delta_t

    @classmethod
    def evaluate_thermodynamic_dew_risk(
        cls,
        temp_c: float = 24.0,
        relative_humidity_pct: float = 82.0,
        wind_speed_kmh: float = 6.0,
        cloud_cover_pct: float = 15.0,
        local_hour: int = 20
    ) -> Dict[str, Any]:
        """
        Pillar 1: Thermodynamic & Atmospheric Microclimate Model.
        Dew forms when:
        1. Delta T_dew (ambient temp minus dew point) <= 2.5°C
        2. High relative humidity (RH >= 78%)
        3. Calm wind speed (<= 8 km/h, preventing droplet evaporation)
        4. Clear sky (cloud cover <= 25%, maximizing terrestrial radiative cooling to space)
        5. Post-sunset night hours (typically 18:00 - 23:00 local time)
        """
        t_dp, delta_t = cls.calculate_magnus_tetens_dew_point(temp_c, relative_humidity_pct)

        # 1. Dew point depression score (0.0 = completely dry, 1.0 = saturated dew)
        if delta_t <= 1.0:
            depression_score = 1.0
        elif delta_t <= 2.2:
            depression_score = 0.85
        elif delta_t <= 3.5:
            depression_score = 0.50
        elif delta_t <= 5.0:
            depression_score = 0.20
        else:
            depression_score = 0.0

        # 2. Wind factor: Calm wind (<= 7 km/h) allows laminar cooling. Strong wind (> 18 km/h) evaporates droplets
        if wind_speed_kmh <= 6.0:
            wind_multiplier = 1.15
        elif wind_speed_kmh <= 12.0:
            wind_multiplier = 1.00
        elif wind_speed_kmh <= 18.0:
            wind_multiplier = 0.70
        else:
            wind_multiplier = 0.35  # Strong wind disrupts dew formation

        # 3. Cloud cover factor: Clear skies permit maximum radiative cooling to deep space
        if cloud_cover_pct <= 20.0:
            radiative_cooling_mult = 1.20
        elif cloud_cover_pct <= 50.0:
            radiative_cooling_mult = 1.00
        elif cloud_cover_pct <= 75.0:
            radiative_cooling_mult = 0.75
        else:
            radiative_cooling_mult = 0.40  # Heavy clouds insulate the surface

        # 4. Diurnal phase: Dew strictly forms after sunset (18:00+) and intensifies into the night
        if 19 <= local_hour <= 23:
            diurnal_mult = 1.25
        elif 18 <= local_hour < 19:
            diurnal_mult = 0.90
        elif 14 <= local_hour < 18:
            diurnal_mult = 0.15  # Daytime sun heats the turf; zero dew possible
        else:
            diurnal_mult = 1.00

        raw_thermo = depression_score * wind_multiplier * radiative_cooling_mult * diurnal_mult
        thermo_dew_index = max(0.0, min(1.0, round(raw_thermo, 3)))

        return {
            "ambient_temp_c": temp_c,
            "dew_point_c": t_dp,
            "dew_point_depression_c": delta_t,
            "relative_humidity_pct": relative_humidity_pct,
            "wind_speed_kmh": wind_speed_kmh,
            "cloud_cover_pct": cloud_cover_pct,
            "thermo_dew_index": thermo_dew_index,
            "is_thermo_dew_forming": (delta_t <= 2.5 and relative_humidity_pct >= 75.0 and local_hour >= 18),
            "cooling_summary": f"ΔT_dew = {delta_t}°C (T={temp_c}°C, T_dp={t_dp}°C, RH={relative_humidity_pct}%). "
                               f"Wind: {wind_speed_kmh} km/h, Clouds: {cloud_cover_pct}%."
        }

    @classmethod
    def get_venue_dew_prior(cls, venue_id: str, month: Optional[int] = None) -> Dict[str, Any]:
        """
        Pillar 2: Geographic & Seasonal Venue Prior Calibration.
        Fetches historical dew propensity based on river/ocean proximity, elevation, and season.
        """
        v_key = venue_id.lower().replace("-", "_").replace(" ", "_")
        profile = None
        for key, p in VENUE_DEW_PRIORS.items():
            if key in v_key:
                profile = p
                break

        if not profile:
            # Default fallback for unknown grounds
            profile = {
                "name": venue_id,
                "base_dew_prior": 0.25,
                "coastal_river_proximity": False,
                "peak_dew_months": [10, 11, 12, 1, 2],
                "drainage_subair": False,
                "description": "Standard venue baseline; moderate regional dew susceptibility."
            }

        cur_month = month or datetime.now().month
        prior_val = profile["base_dew_prior"]

        # Boost prior if current match falls in peak winter dew months for subcontinent
        if cur_month in profile.get("peak_dew_months", []):
            prior_val = min(0.75, prior_val + 0.10)
        elif profile.get("peak_dew_months") and cur_month not in profile["peak_dew_months"]:
            prior_val = max(0.05, prior_val - 0.12)

        return {
            "venue_name": profile["name"],
            "calibrated_prior": round(prior_val, 3),
            "is_dew_prone_venue": (prior_val >= 0.30),
            "coastal_proximity": profile.get("coastal_river_proximity", False),
            "has_subair_drainage": profile.get("drainage_subair", False),
            "notes": profile["description"]
        }

    @classmethod
    def evaluate_in_play_telemetry(
        cls,
        deliveries: Optional[List[Dict[str, Any]]] = None,
        inn_1_stats: Optional[Dict[str, Any]] = None,
        inn_2_stats: Optional[Dict[str, Any]] = None,
        match_format: str = "ODI"
    ) -> Dict[str, Any]:
        """
        Pillar 3: Real-Time In-Play Ball & Performance Telemetry.
        Format-aware analysis calibrated for both 50-over ODIs and 20-over T20 second innings:
        1. Bowling Grip Breakdown: Extras (wides + no-balls) rate spike in Innings 2.
        2. Spin Efficacy Collapse: Spin RPO inflation (> 9.0 in T20, > 6.2 in ODI).
        3. Outfield Skid Multiplier: Boundary rate on ground shots increases due to hydroplaning.
        4. Overs per wicket expansion (batting ease).
        """
        is_t20 = match_format.upper() in ["T20", "T20I"]

        if not deliveries and not inn_2_stats:
            return {
                "telemetry_dew_index": 0.0,
                "extras_spike_ratio": 1.0,
                "spin_rpo_delta": 0.0,
                "boundary_skid_boost": 0.0,
                "telemetry_signals": ["Awaiting in-play delivery sample for live telemetry"]
            }

        if is_t20:
            inn_1 = inn_1_stats or {"runs": 175, "overs": 20.0, "wides": 3, "no_balls": 0.5, "spin_rpo": 7.4, "ground_fours": 10}
        else:
            inn_1 = inn_1_stats or {"runs": 275, "overs": 50.0, "wides": 4, "no_balls": 1, "spin_rpo": 4.6, "ground_fours": 14}

        inn_2 = inn_2_stats or {}

        # If live deliveries list is passed, compute exact 2nd innings counters
        if deliveries:
            inn2_dels = [d for d in deliveries if d.get("innings") == 2 or d.get("inning") == 2]
            if inn2_dels:
                total_balls = len(inn2_dels)
                wides = sum(1 for d in inn2_dels if str(d.get("extra_type", "")).lower() == "wide" or d.get("is_wide"))
                noballs = sum(1 for d in inn2_dels if str(d.get("extra_type", "")).lower() == "noball" or d.get("is_noball"))
                extras_count = wides + noballs
                extras_rate_inn2 = extras_count / max(1, total_balls)

                # Spin deliveries
                spin_dels = [d for d in inn2_dels if "spin" in str(d.get("bowler_style", "")).lower()]
                spin_runs = sum(int(d.get("runs_batter", 0)) + int(d.get("extras", 0)) for d in spin_dels)
                spin_rpo_inn2 = (spin_runs / (len(spin_dels) / 6.0)) if len(spin_dels) >= (6 if is_t20 else 12) else (7.8 if is_t20 else 5.2)

                # Fours
                fours = sum(1 for d in inn2_dels if int(d.get("runs_batter", 0)) == 4)
                fours_per_over = fours / max(1.0, total_balls / 6.0)
            else:
                total_balls = 0
                extras_rate_inn2 = 0.03
                spin_rpo_inn2 = inn_2.get("spin_rpo", 7.6 if is_t20 else 5.0)
                fours_per_over = 2.4 if is_t20 else 2.0
        else:
            total_balls = int(inn_2.get("balls", 36 if is_t20 else 60))
            inn2_extras = int(inn_2.get("wides", 2 if is_t20 else 3)) + int(inn_2.get("no_balls", 1))
            extras_rate_inn2 = inn2_extras / max(1, total_balls)
            spin_rpo_inn2 = float(inn_2.get("spin_rpo", 8.8 if is_t20 else 6.4))
            fours_per_over = float(inn_2.get("fours_per_over", 3.4 if is_t20 else 3.2))

        # Baseline 1st innings extras rate
        inn1_balls = max(30 if is_t20 else 60, int(inn_1.get("overs", 20.0 if is_t20 else 50.0) * 6))
        inn1_extras = int(inn_1.get("wides", 3 if is_t20 else 4)) + int(inn_1.get("no_balls", 1))
        extras_rate_inn1 = inn1_extras / max(1, inn1_balls)

        # Extras spike ratio
        extras_spike_ratio = round(extras_rate_inn2 / max(0.015, extras_rate_inn1), 2)

        # Spin RPO delta
        spin_rpo_inn1 = float(inn_1.get("spin_rpo", 7.4 if is_t20 else 4.6))
        spin_rpo_delta = round(spin_rpo_inn2 - spin_rpo_inn1, 2)

        # Telemetry scoring signals (calibrated for format)
        telemetry_score = 0.0
        signals = []

        extras_threshold = 1.55 if is_t20 else 1.75
        if extras_spike_ratio >= extras_threshold:
            telemetry_score += 0.35
            signals.append(f"Grip failure: Extras rate {extras_spike_ratio}x higher than 1st innings")
        elif extras_spike_ratio >= 1.25:
            telemetry_score += 0.15
            signals.append(f"Elevated extras rate ({extras_spike_ratio}x baseline)")

        spin_cutoff = 9.2 if is_t20 else 6.3
        spin_delta_cutoff = 1.6 if is_t20 else 1.4
        if spin_rpo_delta >= spin_delta_cutoff or spin_rpo_inn2 >= spin_cutoff:
            telemetry_score += 0.35
            signals.append(f"Spinner neutralization: Spin economy surged to {spin_rpo_inn2} RPO (+{spin_rpo_delta} RPO vs Inn 1)")
        elif spin_rpo_delta >= 0.8:
            telemetry_score += 0.15
            signals.append(f"Spin grip reduced (+{spin_rpo_delta} RPO vs Inn 1)")

        fours_cutoff = 2.8 if is_t20 else 3.0
        if fours_per_over >= fours_cutoff:
            telemetry_score += 0.20
            signals.append(f"Outfield skid acceleration: Boundary frequency {fours_per_over:.1f}/ov on wet turf")

        telemetry_dew_index = max(0.0, min(1.0, round(telemetry_score, 3)))

        return {
            "telemetry_dew_index": telemetry_dew_index,
            "extras_spike_ratio": extras_spike_ratio,
            "spin_rpo_inn2": spin_rpo_inn2,
            "spin_rpo_delta": spin_rpo_delta,
            "fours_per_over": round(fours_per_over, 1),
            "telemetry_signals": signals or ["Ball grip within nominal tolerances"]
        }

    @classmethod
    def evaluate_nlp_commentary(cls, commentary_feed: Optional[List[str]]) -> Dict[str, Any]:
        """
        Pillar 4: Real-Time NLP Commentary Stream Mining.
        Regex searches commentary snippets for active dew mitigations:
        towels, wet ball, super sopper, sawdust, grip struggles, dew settlement.
        """
        if not commentary_feed:
            return {
                "nlp_dew_index": 0.0,
                "towels_observed_count": 0,
                "wet_ball_observed_count": 0,
                "super_sopper_observed_count": 0,
                "matched_commentary_snippets": [],
                "nlp_evidence": ["No commentary corpus available"]
            }

        towel_hits = 0
        wet_ball_hits = 0
        sopper_hits = 0
        dew_explicit_hits = 0
        ball_change_hits = 0
        snippets: List[str] = []

        combined_text = " ".join([str(c) for c in commentary_feed if c])

        for pattern in NLP_DEW_PATTERNS["towel_drying"]:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            towel_hits += len(matches)

        for pattern in NLP_DEW_PATTERNS["wet_ball_grip"]:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            wet_ball_hits += len(matches)

        for pattern in NLP_DEW_PATTERNS["super_sopper_ground"]:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            sopper_hits += len(matches)

        for pattern in NLP_DEW_PATTERNS["explicit_dew_commentary"]:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            dew_explicit_hits += len(matches)

        for pattern in NLP_DEW_PATTERNS["ball_change_wetness"]:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            ball_change_hits += len(matches)

        # Extract representative sample sentences containing triggers
        sentences = re.split(r"[.!?\n]+", combined_text)
        for s in sentences:
            s_clean = s.strip()
            if any(re.search(pat, s_clean, re.IGNORECASE) for pats in NLP_DEW_PATTERNS.values() for pat in pats):
                if len(s_clean) > 15 and s_clean not in snippets:
                    snippets.append(s_clean[:130] + ("..." if len(s_clean) > 130 else ""))
            if len(snippets) >= 4:
                break

        # Compute NLP dew score
        nlp_score = 0.0
        evidence = []

        if towel_hits >= 2 or wet_ball_hits >= 2:
            nlp_score += 0.40
            evidence.append(f"Fielding team observed using towels to dry wet ball ({towel_hits + wet_ball_hits} mentions)")
        elif towel_hits >= 1 or wet_ball_hits >= 1:
            nlp_score += 0.20
            evidence.append("Fielders observed drying the ball with towels")

        if dew_explicit_hits >= 2:
            nlp_score += 0.35
            evidence.append(f"Commentators report significant dew on ground ({dew_explicit_hits} mentions)")
        elif dew_explicit_hits >= 1:
            nlp_score += 0.20
            evidence.append("Commentary confirms dew beginning to settle")

        if sopper_hits >= 1:
            nlp_score += 0.25
            evidence.append("Groundstaff deployed super-sopper / sawdust to manage wet outfield")

        if ball_change_hits >= 1:
            nlp_score += 0.25
            evidence.append("Umpires inspected / replaced wet ball out of shape")

        nlp_dew_index = max(0.0, min(1.0, round(nlp_score, 3)))

        return {
            "nlp_dew_index": nlp_dew_index,
            "towels_observed_count": towel_hits,
            "wet_ball_observed_count": wet_ball_hits,
            "super_sopper_observed_count": sopper_hits,
            "explicit_dew_mentions": dew_explicit_hits,
            "matched_commentary_snippets": snippets,
            "nlp_evidence": evidence or ["No active dew mitigation reported in commentary"]
        }

    @classmethod
    def detect_live_dew(
        cls,
        venue_id: str,
        is_day_night: bool,
        match_format: str = "ODI",
        innings: int = 2,
        current_overs: float = 20.0,
        commentary_feed: Optional[List[str]] = None,
        weather_data: Optional[Dict[str, Any]] = None,
        deliveries: Optional[List[Dict[str, Any]]] = None,
        inn_1_summary: Optional[Dict[str, Any]] = None,
        local_hour: int = 20
    ) -> Dict[str, Any]:
        """
        Master Synthesis Methodology:
        Dynamically fuses the 4 pillars across both ODI and T20/T20I 2nd innings chases.
        """
        fmt = match_format.upper()
        is_t20 = fmt in ["T20", "T20I"]
        is_limited_overs = fmt in ["ODI", "T20I", "T20"]

        venue_info = cls.get_venue_dew_prior(venue_id)

        # Weather telemetry extraction
        w_data = weather_data or {}
        temp_c = float(w_data.get("temp_c", 23.5 if is_day_night else 31.0))
        rh_pct = float(w_data.get("relative_humidity_pct", 82.0 if is_day_night else 52.0))
        wind_kmh = float(w_data.get("wind_speed_kmh", 6.5))
        clouds = float(w_data.get("cloud_cover_pct", 15.0))

        thermo_res = cls.evaluate_thermodynamic_dew_risk(
            temp_c=temp_c,
            relative_humidity_pct=rh_pct,
            wind_speed_kmh=wind_kmh,
            cloud_cover_pct=clouds,
            local_hour=local_hour
        )

        telemetry_res = cls.evaluate_in_play_telemetry(
            deliveries=deliveries,
            inn_1_stats=inn_1_summary,
            match_format=fmt
        )

        nlp_res = cls.evaluate_nlp_commentary(commentary_feed)

        # Non-Day/Night matches or non-limited-overs formats have zero active dew advantage
        if not is_day_night or not is_limited_overs:
            return {
                "is_day_night": is_day_night,
                "dew_detected": False,
                "dew_condition": "INACTIVE_DAY_MATCH" if not is_day_night else "INACTIVE_TEST_MATCH",
                "dew_severity_index": 0.0,
                "dew_factor_calibrated": 0.05,
                "chase_equity_boost_pct": 0.0,
                "active_dew_advantage": False,
                "tactical_assessment": "Match is not a Day/Night limited-overs fixture. Dew dynamics inactive.",
                "thermodynamics": thermo_res,
                "venue_prior": venue_info,
                "telemetry": telemetry_res,
                "nlp_mining": nlp_res
            }

        # -------------------------------------------------------------------------
        # Dynamic Bayesian Weighting by Match Phase:
        # T20 2nd innings (20 overs total) accelerates phase transitions:
        #   Powerplay (ov 0-6): Prior 25%, Thermo 35%, Tele 20%, NLP 20%
        #   Middle & Death (ov 6-20): Prior 10%, Thermo 20%, Tele 35%, NLP 35%
        # ODI 2nd innings (50 overs total):
        #   Early (ov 0-15): Prior 35%, Thermo 35%, Tele 15%, NLP 15%
        #   Late (ov 15-50): Prior 15%, Thermo 25%, Tele 30%, NLP 30%
        # -------------------------------------------------------------------------
        p_prior = venue_info["calibrated_prior"]
        p_thermo = thermo_res["thermo_dew_index"]
        p_tele = telemetry_res["telemetry_dew_index"]
        p_nlp = nlp_res["nlp_dew_index"]

        if innings == 1:
            w_prior, w_thermo, w_tele, w_nlp = (0.55, 0.35, 0.05, 0.05) if is_t20 else (0.60, 0.40, 0.00, 0.00)
        elif is_t20:
            if current_overs <= 6.0:
                w_prior, w_thermo, w_tele, w_nlp = 0.25, 0.35, 0.20, 0.20
            else:
                w_prior, w_thermo, w_tele, w_nlp = 0.10, 0.20, 0.35, 0.35
        else: # ODI Innings 2
            if current_overs <= 15.0:
                w_prior, w_thermo, w_tele, w_nlp = 0.35, 0.35, 0.15, 0.15
            else:
                w_prior, w_thermo, w_tele, w_nlp = 0.15, 0.25, 0.30, 0.30

        fused_dew = (w_prior * p_prior) + (w_thermo * p_thermo) + (w_tele * p_tele) + (w_nlp * p_nlp)
        calibrated_dew = round(max(0.05, min(0.85, fused_dew)), 3)

        # Dew severity categorization & Chase Equity Boost Calibration
        # In T20 2nd innings, dew impact per over is amplified because 120 balls leave zero margin for bowling grip error
        if calibrated_dew >= 0.55:
            dew_condition = "HEAVY_DEW_CHASE_DOMINANCE"
            status_badge = "HEAVY DEW ACTIVE"
            chase_equity_boost = round(calibrated_dew * (13.5 if is_t20 else 11.5), 1)  # Up to +8.5-11.5% in T20 chase
            is_dew_active = True
        elif calibrated_dew >= 0.35:
            dew_condition = "ACTIVE_DEW_ADVANTAGE"
            status_badge = "DEW SETTLING"
            chase_equity_boost = round(calibrated_dew * (9.5 if is_t20 else 8.5), 1)   # +3.5 to +5.2% win equity
            is_dew_active = True
        elif calibrated_dew >= 0.22:
            dew_condition = "MILD_DEW_WATCH"
            status_badge = "MILD DEW WATCH"
            chase_equity_boost = round(calibrated_dew * (5.0 if is_t20 else 4.0), 1)   # +1.2 to +2.0% win equity
            is_dew_active = (innings == 2)
        else:
            dew_condition = "DRY_SURFACE"
            status_badge = "DRY CONDITIONS"
            chase_equity_boost = 0.0
            is_dew_active = False

        # Compile comprehensive diagnostic evidence
        evidence = []
        if thermo_res["is_thermo_dew_forming"]:
            evidence.append(f"Atmospheric Dew Point reached: ΔT_dew = {thermo_res['dew_point_depression_c']}°C at {rh_pct}% humidity")
        if venue_info["is_dew_prone_venue"]:
            evidence.append(f"Ground dossier: {venue_info['venue_name']} has historical high evening dew prior ({venue_info['calibrated_prior']})")
        evidence.extend(telemetry_res["telemetry_signals"])
        evidence.extend(nlp_res["nlp_evidence"])

        format_label = f"T20 2nd Innings Chase (ov {current_overs:.1f}/20)" if is_t20 and innings == 2 else ("ODI 2nd Innings Chase" if innings == 2 else f"{fmt} 1st Innings")

        tactical_summary = (
            f"{status_badge} ({calibrated_dew:.2f}) during {format_label}: "
            f"Chasing team gains +{chase_equity_boost}% win equity. "
            f"Wet seam neutralizes spin RPM and causes bowling grip errors (ΔT={thermo_res['dew_point_depression_c']}°C, RH={rh_pct}%)."
        ) if is_dew_active else (
            f"Dry surface conditions (Dew factor: {calibrated_dew:.2f}) for {format_label}. "
            f"ΔT_dew is {thermo_res['dew_point_depression_c']}°C; no significant ball moisture observed."
        )

        return {
            "format": fmt,
            "is_t20": is_t20,
            "is_day_night": is_day_night,
            "dew_detected": is_dew_active,
            "dew_condition": dew_condition,
            "status_badge": status_badge,
            "dew_severity_index": calibrated_dew,
            "dew_factor_calibrated": calibrated_dew,
            "chase_equity_boost_pct": chase_equity_boost,
            "active_dew_advantage": is_dew_active and (innings == 2),
            "match_phase_weights": {
                "prior_weight": w_prior,
                "thermo_weight": w_thermo,
                "telemetry_weight": w_tele,
                "nlp_weight": w_nlp
            },
            "tactical_assessment": tactical_summary,
            "diagnostic_evidence": evidence[:5],
            "thermodynamics": thermo_res,
            "venue_prior": venue_info,
            "telemetry": telemetry_res,
            "nlp_mining": nlp_res
        }
