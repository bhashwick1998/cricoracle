"""
CricOracle Historical Pitch Intelligence Engine
Models international venue pitch friction, historical pace vs spin ratios,
phase-by-phase scoring rates, and chase/dew bias derived from international match history.
"""

from typing import Dict, Any, Tuple

# Comprehensive historical venue database from international T20I & ODI matches
HISTORICAL_VENUE_DOSSIERS: Dict[str, Dict[str, Any]] = {
    "wellington": {
        "id": "wellington",
        "name": "Sky Stadium (Westpac)",
        "city": "Wellington",
        "country": "New Zealand",
        "region_cluster": "sena",
        "pitch_type": "pace_friendly",
        "pitch_description": "True maritime seam & bounce with coastal crosswinds aiding early lateral swing.",
        "sample_matches": 38,
        "spin_wicket_ratio": 0.28,
        "pace_wicket_ratio": 0.72,
        "pace_avg_economy": 7.42,
        "spin_avg_economy": 8.15,
        "avg_pp_score": 46.8,
        "avg_mid_score": 74.2,
        "avg_death_score": 45.0,
        "avg_first_innings_score_t20": 166,
        "avg_first_innings_score_odi": 285,
        "chase_win_pct": 0.526,
        "dew_factor": 0.20,
        "boundary_avg_m": 72.0,
        "favored_archetypes": ["RF", "RMF"],
        "troubled_archetypes": ["SLA", "OB"]
    },
    "seddon_park": {
        "id": "seddon_park",
        "name": "Seddon Park",
        "city": "Hamilton",
        "country": "New Zealand",
        "region_cluster": "sena",
        "pitch_type": "balanced",
        "pitch_description": "Firm drop-in wicket with even bounce; slows slightly in middle overs before accelerating.",
        "sample_matches": 32,
        "spin_wicket_ratio": 0.35,
        "pace_wicket_ratio": 0.65,
        "pace_avg_economy": 7.85,
        "spin_avg_economy": 7.60,
        "avg_pp_score": 49.5,
        "avg_mid_score": 76.8,
        "avg_death_score": 46.2,
        "avg_first_innings_score_t20": 172,
        "avg_first_innings_score_odi": 292,
        "chase_win_pct": 0.531,
        "dew_factor": 0.25,
        "boundary_avg_m": 68.0,
        "favored_archetypes": ["RMF", "LBG"],
        "troubled_archetypes": ["OB"]
    },
    "eden_park": {
        "id": "eden_park",
        "name": "Eden Park",
        "city": "Auckland",
        "country": "New Zealand",
        "region_cluster": "sena",
        "pitch_type": "batting_paradise",
        "pitch_description": "Unusually short straight boundaries (65m) and fast outfield produce extreme boundary density.",
        "sample_matches": 44,
        "spin_wicket_ratio": 0.30,
        "pace_wicket_ratio": 0.70,
        "pace_avg_economy": 8.65,
        "spin_avg_economy": 8.95,
        "avg_pp_score": 53.2,
        "avg_mid_score": 82.0,
        "avg_death_score": 49.5,
        "avg_first_innings_score_t20": 184,
        "avg_first_innings_score_odi": 305,
        "chase_win_pct": 0.523,
        "dew_factor": 0.25,
        "boundary_avg_m": 65.0,
        "favored_archetypes": ["RF"],
        "troubled_archetypes": ["SLA", "OB"]
    },
    "sharjah": {
        "id": "sharjah",
        "name": "Sharjah Cricket Stadium",
        "city": "Sharjah",
        "country": "UAE",
        "region_cluster": "subcontinent",
        "pitch_type": "spin_friendly",
        "pitch_description": "Dry low-bounce clay deck offering sharp spin friction and low natural carry for pace.",
        "sample_matches": 62,
        "spin_wicket_ratio": 0.58,
        "pace_wicket_ratio": 0.42,
        "pace_avg_economy": 8.45,
        "spin_avg_economy": 6.82,
        "avg_pp_score": 44.5,
        "avg_mid_score": 68.2,
        "avg_death_score": 41.5,
        "avg_first_innings_score_t20": 154,
        "avg_first_innings_score_odi": 250,
        "chase_win_pct": 0.565,
        "dew_factor": 0.45,
        "boundary_avg_m": 64.0,
        "favored_archetypes": ["LBG", "SLA", "LAWS"],
        "troubled_archetypes": ["RF", "RMF"]
    },
    "dubai_dsc": {
        "id": "dubai_dsc",
        "name": "Dubai International Stadium",
        "city": "Dubai",
        "country": "UAE",
        "region_cluster": "subcontinent",
        "pitch_type": "spin_friendly",
        "pitch_description": "Ring of Fire stadium; heavy evening dew causes major second-innings chase advantage.",
        "sample_matches": 78,
        "spin_wicket_ratio": 0.62,
        "pace_wicket_ratio": 0.38,
        "pace_avg_economy": 8.20,
        "spin_avg_economy": 6.75,
        "avg_pp_score": 45.0,
        "avg_mid_score": 71.0,
        "avg_death_score": 42.0,
        "avg_first_innings_score_t20": 158,
        "avg_first_innings_score_odi": 255,
        "chase_win_pct": 0.628,
        "dew_factor": 0.60,
        "boundary_avg_m": 68.0,
        "favored_archetypes": ["SLA", "LBG", "LAWS"],
        "troubled_archetypes": ["RMF"]
    },
    "mcg": {
        "id": "mcg",
        "name": "Melbourne Cricket Ground",
        "city": "Melbourne",
        "country": "Australia",
        "region_cluster": "sena",
        "pitch_type": "pace_friendly",
        "pitch_description": "Massive ground dimensions (82m sq) reward fast running, tall seamers, and deep boundary catches.",
        "sample_matches": 55,
        "spin_wicket_ratio": 0.32,
        "pace_wicket_ratio": 0.68,
        "pace_avg_economy": 7.30,
        "spin_avg_economy": 7.90,
        "avg_pp_score": 46.0,
        "avg_mid_score": 73.5,
        "avg_death_score": 44.5,
        "avg_first_innings_score_t20": 164,
        "avg_first_innings_score_odi": 285,
        "chase_win_pct": 0.545,
        "dew_factor": 0.30,
        "boundary_avg_m": 82.0,
        "favored_archetypes": ["RF", "RMF"],
        "troubled_archetypes": ["OB"]
    },
    "scg": {
        "id": "scg",
        "name": "Sydney Cricket Ground",
        "city": "Sydney",
        "country": "Australia",
        "region_cluster": "sena",
        "pitch_type": "spin_friendly",
        "pitch_description": "Traditional Australian spin surface with bulli soil providing grip and turn for wrist spinners.",
        "sample_matches": 48,
        "spin_wicket_ratio": 0.48,
        "pace_wicket_ratio": 0.52,
        "pace_avg_economy": 7.95,
        "spin_avg_economy": 7.20,
        "avg_pp_score": 47.5,
        "avg_mid_score": 74.0,
        "avg_death_score": 45.0,
        "avg_first_innings_score_t20": 166,
        "avg_first_innings_score_odi": 280,
        "chase_win_pct": 0.500,
        "dew_factor": 0.35,
        "boundary_avg_m": 76.0,
        "favored_archetypes": ["LBG", "SLA", "RF"],
        "troubled_archetypes": ["RMF"]
    },
    "adelaide": {
        "id": "adelaide",
        "name": "Adelaide Oval",
        "city": "Adelaide",
        "country": "Australia",
        "region_cluster": "sena",
        "pitch_type": "balanced",
        "pitch_description": "Short square boundaries and long straight boundaries; excellent batting surface with consistent carry.",
        "sample_matches": 40,
        "spin_wicket_ratio": 0.38,
        "pace_wicket_ratio": 0.62,
        "pace_avg_economy": 7.80,
        "spin_avg_economy": 7.75,
        "avg_pp_score": 48.0,
        "avg_mid_score": 75.0,
        "avg_death_score": 45.5,
        "avg_first_innings_score_t20": 168,
        "avg_first_innings_score_odi": 288,
        "chase_win_pct": 0.512,
        "dew_factor": 0.30,
        "boundary_avg_m": 75.0,
        "favored_archetypes": ["RF", "LBG"],
        "troubled_archetypes": ["SLA"]
    },
    "perth": {
        "id": "perth",
        "name": "Perth Stadium (Optus Stadium)",
        "city": "Perth",
        "country": "Australia",
        "region_cluster": "sena",
        "pitch_type": "bounce",
        "pitch_description": "Hard-packed Western Australian deck offering steep vertical bounce, rapid pace carry, and true seam movement.",
        "sample_matches": 36,
        "spin_wicket_ratio": 0.26,
        "pace_wicket_ratio": 0.74,
        "pace_avg_economy": 7.35,
        "spin_avg_economy": 8.05,
        "avg_pp_score": 45.2,
        "avg_mid_score": 72.8,
        "avg_death_score": 44.0,
        "avg_first_innings_score_t20": 165,
        "avg_first_innings_score_odi": 268,
        "avg_first_innings_score_test": 310,
        "chase_win_pct": 0.480,
        "dew_factor": 0.12,
        "boundary_avg_m": 72.0,
        "favored_archetypes": ["RF", "LF"],
        "troubled_archetypes": ["SLA", "OB"]
    },
    "hobart": {
        "id": "hobart",
        "name": "Bellerive Oval",
        "city": "Hobart",
        "country": "Australia",
        "region_cluster": "sena",
        "pitch_type": "bouncy",
        "pitch_description": "True bounce and carry with breezy Derwent River coastal conditions aiding swing; fast outfield rewarding strokeplay.",
        "sample_matches": 38,
        "spin_wicket_ratio": 0.35,
        "pace_wicket_ratio": 0.65,
        "pace_avg_economy": 7.45,
        "spin_avg_economy": 7.60,
        "avg_pp_score": 48.0,
        "avg_mid_score": 75.0,
        "avg_death_score": 46.0,
        "avg_first_innings_score_t20": 172,
        "avg_first_innings_score_odi": 275,
        "avg_first_innings_score_test": 320,
        "chase_win_pct": 0.520,
        "dew_factor": 0.20,
        "boundary_avg_m": 70.0,
        "favored_archetypes": ["RF", "RMF", "LF"],
        "troubled_archetypes": ["OB", "SLA"]
    },
    "bellerive_oval": {
        "id": "hobart",
        "name": "Bellerive Oval",
        "city": "Hobart",
        "country": "Australia",
        "region_cluster": "sena",
        "pitch_type": "bouncy",
        "pitch_description": "True bounce and carry with breezy Derwent River coastal conditions aiding swing; fast outfield rewarding strokeplay.",
        "sample_matches": 38,
        "spin_wicket_ratio": 0.35,
        "pace_wicket_ratio": 0.65,
        "pace_avg_economy": 7.45,
        "spin_avg_economy": 7.60,
        "avg_pp_score": 48.0,
        "avg_mid_score": 75.0,
        "avg_death_score": 46.0,
        "avg_first_innings_score_t20": 172,
        "avg_first_innings_score_odi": 275,
        "avg_first_innings_score_test": 320,
        "chase_win_pct": 0.520,
        "dew_factor": 0.20,
        "boundary_avg_m": 70.0,
        "favored_archetypes": ["RF", "RMF", "LF"],
        "troubled_archetypes": ["OB", "SLA"]
    },
    "centurion": {
        "id": "centurion",
        "name": "SuperSport Park",
        "city": "Centurion",
        "country": "South Africa",
        "region_cluster": "sena",
        "pitch_type": "pace_friendly",
        "pitch_description": "High altitude (1400m above sea level) reduces air drag; express pacers generate wicked bounce.",
        "sample_matches": 36,
        "spin_wicket_ratio": 0.24,
        "pace_wicket_ratio": 0.76,
        "pace_avg_economy": 7.55,
        "spin_avg_economy": 8.60,
        "avg_pp_score": 51.0,
        "avg_mid_score": 80.5,
        "avg_death_score": 48.5,
        "avg_first_innings_score_t20": 180,
        "avg_first_innings_score_odi": 295,
        "chase_win_pct": 0.543,
        "dew_factor": 0.20,
        "boundary_avg_m": 75.0,
        "favored_archetypes": ["RF"],
        "troubled_archetypes": ["OB", "SLA"]
    },
    "karachi": {
        "id": "karachi",
        "name": "National Stadium",
        "city": "Karachi",
        "country": "Pakistan",
        "region_cluster": "subcontinent",
        "pitch_type": "batting_paradise",
        "pitch_description": "True bounce coastal flat pitch with low seam movement; favors fearless powerplay strokeplay.",
        "sample_matches": 42,
        "spin_wicket_ratio": 0.44,
        "pace_wicket_ratio": 0.56,
        "pace_avg_economy": 8.50,
        "spin_avg_economy": 8.10,
        "avg_pp_score": 52.0,
        "avg_mid_score": 81.0,
        "avg_death_score": 49.0,
        "avg_first_innings_score_t20": 182,
        "avg_first_innings_score_odi": 290,
        "chase_win_pct": 0.552,
        "dew_factor": 0.50,
        "boundary_avg_m": 70.0,
        "favored_archetypes": ["RF", "LBG"],
        "troubled_archetypes": ["RMF"]
    },
    "rawalpindi": {
        "id": "rawalpindi",
        "name": "Rawalpindi Cricket Stadium",
        "city": "Rawalpindi",
        "country": "Pakistan",
        "region_cluster": "subcontinent",
        "pitch_type": "batting_paradise",
        "pitch_description": "True bounce flat track offering high run scoring early; reverse swing and spinner wear decisive late.",
        "sample_matches": 36,
        "spin_wicket_ratio": 0.38,
        "pace_wicket_ratio": 0.62,
        "pace_avg_economy": 8.35,
        "spin_avg_economy": 7.95,
        "avg_pp_score": 53.5,
        "avg_mid_score": 83.0,
        "avg_death_score": 50.5,
        "avg_first_innings_score_t20": 185,
        "avg_first_innings_score_odi": 295,
        "chase_win_pct": 0.540,
        "dew_factor": 0.40,
        "boundary_avg_m": 71.5,
        "favored_archetypes": ["RF", "OB"],
        "troubled_archetypes": ["SLA"]
    },
    "lahore": {
        "id": "lahore",
        "name": "Gaddafi Stadium",
        "city": "Lahore",
        "country": "Pakistan",
        "region_cluster": "subcontinent",
        "pitch_type": "batting_paradise",
        "pitch_description": "Red soil wicket with true bounce favoring strokeplay for the first half before abrasive spin takes over.",
        "sample_matches": 45,
        "spin_wicket_ratio": 0.44,
        "pace_wicket_ratio": 0.56,
        "pace_avg_economy": 8.40,
        "spin_avg_economy": 8.05,
        "avg_pp_score": 52.0,
        "avg_mid_score": 82.0,
        "avg_death_score": 49.5,
        "avg_first_innings_score_t20": 180,
        "avg_first_innings_score_odi": 290,
        "chase_win_pct": 0.535,
        "dew_factor": 0.50,
        "boundary_avg_m": 71.0,
        "favored_archetypes": ["RF", "LBG", "OB"],
        "troubled_archetypes": ["RMF"]
    },
    "mirpur": {
        "id": "mirpur",
        "name": "Sher-e-Bangla National Stadium",
        "city": "Mirpur",
        "country": "Bangladesh",
        "region_cluster": "subcontinent",
        "pitch_type": "spin_friendly",
        "pitch_description": "Heavy black clay surface with sluggish pace and dramatic spin; lowest scoring venue among full members.",
        "sample_matches": 68,
        "spin_wicket_ratio": 0.65,
        "pace_wicket_ratio": 0.35,
        "pace_avg_economy": 7.90,
        "spin_avg_economy": 6.35,
        "avg_pp_score": 41.0,
        "avg_mid_score": 66.0,
        "avg_death_score": 40.5,
        "avg_first_innings_score_t20": 148,
        "avg_first_innings_score_odi": 240,
        "chase_win_pct": 0.521,
        "dew_factor": 0.40,
        "boundary_avg_m": 68.0,
        "favored_archetypes": ["SLA", "OB", "LAWS"],
        "troubled_archetypes": ["RF"]
    },
    "wankhede": {
        "id": "wankhede",
        "name": "Wankhede Stadium",
        "city": "Mumbai",
        "country": "India",
        "region_cluster": "subcontinent",
        "pitch_type": "batting_paradise",
        "pitch_description": "Red soil surface provides electric pace and bounce; evening Arabian Sea humidity creates massive dew.",
        "sample_matches": 52,
        "spin_wicket_ratio": 0.42,
        "pace_wicket_ratio": 0.58,
        "pace_avg_economy": 8.75,
        "spin_avg_economy": 8.60,
        "avg_pp_score": 54.0,
        "avg_mid_score": 85.0,
        "avg_death_score": 53.0,
        "avg_first_innings_score_t20": 192,
        "avg_first_innings_score_odi": 310,
        "chase_win_pct": 0.640,
        "dew_factor": 0.65,
        "boundary_avg_m": 68.0,
        "favored_archetypes": ["RF", "LBG"],
        "troubled_archetypes": ["SLA", "OB"]
    },
    "eden_gardens": {
        "id": "eden_gardens",
        "name": "Eden Gardens",
        "city": "Kolkata",
        "country": "India",
        "region_cluster": "subcontinent",
        "pitch_type": "spin_friendly",
        "pitch_description": "Traditional historic clay surface; good carry under lights with significant grip for mystery spinners.",
        "sample_matches": 46,
        "spin_wicket_ratio": 0.52,
        "pace_wicket_ratio": 0.48,
        "pace_avg_economy": 8.10,
        "spin_avg_economy": 7.30,
        "avg_pp_score": 49.0,
        "avg_mid_score": 77.0,
        "avg_death_score": 50.0,
        "avg_first_innings_score_t20": 176,
        "avg_first_innings_score_odi": 280,
        "chase_win_pct": 0.587,
        "dew_factor": 0.55,
        "boundary_avg_m": 71.0,
        "favored_archetypes": ["LBG", "LAWS", "RF"],
        "troubled_archetypes": ["RMF"]
    },
    "sylhet": {
        "id": "sylhet",
        "name": "Sylhet International Cricket Stadium",
        "city": "Sylhet",
        "country": "Bangladesh",
        "region_cluster": "subcontinent",
        "pitch_type": "spin_friendly",
        "pitch_description": "Picturesque green amphitheatre; early seam movement gives way to sharp, low-bounce spin from day 1 afternoon.",
        "sample_matches": 35,
        "spin_wicket_ratio": 0.62,
        "pace_wicket_ratio": 0.38,
        "pace_avg_economy": 7.45,
        "spin_avg_economy": 6.20,
        "avg_pp_score": 43.5,
        "avg_mid_score": 68.0,
        "avg_death_score": 42.0,
        "avg_first_innings_score_t20": 154,
        "avg_first_innings_score_odi": 248,
        "chase_win_pct": 0.514,
        "dew_factor": 0.30,
        "boundary_avg_m": 70.0,
        "favored_archetypes": ["SLA", "OB", "RF"],
        "troubled_archetypes": ["RMF"]
    }
}

# Format-stratified historical bowling economies (TEST, ODI, T20I)
# Derived from international match history across each venue
VENUE_BOWLING_ECONOMIES: Dict[str, Dict[str, Tuple[float, float]]] = {
    # format: (pace_avg_economy, spin_avg_economy)
    "sharjah": {
        "TEST": (2.92, 2.64),
        "ODI": (5.42, 4.88),
        "T20I": (8.45, 6.82),
    },
    "sylhet": {
        "TEST": (3.12, 2.76),
        "ODI": (5.35, 4.95),
        "T20I": (7.95, 7.20),
    },
    "wellington": {
        "TEST": (2.98, 3.15),
        "ODI": (5.25, 5.65),
        "T20I": (7.42, 8.15),
    },
    "seddon_park": {
        "TEST": (3.02, 3.08),
        "ODI": (5.38, 5.42),
        "T20I": (7.85, 7.60),
    },
    "bay_oval": {
        "TEST": (3.05, 3.10),
        "ODI": (5.30, 5.40),
        "T20I": (7.80, 7.70),
    },
    "eden_park": {
        "TEST": (3.25, 3.35),
        "ODI": (5.75, 5.80),
        "T20I": (8.65, 8.95),
    },
    "mcg": {
        "TEST": (2.90, 3.05),
        "ODI": (5.15, 5.10),
        "T20I": (7.60, 7.40),
    },
    "scg": {
        "TEST": (3.05, 2.85),
        "ODI": (5.40, 5.05),
        "T20I": (8.10, 7.30),
    },
    "adelaide": {
        "TEST": (2.95, 2.90),
        "ODI": (5.25, 5.15),
        "T20I": (7.85, 7.50),
    },
    "hobart": {
        "TEST": (2.95, 3.10),
        "ODI": (5.25, 5.30),
        "T20I": (7.65, 7.55),
    },
    "bellerive_oval": {
        "TEST": (2.95, 3.10),
        "ODI": (5.25, 5.30),
        "T20I": (7.65, 7.55),
    },
    "perth": {
        "TEST": (2.95, 3.15),
        "ODI": (5.20, 5.30),
        "T20I": (7.35, 8.05),
    },
    "wankhede": {
        "TEST": (3.20, 2.95),
        "ODI": (5.65, 5.25),
        "T20I": (8.55, 7.90),
    },
    "eden_gardens": {
        "TEST": (3.10, 2.80),
        "ODI": (5.35, 4.95),
        "T20I": (8.20, 7.45),
    },
    "karachi": {
        "TEST": (3.25, 3.10),
        "ODI": (5.50, 5.20),
        "T20I": (8.40, 7.85),
    },
    "rawalpindi": {
        "TEST": (3.35, 3.20),
        "ODI": (5.60, 5.30),
        "T20I": (8.55, 7.95),
    },
    "lahore": {
        "TEST": (3.30, 3.15),
        "ODI": (5.55, 5.25),
        "T20I": (8.45, 7.90),
    },
    "centurion": {
        "TEST": (3.15, 3.30),
        "ODI": (5.55, 5.60),
        "T20I": (8.50, 8.20),
    },
    "dubai_dsc": {
        "TEST": (2.95, 2.70),
        "ODI": (5.30, 4.80),
        "T20I": (8.20, 6.75),
    },
    "mirpur": {
        "TEST": (3.08, 2.68),
        "ODI": (5.20, 4.75),
        "T20I": (7.70, 6.95),
    },
    "premadasa": {
        "TEST": (3.10, 2.65),
        "ODI": (5.25, 4.80),
        "T20I": (7.80, 6.90),
    },
    "lords": {
        "TEST": (2.92, 3.12),
        "ODI": (5.20, 5.35),
        "T20I": (7.90, 7.80),
    },
    "the_oval": {
        "TEST": (3.02, 2.95),
        "ODI": (5.45, 5.20),
        "T20I": (8.30, 7.60),
    }
}

def get_venue_bowling_economy(venue_id: str, match_format: str = "T20I") -> Tuple[float, float]:
    """
    Returns authentic format-stratified bowling economy rates: (pace_avg_economy, spin_avg_economy).
    Pace and spin economies vary dramatically by format:
    - TEST: Pace 2.85 - 3.25 RPO, Spin 2.50 - 2.95 RPO
    - ODI: Pace 5.10 - 5.65 RPO, Spin 4.70 - 5.20 RPO
    - T20I: Pace 7.60 - 8.65 RPO, Spin 6.80 - 7.60 RPO
    """
    fmt = (match_format or "T20I").upper()
    if "TEST" in fmt:
        target_fmt = "TEST"
    elif "ODI" in fmt or "50" in fmt:
        target_fmt = "ODI"
    else:
        target_fmt = "T20I"

    vid = (venue_id or "").lower().replace("-", "_").replace(" ", "_")
    matched = None
    if vid in VENUE_BOWLING_ECONOMIES:
        matched = VENUE_BOWLING_ECONOMIES[vid]
    else:
        for k, v in VENUE_BOWLING_ECONOMIES.items():
            if k in vid or vid in k:
                matched = v
                break

    if matched and target_fmt in matched:
        return matched[target_fmt]

    # Global standard defaults by format
    if target_fmt == "TEST":
        return (2.98, 2.75)
    elif target_fmt == "ODI":
        return (5.35, 5.05)
    else:
        return (8.15, 7.50)

def get_historical_venue_profile(venue_id: str, match_format: str = "T20I") -> Dict[str, Any]:
    """
    Retrieves the complete historical pitch and venue profile format-stratified.
    Falls back gracefully if venue_id is unfamiliar.
    """
    vid = venue_id.lower().replace("-", "_").replace(" ", "_")
    pace_econ, spin_econ = get_venue_bowling_economy(vid, match_format)

    if vid in HISTORICAL_VENUE_DOSSIERS:
        profile = dict(HISTORICAL_VENUE_DOSSIERS[vid])
        profile["pace_avg_economy"] = pace_econ
        profile["spin_avg_economy"] = spin_econ
        profile["bowling_economy_format"] = match_format
        if "avg_first_innings_score_test" not in profile:
            profile["avg_first_innings_score_test"] = int(profile.get("avg_first_innings_score_odi", 260) * 1.38)
        return profile

    # Partial match
    for k, v in HISTORICAL_VENUE_DOSSIERS.items():
        if k in vid or vid in k:
            profile = dict(v)
            profile["pace_avg_economy"] = pace_econ
            profile["spin_avg_economy"] = spin_econ
            profile["bowling_economy_format"] = match_format
            if "avg_first_innings_score_test" not in profile:
                profile["avg_first_innings_score_test"] = int(profile.get("avg_first_innings_score_odi", 260) * 1.38)
            return profile

    # Fallback balanced international standard
    return {
        "id": venue_id,
        "name": venue_id.title(),
        "city": "International Ground",
        "country": "Neutral",
        "region_cluster": "sena",
        "pitch_type": "balanced",
        "pitch_description": "Standard international balanced surface.",
        "sample_matches": 25,
        "spin_wicket_ratio": 0.40,
        "pace_wicket_ratio": 0.60,
        "pace_avg_economy": pace_econ,
        "spin_avg_economy": spin_econ,
        "bowling_economy_format": match_format,
        "avg_pp_score": 48.0,
        "avg_mid_score": 74.0,
        "avg_death_score": 45.0,
        "avg_first_innings_score_t20": 167,
        "avg_first_innings_score_odi": 280,
        "chase_win_pct": 0.520,
        "dew_factor": 0.35,
        "boundary_avg_m": 72.0,
        "favored_archetypes": ["RF", "LBG"],
        "troubled_archetypes": []
    }

def compute_venue_pitch_condition_delta(
    venue_profile: Dict[str, Any],
    team_a_spin_index: float,
    team_a_pace_index: float,
    team_b_spin_index: float,
    team_b_pace_index: float
) -> Tuple[float, str]:
    """
    Computes exact condition differential from historical spin/pace wicket ratios:
    Delta_Pitch = (TeamA_Spin - TeamB_Spin) * SpinRatio + (TeamA_Pace - TeamB_Pace) * PaceRatio
    Returns the differential score and human-readable diagnostic.
    """
    spin_w = venue_profile.get("spin_wicket_ratio", 0.40)
    pace_w = venue_profile.get("pace_wicket_ratio", 0.60)

    spin_diff = (team_a_spin_index - team_b_spin_index) * spin_w
    pace_diff = (team_a_pace_index - team_b_pace_index) * pace_w

    delta_pitch = round(spin_diff + pace_diff, 2)

    if spin_w >= 0.52:
        reason = f"High historical spin ratio ({int(spin_w*100)}%) creates decisive turning friction"
    elif pace_w >= 0.68:
        reason = f"Pace-dominant pitch ({int(pace_w*100)}% pace wickets) heavily rewards seam bounce"
    else:
        reason = f"Even pace/spin balance ({int(pace_w*100)}% pace / {int(spin_w*100)}% spin) tests all phases"

    return delta_pitch, reason
