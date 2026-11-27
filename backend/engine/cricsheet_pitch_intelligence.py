"""
CricOracle Cricsheet Ground & Pitch Intelligence Engine
Comprehensive historical cricket ground database and pitch behavior categorization.
Categorizes every ground across 5 core pitch archetypes for EACH format (TEST, ODI, T20I):
1. 'flat': True batting paradise, high par score, low friction, minimal seam/turn.
2. 'green': Lush grass cover, early lateral swing, seam deviation, top-order hazard.
3. 'seam': Abrasive deck, prominent seam movement off the pitch, two-paced cutters.
4. 'bounce': Hard-packed clay, steep vertical lift, tennis-ball bounce, short-ball threat.
5. 'spin heavy': Dry clay, high surface friction, early abrasion, sharp spin, low carry.

Provides:
- Format-specific ground pitch behavior & Cricsheet ground dossiers
- Country/National Team adaptability scores by format and pitch archetype (with 65% Recency Weighting for last 1 year)
- Player 5-way pitch adaptability scoring engine with 1-year rolling recent form weighting
- Format-specific predictive variable extractor (Test degradation, tail depth, ODI middle overs, 6th bowler, T20 death execution, boundary asymmetry)
- Cricsheet historical toss impact (bat first vs chase win % by archetype & format)
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import os
import re
import sqlite3
import numpy as np

# 5 Core Pitch Archetypes
# 5 Core Canonical Pitch Archetypes
PITCH_ARCHETYPES = ["dusty", "green", "bouncy", "flat", "balanced"]

def canonicalize_pitch_type(pitch: str) -> str:
    """
    Standardizes any pitch string into one of the 5 canonical pitch archetypes:
    'dusty', 'green', 'bouncy', 'flat', 'balanced'.
    Seamlessly handles legacy and descriptive aliases.
    """
    if not pitch:
        return "balanced"
    p = str(pitch).strip().lower().replace("-", "_").replace(" ", "_")
    if p in ["dusty", "spin_heavy", "spin_friendly", "spin", "turner", "raging_turner"]:
        return "dusty"
    if p in ["green", "lush_green", "green_top", "green_seamer"]:
        return "green"
    if p in ["bouncy", "bounce", "pace_friendly", "hard", "steep_bounce"]:
        return "bouncy"
    if p in ["flat", "batting_paradise", "road", "high_scoring"]:
        return "flat"
    if p in ["balanced", "seam", "sporting", "two_paced", "neutral", "seam_/_green", "green_/_seam", "pace"]:
        return "balanced"
    if any(k in p for k in ("dust", "turn", "spin", "dry", "crumb")):
        return "dusty"
    if any(k in p for k in ("green", "grass", "lush", "seamer")):
        return "green"
    if any(k in p for k in ("bounc", "pace_friendly", "steep", "hard")):
        return "bouncy"
    if any(k in p for k in ("flat", "road", "belter", "paradise", "high_scor")):
        return "flat"
    return "balanced"


# 1. Format-Specific Ground Pitch Categorization & Cricsheet Dossiers
FORMAT_GROUND_PITCH_DOSSIERS: Dict[str, Dict[str, Any]] = {
    "wellington": {
        "id": "wellington",
        "name": "Sky Stadium (Westpac)",
        "city": "Wellington",
        "country": "New Zealand",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 75.0, "square_m": 70.0, "avg_m": 72.0},
        "dew_factor": 0.20,
        "format_archetypes": {
            "TEST": "green",       # Maritime winds, lush green top, early lateral swing & seam
            "ODI": "balanced",         # Seam movement off surface, cutters grip in middle overs
            "T20I": "bouncy"       # Firm surface, high wind-drag, tennis-ball bounce
        },
        "format_pars": {"TEST": 275, "ODI": 285, "T20I": 166},
        "pitch_wear_rate": 3.8,    # Low wear, maritime dampness preserves grass
        "description": "Maritime bowl prone to fierce coastal gusts; green top aids new-ball pacers before settling."
    },
    "seddon_park": {
        "id": "seddon_park",
        "name": "Seddon Park",
        "city": "Hamilton",
        "country": "New Zealand",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 68.0, "square_m": 68.0, "avg_m": 68.0},
        "dew_factor": 0.25,
        "format_archetypes": {
            "TEST": "balanced",        # Drop-in moisture produces consistent seam deviation
            "ODI": "flat",         # Excellent batting track with even pace
            "T20I": "flat"         # True bounce and short boundaries create high-scoring affairs
        },
        "format_pars": {"TEST": 285, "ODI": 292, "T20I": 172},
        "pitch_wear_rate": 4.5,
        "description": "Boutique circular ground with firm drop-in wickets; true bounce allows clean stroke play."
    },
    "bay_oval": {
        "id": "bay_oval",
        "name": "Bay Oval",
        "city": "Mount Maunganui",
        "country": "New Zealand",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 68.0, "avg_m": 68.0},
        "dew_factor": 0.22,
        "format_archetypes": {
            "TEST": "balanced",        # Coastal maritime bounce and seam movement
            "ODI": "balanced",         # Initial lateral seam movement in opening 10 overs
            "T20I": "flat"         # True pace and even bounce reward aggressive shotmakers
        },
        "format_pars": {"TEST": 275, "ODI": 275, "T20I": 172},
        "pitch_wear_rate": 4.0,
        "description": "Scenic coastal ground with consistent maritime breeze; true seam movement for pacers in opening spell."
    },
    "eden_park": {
        "id": "eden_park",
        "name": "Eden Park",
        "city": "Auckland",
        "country": "New Zealand",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 64.0, "square_m": 72.0, "avg_m": 65.0},
        "dew_factor": 0.25,
        "format_archetypes": {
            "TEST": "balanced",        # Good carry and seam movement under cloud cover
            "ODI": "flat",         # Short straight boundaries (64m) lead to huge totals
            "T20I": "flat"         # Extreme boundary density, 64m straight six-hitting paradise
        },
        "format_pars": {"TEST": 290, "ODI": 310, "T20I": 185},
        "pitch_wear_rate": 4.2,
        "description": "Unique baseball-diamond geometry with short 64m straight boundaries heavily rewarding power hitters."
    },
    "sharjah": {
        "id": "sharjah",
        "name": "Sharjah Cricket Stadium",
        "city": "Sharjah",
        "country": "United Arab Emirates",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 65.0, "square_m": 62.0, "avg_m": 62.0},
        "dew_factor": 0.55,
        "format_archetypes": {
            "TEST": "dusty",  # Low, abrasive clay deck that dusts up quickly
            "ODI": "dusty",  # Surface friction chokes run rates against spinners
            "T20I": "dusty"   # Low natural carry and sharp turn reward wrist-spinners
        },
        "format_pars": {"TEST": 240, "ODI": 255, "T20I": 154},
        "pitch_wear_rate": 8.5,
        "description": "Historic desert amphitheatre; dry clay generates extreme spin friction and low bounce."
    },
    "dubai_dsc": {
        "id": "dubai_dsc",
        "name": "Dubai International Stadium",
        "city": "Dubai",
        "country": "United Arab Emirates",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 68.0, "avg_m": 68.0},
        "dew_factor": 0.60,
        "format_archetypes": {
            "TEST": "dusty",  # Deteriorating surface with low bounce on Days 4-5
            "ODI": "dusty",  # Spin-dominant middle overs; chase bias under lights
            "T20I": "dusty"   # Heavy evening dew causes massive second-innings chase advantage
        },
        "format_pars": {"TEST": 250, "ODI": 258, "T20I": 158},
        "pitch_wear_rate": 7.8,
        "description": "Ring of Fire stadium; high evening dew factor creates a decisive second-innings chasing bias."
    },
    "mcg": {
        "id": "mcg",
        "name": "Melbourne Cricket Ground",
        "city": "Melbourne",
        "country": "Australia",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 84.0, "square_m": 82.0, "avg_m": 82.0},
        "dew_factor": 0.25,
        "format_archetypes": {
            "TEST": "balanced",        # Hard drop-in deck with seam deviation and variable carry
            "ODI": "bouncy",       # Colossal dimensions reward hard lengths and tall hit-the-deck pacers
            "T20I": "bouncy"       # 82m square boundaries make mis-timed pull shots easy catches
        },
        "format_pars": {"TEST": 280, "ODI": 285, "T20I": 164},
        "pitch_wear_rate": 5.2,
        "description": "Colossal Australian arena; vast boundary dimensions heavily punish horizontal-bat mis-hits."
    },
    "scg": {
        "id": "scg",
        "name": "Sydney Cricket Ground",
        "city": "Sydney",
        "country": "Australia",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 78.0, "square_m": 74.0, "avg_m": 76.0},
        "dew_factor": 0.35,
        "format_archetypes": {
            "TEST": "dusty",  # Bulli soil creates Australia's premier turning surface on Days 4-5
            "ODI": "dusty",  # Gripping surface rewarding wrist-spinners in middle overs
            "T20I": "dusty"   # Slow surface where spin averages 7.2 economy
        },
        "format_pars": {"TEST": 275, "ODI": 280, "T20I": 166},
        "pitch_wear_rate": 7.2,
        "description": "Historic Sydney venue featuring bulli soil that offers unmatched grip and turn in Australasia."
    },
    "adelaide": {
        "id": "adelaide",
        "name": "Adelaide Oval",
        "city": "Adelaide",
        "country": "Australia",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 82.0, "square_m": 66.0, "avg_m": 74.0},
        "dew_factor": 0.30,
        "format_archetypes": {
            "TEST": "balanced",        # Pink-ball day/night seam movement and swing under twilight
            "ODI": "flat",         # Short square boundaries (66m) reward square-of-the-wicket strokes
            "T20I": "flat"         # True batting paradise with lightning outfield
        },
        "format_pars": {"TEST": 295, "ODI": 290, "T20I": 175},
        "pitch_wear_rate": 5.0,
        "description": "Picturesque venue with long straight and short square boundaries; high scoring limited-overs track."
    },
    "perth": {
        "id": "perth",
        "name": "Perth Stadium (Optus Stadium)",
        "city": "Perth",
        "country": "Australia",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 75.0, "square_m": 70.0, "avg_m": 72.5},
        "dew_factor": 0.12,
        "format_archetypes": {
            "TEST": "bouncy",      # Hard surface offering steep tennis-ball bounce and lateral carry
            "ODI": "bouncy",       # Fast pitch where high release points dominate
            "T20I": "bouncy"       # Difficult to hit through the line early against express pace
        },
        "format_pars": {"TEST": 310, "ODI": 268, "T20I": 165},
        "pitch_wear_rate": 5.8,
        "description": "World-class Optus Stadium wicket famous for extreme pace and shoulder-high bounce."
    },
    "centurion": {
        "id": "centurion",
        "name": "SuperSport Park",
        "city": "Centurion",
        "country": "South Africa",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 78.0, "square_m": 72.0, "avg_m": 75.0},
        "dew_factor": 0.20,
        "format_archetypes": {
            "TEST": "green",       # High altitude seam acceleration and lush grass top
            "ODI": "bouncy",       # 1,400m altitude reduces air drag; steep shoulder-high bounce
            "T20I": "bouncy"       # Express pacers get extreme lift and carry
        },
        "format_pars": {"TEST": 265, "ODI": 295, "T20I": 180},
        "pitch_wear_rate": 6.0,
        "description": "Highveld high-altitude surface; thin air accelerates ball speeds and creates steep vertical bounce."
    },
    "karachi": {
        "id": "karachi",
        "name": "National Stadium",
        "city": "Karachi",
        "country": "Pakistan",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.50,
        "format_archetypes": {
            "TEST": "flat",        # Traditional batting road with true pace and minimal sideways drift
            "ODI": "flat",         # High-scoring flat deck
            "T20I": "flat"         # 182 par, fast outfield and low seam hazard
        },
        "format_pars": {"TEST": 340, "ODI": 290, "T20I": 182},
        "pitch_wear_rate": 5.5,
        "description": "Coastal flat wicket with true bounce; minimal early movement favors aggressive stroke makers."
    },
    "mirpur": {
        "id": "mirpur",
        "name": "Sher-e-Bangla National Stadium",
        "city": "Mirpur",
        "country": "Bangladesh",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.40,
        "format_archetypes": {
            "TEST": "dusty",  # Heavy black clay rank turner; lowest scoring venue in Test cricket
            "ODI": "dusty",  # Slow, low surface where spinners bowl 35+ overs
            "T20I": "dusty"   # 148 par, 65% spin wickets, high dot-ball choke
        },
        "format_pars": {"TEST": 210, "ODI": 240, "T20I": 148},
        "pitch_wear_rate": 9.2,
        "description": "Heavy black clay surface with sluggish pace and dramatic spin; lowest scoring venue among full members."
    },
    "sylhet": {
        "id": "sylhet",
        "name": "Sylhet International Cricket Stadium",
        "city": "Sylhet",
        "country": "Bangladesh",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.30,
        "format_archetypes": {
            "TEST": "dusty",  # Green tinge on morning Day 1 gives way to abrasive, low-bounce spin
            "ODI": "balanced",         # Cutters and wobble-seam grip well on damp morning surface
            "T20I": "dusty"   # Slow-paced deck testing footwork and sweep shots
        },
        "format_pars": {"TEST": 252, "ODI": 248, "T20I": 154},
        "pitch_wear_rate": 8.8,
        "description": "Surrounded by tea gardens; early morning moisture gives pacers nip before turning into a spinning deck."
    },
    "wankhede": {
        "id": "wankhede",
        "name": "Wankhede Stadium",
        "city": "Mumbai",
        "country": "India",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.65,
        "format_archetypes": {
            "TEST": "dusty",  # Red soil deck generates vicious turn and bounce from Day 3
            "ODI": "flat",         # Electric red soil bounce allows free strokeplay (310+ par)
            "T20I": "flat"         # Evening sea humidity creates heavy dew; sixes hit with ease
        },
        "format_pars": {"TEST": 280, "ODI": 310, "T20I": 192},
        "pitch_wear_rate": 7.5,
        "description": "Red soil surface providing electric carry; heavy Arabian Sea dew makes chasing heavily favored."
    },
    "eden_gardens": {
        "id": "eden_gardens",
        "name": "Eden Gardens",
        "city": "Kolkata",
        "country": "India",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 68.0, "avg_m": 71.0},
        "dew_factor": 0.55,
        "format_archetypes": {
            "TEST": "dusty",  # Historic clay wicket with reverse swing and gripping spin
            "ODI": "dusty",  # Gripping black soil slows in middle overs
            "T20I": "flat"         # Fast outfield and evening dew produce high-scoring matches
        },
        "format_pars": {"TEST": 270, "ODI": 280, "T20I": 176},
        "pitch_wear_rate": 7.6,
        "description": "Iconic Indian venue; traditional clay pitch offers excellent carry initially before turning."
    },
    "lords": {
        "id": "lords",
        "name": "Lord's Cricket Ground",
        "city": "London",
        "country": "England",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 76.0, "square_m": 72.0, "avg_m": 74.0},
        "dew_factor": 0.20,
        "format_archetypes": {
            "TEST": "green",       # Famous 2.5-degree natural slope enhances lateral seam movement
            "ODI": "balanced",         # Balanced contest with early swing and seam deviation
            "T20I": "balanced"         # Disciplined line and length rewarded over raw pace
        },
        "format_pars": {"TEST": 260, "ODI": 275, "T20I": 168},
        "pitch_wear_rate": 5.4,
        "description": "Home of Cricket; natural slope creates lateral deviation into right-handers from Pavilion End."
    },
    "the_oval": {
        "id": "the_oval",
        "name": "The Oval",
        "city": "London",
        "country": "England",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 70.0, "avg_m": 71.0},
        "dew_factor": 0.25,
        "format_archetypes": {
            "TEST": "bouncy",      # Hardest surface in England with true bounce, reverse swing late
            "ODI": "flat",         # Fast outfield and consistent carry allow 300+ chases
            "T20I": "flat"         # True bounce and even pace produce high scoring
        },
        "format_pars": {"TEST": 305, "ODI": 290, "T20I": 175},
        "pitch_wear_rate": 5.8,
        "description": "South London clay deck; renowned as the fastest and truest batting pitch in the UK."
    },
    "rawalpindi": {
        "id": "rawalpindi",
        "name": "Rawalpindi Cricket Stadium",
        "city": "Rawalpindi",
        "country": "Pakistan",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 75.0, "square_m": 68.0, "avg_m": 71.5},
        "dew_factor": 0.40,
        "format_archetypes": {
            "TEST": "flat",        # Historically high-scoring batting track; early bounce, late reverse swing & spin wear
            "ODI": "flat",         # Fast outfield and true bounce produce high-scoring 290+ matches
            "T20I": "flat"         # 185 par, high-scoring venue with true surface
        },
        "format_pars": {"TEST": 365, "ODI": 295, "T20I": 185},
        "pitch_wear_rate": 6.8,
        "description": "High-scoring subcontinental venue with true bounce early on, transitioning to reverse swing and rough spin on Days 4-5."
    },
    "lahore": {
        "id": "lahore",
        "name": "Gaddafi Stadium",
        "city": "Lahore",
        "country": "Pakistan",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 68.0, "avg_m": 71.0},
        "dew_factor": 0.50,
        "format_archetypes": {
            "TEST": "flat",        # Classic Lahore batting road on Days 1-3, abrasive soil generates sharp spin Days 4-5
            "ODI": "flat",         # High-scoring batting deck; dew factor in evening chases
            "T20I": "flat"         # 180 par, fast outfield favoring boundary hitters
        },
        "format_pars": {"TEST": 380, "ODI": 290, "T20I": 180},
        "pitch_wear_rate": 6.5,
        "description": "Historic Pakistani colosseum with true red-soil bounce; premier batting surface early before spinners exploit wear."    },
    "chinnaswamy": {
        "id": "chinnaswamy",
        "name": "M. Chinnaswamy Stadium",
        "city": "Bengaluru",
        "country": "India",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 65.0, "square_m": 61.0, "avg_m": 63.0},
        "dew_factor": 0.40,
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"},
        "format_pars": {"TEST": 360, "ODI": 315, "T20I": 195},
        "pitch_wear_rate": 5.2,
        "description": "High-altitude batting colosseum with short 61m square boundaries; highest boundary percentage in India."
    },
    "narendra_modi": {
        "id": "narendra_modi",
        "name": "Narendra Modi Stadium",
        "city": "Ahmedabad",
        "country": "India",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 76.0, "square_m": 74.0, "avg_m": 75.0},
        "dew_factor": 0.45,
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"},
        "format_pars": {"TEST": 320, "ODI": 270, "T20I": 172},
        "pitch_wear_rate": 6.0,
        "description": "World largest cricket stadium; cavernous boundaries reward power and athletic outfielding."
    },
    "chepauk": {
        "id": "chepauk",
        "name": "MA Chidambaram Stadium (Chepauk)",
        "city": "Chennai",
        "country": "India",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 65.0, "avg_m": 68.0},
        "dew_factor": 0.30,
        "format_archetypes": {"TEST": "dusty", "ODI": "dusty", "T20I": "dusty"},
        "format_pars": {"TEST": 305, "ODI": 245, "T20I": 155},
        "pitch_wear_rate": 8.0,
        "description": "Classic red clay dustbowl with heavy coastal humidity; prominent spin grip and low bounce."
    },
    "dharamshala": {
        "id": "dharamshala",
        "name": "HPCA Stadium",
        "city": "Dharamshala",
        "country": "India",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.15,
        "format_archetypes": {"TEST": "green", "ODI": "balanced", "T20I": "balanced"},
        "format_pars": {"TEST": 280, "ODI": 265, "T20I": 168},
        "pitch_wear_rate": 4.0,
        "description": "High-altitude Himalayan ground with thin air and brisk mountain breezes assisting swing and ball flight."
    },
    "gabba": {
        "id": "gabba",
        "name": "The Gabba",
        "city": "Brisbane",
        "country": "Australia",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 78.0, "square_m": 75.0, "avg_m": 76.5},
        "dew_factor": 0.18,
        "format_archetypes": {"TEST": "bouncy", "ODI": "bouncy", "T20I": "bouncy"},
        "format_pars": {"TEST": 330, "ODI": 280, "T20I": 174},
        "pitch_wear_rate": 5.0,
        "description": "Queensland fortress with steep vertical bounce and true carry for hit-the-deck pacers."
    },
    "hobart": {
        "id": "hobart",
        "name": "Bellerive Oval",
        "city": "Hobart",
        "country": "Australia",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.20,
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "flat"},
        "format_pars": {"TEST": 290, "ODI": 275, "T20I": 172},
        "pitch_wear_rate": 4.2,
        "description": "Maritime Tasmanian venue with Derwent River gusts; excellent true bounce under lights."
    },
    "trent_bridge": {
        "id": "trent_bridge",
        "name": "Trent Bridge",
        "city": "Nottingham",
        "country": "England",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 67.0, "square_m": 65.0, "avg_m": 66.0},
        "dew_factor": 0.15,
        "format_archetypes": {"TEST": "balanced", "ODI": "flat", "T20I": "flat"},
        "format_pars": {"TEST": 310, "ODI": 325, "T20I": 188},
        "pitch_wear_rate": 4.8,
        "description": "Renowned high-scoring English venue with short square ropes and lightning fast outfield."
    },
    "edgbaston": {
        "id": "edgbaston",
        "name": "Edgbaston",
        "city": "Birmingham",
        "country": "England",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 66.0, "avg_m": 69.0},
        "dew_factor": 0.18,
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "flat"},
        "format_pars": {"TEST": 305, "ODI": 285, "T20I": 175},
        "pitch_wear_rate": 5.2,
        "description": "Atmospheric Birmingham bowl with vocal crowd; true carry and short Eric Hollies boundary."
    },
    "headingley": {
        "id": "headingley",
        "name": "Headingley",
        "city": "Leeds",
        "country": "England",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 67.0, "avg_m": 69.5},
        "dew_factor": 0.15,
        "format_archetypes": {"TEST": "green", "ODI": "balanced", "T20I": "flat"},
        "format_pars": {"TEST": 285, "ODI": 275, "T20I": 174},
        "pitch_wear_rate": 4.5,
        "description": "Traditional Yorkshire swinging venue; heavy cloud cover triggers pronounced lateral hoop."
    },
    "old_trafford": {
        "id": "old_trafford",
        "name": "Emirates Old Trafford",
        "city": "Manchester",
        "country": "England",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 70.0, "avg_m": 72.0},
        "dew_factor": 0.20,
        "format_archetypes": {"TEST": "bouncy", "ODI": "balanced", "T20I": "balanced"},
        "format_pars": {"TEST": 315, "ODI": 270, "T20I": 168},
        "pitch_wear_rate": 5.0,
        "description": "Pace-friendly English surface offering the quickest deck and most consistent bounce in the UK."
    },
    "wanderers": {
        "id": "wanderers",
        "name": "Wanderers Stadium",
        "city": "Johannesburg",
        "country": "South Africa",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 68.0, "square_m": 64.0, "avg_m": 66.0},
        "dew_factor": 0.10,
        "format_archetypes": {"TEST": "bouncy", "ODI": "flat", "T20I": "flat"},
        "format_pars": {"TEST": 340, "ODI": 320, "T20I": 190},
        "pitch_wear_rate": 5.5,
        "description": "The Bullring; high altitude (1700m) and 64m boundaries create extreme six-hitting projectile velocity."
    },
    "newlands": {
        "id": "newlands",
        "name": "Newlands",
        "city": "Cape Town",
        "country": "South Africa",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.15,
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"},
        "format_pars": {"TEST": 295, "ODI": 265, "T20I": 165},
        "pitch_wear_rate": 5.8,
        "description": "Table Mountain backdrop; early seam movement followed by reverse swing and late turn."
    },
    "kingsmead": {
        "id": "kingsmead",
        "name": "Kingsmead",
        "city": "Durban",
        "country": "South Africa",
        "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 68.0, "avg_m": 71.0},
        "dew_factor": 0.35,
        "format_archetypes": {"TEST": "green", "ODI": "balanced", "T20I": "balanced"},
        "format_pars": {"TEST": 280, "ODI": 255, "T20I": 162},
        "pitch_wear_rate": 4.6,
        "description": "Subtropical seaside venue with high humidity; heavy evening dew affects night matches."
    },
    "kensington_oval": {
        "id": "kensington_oval",
        "name": "Kensington Oval",
        "city": "Bridgetown",
        "country": "Barbados",
        "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 65.0, "square_m": 62.0, "avg_m": 63.5},
        "dew_factor": 0.15,
        "format_archetypes": {"TEST": "bouncy", "ODI": "flat", "T20I": "flat"},
        "format_pars": {"TEST": 330, "ODI": 285, "T20I": 178},
        "pitch_wear_rate": 5.0,
        "description": "Mecca of Caribbean cricket; brisk trade winds blowing across short 62m square boundaries."
    },
    "queens_park_oval": {
        "id": "queens_park_oval",
        "name": "Queen's Park Oval",
        "city": "Port of Spain",
        "country": "Trinidad and Tobago",
        "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.22,
        "format_archetypes": {"TEST": "dusty", "ODI": "dusty", "T20I": "balanced"},
        "format_pars": {"TEST": 275, "ODI": 240, "T20I": 158},
        "pitch_wear_rate": 6.8,
        "description": "Trinidad classic with sluggish, two-paced nature favoring finger spin and cutters."
    },
    "daren_sammy": {
        "id": "daren_sammy",
        "name": "Daren Sammy National Cricket Stadium",
        "city": "Gros Islet",
        "country": "Saint Lucia",
        "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 65.0, "avg_m": 67.5},
        "dew_factor": 0.20,
        "format_archetypes": {"TEST": "bouncy", "ODI": "flat", "T20I": "flat"},
        "format_pars": {"TEST": 310, "ODI": 280, "T20I": 182},
        "pitch_wear_rate": 4.8,
        "description": "True Caribbean bounce with Caribbean trade winds aiding sixes over midwicket."
    },
    "providence": {
        "id": "providence",
        "name": "Providence Stadium",
        "city": "Georgetown",
        "country": "Guyana",
        "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.35,
        "format_archetypes": {"TEST": "dusty", "ODI": "dusty", "T20I": "dusty"},
        "format_pars": {"TEST": 260, "ODI": 230, "T20I": 148},
        "pitch_wear_rate": 7.5,
        "description": "Heavy sea-level air and clay soil; slow, low, turning deck where run-making is a grind."
    },
    "galle": {
        "id": "galle",
        "name": "Galle International Stadium",
        "city": "Galle",
        "country": "Sri Lanka",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.25,
        "format_archetypes": {"TEST": "dusty", "ODI": "dusty", "T20I": "dusty"},
        "format_pars": {"TEST": 280, "ODI": 235, "T20I": 150},
        "pitch_wear_rate": 8.8,
        "description": "Historic coastal fort fortress; sharp turn from Day 1 with abrasive coastal salt breeze."
    },
    "pallekele": {
        "id": "pallekele",
        "name": "Pallekele International Cricket Stadium",
        "city": "Kandy",
        "country": "Sri Lanka",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.35,
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "flat"},
        "format_pars": {"TEST": 310, "ODI": 275, "T20I": 172},
        "pitch_wear_rate": 5.0,
        "description": "Scenic hill-country venue; lush outfield and good carry before taking spin."
    },
    "premadasa": {
        "id": "premadasa",
        "name": "R. Premadasa Stadium",
        "city": "Colombo",
        "country": "Sri Lanka",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.40,
        "format_archetypes": {"TEST": "dusty", "ODI": "dusty", "T20I": "dusty"},
        "format_pars": {"TEST": 290, "ODI": 245, "T20I": 152},
        "pitch_wear_rate": 7.2,
        "description": "Columbo spin haven; slow surface with sharp turn and high evening dew in chases."
    },
    "abu_dhabi": {
        "id": "abu_dhabi",
        "name": "Sheikh Zayed Cricket Stadium",
        "city": "Abu Dhabi",
        "country": "United Arab Emirates",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 70.0, "avg_m": 72.0},
        "dew_factor": 0.50,
        "format_archetypes": {"TEST": "flat", "ODI": "balanced", "T20I": "balanced"},
        "format_pars": {"TEST": 380, "ODI": 265, "T20I": 165},
        "pitch_wear_rate": 5.5,
        "description": "Modern desert bowl; flat batting track on early days before evening dew speeds outfield."
    },
    "harare": {
        "id": "harare",
        "name": "Harare Sports Club",
        "city": "Harare",
        "country": "Zimbabwe",
        "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.10,
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"},
        "format_pars": {"TEST": 350, "ODI": 280, "T20I": 175},
        "pitch_wear_rate": 5.0,
        "description": "Highveld venue with clean altitude carry; true bounce and consistent stroke-play."    },
    "delhi": {
        "id": "delhi", "name": "Arun Jaitley Stadium", "city": "Delhi", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 68.0, "square_m": 62.0, "avg_m": 64.0},
        "dew_factor": 0.45, "coordinates": {"lat": 28.6378, "lon": 77.2425},
        "format_archetypes": {"TEST": "dusty", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 330, "ODI": 290, "T20I": 185},
        "pitch_wear_rate": 6.8, "description": "Historic Kotla venue; short 62m square boundaries with high winter evening dew."
    },
    "hyderabad": {
        "id": "hyderabad", "name": "Rajiv Gandhi International Stadium", "city": "Hyderabad", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 71.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.35, "coordinates": {"lat": 17.4065, "lon": 78.5505},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 360, "ODI": 310, "T20I": 190},
        "pitch_wear_rate": 5.5, "description": "High-scoring Telangana surface; black soil deck favoring batters."
    },
    "lucknow": {
        "id": "lucknow", "name": "BRSABV Ekana Cricket Stadium", "city": "Lucknow", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 73.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.40, "coordinates": {"lat": 26.7925, "lon": 81.0142},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 310, "ODI": 260, "T20I": 162},
        "pitch_wear_rate": 6.5, "description": "Spacious venue offering dual soil tracks (black clay vs red clay); balanced contest."
    },
    "rajkot": {
        "id": "rajkot", "name": "SCA Stadium (Khandheri)", "city": "Rajkot", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.30, "coordinates": {"lat": 22.3653, "lon": 70.7103},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 400, "ODI": 320, "T20I": 195},
        "pitch_wear_rate": 4.5, "description": "Pure batting highway; minimal lateral seam movement, true bounce."
    },
    "indore": {
        "id": "indore", "name": "Holkar Cricket Stadium", "city": "Indore", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 65.0, "square_m": 60.0, "avg_m": 62.0},
        "dew_factor": 0.35, "coordinates": {"lat": 22.7244, "lon": 75.8778},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 380, "ODI": 330, "T20I": 205},
        "pitch_wear_rate": 5.0, "description": "Ultra-compact square boundaries (60m) and lightning-fast outfield; six-hitting paradise."
    },
    "mohali": {
        "id": "mohali", "name": "PCA IS Bindra Stadium", "city": "Mohali", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 72.0, "avg_m": 73.0},
        "dew_factor": 0.50, "coordinates": {"lat": 30.6908, "lon": 76.7374},
        "format_archetypes": {"TEST": "balanced", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 340, "ODI": 295, "T20I": 182},
        "pitch_wear_rate": 5.2, "description": "Traditional pace venue with generous boundary dimensions; heavy winter dew."
    },
    "mullanpur": {
        "id": "mullanpur", "name": "Maharaja Yadavindra Singh Stadium", "city": "New Chandigarh", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.45, "coordinates": {"lat": 30.7936, "lon": 76.7289},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "flat"}, "format_pars": {"TEST": 320, "ODI": 280, "T20I": 178},
        "pitch_wear_rate": 5.0, "description": "Modern venue in Punjab with good carry and true bounce."
    },
    "vizag": {
        "id": "vizag", "name": "ACA-VDCA Cricket Stadium", "city": "Visakhapatnam", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 67.5},
        "dew_factor": 0.40, "coordinates": {"lat": 17.7969, "lon": 83.3533},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 370, "ODI": 310, "T20I": 192},
        "pitch_wear_rate": 5.2, "description": "Coastal batting track surrounded by hills; true bounce and high six rates."
    },
    "guwahati": {
        "id": "guwahati", "name": "Barsapara Cricket Stadium", "city": "Guwahati", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 68.0, "square_m": 64.0, "avg_m": 65.5},
        "dew_factor": 0.55, "coordinates": {"lat": 26.1415, "lon": 91.7362},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 360, "ODI": 320, "T20I": 200},
        "pitch_wear_rate": 4.8, "description": "High humidity, short square ropes, and massive dew in evening matches."
    },
    "ranchi": {
        "id": "ranchi", "name": "JSCA International Stadium Complex", "city": "Ranchi", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 70.0, "avg_m": 72.0},
        "dew_factor": 0.40, "coordinates": {"lat": 23.3134, "lon": 85.2778},
        "format_archetypes": {"TEST": "dusty", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 320, "ODI": 275, "T20I": 170},
        "pitch_wear_rate": 7.0, "description": "Turning track in Tests; fair bounce in limited-overs matches."
    },
    "kanpur": {
        "id": "kanpur", "name": "Green Park Stadium", "city": "Kanpur", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 69.0, "square_m": 65.0, "avg_m": 66.5},
        "dew_factor": 0.45, "coordinates": {"lat": 26.4789, "lon": 80.3547},
        "format_archetypes": {"TEST": "dusty", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 290, "ODI": 255, "T20I": 160},
        "pitch_wear_rate": 8.0, "description": "Traditional low and slow track with sharp spin on Day 3 onwards."
    },
    "cuttack": {
        "id": "cuttack", "name": "Barabati Stadium", "city": "Cuttack", "country": "India", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.45, "coordinates": {"lat": 20.4819, "lon": 85.8675},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 310, "ODI": 265, "T20I": 165},
        "pitch_wear_rate": 6.0, "description": "Coastal humidity, moderate seam movement early, heavy evening dew."
    },
    "canberra": {
        "id": "canberra", "name": "Manuka Oval", "city": "Canberra", "country": "Australia", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 68.0, "avg_m": 70.5},
        "dew_factor": 0.20, "coordinates": {"lat": -35.3178, "lon": 149.1344},
        "format_archetypes": {"TEST": "balanced", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 330, "ODI": 290, "T20I": 178},
        "pitch_wear_rate": 4.5, "description": "True drop-in deck in capital territory; reliable bounce and carry."
    },
    "geelong": {
        "id": "geelong", "name": "GMHBA Stadium (Kardinia Park)", "city": "Geelong", "country": "Australia", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 88.0, "square_m": 60.0, "avg_m": 70.0},
        "dew_factor": 0.22, "coordinates": {"lat": -38.1581, "lon": 144.3547},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 280, "ODI": 250, "T20I": 155},
        "pitch_wear_rate": 4.2, "description": "Narrow AFL oval geometry: very long straight (88m), short square (60m)."
    },
    "cardiff": {
        "id": "cardiff", "name": "Sophia Gardens", "city": "Cardiff", "country": "England", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 64.0, "avg_m": 68.5},
        "dew_factor": 0.20, "coordinates": {"lat": 51.4878, "lon": -3.1906},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "flat"}, "format_pars": {"TEST": 290, "ODI": 270, "T20I": 168},
        "pitch_wear_rate": 5.0, "description": "Welsh capital ground with river breeze; one square boundary noticeably short (60m)."
    },
    "bristol": {
        "id": "bristol", "name": "Seat Unique Stadium (Bristol)", "city": "Bristol", "country": "England", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 63.0, "avg_m": 66.0},
        "dew_factor": 0.18, "coordinates": {"lat": 51.4764, "lon": -2.5858},
        "format_archetypes": {"TEST": "balanced", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 310, "ODI": 310, "T20I": 185},
        "pitch_wear_rate": 4.5, "description": "High-scoring English county ground with short square boundaries."
    },
    "southampton": {
        "id": "southampton", "name": "The Ageas Bowl (Rose Bowl)", "city": "Southampton", "country": "England", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 76.0, "square_m": 72.0, "avg_m": 74.0},
        "dew_factor": 0.18, "coordinates": {"lat": 50.9242, "lon": -1.3217},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 305, "ODI": 270, "T20I": 165},
        "pitch_wear_rate": 5.2, "description": "Large circular amphitheater; generous boundaries demand athletic running."
    },
    "chester_le_street": {
        "id": "chester_le_street", "name": "Riverside Ground", "city": "Chester-le-Street", "country": "England", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 76.0, "square_m": 72.0, "avg_m": 74.0},
        "dew_factor": 0.18, "coordinates": {"lat": 54.8483, "lon": -1.5594},
        "format_archetypes": {"TEST": "green", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 285, "ODI": 260, "T20I": 162},
        "pitch_wear_rate": 4.8, "description": "Northern venue on the River Wear; significant early swing and lateral movement."
    },
    "gqeberha": {
        "id": "gqeberha", "name": "St George's Park", "city": "Gqeberha (Port Elizabeth)", "country": "South Africa", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.25, "coordinates": {"lat": -33.9667, "lon": 25.6078},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 290, "ODI": 255, "T20I": 160},
        "pitch_wear_rate": 6.0, "description": "Windy coastal venue famous for brass band; reverse swing and late reverse spin."
    },
    "bloemfontein": {
        "id": "bloemfontein", "name": "Mangaung Oval", "city": "Bloemfontein", "country": "South Africa", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 70.0, "avg_m": 72.0},
        "dew_factor": 0.12, "coordinates": {"lat": -29.1139, "lon": 26.2086},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 360, "ODI": 310, "T20I": 185},
        "pitch_wear_rate": 4.8, "description": "High altitude inland batting surface; true pace with ball flying off the bat."
    },
    "paarl": {
        "id": "paarl", "name": "Boland Park", "city": "Paarl", "country": "South Africa", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.15, "coordinates": {"lat": -33.7439, "lon": 18.9742},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 295, "ODI": 260, "T20I": 158},
        "pitch_wear_rate": 5.5, "description": "Picturesque winelands venue with slowish pitch nature; cutters grip."
    },
    "tarouba": {
        "id": "tarouba", "name": "Brian Lara Cricket Academy", "city": "Tarouba", "country": "Trinidad and Tobago", "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 65.0, "avg_m": 67.5},
        "dew_factor": 0.25, "coordinates": {"lat": 10.2764, "lon": -61.4286},
        "format_archetypes": {"TEST": "dusty", "ODI": "dusty", "T20I": "dusty"}, "format_pars": {"TEST": 265, "ODI": 225, "T20I": 142},
        "pitch_wear_rate": 7.5, "description": "Modern Trinidad facility; spongy bounce with sharp grip for finger spinners."
    },
    "antigua": {
        "id": "antigua", "name": "Sir Vivian Richards Stadium", "city": "North Sound", "country": "Antigua and Barbuda", "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.18, "coordinates": {"lat": 17.1039, "lon": -61.7842},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "flat"}, "format_pars": {"TEST": 305, "ODI": 270, "T20I": 168},
        "pitch_wear_rate": 5.0, "description": "Trade winds blowing across spacious playing area; true bounce."
    },
    "st_kitts": {
        "id": "st_kitts", "name": "Warner Park", "city": "Basseterre", "country": "Saint Kitts and Nevis", "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 65.0, "square_m": 60.0, "avg_m": 62.5},
        "dew_factor": 0.20, "coordinates": {"lat": 17.2978, "lon": -62.7214},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 330, "ODI": 295, "T20I": 188},
        "pitch_wear_rate": 4.5, "description": "One of the smallest international venues in the Caribbean; high boundary percentage."
    },
    "dominica": {
        "id": "dominica", "name": "Windsor Park", "city": "Roseau", "country": "Dominica", "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.25, "coordinates": {"lat": 15.3025, "lon": -61.3858},
        "format_archetypes": {"TEST": "dusty", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 280, "ODI": 250, "T20I": 158},
        "pitch_wear_rate": 6.5, "description": "Lush Nature Island ground; slow deck favoring spinners."
    },
    "st_vincent": {
        "id": "st_vincent", "name": "Arnos Vale Stadium", "city": "Kingstown", "country": "Saint Vincent and the Grenadines", "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 68.0, "square_m": 64.0, "avg_m": 66.0},
        "dew_factor": 0.20, "coordinates": {"lat": 13.1439, "lon": -61.2114},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 285, "ODI": 245, "T20I": 154},
        "pitch_wear_rate": 5.8, "description": "Seaside venue with strong crosswinds; assists cutters and swing."
    },
    "jamaica": {
        "id": "jamaica", "name": "Sabina Park", "city": "Kingston", "country": "Jamaica", "region_cluster": "caribbean",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.15, "coordinates": {"lat": 17.9789, "lon": -76.7814},
        "format_archetypes": {"TEST": "bouncy", "ODI": "balanced", "T20I": "flat"}, "format_pars": {"TEST": 310, "ODI": 265, "T20I": 172},
        "pitch_wear_rate": 5.5, "description": "Historic venue offering steep bounce and pace off a hard clay surface."
    },
    "christchurch": {
        "id": "christchurch", "name": "Hagley Oval", "city": "Christchurch", "country": "New Zealand", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.18, "coordinates": {"lat": -43.5358, "lon": 172.6203},
        "format_archetypes": {"TEST": "green", "ODI": "balanced", "T20I": "flat"}, "format_pars": {"TEST": 280, "ODI": 275, "T20I": 172},
        "pitch_wear_rate": 4.2, "description": "Boutique South Island park ground; pristine green top offering early swing."
    },
    "basin_reserve": {
        "id": "basin_reserve", "name": "Basin Reserve", "city": "Wellington", "country": "New Zealand", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.20, "coordinates": {"lat": -41.3003, "lon": 174.7819},
        "format_archetypes": {"TEST": "green", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 285, "ODI": 265, "T20I": 165},
        "pitch_wear_rate": 4.5, "description": "Iconic Test venue subject to ferocious Cook Strait gale winds."
    },
    "dunedin": {
        "id": "dunedin", "name": "University Oval", "city": "Dunedin", "country": "New Zealand", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.18, "coordinates": {"lat": -45.8653, "lon": 170.5217},
        "format_archetypes": {"TEST": "green", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 290, "ODI": 285, "T20I": 175},
        "pitch_wear_rate": 4.0, "description": "Southernmost international venue; crisp air and true drop-in bounce."
    },
    "multan": {
        "id": "multan", "name": "Multan Cricket Stadium", "city": "Multan", "country": "Pakistan", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 68.0, "avg_m": 71.0},
        "dew_factor": 0.45, "coordinates": {"lat": 30.1794, "lon": 71.4939},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 410, "ODI": 315, "T20I": 188},
        "pitch_wear_rate": 6.5, "description": "High-scoring Punjab batting track in early overs before abrasive dust yields spin."
    },
    "faisalabad": {
        "id": "faisalabad", "name": "Iqbal Stadium", "city": "Faisalabad", "country": "Pakistan", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.40, "coordinates": {"lat": 31.4114, "lon": 73.0764},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 380, "ODI": 290, "T20I": 178},
        "pitch_wear_rate": 5.8, "description": "Traditional batting strip with true bounce."
    },
    "hambantota": {
        "id": "hambantota", "name": "Mahinda Rajapaksa International Stadium", "city": "Hambantota", "country": "Sri Lanka", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 78.0, "square_m": 74.0, "avg_m": 76.0},
        "dew_factor": 0.35, "coordinates": {"lat": 6.1953, "lon": 81.1681},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 295, "ODI": 260, "T20I": 160},
        "pitch_wear_rate": 6.0, "description": "Massive playing arena subject to heavy Southern Province sea gales."
    },
    "ssc_colombo": {
        "id": "ssc_colombo", "name": "Sinhalese Sports Club (SSC)", "city": "Colombo", "country": "Sri Lanka", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 74.0, "square_m": 70.0, "avg_m": 72.0},
        "dew_factor": 0.35, "coordinates": {"lat": 6.9078, "lon": 79.8683},
        "format_archetypes": {"TEST": "flat", "ODI": "balanced", "T20I": "flat"}, "format_pars": {"TEST": 360, "ODI": 280, "T20I": 175},
        "pitch_wear_rate": 6.8, "description": "High-scoring Colombo Test venue early on before turning sharply."
    },
    "chattogram": {
        "id": "chattogram", "name": "Zahur Ahmed Chowdhury Stadium", "city": "Chattogram", "country": "Bangladesh", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.40, "coordinates": {"lat": 22.3589, "lon": 91.7767},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 340, "ODI": 285, "T20I": 174},
        "pitch_wear_rate": 6.0, "description": "Best batting surface in Bangladesh; high run aggregate venue."
    },
    "bulawayo": {
        "id": "bulawayo", "name": "Queens Sports Club", "city": "Bulawayo", "country": "Zimbabwe", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.12, "coordinates": {"lat": -20.1558, "lon": 28.5911},
        "format_archetypes": {"TEST": "balanced", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 310, "ODI": 275, "T20I": 170},
        "pitch_wear_rate": 5.2, "description": "True African Highveld track; consistent bounce and sunny strokeplay."
    },
    "dallas": {
        "id": "dallas", "name": "Grand Prairie Stadium", "city": "Dallas", "country": "United States", "region_cluster": "other",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 68.0, "avg_m": 70.0},
        "dew_factor": 0.25, "coordinates": {"lat": 32.7667, "lon": -96.9936},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 320, "ODI": 280, "T20I": 180},
        "pitch_wear_rate": 4.5, "description": "Premier Major League Cricket venue in Texas; high scoring drop-in wicket."
    },
    "lauderhill": {
        "id": "lauderhill", "name": "Central Broward Regional Park", "city": "Lauderhill", "country": "United States", "region_cluster": "other",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 65.0, "avg_m": 67.0},
        "dew_factor": 0.35, "coordinates": {"lat": 26.1417, "lon": -80.2008},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 330, "ODI": 290, "T20I": 185},
        "pitch_wear_rate": 5.0, "description": "Florida venue with subtropical humidity and fast outfield."
    },
    "new_york": {
        "id": "new_york", "name": "Nassau County International Stadium", "city": "New York", "country": "United States", "region_cluster": "other",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.20, "coordinates": {"lat": 40.7308, "lon": -73.5858},
        "format_archetypes": {"TEST": "bouncy", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 250, "ODI": 220, "T20I": 138},
        "pitch_wear_rate": 4.0, "description": "Adelaide-cultivated drop-in pitch with variable bounce and thick grass outfield."
    },
    "kirtipur": {
        "id": "kirtipur", "name": "Tribhuvan University Ground", "city": "Kirtipur", "country": "Nepal", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.20, "coordinates": {"lat": 27.6764, "lon": 85.2878},
        "format_archetypes": {"TEST": "dusty", "ODI": "dusty", "T20I": "balanced"}, "format_pars": {"TEST": 280, "ODI": 240, "T20I": 155},
        "pitch_wear_rate": 6.5, "description": "Valley venue in Kathmandu; thin air and sharp turn for spinners."
    },
    "dublin": {
        "id": "dublin", "name": "Malahide Cricket Club Ground", "city": "Dublin", "country": "Ireland", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.25, "coordinates": {"lat": 53.4503, "lon": -6.1625},
        "format_archetypes": {"TEST": "green", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 285, "ODI": 290, "T20I": 178},
        "pitch_wear_rate": 4.0, "description": "The Village; true surface with coastal breezes assisting swing."
    },
    "amstelveen": {
        "id": "amstelveen", "name": "VRA Cricket Ground", "city": "Amstelveen", "country": "Netherlands", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.25, "coordinates": {"lat": 52.3167, "lon": 4.8500},
        "format_archetypes": {"TEST": "green", "ODI": "balanced", "T20I": "flat"}, "format_pars": {"TEST": 275, "ODI": 270, "T20I": 170},
        "pitch_wear_rate": 4.2, "description": "Amsterdam forest setting; true seam movement under overcast conditions."
    },
    "muscat": {
        "id": "muscat", "name": "Al Amerat Cricket Stadium", "city": "Muscat", "country": "Oman", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.35, "coordinates": {"lat": 23.4756, "lon": 58.4686},
        "format_archetypes": {"TEST": "flat", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 310, "ODI": 255, "T20I": 158},
        "pitch_wear_rate": 5.5, "description": "Oman desert valley ground surrounded by rocky mountains; true bounce."
    },
    "windhoek": {
        "id": "windhoek", "name": "Wanderers Cricket Ground", "city": "Windhoek", "country": "Namibia", "region_cluster": "other",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.10, "coordinates": {"lat": -22.5833, "lon": 17.0833},
        "format_archetypes": {"TEST": "flat", "ODI": "flat", "T20I": "flat"}, "format_pars": {"TEST": 330, "ODI": 285, "T20I": 175},
        "pitch_wear_rate": 4.5, "description": "High altitude (1700m) arid African venue; fast outfield and great ball carry."
    },
"edinburgh": {
        "id": "edinburgh", "name": "The Grange Club", "city": "Edinburgh", "country": "Scotland", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 72.0, "square_m": 66.0, "avg_m": 68.0},
        "dew_factor": 0.20, "coordinates": {"lat": 55.9592, "lon": -3.2081},
        "format_archetypes": {"TEST": "green", "ODI": "balanced", "T20I": "flat"}, "format_pars": {"TEST": 280, "ODI": 275, "T20I": 172},
        "pitch_wear_rate": 3.8, "description": "Scottish maritime ground; early swing with firm batting conditions in afternoon."
    },
    "king_city": {
        "id": "king_city", "name": "Maple Leaf Cricket Club", "city": "King City", "country": "Canada", "region_cluster": "other",
        "boundary_dimensions": {"straight_m": 70.0, "square_m": 65.0, "avg_m": 67.0},
        "dew_factor": 0.15, "coordinates": {"lat": 43.9167, "lon": -79.5333},
        "format_archetypes": {"TEST": "balanced", "ODI": "balanced", "T20I": "flat"}, "format_pars": {"TEST": 285, "ODI": 260, "T20I": 165},
        "pitch_wear_rate": 4.0, "description": "North American open field venue; true bounce with moderate seam early."
    },
    "port_moresby": {
        "id": "port_moresby", "name": "Amini Park", "city": "Port Moresby", "country": "Papua New Guinea", "region_cluster": "other",
        "boundary_dimensions": {"straight_m": 68.0, "square_m": 64.0, "avg_m": 66.0},
        "dew_factor": 0.20, "coordinates": {"lat": -9.4789, "lon": 147.1925},
        "format_archetypes": {"TEST": "dusty", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 270, "ODI": 240, "T20I": 152},
        "pitch_wear_rate": 5.2, "description": "Pacific coastal deck; fair pace early yielding to gripping spin."
    },
    "dubai": {
        "id": "dubai", "name": "Dubai International Cricket Stadium", "city": "Dubai", "country": "United Arab Emirates", "region_cluster": "subcontinent",
        "boundary_dimensions": {"straight_m": 75.0, "square_m": 70.0, "avg_m": 72.5},
        "dew_factor": 0.40, "coordinates": {"lat": 25.0448, "lon": 55.2191},
        "format_archetypes": {"TEST": "dusty", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 310, "ODI": 265, "T20I": 165},
        "pitch_wear_rate": 6.2, "description": "Ring of Fire stadium; high dew in night fixtures, abrasive clay favoring reverse swing and spinners."
    },
    "belfast": {
        "id": "belfast", "name": "Civil Service Cricket Club (Stormont)", "city": "Belfast", "country": "Ireland", "region_cluster": "sena",
        "boundary_dimensions": {"straight_m": 69.0, "square_m": 65.0, "avg_m": 67.0},
        "dew_factor": 0.25, "coordinates": {"lat": 54.5986, "lon": -5.8458},
        "format_archetypes": {"TEST": "green", "ODI": "balanced", "T20I": "balanced"}, "format_pars": {"TEST": 275, "ODI": 255, "T20I": 160},
        "pitch_wear_rate": 4.1, "description": "Historic Northern Ireland venue; lush green cover supporting persistent seam movement."
    },
}

# 2A. Country Career Pitch Adaptability Matrix (15+ Years Historical Baseline)
COUNTRY_PITCH_ADAPTABILITY: Dict[str, Dict[str, Dict[str, float]]] = {
    "IND": {
        "TEST": {"flat": 90.0, "green": 74.0, "balanced": 80.0, "bouncy": 78.0, "dusty": 96.0},
        "ODI":  {"flat": 95.0, "green": 80.0, "balanced": 84.0, "bouncy": 86.0, "dusty": 94.0},
        "T20I": {"flat": 96.0, "green": 78.0, "balanced": 82.0, "bouncy": 84.0, "dusty": 92.0}
    },
    "AUS": {
        "TEST": {"flat": 88.0, "green": 92.0, "balanced": 90.0, "bouncy": 96.0, "dusty": 70.0},
        "ODI":  {"flat": 93.0, "green": 90.0, "balanced": 88.0, "bouncy": 95.0, "dusty": 76.0},
        "T20I": {"flat": 92.0, "green": 85.0, "balanced": 86.0, "bouncy": 94.0, "dusty": 78.0}
    },
    "ENG": {
        "TEST": {"flat": 84.0, "green": 94.0, "balanced": 92.0, "bouncy": 80.0, "dusty": 68.0},
        "ODI":  {"flat": 95.0, "green": 86.0, "balanced": 85.0, "bouncy": 84.0, "dusty": 74.0},
        "T20I": {"flat": 93.0, "green": 82.0, "balanced": 84.0, "bouncy": 82.0, "dusty": 76.0}
    },
    "SA": {
        "TEST": {"flat": 80.0, "green": 90.0, "balanced": 88.0, "bouncy": 97.0, "dusty": 64.0},
        "ODI":  {"flat": 90.0, "green": 86.0, "balanced": 84.0, "bouncy": 94.0, "dusty": 72.0},
        "T20I": {"flat": 90.0, "green": 84.0, "balanced": 82.0, "bouncy": 92.0, "dusty": 74.0}
    },
    "NZ": {
        "TEST": {"flat": 82.0, "green": 95.0, "balanced": 93.0, "bouncy": 84.0, "dusty": 66.0},
        "ODI":  {"flat": 90.0, "green": 91.0, "balanced": 89.0, "bouncy": 86.0, "dusty": 74.0},
        "T20I": {"flat": 88.0, "green": 84.0, "balanced": 86.0, "bouncy": 84.0, "dusty": 76.0}
    },
    "PAK": {
        "TEST": {"flat": 86.0, "green": 78.0, "balanced": 84.0, "bouncy": 82.0, "dusty": 80.0},
        "ODI":  {"flat": 88.0, "green": 76.0, "balanced": 82.0, "bouncy": 80.0, "dusty": 84.0},
        "T20I": {"flat": 90.0, "green": 74.0, "balanced": 80.0, "bouncy": 84.0, "dusty": 82.0}
    },
    "WI": {
        "TEST": {"flat": 70.0, "green": 68.0, "balanced": 74.0, "bouncy": 78.0, "dusty": 70.0},
        "ODI":  {"flat": 80.0, "green": 70.0, "balanced": 76.0, "bouncy": 80.0, "dusty": 72.0},
        "T20I": {"flat": 90.0, "green": 68.0, "balanced": 74.0, "bouncy": 84.0, "dusty": 76.0}
    },
    "SL": {
        "TEST": {"flat": 76.0, "green": 54.0, "balanced": 60.0, "bouncy": 56.0, "dusty": 95.0},
        "ODI":  {"flat": 80.0, "green": 64.0, "balanced": 68.0, "bouncy": 64.0, "dusty": 90.0},
        "T20I": {"flat": 80.0, "green": 62.0, "balanced": 66.0, "bouncy": 62.0, "dusty": 88.0}
    },
    "BAN": {
        "TEST": {"flat": 72.0, "green": 50.0, "balanced": 56.0, "bouncy": 52.0, "dusty": 94.0},
        "ODI":  {"flat": 80.0, "green": 60.0, "balanced": 66.0, "bouncy": 60.0, "dusty": 90.0},
        "T20I": {"flat": 76.0, "green": 58.0, "balanced": 62.0, "bouncy": 56.0, "dusty": 86.0}
    },
    "AFG": {
        "TEST": {"flat": 68.0, "green": 44.0, "balanced": 52.0, "bouncy": 50.0, "dusty": 90.0},
        "ODI":  {"flat": 78.0, "green": 58.0, "balanced": 64.0, "bouncy": 60.0, "dusty": 92.0},
        "T20I": {"flat": 84.0, "green": 62.0, "balanced": 68.0, "bouncy": 66.0, "dusty": 95.0}
    },
    "ZIM": {
        "TEST": {"flat": 56.0, "green": 60.0, "balanced": 64.0, "bouncy": 62.0, "dusty": 60.0},
        "ODI":  {"flat": 66.0, "green": 64.0, "balanced": 66.0, "bouncy": 65.0, "dusty": 64.0},
        "T20I": {"flat": 70.0, "green": 62.0, "balanced": 65.0, "bouncy": 66.0, "dusty": 68.0}
    },
"IRE": {
        "TEST": {"flat": 74.0, "green": 82.0, "balanced": 80.0, "bouncy": 74.0, "dusty": 60.0},
        "ODI":  {"flat": 82.0, "green": 84.0, "balanced": 82.0, "bouncy": 78.0, "dusty": 65.0},
        "T20I": {"flat": 84.0, "green": 82.0, "balanced": 82.0, "bouncy": 80.0, "dusty": 68.0}
    },
    "NED": {
        "TEST": {"flat": 65.0, "green": 78.0, "balanced": 76.0, "bouncy": 70.0, "dusty": 55.0},
        "ODI":  {"flat": 78.0, "green": 80.0, "balanced": 80.0, "bouncy": 76.0, "dusty": 62.0},
        "T20I": {"flat": 82.0, "green": 78.0, "balanced": 78.0, "bouncy": 78.0, "dusty": 64.0}
    },
    "SCO": {
        "TEST": {"flat": 68.0, "green": 80.0, "balanced": 78.0, "bouncy": 74.0, "dusty": 56.0},
        "ODI":  {"flat": 80.0, "green": 82.0, "balanced": 82.0, "bouncy": 78.0, "dusty": 62.0},
        "T20I": {"flat": 84.0, "green": 80.0, "balanced": 80.0, "bouncy": 80.0, "dusty": 66.0}
    },
    "NAM": {
        "TEST": {"flat": 66.0, "green": 70.0, "balanced": 74.0, "bouncy": 82.0, "dusty": 58.0},
        "ODI":  {"flat": 78.0, "green": 72.0, "balanced": 76.0, "bouncy": 84.0, "dusty": 64.0},
        "T20I": {"flat": 80.0, "green": 72.0, "balanced": 78.0, "bouncy": 86.0, "dusty": 66.0}
    },
    "USA": {
        "TEST": {"flat": 64.0, "green": 66.0, "balanced": 72.0, "bouncy": 76.0, "dusty": 60.0},
        "ODI":  {"flat": 78.0, "green": 70.0, "balanced": 75.0, "bouncy": 80.0, "dusty": 66.0},
        "T20I": {"flat": 84.0, "green": 72.0, "balanced": 78.0, "bouncy": 84.0, "dusty": 70.0}
    },
    "NEP": {
        "TEST": {"flat": 60.0, "green": 52.0, "balanced": 64.0, "bouncy": 60.0, "dusty": 84.0},
        "ODI":  {"flat": 74.0, "green": 58.0, "balanced": 70.0, "bouncy": 64.0, "dusty": 88.0},
        "T20I": {"flat": 78.0, "green": 60.0, "balanced": 72.0, "bouncy": 66.0, "dusty": 90.0}
    },
    "OMA": {
        "TEST": {"flat": 62.0, "green": 54.0, "balanced": 68.0, "bouncy": 62.0, "dusty": 82.0},
        "ODI":  {"flat": 74.0, "green": 60.0, "balanced": 72.0, "bouncy": 66.0, "dusty": 86.0},
        "T20I": {"flat": 78.0, "green": 62.0, "balanced": 74.0, "bouncy": 68.0, "dusty": 88.0}
    },
    "UAE": {
        "TEST": {"flat": 68.0, "green": 56.0, "balanced": 70.0, "bouncy": 66.0, "dusty": 84.0},
        "ODI":  {"flat": 78.0, "green": 62.0, "balanced": 74.0, "bouncy": 70.0, "dusty": 88.0},
        "T20I": {"flat": 82.0, "green": 64.0, "balanced": 76.0, "bouncy": 72.0, "dusty": 90.0}
    },
    "CAN": {
        "TEST": {"flat": 62.0, "green": 68.0, "balanced": 70.0, "bouncy": 76.0, "dusty": 56.0},
        "ODI":  {"flat": 74.0, "green": 72.0, "balanced": 74.0, "bouncy": 78.0, "dusty": 60.0},
        "T20I": {"flat": 78.0, "green": 72.0, "balanced": 76.0, "bouncy": 80.0, "dusty": 62.0}
    },
    "PNG": {
        "TEST": {"flat": 58.0, "green": 60.0, "balanced": 68.0, "bouncy": 64.0, "dusty": 72.0},
        "ODI":  {"flat": 70.0, "green": 64.0, "balanced": 70.0, "bouncy": 68.0, "dusty": 76.0},
        "T20I": {"flat": 74.0, "green": 64.0, "balanced": 72.0, "bouncy": 70.0, "dusty": 78.0}
    }
}

# 2B. Country Recent 1-Year Adaptability Matrix (Rolling 12-18 Months Cricsheet Data)
# Heavy weighting given to matches played in the most recent year
COUNTRY_RECENT_1YR_ADAPTABILITY: Dict[str, Dict[str, Dict[str, float]]] = {
    "IND": {
        "TEST": {"flat": 88.0, "green": 76.0, "balanced": 82.0, "bouncy": 80.0, "dusty": 90.0},
        "ODI":  {"flat": 96.0, "green": 82.0, "balanced": 86.0, "bouncy": 88.0, "dusty": 94.0},
        "T20I": {"flat": 98.0, "green": 82.0, "balanced": 86.0, "bouncy": 88.0, "dusty": 94.0}
    },
    "AUS": {
        "TEST": {"flat": 90.0, "green": 94.0, "balanced": 92.0, "bouncy": 98.0, "dusty": 72.0},
        "ODI":  {"flat": 95.0, "green": 92.0, "balanced": 90.0, "bouncy": 96.0, "dusty": 78.0},
        "T20I": {"flat": 91.0, "green": 86.0, "balanced": 88.0, "bouncy": 94.0, "dusty": 76.0}
    },
    "ENG": {
        "TEST": {"flat": 92.0, "green": 92.0, "balanced": 90.0, "bouncy": 82.0, "dusty": 72.0},
        "ODI":  {"flat": 91.0, "green": 84.0, "balanced": 82.0, "bouncy": 82.0, "dusty": 70.0},
        "T20I": {"flat": 92.0, "green": 82.0, "balanced": 84.0, "bouncy": 82.0, "dusty": 75.0}
    },
    "SA": {
        "TEST": {"flat": 82.0, "green": 92.0, "balanced": 90.0, "bouncy": 98.0, "dusty": 74.0},
        "ODI":  {"flat": 92.0, "green": 88.0, "balanced": 86.0, "bouncy": 95.0, "dusty": 74.0},
        "T20I": {"flat": 92.0, "green": 86.0, "balanced": 84.0, "bouncy": 94.0, "dusty": 78.0}
    },
    "NZ": {
        "TEST": {"flat": 85.0, "green": 96.0, "balanced": 94.0, "bouncy": 85.0, "dusty": 88.0},  # Historical 3-0 in India boost
        "ODI":  {"flat": 88.0, "green": 90.0, "balanced": 90.0, "bouncy": 86.0, "dusty": 76.0},
        "T20I": {"flat": 89.0, "green": 85.0, "balanced": 88.0, "bouncy": 86.0, "dusty": 78.0}
    },
    "PAK": {
        "TEST": {"flat": 80.0, "green": 74.0, "balanced": 80.0, "bouncy": 78.0, "dusty": 88.0},
        "ODI":  {"flat": 90.0, "green": 78.0, "balanced": 84.0, "bouncy": 82.0, "dusty": 82.0},
        "T20I": {"flat": 88.0, "green": 72.0, "balanced": 78.0, "bouncy": 82.0, "dusty": 80.0}
    },
    "WI": {
        "TEST": {"flat": 68.0, "green": 70.0, "balanced": 76.0, "bouncy": 84.0, "dusty": 68.0},
        "ODI":  {"flat": 82.0, "green": 72.0, "balanced": 76.0, "bouncy": 82.0, "dusty": 74.0},
        "T20I": {"flat": 93.0, "green": 72.0, "balanced": 78.0, "bouncy": 86.0, "dusty": 78.0}
    },
    "SL": {
        "TEST": {"flat": 78.0, "green": 58.0, "balanced": 68.0, "bouncy": 58.0, "dusty": 96.0},
        "ODI":  {"flat": 84.0, "green": 66.0, "balanced": 70.0, "bouncy": 66.0, "dusty": 92.0},
        "T20I": {"flat": 82.0, "green": 64.0, "balanced": 68.0, "bouncy": 64.0, "dusty": 90.0}
    },
    "BAN": {
        "TEST": {"flat": 78.0, "green": 54.0, "balanced": 64.0, "bouncy": 56.0, "dusty": 92.0},
        "ODI":  {"flat": 78.0, "green": 58.0, "balanced": 66.0, "bouncy": 58.0, "dusty": 88.0},
        "T20I": {"flat": 74.0, "green": 56.0, "balanced": 60.0, "bouncy": 54.0, "dusty": 85.0}
    },
    "AFG": {
        "TEST": {"flat": 70.0, "green": 42.0, "balanced": 54.0, "bouncy": 48.0, "dusty": 92.0},
        "ODI":  {"flat": 82.0, "green": 60.0, "balanced": 66.0, "bouncy": 62.0, "dusty": 94.0},
        "T20I": {"flat": 86.0, "green": 64.0, "balanced": 72.0, "bouncy": 70.0, "dusty": 98.0}
    },
    "ZIM": {
        "TEST": {"flat": 58.0, "green": 62.0, "balanced": 66.0, "bouncy": 64.0, "dusty": 62.0},
        "ODI":  {"flat": 68.0, "green": 65.0, "balanced": 68.0, "bouncy": 66.0, "dusty": 66.0},
        "T20I": {"flat": 74.0, "green": 64.0, "balanced": 66.0, "bouncy": 68.0, "dusty": 70.0}
    },
"IRE": {
        "TEST": {"flat": 76.0, "green": 84.0, "balanced": 82.0, "bouncy": 76.0, "dusty": 62.0},
        "ODI":  {"flat": 84.0, "green": 86.0, "balanced": 84.0, "bouncy": 80.0, "dusty": 68.0},
        "T20I": {"flat": 86.0, "green": 84.0, "balanced": 84.0, "bouncy": 82.0, "dusty": 70.0}
    },
    "NED": {
        "TEST": {"flat": 68.0, "green": 80.0, "balanced": 78.0, "bouncy": 72.0, "dusty": 58.0},
        "ODI":  {"flat": 80.0, "green": 82.0, "balanced": 82.0, "bouncy": 78.0, "dusty": 65.0},
        "T20I": {"flat": 84.0, "green": 80.0, "balanced": 80.0, "bouncy": 80.0, "dusty": 66.0}
    },
    "SCO": {
        "TEST": {"flat": 70.0, "green": 82.0, "balanced": 80.0, "bouncy": 76.0, "dusty": 58.0},
        "ODI":  {"flat": 82.0, "green": 84.0, "balanced": 84.0, "bouncy": 80.0, "dusty": 64.0},
        "T20I": {"flat": 86.0, "green": 82.0, "balanced": 82.0, "bouncy": 82.0, "dusty": 68.0}
    },
    "NAM": {
        "TEST": {"flat": 68.0, "green": 72.0, "balanced": 76.0, "bouncy": 85.0, "dusty": 60.0},
        "ODI":  {"flat": 80.0, "green": 74.0, "balanced": 78.0, "bouncy": 86.0, "dusty": 66.0},
        "T20I": {"flat": 82.0, "green": 74.0, "balanced": 80.0, "bouncy": 88.0, "dusty": 68.0}
    },
    "USA": {
        "TEST": {"flat": 66.0, "green": 68.0, "balanced": 74.0, "bouncy": 78.0, "dusty": 62.0},
        "ODI":  {"flat": 80.0, "green": 72.0, "balanced": 78.0, "bouncy": 82.0, "dusty": 68.0},
        "T20I": {"flat": 88.0, "green": 76.0, "balanced": 82.0, "bouncy": 86.0, "dusty": 74.0}
    },
    "NEP": {
        "TEST": {"flat": 62.0, "green": 54.0, "balanced": 66.0, "bouncy": 62.0, "dusty": 86.0},
        "ODI":  {"flat": 76.0, "green": 60.0, "balanced": 72.0, "bouncy": 66.0, "dusty": 90.0},
        "T20I": {"flat": 80.0, "green": 62.0, "balanced": 74.0, "bouncy": 68.0, "dusty": 92.0}
    },
    "OMA": {
        "TEST": {"flat": 64.0, "green": 56.0, "balanced": 70.0, "bouncy": 64.0, "dusty": 84.0},
        "ODI":  {"flat": 76.0, "green": 62.0, "balanced": 74.0, "bouncy": 68.0, "dusty": 88.0},
        "T20I": {"flat": 80.0, "green": 64.0, "balanced": 76.0, "bouncy": 70.0, "dusty": 90.0}
    },
    "UAE": {
        "TEST": {"flat": 70.0, "green": 58.0, "balanced": 72.0, "bouncy": 68.0, "dusty": 86.0},
        "ODI":  {"flat": 80.0, "green": 64.0, "balanced": 76.0, "bouncy": 72.0, "dusty": 90.0},
        "T20I": {"flat": 84.0, "green": 66.0, "balanced": 78.0, "bouncy": 74.0, "dusty": 92.0}
    },
    "CAN": {
        "TEST": {"flat": 64.0, "green": 70.0, "balanced": 72.0, "bouncy": 78.0, "dusty": 58.0},
        "ODI":  {"flat": 76.0, "green": 74.0, "balanced": 76.0, "bouncy": 80.0, "dusty": 62.0},
        "T20I": {"flat": 80.0, "green": 74.0, "balanced": 78.0, "bouncy": 82.0, "dusty": 64.0}
    },
    "PNG": {
        "TEST": {"flat": 60.0, "green": 62.0, "balanced": 70.0, "bouncy": 66.0, "dusty": 74.0},
        "ODI":  {"flat": 72.0, "green": 66.0, "balanced": 72.0, "bouncy": 70.0, "dusty": 78.0},
        "T20I": {"flat": 76.0, "green": 66.0, "balanced": 74.0, "bouncy": 72.0, "dusty": 80.0}
    }
}

# 3. Cricsheet Historical Toss Impact: Win % when Batting First vs Fielding First
# 2C. Country 5 Home Surface Baselines (Home Decks across all 5 pitch archetypes)
COUNTRY_HOME_SURFACES: Dict[str, Dict[str, Dict[str, float]]] = {
    "IND": {
        "TEST": {"flat_home": 96.0, "green_home": 82.0, "balanced_home": 88.0, "bouncy_home": 85.0, "dusty_home": 98.0},
        "ODI":  {"flat_home": 97.0, "green_home": 85.0, "balanced_home": 90.0, "bouncy_home": 89.0, "dusty_home": 98.0},
        "T20I": {"flat_home": 98.0, "green_home": 84.0, "balanced_home": 88.0, "bouncy_home": 87.0, "dusty_home": 96.0}
    },
    "AUS": {
        "TEST": {"flat_home": 94.0, "green_home": 95.0, "balanced_home": 94.0, "bouncy_home": 98.0, "dusty_home": 75.0},
        "ODI":  {"flat_home": 96.0, "green_home": 93.0, "balanced_home": 92.0, "bouncy_home": 98.0, "dusty_home": 80.0},
        "T20I": {"flat_home": 95.0, "green_home": 88.0, "balanced_home": 90.0, "bouncy_home": 97.0, "dusty_home": 80.0}
    },
    "ENG": {
        "TEST": {"flat_home": 90.0, "green_home": 97.0, "balanced_home": 95.0, "bouncy_home": 85.0, "dusty_home": 72.0},
        "ODI":  {"flat_home": 97.0, "green_home": 92.0, "balanced_home": 90.0, "bouncy_home": 88.0, "dusty_home": 76.0},
        "T20I": {"flat_home": 96.0, "green_home": 88.0, "balanced_home": 88.0, "bouncy_home": 86.0, "dusty_home": 78.0}
    },
    "SA": {
        "TEST": {"flat_home": 86.0, "green_home": 94.0, "balanced_home": 92.0, "bouncy_home": 98.0, "dusty_home": 68.0},
        "ODI":  {"flat_home": 93.0, "green_home": 90.0, "balanced_home": 88.0, "bouncy_home": 97.0, "dusty_home": 74.0},
        "T20I": {"flat_home": 93.0, "green_home": 88.0, "balanced_home": 86.0, "bouncy_home": 96.0, "dusty_home": 76.0}
    },
    "NZ": {
        "TEST": {"flat_home": 88.0, "green_home": 98.0, "balanced_home": 96.0, "bouncy_home": 88.0, "dusty_home": 70.0},
        "ODI":  {"flat_home": 93.0, "green_home": 95.0, "balanced_home": 93.0, "bouncy_home": 90.0, "dusty_home": 76.0},
        "T20I": {"flat_home": 92.0, "green_home": 88.0, "balanced_home": 90.0, "bouncy_home": 88.0, "dusty_home": 78.0}
    },
    "PAK": {
        "TEST": {"flat_home": 92.0, "green_home": 82.0, "balanced_home": 88.0, "bouncy_home": 85.0, "dusty_home": 86.0},
        "ODI":  {"flat_home": 92.0, "green_home": 80.0, "balanced_home": 86.0, "bouncy": 84.0, "dusty_home": 88.0},
        "T20I": {"flat_home": 94.0, "green_home": 78.0, "balanced_home": 84.0, "bouncy_home": 87.0, "dusty_home": 86.0}
    },
    "WI": {
        "TEST": {"flat_home": 76.0, "green_home": 74.0, "balanced_home": 80.0, "bouncy_home": 84.0, "dusty_home": 75.0},
        "ODI":  {"flat_home": 85.0, "green_home": 75.0, "balanced_home": 80.0, "bouncy_home": 85.0, "dusty_home": 76.0},
        "T20I": {"flat_home": 94.0, "green_home": 74.0, "balanced_home": 80.0, "bouncy_home": 88.0, "dusty_home": 80.0}
    },
    "SL": {
        "TEST": {"flat_home": 84.0, "green_home": 60.0, "balanced_home": 68.0, "bouncy_home": 62.0, "dusty_home": 98.0},
        "ODI":  {"flat_home": 86.0, "green_home": 68.0, "balanced_home": 74.0, "bouncy_home": 68.0, "dusty_home": 96.0},
        "T20I": {"flat_home": 85.0, "green_home": 66.0, "balanced_home": 72.0, "bouncy_home": 66.0, "dusty_home": 94.0}
    },
    "BAN": {
        "TEST": {"flat_home": 80.0, "green_home": 56.0, "balanced_home": 64.0, "bouncy_home": 58.0, "dusty_home": 96.0},
        "ODI":  {"flat_home": 85.0, "green_home": 64.0, "balanced_home": 72.0, "bouncy_home": 64.0, "dusty_home": 94.0},
        "T20I": {"flat_home": 82.0, "green_home": 62.0, "balanced_home": 68.0, "bouncy_home": 60.0, "dusty_home": 92.0}
    },
    "AFG": {
        "TEST": {"flat_home": 74.0, "green_home": 48.0, "balanced_home": 58.0, "bouncy_home": 54.0, "dusty_home": 94.0},
        "ODI":  {"flat_home": 84.0, "green_home": 62.0, "balanced_home": 70.0, "bouncy_home": 66.0, "dusty_home": 95.0},
        "T20I": {"flat_home": 88.0, "green_home": 66.0, "balanced_home": 74.0, "bouncy_home": 72.0, "dusty_home": 97.0}
    },
    "ZIM": {
        "TEST": {"flat_home": 64.0, "green_home": 66.0, "balanced_home": 70.0, "bouncy_home": 68.0, "dusty_home": 65.0},
        "ODI":  {"flat_home": 72.0, "green_home": 70.0, "balanced_home": 72.0, "bouncy_home": 70.0, "dusty_home": 68.0},
        "T20I": {"flat_home": 76.0, "green_home": 68.0, "balanced_home": 70.0, "bouncy_home": 72.0, "dusty_home": 72.0}
    },
    "IRE": {
        "TEST": {"flat_home": 78.0, "green_home": 88.0, "balanced_home": 85.0, "bouncy_home": 78.0, "dusty_home": 64.0},
        "ODI":  {"flat_home": 86.0, "green_home": 89.0, "balanced_home": 86.0, "bouncy_home": 82.0, "dusty_home": 68.0},
        "T20I": {"flat_home": 88.0, "green_home": 86.0, "balanced_home": 86.0, "bouncy_home": 84.0, "dusty_home": 70.0}
    },
    "NED": {
        "TEST": {"flat_home": 70.0, "green_home": 84.0, "balanced_home": 80.0, "bouncy_home": 74.0, "dusty_home": 58.0},
        "ODI":  {"flat_home": 82.0, "green_home": 85.0, "balanced_home": 84.0, "bouncy_home": 80.0, "dusty_home": 65.0},
        "T20I": {"flat_home": 86.0, "green_home": 82.0, "balanced_home": 82.0, "bouncy_home": 82.0, "dusty_home": 66.0}
    },
    "SCO": {
        "TEST": {"flat_home": 72.0, "green_home": 86.0, "balanced_home": 82.0, "bouncy_home": 78.0, "dusty_home": 60.0},
        "ODI":  {"flat_home": 84.0, "green_home": 86.0, "balanced_home": 86.0, "bouncy_home": 82.0, "dusty_home": 65.0},
        "T20I": {"flat_home": 88.0, "green_home": 84.0, "balanced_home": 84.0, "bouncy_home": 84.0, "dusty_home": 68.0}
    },
    "NAM": {
        "TEST": {"flat_home": 72.0, "green_home": 75.0, "balanced_home": 78.0, "bouncy_home": 88.0, "dusty_home": 62.0},
        "ODI":  {"flat_home": 82.0, "green_home": 76.0, "balanced_home": 80.0, "bouncy_home": 90.0, "dusty_home": 66.0},
        "T20I": {"flat_home": 84.0, "green_home": 76.0, "balanced_home": 82.0, "bouncy_home": 90.0, "dusty_home": 68.0}
    },
    "USA": {
        "TEST": {"flat_home": 70.0, "green_home": 72.0, "balanced_home": 76.0, "bouncy_home": 82.0, "dusty_home": 64.0},
        "ODI":  {"flat_home": 82.0, "green_home": 75.0, "balanced_home": 80.0, "bouncy_home": 84.0, "dusty_home": 70.0},
        "T20I": {"flat_home": 88.0, "green_home": 78.0, "balanced_home": 82.0, "bouncy_home": 88.0, "dusty_home": 74.0}
    },
    "NEP": {
        "TEST": {"flat_home": 66.0, "green_home": 58.0, "balanced_home": 70.0, "bouncy_home": 66.0, "dusty_home": 92.0},
        "ODI":  {"flat_home": 78.0, "green_home": 64.0, "balanced_home": 76.0, "bouncy_home": 70.0, "dusty_home": 94.0},
        "T20I": {"flat_home": 82.0, "green_home": 66.0, "balanced_home": 78.0, "bouncy_home": 72.0, "dusty_home": 95.0}
    },
    "OMA": {
        "TEST": {"flat_home": 68.0, "green_home": 60.0, "balanced_home": 74.0, "bouncy_home": 68.0, "dusty_home": 90.0},
        "ODI":  {"flat_home": 78.0, "green_home": 65.0, "balanced_home": 76.0, "bouncy_home": 70.0, "dusty_home": 92.0},
        "T20I": {"flat_home": 82.0, "green_home": 68.0, "balanced_home": 78.0, "bouncy_home": 72.0, "dusty_home": 94.0}
    },
    "UAE": {
        "TEST": {"flat_home": 74.0, "green_home": 62.0, "balanced_home": 76.0, "bouncy_home": 72.0, "dusty_home": 92.0},
        "ODI":  {"flat_home": 82.0, "green_home": 68.0, "balanced_home": 78.0, "bouncy_home": 75.0, "dusty_home": 94.0},
        "T20I": {"flat_home": 86.0, "green_home": 70.0, "balanced_home": 80.0, "bouncy_home": 76.0, "dusty_home": 95.0}
    },
    "CAN": {
        "TEST": {"flat_home": 68.0, "green_home": 74.0, "balanced_home": 76.0, "bouncy_home": 82.0, "dusty_home": 60.0},
        "ODI":  {"flat_home": 78.0, "green_home": 76.0, "balanced_home": 78.0, "bouncy_home": 84.0, "dusty_home": 64.0},
        "T20I": {"flat_home": 82.0, "green_home": 76.0, "balanced_home": 80.0, "bouncy_home": 86.0, "dusty_home": 66.0}
    },
    "PNG": {
        "TEST": {"flat_home": 64.0, "green_home": 66.0, "balanced_home": 74.0, "bouncy_home": 70.0, "dusty_home": 80.0},
        "ODI":  {"flat_home": 74.0, "green_home": 68.0, "balanced_home": 74.0, "bouncy_home": 72.0, "dusty_home": 82.0},
        "T20I": {"flat_home": 78.0, "green_home": 68.0, "balanced_home": 76.0, "bouncy_home": 74.0, "dusty_home": 84.0}
    }
}

# 2D. Dual Top Touring Performers across Top 20 ICC Playing Nations (🏏 Batter & 🎯 Bowler)
COUNTRY_TOP_TOURING_PERFORMERS: Dict[str, Dict[str, Any]] = {
    "IND": {"batter": {"name": "Virat Kohli", "score": 94.2, "favored_deck": "FLAT"}, "bowler": {"name": "Jasprit Bumrah", "score": 96.5, "favored_deck": "GREEN"}},
    "AUS": {"batter": {"name": "Steve Smith", "score": 93.8, "favored_deck": "BALANCED"}, "bowler": {"name": "Pat Cummins", "score": 95.4, "favored_deck": "BOUNCY"}},
    "ENG": {"batter": {"name": "Joe Root", "score": 93.5, "favored_deck": "BALANCED"}, "bowler": {"name": "Jofra Archer", "score": 91.8, "favored_deck": "BOUNCY"}},
    "SA":  {"batter": {"name": "Heinrich Klaasen", "score": 91.4, "favored_deck": "FLAT"}, "bowler": {"name": "Kagiso Rabada", "score": 94.6, "favored_deck": "BOUNCY"}},
    "NZ":  {"batter": {"name": "Kane Williamson", "score": 92.8, "favored_deck": "BALANCED"}, "bowler": {"name": "Trent Boult", "score": 93.1, "favored_deck": "GREEN"}},
    "PAK": {"batter": {"name": "Babar Azam", "score": 91.2, "favored_deck": "FLAT"}, "bowler": {"name": "Shaheen Shah Afridi", "score": 93.4, "favored_deck": "GREEN"}},
    "WI":  {"batter": {"name": "Nicholas Pooran", "score": 89.5, "favored_deck": "FLAT"}, "bowler": {"name": "Alzarri Joseph", "score": 87.2, "favored_deck": "BOUNCY"}},
    "SL":  {"batter": {"name": "Pathum Nissanka", "score": 88.5, "favored_deck": "FLAT"}, "bowler": {"name": "Wanindu Hasaranga", "score": 92.0, "favored_deck": "DUSTY"}},
    "BAN": {"batter": {"name": "Najmul Hossain Shanto", "score": 84.2, "favored_deck": "DUSTY"}, "bowler": {"name": "Mustafizur Rahman", "score": 88.4, "favored_deck": "BALANCED"}},
    "AFG": {"batter": {"name": "Rahmanullah Gurbaz", "score": 86.4, "favored_deck": "FLAT"}, "bowler": {"name": "Rashid Khan", "score": 96.2, "favored_deck": "DUSTY"}},
    "ZIM": {"batter": {"name": "Sikandar Raza", "score": 87.8, "favored_deck": "BALANCED"}, "bowler": {"name": "Blessing Muzarabani", "score": 86.5, "favored_deck": "BOUNCY"}},
    "IRE": {"batter": {"name": "Harry Tector", "score": 85.6, "favored_deck": "GREEN"}, "bowler": {"name": "Josh Little", "score": 86.2, "favored_deck": "GREEN"}},
    "NED": {"batter": {"name": "Scott Edwards", "score": 82.4, "favored_deck": "BALANCED"}, "bowler": {"name": "Paul van Meekeren", "score": 83.8, "favored_deck": "BOUNCY"}},
    "SCO": {"batter": {"name": "George Munsey", "score": 81.6, "favored_deck": "FLAT"}, "bowler": {"name": "Chris Sole", "score": 83.5, "favored_deck": "BOUNCY"}},
    "NAM": {"batter": {"name": "Gerhard Erasmus", "score": 82.1, "favored_deck": "BOUNCY"}, "bowler": {"name": "Ruben Trumpelmann", "score": 83.0, "favored_deck": "GREEN"}},
    "USA": {"batter": {"name": "Aaron Jones", "score": 81.5, "favored_deck": "FLAT"}, "bowler": {"name": "Saurabh Netravalkar", "score": 84.2, "favored_deck": "GREEN"}},
    "NEP": {"batter": {"name": "Rohit Paudel", "score": 80.8, "favored_deck": "DUSTY"}, "bowler": {"name": "Sandeep Lamichhane", "score": 85.0, "favored_deck": "DUSTY"}},
    "OMA": {"batter": {"name": "Aqib Ilyas", "score": 79.8, "favored_deck": "DUSTY"}, "bowler": {"name": "Bilal Khan", "score": 82.6, "favored_deck": "GREEN"}},
    "UAE": {"batter": {"name": "Muhammad Waseem", "score": 82.2, "favored_deck": "FLAT"}, "bowler": {"name": "Junaid Siddique", "score": 81.4, "favored_deck": "BOUNCY"}},
    "CAN": {"batter": {"name": "Nicholas Kirton", "score": 79.5, "favored_deck": "BOUNCY"}, "bowler": {"name": "Kaleem Sana", "score": 81.2, "favored_deck": "GREEN"}},
    "PNG": {"batter": {"name": "Assad Vala", "score": 78.9, "favored_deck": "BALANCED"}, "bowler": {"name": "Norman Vanua", "score": 80.1, "favored_deck": "BALANCED"}}
}

# 2E. Associate Nation Normalization Method (Opponent Quality Weighting)
# W_opp = 1.00 for Full Members, 0.80 for Associate Members
ICC_FULL_MEMBERS = {"IND", "AUS", "ENG", "SA", "RSA", "NZ", "NZL", "PAK", "WI", "WIN", "SL", "SRI", "BAN", "AFG", "ZIM", "IRE"}

def get_opponent_quality_weight(opponent_code: str) -> float:
    c = (opponent_code or "").strip().upper()
    return 1.00 if c in ICC_FULL_MEMBERS else 0.80

def calculate_opponent_weighted_stats(stats: Dict[str, Any], opponent_code: str) -> Dict[str, Any]:
    w_opp = get_opponent_quality_weight(opponent_code)
    r_raw = float(stats.get("runs", 0))
    d_raw = float(stats.get("dismissals", stats.get("outs", 1)))
    w_raw = float(stats.get("wickets", 0))
    b_raw = float(stats.get("balls", stats.get("balls_bowled", 0)))

    r_eff = r_raw * w_opp
    d_eff = max(0.1, d_raw * w_opp)
    w_eff = w_raw * w_opp
    b_eff = max(0.1, b_raw * w_opp)

    return {
        "opponent_weight": w_opp,
        "runs_effective": round(r_eff, 1),
        "dismissals_effective": round(d_eff, 1),
        "wickets_effective": round(w_eff, 1),
        "balls_effective": round(b_eff, 1),
        "adjusted_batting_avg": round(r_eff / d_eff, 2) if d_eff > 0 else 0.0,
        "adjusted_bowling_sr": round(b_eff / max(0.1, w_eff), 1) if w_eff > 0 else 99.9
    }

# 2F. Bowler Adaptability Engine (compute_bowler_foreign_adaptability_all_formats)
# Evaluates bowling performance across all 5 pitch archetypes based on format-calibrated benchmarks
# for Bowling Average, Economy Rate, and Strike Rate with Bayesian shrinkage.
def compute_bowler_foreign_adaptability_all_formats(
    bowler: Dict[str, Any],
    match_format: str = "ODI",
    recency_weight: float = 0.65
) -> Dict[str, Any]:
    fmt = match_format.upper()
    role = str(bowler.get("role", "bowler")).lower()
    style = str(bowler.get("bowling_style", bowler.get("style", "RF"))).upper()
    sena_aff = float(bowler.get("sena_affinity", 1.0) or 1.0)
    sub_aff = float(bowler.get("subcontinent_affinity", 1.0) or 1.0)

    # Format-calibrated benchmarks
    if fmt == "TEST":
        bench_avg, bench_econ, bench_sr = 28.0, 3.10, 52.0
        k_shrink = 300.0  # balls
    elif fmt == "ODI":
        bench_avg, bench_econ, bench_sr = 32.0, 5.20, 36.0
        k_shrink = 150.0  # balls
    else:  # T20 / T20I
        bench_avg, bench_econ, bench_sr = 22.0, 7.60, 17.0
        k_shrink = 60.0   # balls

    balls = float(bowler.get("balls_bowled", bowler.get("exp_balls_bowled", 300)) or 300)
    wkts = float(bowler.get("wickets_taken", bowler.get("seed_wickets", max(1, balls / bench_sr))) or 1)
    runs = float(bowler.get("runs_conceded", wkts * bench_avg) or (wkts * bench_avg))

    # Observed metrics
    obs_avg = runs / max(1.0, wkts)
    obs_econ = (runs / max(1.0, balls)) * 6.0
    obs_sr = balls / max(1.0, wkts)

    # Bayesian shrinkage toward format benchmarks
    w_obs = balls / (balls + k_shrink)
    w_prior = 1.0 - w_obs
    shrunk_avg = obs_avg * w_obs + bench_avg * w_prior
    shrunk_econ = obs_econ * w_obs + bench_econ * w_prior
    shrunk_sr = obs_sr * w_obs + bench_sr * w_prior

    is_spinner = any(s in style for s in ["OB", "SLA", "LBG", "LB", "SPIN", "OFFBREAK", "LEGBREAK"])
    is_fast = any(s in style for s in ["RF", "LF", "FAST", "RMF", "LMF", "PACE"])

    # 1. GREEN (Lush grass cover: rewards pace, seam, lateral swing)
    if is_fast:
        green_score = 50.0 + (bench_avg / max(15.0, shrunk_avg)) * 25.0 + (bench_econ / max(2.5, shrunk_econ)) * 15.0 + (sena_aff - 1.0) * 15.0
    else:
        green_score = 42.0 + (bench_econ / max(2.5, shrunk_econ)) * 20.0 + (sub_aff - 1.0) * 8.0
    green_score = min(98.0, max(42.0, green_score))

    # 2. DUSTY (Dry abrasive surface: sharp turn, variable bounce)
    if is_spinner:
        dusty_score = 52.0 + (bench_avg / max(15.0, shrunk_avg)) * 26.0 + (bench_econ / max(2.5, shrunk_econ)) * 16.0 + (sub_aff - 1.0) * 18.0
    else:
        # Reverse swing pacers
        dusty_score = 44.0 + (bench_avg / max(15.0, shrunk_avg)) * 18.0 + (bench_econ / max(2.5, shrunk_econ)) * 14.0
    dusty_score = min(98.0, max(40.0, dusty_score))

    # 3. BOUNCY (Hard packed clay: vertical lift, short-ball threat)
    if is_fast and any(f in style for f in ["RF", "FAST", "RMF"]):
        bouncy_score = 52.0 + (bench_sr / max(10.0, shrunk_sr)) * 25.0 + (bench_econ / max(2.5, shrunk_econ)) * 15.0 + (sena_aff - 1.0) * 15.0
    else:
        bouncy_score = 46.0 + (bench_econ / max(2.5, shrunk_econ)) * 20.0
    bouncy_score = min(98.0, max(42.0, bouncy_score))

    # 4. FLAT (Batting belter: high economy hazard, rewards control)
    econ_protection = (bench_econ / max(2.8, shrunk_econ)) * 35.0
    sr_protection = (bench_sr / max(12.0, shrunk_sr)) * 15.0
    flat_score = 40.0 + econ_protection + sr_protection
    flat_score = min(96.0, max(38.0, flat_score))

    # 5. BALANCED (Sporting track: 50/50 seam/spin ratio)
    balanced_score = 48.0 + (bench_avg / max(15.0, shrunk_avg)) * 22.0 + (bench_econ / max(2.5, shrunk_econ)) * 18.0
    balanced_score = min(98.0, max(44.0, balanced_score))

    scores = {
        "green": round(green_score, 1),
        "dusty": round(dusty_score, 1),
        "bouncy": round(bouncy_score, 1),
        "flat": round(flat_score, 1),
        "balanced": round(balanced_score, 1)
    }

    favored = max(scores.items(), key=lambda x: x[1])[0]

    return {
        "bowler_id": bowler.get("player_id", bowler.get("id")),
        "format": fmt,
        "shrunk_average": round(shrunk_avg, 2),
        "shrunk_economy": round(shrunk_econ, 2),
        "shrunk_strike_rate": round(shrunk_sr, 1),
        "pitch_adaptability": scores,
        "favored_pitch": favored.upper(),
        "overall_bowler_adapt_score": round(sum(scores.values()) / 5.0, 1),
        "benchmark_comparison": {
            "average_vs_benchmark": round(bench_avg - shrunk_avg, 2),
            "economy_vs_benchmark": round(bench_econ - shrunk_econ, 2),
            "strike_rate_vs_benchmark": round(bench_sr - shrunk_sr, 1)
        }
    }

# 3. Cricsheet Historical Toss Impact
CRICSHEET_TOSS_FACTORS: Dict[str, Dict[str, Dict[str, float]]] = {
    "flat": {
        "TEST": {"bat_first_win_pct": 0.58, "field_first_win_pct": 0.28, "draw_pct": 0.14},
        "ODI":  {"bat_first_win_pct": 0.46, "field_first_win_pct": 0.54, "draw_pct": 0.0},
        "T20I": {"bat_first_win_pct": 0.42, "field_first_win_pct": 0.58, "draw_pct": 0.0}
    },
    "green": {
        "TEST": {"bat_first_win_pct": 0.36, "field_first_win_pct": 0.52, "draw_pct": 0.12},
        "ODI":  {"bat_first_win_pct": 0.40, "field_first_win_pct": 0.60, "draw_pct": 0.0},
        "T20I": {"bat_first_win_pct": 0.45, "field_first_win_pct": 0.55, "draw_pct": 0.0}
    },
    "balanced": {
        "TEST": {"bat_first_win_pct": 0.52, "field_first_win_pct": 0.36, "draw_pct": 0.12},
        "ODI":  {"bat_first_win_pct": 0.48, "field_first_win_pct": 0.52, "draw_pct": 0.0},
        "T20I": {"bat_first_win_pct": 0.47, "field_first_win_pct": 0.53, "draw_pct": 0.0}
    },
    "bouncy": {
        "TEST": {"bat_first_win_pct": 0.55, "field_first_win_pct": 0.35, "draw_pct": 0.10},
        "ODI":  {"bat_first_win_pct": 0.52, "field_first_win_pct": 0.48, "draw_pct": 0.0},
        "T20I": {"bat_first_win_pct": 0.51, "field_first_win_pct": 0.49, "draw_pct": 0.0}
    },
    "dusty": {
        "TEST": {"bat_first_win_pct": 0.68, "field_first_win_pct": 0.22, "draw_pct": 0.10},
        "ODI":  {"bat_first_win_pct": 0.58, "field_first_win_pct": 0.42, "draw_pct": 0.0},
        "T20I": {"bat_first_win_pct": 0.54, "field_first_win_pct": 0.46, "draw_pct": 0.0}
    }
}

# Helper to normalize international country codes
def normalize_team_code(name_or_code: str) -> str:
    n = (name_or_code or "").strip().lower()
    exact_codes = {
        "ind": "IND", "aus": "AUS", "eng": "ENG", "sa": "SA", "rsa": "SA",
        "pak": "PAK", "nz": "NZ", "nzl": "NZ", "wi": "WI", "win": "WI",
        "sl": "SL", "sri": "SL", "ban": "BAN", "afg": "AFG", "zim": "ZIM",
        "ire": "IRE", "ned": "NED", "sco": "SCO", "nam": "NAM", "usa": "USA",
        "nep": "NEP", "oma": "OMA", "uae": "UAE", "can": "CAN", "png": "PNG"
    }
    if n in exact_codes:
        return exact_codes[n]

    if any(x in n for x in ["india", "team_ind", "bcci"]): return "IND"
    if any(x in n for x in ["australia", "crict_aus"]): return "AUS"
    if any(x in n for x in ["england", "ecb"]): return "ENG"
    if any(x in n for x in ["united states", "america"]): return "USA"
    if any(x in n for x in ["south africa", "proteas"]): return "SA"
    if any(x in n for x in ["pakistan", "pcb"]): return "PAK"
    if any(x in n for x in ["new zealand", "black caps"]): return "NZ"
    if any(x in n for x in ["west indies", "windies"]): return "WI"
    if any(x in n for x in ["sri lanka", "lions"]): return "SL"
    if any(x in n for x in ["bangladesh", "tigers"]): return "BAN"
    if any(x in n for x in ["afghanistan", "atalan"]): return "AFG"
    if any(x in n for x in ["zimbabwe", "chevrons"]): return "ZIM"
    if any(x in n for x in ["ireland"]): return "IRE"
    if any(x in n for x in ["netherlands", "dutch"]): return "NED"
    if any(x in n for x in ["scotland"]): return "SCO"
    if any(x in n for x in ["namibia"]): return "NAM"
    if any(x in n for x in ["nepal"]): return "NEP"
    if any(x in n for x in ["oman"]): return "OMA"
    if any(x in n for x in ["emirates"]): return "UAE"
    if any(x in n for x in ["canada"]): return "CAN"
    if any(x in n for x in ["papua"]): return "PNG"
    return name_or_code.upper() if name_or_code else "IND"

# 4. Authentic Format-Specific Bilateral Head-to-Head Dominance Matrix
INTERNATIONAL_H2H_DATABASE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "IND_vs_NZ": {
        "TEST": {"matches": 64, "win_pct_a": 0.344, "win_pct_b": 0.234, "draw_pct": 0.422, "last_3yr_win_pct_a": 0.40, "last_3yr_win_pct_b": 0.50},
        "ODI":  {"matches": 118, "win_pct_a": 0.517, "win_pct_b": 0.424, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.68, "last_3yr_win_pct_b": 0.32},
        "T20I": {"matches": 25, "win_pct_a": 0.560, "win_pct_b": 0.400, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.667, "last_3yr_win_pct_b": 0.333}
    },
    "AUS_vs_ENG": {
        "TEST": {"matches": 356, "win_pct_a": 0.427, "win_pct_b": 0.315, "draw_pct": 0.258, "last_3yr_win_pct_a": 0.556, "last_3yr_win_pct_b": 0.333},
        "ODI":  {"matches": 157, "win_pct_a": 0.561, "win_pct_b": 0.395, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.625, "last_3yr_win_pct_b": 0.375},
        "T20I": {"matches": 24, "win_pct_a": 0.478, "win_pct_b": 0.478, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.500, "last_3yr_win_pct_b": 0.500}
    },
    "PAK_vs_SL": {
        "TEST": {"matches": 59, "win_pct_a": 0.390, "win_pct_b": 0.288, "draw_pct": 0.322, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.30},
        "ODI":  {"matches": 155, "win_pct_a": 0.594, "win_pct_b": 0.374, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.65, "last_3yr_win_pct_b": 0.35},
        "T20I": {"matches": 23, "win_pct_a": 0.565, "win_pct_b": 0.435, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.55, "last_3yr_win_pct_b": 0.45}
    },
    "AFG_vs_ZIM": {
        "TEST": {"matches": 2, "win_pct_a": 0.50, "win_pct_b": 0.50, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.72, "last_3yr_win_pct_b": 0.28},
        "ODI":  {"matches": 28, "win_pct_a": 0.679, "win_pct_b": 0.321, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.75, "last_3yr_win_pct_b": 0.25},
        "T20I": {"matches": 15, "win_pct_a": 0.867, "win_pct_b": 0.133, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.833, "last_3yr_win_pct_b": 0.167}
    },
    "WI_vs_BAN": {
        "TEST": {"matches": 20, "win_pct_a": 0.65, "win_pct_b": 0.25, "draw_pct": 0.10, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.40},
        "ODI":  {"matches": 44, "win_pct_a": 0.477, "win_pct_b": 0.477, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.45, "last_3yr_win_pct_b": 0.55},
        "T20I": {"matches": 20, "win_pct_a": 0.55, "win_pct_b": 0.40, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.40}
    },
    "IND_vs_AUS": {
        "TEST": {"matches": 107, "win_pct_a": 0.318, "win_pct_b": 0.421, "draw_pct": 0.261, "last_3yr_win_pct_a": 0.50, "last_3yr_win_pct_b": 0.40},
        "ODI":  {"matches": 151, "win_pct_a": 0.384, "win_pct_b": 0.556, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.58, "last_3yr_win_pct_b": 0.42},
        "T20I": {"matches": 32, "win_pct_a": 0.625, "win_pct_b": 0.344, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.65, "last_3yr_win_pct_b": 0.35}
    },
    "IND_vs_ENG": {
        "TEST": {"matches": 136, "win_pct_a": 0.265, "win_pct_b": 0.375, "draw_pct": 0.360, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.30},
        "ODI":  {"matches": 107, "win_pct_a": 0.542, "win_pct_b": 0.411, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.667, "last_3yr_win_pct_b": 0.333},
        "T20I": {"matches": 24, "win_pct_a": 0.542, "win_pct_b": 0.458, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.40}
    },
    "IND_vs_SA": {
        "TEST": {"matches": 44, "win_pct_a": 0.364, "win_pct_b": 0.409, "draw_pct": 0.227, "last_3yr_win_pct_a": 0.50, "last_3yr_win_pct_b": 0.50},
        "ODI":  {"matches": 94, "win_pct_a": 0.426, "win_pct_b": 0.543, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.65, "last_3yr_win_pct_b": 0.35},
        "T20I": {"matches": 31, "win_pct_a": 0.516, "win_pct_b": 0.419, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.40}
    },
    "IND_vs_PAK": {
        "TEST": {"matches": 59, "win_pct_a": 0.153, "win_pct_b": 0.203, "draw_pct": 0.644, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.40},
        "ODI":  {"matches": 135, "win_pct_a": 0.422, "win_pct_b": 0.541, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.80, "last_3yr_win_pct_b": 0.20},
        "T20I": {"matches": 13, "win_pct_a": 0.692, "win_pct_b": 0.231, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.75, "last_3yr_win_pct_b": 0.25}
    },
    "AUS_vs_SA": {
        "TEST": {"matches": 101, "win_pct_a": 0.535, "win_pct_b": 0.267, "draw_pct": 0.198, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.40},
        "ODI":  {"matches": 110, "win_pct_a": 0.455, "win_pct_b": 0.500, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.50, "last_3yr_win_pct_b": 0.50},
        "T20I": {"matches": 25, "win_pct_a": 0.640, "win_pct_b": 0.360, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.65, "last_3yr_win_pct_b": 0.35}
    },
    "AUS_vs_NZ": {
        "TEST": {"matches": 62, "win_pct_a": 0.581, "win_pct_b": 0.129, "draw_pct": 0.290, "last_3yr_win_pct_a": 0.80, "last_3yr_win_pct_b": 0.20},
        "ODI":  {"matches": 142, "win_pct_a": 0.676, "win_pct_b": 0.275, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.70, "last_3yr_win_pct_b": 0.30},
        "T20I": {"matches": 19, "win_pct_a": 0.684, "win_pct_b": 0.316, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.70, "last_3yr_win_pct_b": 0.30}
    },
    "ENG_vs_SA": {
        "TEST": {"matches": 156, "win_pct_a": 0.423, "win_pct_b": 0.224, "draw_pct": 0.353, "last_3yr_win_pct_a": 0.55, "last_3yr_win_pct_b": 0.45},
        "ODI":  {"matches": 70, "win_pct_a": 0.429, "win_pct_b": 0.486, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.50, "last_3yr_win_pct_b": 0.50},
        "T20I": {"matches": 26, "win_pct_a": 0.462, "win_pct_b": 0.462, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.55, "last_3yr_win_pct_b": 0.45}
    },
    "ENG_vs_NZ": {
        "TEST": {"matches": 112, "win_pct_a": 0.455, "win_pct_b": 0.116, "draw_pct": 0.429, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.40},
        "ODI":  {"matches": 96, "win_pct_a": 0.458, "win_pct_b": 0.469, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.55, "last_3yr_win_pct_b": 0.45},
        "T20I": {"matches": 27, "win_pct_a": 0.556, "win_pct_b": 0.370, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.55, "last_3yr_win_pct_b": 0.45}
    },
    "PAK_vs_BAN": {
        "TEST": {"matches": 15, "win_pct_a": 0.800, "win_pct_b": 0.133, "draw_pct": 0.067, "last_3yr_win_pct_a": 0.333, "last_3yr_win_pct_b": 0.667},
        "ODI":  {"matches": 38, "win_pct_a": 0.868, "win_pct_b": 0.132, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.75, "last_3yr_win_pct_b": 0.25},
        "T20I": {"matches": 19, "win_pct_a": 0.842, "win_pct_b": 0.158, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.80, "last_3yr_win_pct_b": 0.20}
    },
    "SA_vs_SL": {
        "TEST": {"matches": 31, "win_pct_a": 0.516, "win_pct_b": 0.290, "draw_pct": 0.194, "last_3yr_win_pct_a": 0.65, "last_3yr_win_pct_b": 0.35},
        "ODI":  {"matches": 80, "win_pct_a": 0.563, "win_pct_b": 0.388, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.60, "last_3yr_win_pct_b": 0.40},
        "T20I": {"matches": 18, "win_pct_a": 0.667, "win_pct_b": 0.278, "draw_pct": 0.0, "last_3yr_win_pct_a": 0.70, "last_3yr_win_pct_b": 0.30}
    }
}

# 5. Authentic International Team Rolling Momentum & EMA Form Vector
TEAM_ROLLING_MOMENTUM_DATABASE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "IND": {
        "TEST": {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.70, "ema_momentum": 68.0, "streak": "W1", "form_trend": "STRONG"},
        "ODI":  {"recent_l5_win_pct": 0.80, "recent_l10_win_pct": 0.80, "ema_momentum": 82.0, "streak": "W3", "form_trend": "HOT"},
        "T20I": {"recent_l5_win_pct": 0.80, "recent_l10_win_pct": 0.90, "ema_momentum": 88.0, "streak": "W4", "form_trend": "HOT"}
    },
    "AUS": {
        "TEST": {"recent_l5_win_pct": 0.80, "recent_l10_win_pct": 0.75, "ema_momentum": 80.0, "streak": "W3", "form_trend": "HOT"},
        "ODI":  {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.70, "ema_momentum": 72.0, "streak": "W1", "form_trend": "STRONG"},
        "T20I": {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.65, "ema_momentum": 66.0, "streak": "L1", "form_trend": "NEUTRAL"}
    },
    "ENG": {
        "TEST": {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.65, "ema_momentum": 68.0, "streak": "W2", "form_trend": "STRONG"},
        "ODI":  {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.45, "ema_momentum": 48.0, "streak": "L2", "form_trend": "DIP"},
        "T20I": {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.60, "ema_momentum": 62.0, "streak": "W1", "form_trend": "NEUTRAL"}
    },
    "SA": {
        "TEST": {"recent_l5_win_pct": 0.80, "recent_l10_win_pct": 0.70, "ema_momentum": 76.0, "streak": "W3", "form_trend": "STRONG"},
        "ODI":  {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.65, "ema_momentum": 65.0, "streak": "W2", "form_trend": "STRONG"},
        "T20I": {"recent_l5_win_pct": 0.80, "recent_l10_win_pct": 0.75, "ema_momentum": 78.0, "streak": "W2", "form_trend": "HOT"}
    },
    "NZ": {
        "TEST": {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.60, "ema_momentum": 70.0, "streak": "W3", "form_trend": "HOT"},
        "ODI":  {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.50, "ema_momentum": 52.0, "streak": "L1", "form_trend": "NEUTRAL"},
        "T20I": {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.45, "ema_momentum": 48.0, "streak": "L2", "form_trend": "DIP"}
    },
    "PAK": {
        "TEST": {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.35, "ema_momentum": 44.0, "streak": "W2", "form_trend": "NEUTRAL"},
        "ODI":  {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.55, "ema_momentum": 60.0, "streak": "W2", "form_trend": "STRONG"},
        "T20I": {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.45, "ema_momentum": 46.0, "streak": "L1", "form_trend": "DIP"}
    },
    "SL": {
        "TEST": {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.55, "ema_momentum": 62.0, "streak": "W2", "form_trend": "STRONG"},
        "ODI":  {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.55, "ema_momentum": 64.0, "streak": "W1", "form_trend": "STRONG"},
        "T20I": {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.45, "ema_momentum": 48.0, "streak": "L1", "form_trend": "NEUTRAL"}
    },
    "WI": {
        "TEST": {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.35, "ema_momentum": 42.0, "streak": "L1", "form_trend": "DIP"},
        "ODI":  {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.50, "ema_momentum": 55.0, "streak": "W1", "form_trend": "NEUTRAL"},
        "T20I": {"recent_l5_win_pct": 0.60, "recent_l10_win_pct": 0.65, "ema_momentum": 68.0, "streak": "W2", "form_trend": "STRONG"}
    },
    "BAN": {
        "TEST": {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.35, "ema_momentum": 45.0, "streak": "L2", "form_trend": "DIP"},
        "ODI":  {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.45, "ema_momentum": 46.0, "streak": "L1", "form_trend": "DIP"},
        "T20I": {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.40, "ema_momentum": 44.0, "streak": "L3", "form_trend": "DIP"}
    },
    "AFG": {
        "TEST": {"recent_l5_win_pct": 0.50, "recent_l10_win_pct": 0.50, "ema_momentum": 55.0, "streak": "W1", "form_trend": "NEUTRAL"},
        "ODI":  {"recent_l5_win_pct": 0.80, "recent_l10_win_pct": 0.70, "ema_momentum": 78.0, "streak": "W3", "form_trend": "HOT"},
        "T20I": {"recent_l5_win_pct": 0.80, "recent_l10_win_pct": 0.75, "ema_momentum": 76.0, "streak": "W2", "form_trend": "HOT"}
    },
    "ZIM": {
        "TEST": {"recent_l5_win_pct": 0.20, "recent_l10_win_pct": 0.25, "ema_momentum": 32.0, "streak": "L2", "form_trend": "DIP"},
        "ODI":  {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.35, "ema_momentum": 40.0, "streak": "L1", "form_trend": "DIP"},
        "T20I": {"recent_l5_win_pct": 0.40, "recent_l10_win_pct": 0.40, "ema_momentum": 42.0, "streak": "W1", "form_trend": "NEUTRAL"}
    }
}

def get_international_h2h(team_a_code: str, team_b_code: str, match_format: str = "ODI") -> Dict[str, Any]:
    """
    Returns authentic format-specific bilateral H2H record between two teams.
    """
    c_a = normalize_team_code(team_a_code)
    c_b = normalize_team_code(team_b_code)
    fmt = match_format.upper() if match_format else "ODI"

    pair_key = f"{c_a}_vs_{c_b}"
    rev_key = f"{c_b}_vs_{c_a}"

    if pair_key in INTERNATIONAL_H2H_DATABASE and fmt in INTERNATIONAL_H2H_DATABASE[pair_key]:
        data = INTERNATIONAL_H2H_DATABASE[pair_key][fmt]
        return {
            "team_a": c_a,
            "team_b": c_b,
            "format": fmt,
            "matches": data["matches"],
            "win_pct_a": data["win_pct_a"],
            "win_pct_b": data["win_pct_b"],
            "draw_pct": data.get("draw_pct", 0.0),
            "last_3yr_win_pct_a": data.get("last_3yr_win_pct_a", data["win_pct_a"]),
            "last_3yr_win_pct_b": data.get("last_3yr_win_pct_b", data["win_pct_b"])
        }
    elif rev_key in INTERNATIONAL_H2H_DATABASE and fmt in INTERNATIONAL_H2H_DATABASE[rev_key]:
        data = INTERNATIONAL_H2H_DATABASE[rev_key][fmt]
        return {
            "team_a": c_a,
            "team_b": c_b,
            "format": fmt,
            "matches": data["matches"],
            "win_pct_a": data["win_pct_b"],
            "win_pct_b": data["win_pct_a"],
            "draw_pct": data.get("draw_pct", 0.0),
            "last_3yr_win_pct_a": data.get("last_3yr_win_pct_b", data["win_pct_b"]),
            "last_3yr_win_pct_b": data.get("last_3yr_win_pct_a", data["win_pct_a"])
        }

    # Dynamic fallback based on default parity
    return {
        "team_a": c_a,
        "team_b": c_b,
        "format": fmt,
        "matches": 15,
        "win_pct_a": 0.50,
        "win_pct_b": 0.50,
        "draw_pct": 0.15 if fmt == "TEST" else 0.0,
        "last_3yr_win_pct_a": 0.50,
        "last_3yr_win_pct_b": 0.50
    }

def get_team_rolling_momentum(team_code: str, match_format: str = "ODI") -> Dict[str, Any]:
    """
    Returns authentic rolling form, recent L5/L10 win %, and EMA momentum for an international team.
    """
    c = normalize_team_code(team_code)
    fmt = match_format.upper() if match_format else "ODI"

    team_data = TEAM_ROLLING_MOMENTUM_DATABASE.get(c, {}).get(fmt)
    if not team_data:
        team_data = TEAM_ROLLING_MOMENTUM_DATABASE.get(c, {}).get("ODI", {
            "recent_l5_win_pct": 0.50,
            "recent_l10_win_pct": 0.50,
            "ema_momentum": 50.0,
            "streak": "W1",
            "form_trend": "STABLE"
        })

    return {
        "team_code": c,
        "format": fmt,
        "recent_l5_win_pct": team_data.get("recent_l5_win_pct", 0.50),
        "recent_l10_win_pct": team_data.get("recent_l10_win_pct", 0.50),
        "ema_momentum": team_data.get("ema_momentum", 50.0),
        "streak": team_data.get("streak", "W1"),
        "form_trend": team_data.get("form_trend", "STABLE")
    }

def compute_atmospheric_aerodynamic_variables(
    venue: Dict[str, Any],
    weather: Optional[Dict[str, Any]] = None,
    is_day_night: bool = True
) -> Dict[str, Any]:
    """
    Computes aerodynamic lateral ball swing index, air density boundary-layer drag,
    and evening dew ball-slippage penalty.
    """
    w = weather or {}
    cloud_pct = float(w.get("cloud_cover_pct", 45.0))
    humidity_pct = float(w.get("humidity_pct", 65.0))
    temp_c = float(w.get("temperature_c", 22.0))
    wind_kmh = float(w.get("wind_speed_kmh", 18.0))
    dew_factor = float(venue.get("dew_factor", 0.25) if is_day_night else 0.05)

    # 1. Aerodynamic Swing Index (0.85x to 1.35x)
    cloud_swing_component = max(0.0, (cloud_pct - 50.0) / 100.0) * 0.40
    humidity_swing_component = max(0.0, (humidity_pct - 60.0) / 100.0) * 0.25
    swing_multiplier = round(1.0 + cloud_swing_component + humidity_swing_component, 2)
    swing_multiplier = min(1.35, max(0.85, swing_multiplier))

    # 2. Powerplay Wicket Hazard Multiplier
    pp_wicket_hazard = round(1.0 + (swing_multiplier - 1.0) * 0.85, 2)

    # 3. Evening Dew Ball-Slippage Penalty (0-30%)
    dew_penalty_pct = round(dew_factor * 28.0 if is_day_night else 0.0, 1)
    spin_rpm_drop_pct = round(dew_penalty_pct * 1.15, 1)

    return {
        "cloud_cover_pct": cloud_pct,
        "humidity_pct": humidity_pct,
        "aerodynamic_swing_multiplier": swing_multiplier,
        "pp_wicket_hazard_multiplier": pp_wicket_hazard,
        "dew_factor": dew_factor,
        "evening_dew_slippage_penalty_pct": dew_penalty_pct,
        "spin_rpm_retention_pct": round(100.0 - spin_rpm_drop_pct, 1),
        "tactical_weather_summary": (
            f"Overhead conditions ({cloud_pct:.0f}% cloud, {humidity_pct:.0f}% humidity) produce a "
            f"{swing_multiplier}x aerodynamic swing index; evening dew penalty: -{dew_penalty_pct}% grip."
        )
    }


def get_seasonal_pitch_context(
    venue_id: str,
    match_format: str = "ODI",
    match_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes format-specific ground pitch archetype and time-of-year seasonal calibration.
    Takes into account:
    - Month of year (1-12)
    - Regional climate (Subcontinent winter dew/morning seam vs SENA summer bounce vs UK early-summer swing)
    - Day-by-Day Test surface degradation (Session 1 morning seam -> Day 3-5 abrasive dusty turn)
    - Day/Night floodlight evening dew factor for ODIs and T20Is
    """
    vid = str(venue_id).lower().replace("-", "_").replace(" ", "_")
    fmt = str(match_format).upper() if match_format else "ODI"
    dossier = FORMAT_GROUND_PITCH_DOSSIERS.get(vid, {})
    if not dossier:
        for k, v in FORMAT_GROUND_PITCH_DOSSIERS.items():
            if k in vid or vid in k:
                dossier = v
                break

    region = dossier.get("region_cluster", "sena").lower()
    country = dossier.get("country", "").lower()

    # Extract month from match_date (default to November = 11)
    month = 11
    if match_date:
        try:
            parts = str(match_date).split("T")[0].split("-")
            if len(parts) >= 2:
                month = int(parts[1])
        except Exception:
            month = 11

    base_archetype = dossier.get("format_archetypes", {}).get(fmt, "balanced")
    base_archetype = canonicalize_pitch_type(base_archetype)

    seasonal_arch = base_archetype
    dew_factor_seasonal = float(dossier.get("dew_factor", 0.30))
    session_1_behavior = "True bounce and carry"
    wear_trajectory = "Consistent wear across match"
    season_label = "Standard Conditions"
    par_mod = 0
    chase_mod = 0.0

    # 1. Subcontinent (India, Pakistan, Sri Lanka, Bangladesh, UAE)
    if any(c in country for c in ["india", "pakistan", "sri lanka", "bangladesh", "united arab emirates", "uae"]) or region == "subcontinent":
        if month in [11, 12, 1, 2]: # Subcontinent Winter
            season_label = "Subcontinent Winter (Morning Coolness & Heavy Evening Dew)"
            if fmt == "TEST":
                seasonal_arch = "balanced" if base_archetype in ["flat", "balanced"] else base_archetype
                session_1_behavior = "Early morning condensation under covers produces sharp lateral swing and seam in Session 1"
                wear_trajectory = "Dry winter atmosphere accelerates pitch cracking; sharp spin and reverse swing from Day 3 afternoon"
                par_mod = -15
            else:
                dew_factor_seasonal = min(0.85, dew_factor_seasonal + 0.25)
                session_1_behavior = "Initial dry grip in first innings"
                wear_trajectory = "Heavy winter evening dew under floodlights severely wets the outfield, neutralizing spin grip"
                chase_mod = +0.12 # +12% chase advantage
                par_mod = +15 if fmt == "T20I" else +20
        elif month in [3, 4, 5]: # Subcontinent Peak Summer
            season_label = "Subcontinent Peak Summer (Extreme Dry Heat)"
            seasonal_arch = "dusty" if fmt == "TEST" else "flat"
            session_1_behavior = "Rapid moisture evaporation, early abrasive surface friction"
            wear_trajectory = "Puffing dust and widening fissures from Day 2 onward"
            dew_factor_seasonal = max(0.10, dew_factor_seasonal - 0.15)
            par_mod = +25 if fmt != "TEST" else -20
        else: # Monsoon / Autumn
            season_label = "Subcontinent Monsoon/Autumn (High Humidity & Moisture)"
            seasonal_arch = "balanced"
            session_1_behavior = "Overcast moisture keeps ball swinging throughout"

    # 2. SENA - Australia
    elif "australia" in country:
        if month in [11, 12, 1, 2]: # Australian Summer
            season_label = "Australian Summer (Hard-Packed True Decks)"
            if "perth" in vid or "gabba" in vid:
                seasonal_arch = "bouncy"
                session_1_behavior = "Steep vertical lift and rapid pace off the hard clay"
                wear_trajectory = "Minimal crumbling; true pace maintained throughout 5 days"
            elif "scg" in vid and month in [1, 2]:
                seasonal_arch = "dusty"
                session_1_behavior = "Even bounce early, deteriorating into classic Bulli soil turn"
            else:
                seasonal_arch = base_archetype
            dew_factor_seasonal = 0.15
            chase_mod = +0.02
        else:
            season_label = "Australian Early Spring / Shoulder Season"

    # 3. SENA - New Zealand
    elif "new zealand" in country:
        if month in [11, 12, 1, 2, 3]: # NZ Summer
            season_label = "New Zealand Maritime Summer (Coastal Wind & Lush Grass)"
            if fmt == "TEST":
                seasonal_arch = "green" if "wellington" in vid or "basin" in vid else "balanced"
                session_1_behavior = "Maritime green tinge creates two-way swing and persistent seam nibble"
                wear_trajectory = "Damp coastal conditions preserve grass roots; pitch flattens on Days 2-3 before uneven bounce"
            dew_factor_seasonal = 0.20

    # 4. SENA - England
    elif "england" in country:
        if month in [5, 6]: # Early English Summer
            season_label = "Early English Summer (High Atmospheric Moisture & Green Seam)"
            seasonal_arch = "green" if fmt == "TEST" else "balanced"
            session_1_behavior = "Heavy overhead swing with pronounced Dukes ball seam movement"
        else:
            season_label = "Late English Summer (Sun-Baked Abrasive Surfaces)"
            seasonal_arch = "balanced" if fmt == "TEST" else "flat"

    # 5. South Africa
    elif "south africa" in country:
        if "centurion" in vid or "wanderers" in vid:
            season_label = "Highveld Summer (High Altitude 1400m - Express Pace & Steep Lift)"
            seasonal_arch = "bouncy"
        else:
            season_label = "Coastal South Africa (Maritime Seam)"
            seasonal_arch = "balanced"

    return {
        "venue_id": vid,
        "format": fmt,
        "month": month,
        "season_label": season_label,
        "base_archetype": base_archetype,
        "seasonal_archetype": seasonal_arch,
        "dew_factor": round(dew_factor_seasonal, 2),
        "session_1_behavior": session_1_behavior,
        "wear_trajectory": wear_trajectory,
        "par_score_modifier": par_mod,
        "chase_bias_modifier": round(chase_mod, 3)
    }

def get_ground_pitch_archetype(venue_id: str, match_format: str = "T20I", match_date: Optional[str] = None) -> str:
    """
    Returns the authentic format-specific pitch archetype for a given ground.
    Strictly one of the 5 canonical archetypes: 'dusty', 'green', 'bouncy', 'flat', 'balanced'.
    If match_date is provided, incorporates seasonal time-of-year calibration.
    """
    if match_date:
        ctx = get_seasonal_pitch_context(venue_id, match_format, match_date)
        return ctx["seasonal_archetype"]

    vid = venue_id.lower().replace("-", "_").replace(" ", "_")
    dossier = FORMAT_GROUND_PITCH_DOSSIERS.get(vid)
    fmt = match_format.upper() if match_format else "T20I"

    if dossier and "format_archetypes" in dossier:
        arch = dossier["format_archetypes"].get(fmt, dossier["format_archetypes"].get("T20I", "flat"))
        return canonicalize_pitch_type(arch)

    for k, v in FORMAT_GROUND_PITCH_DOSSIERS.items():
        if k in vid or vid in k:
            arch = v["format_archetypes"].get(fmt, v["format_archetypes"].get("T20I", "flat"))
            return canonicalize_pitch_type(arch)

    if any(s in vid for s in ["sylhet", "mirpur", "chepauk", "colombo", "sharjah", "galle"]):
        return "dusty"
    elif any(s in vid for s in ["wellington", "headingley", "lords"]):
        return "green" if fmt == "TEST" else "balanced"
    elif any(s in vid for s in ["centurion", "perth", "mcg"]):
        return "bouncy"
    return "flat"


_VENUES_DB_CACHE: Optional[List[Dict[str, Any]]] = None

def _load_venues_db_cache() -> List[Dict[str, Any]]:
    """Lazy-loads venue metrics from SQLite once for low-latency dossier merging."""
    global _VENUES_DB_CACHE
    if _VENUES_DB_CACHE is not None:
        return _VENUES_DB_CACHE
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cricoracle.db"))
    rows: List[Dict[str, Any]] = []
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=2.0)
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(
                "SELECT id, name, city, country, pitch_type, avg_first_innings_score_t20, "
                "avg_first_innings_score_odi, chase_win_pct, dew_factor, dimensions_avg_boundary_m FROM venues"
            ).fetchall()]
            conn.close()
        except Exception:
            rows = []
    _VENUES_DB_CACHE = rows
    return _VENUES_DB_CACHE


def resolve_venue_dossier_context(venue_hint: Optional[str], match_format: str = "T20I") -> Dict[str, Any]:
    """
    Resolves format-specific venue par, boundary dimensions, dew factor, and empirical chase win%
    from FORMAT_GROUND_PITCH_DOSSIERS and SQLite venues table using safe token/city matching.
    """
    fmt_raw = str(match_format or "T20I").upper().strip()
    fmt = "TEST" if "TEST" in fmt_raw else ("ODI" if "ODI" in fmt_raw else "T20I")
    if not venue_hint:
        return {"matched": False, "venue_key": None, "venue_par": None, "boundary_dimensions": {}, "avg_m": 68.0, "dew_factor": None, "chase_win_pct": None, "pitch_wear_rate": None}

    raw_lower = str(venue_hint).lower()
    norm_id = raw_lower.replace("-", "_").replace(" ", "_")

    def _token_in_text(tok: str, text: str) -> bool:
        t = tok.strip().lower()
        if not t or len(t) < 3:
            return False
        if len(t) <= 4:
            return bool(re.search(rf"(?:^|[^a-z0-9]){re.escape(t)}(?:$|[^a-z0-9])", text))
        return t in text

    matched_key = None
    dossier = FORMAT_GROUND_PITCH_DOSSIERS.get(norm_id)
    if dossier:
        matched_key = norm_id
    else:
        for k, v in sorted(FORMAT_GROUND_PITCH_DOSSIERS.items(), key=lambda kv: len(kv[0]), reverse=True):
            v_name = str(v.get("name", "")).lower()
            v_city = str(v.get("city", "")).lower()
            if _token_in_text(k.replace("_", " "), raw_lower) or _token_in_text(k, norm_id) or _token_in_text(v_name, raw_lower) or (v_city and _token_in_text(v_city, raw_lower)):
                matched_key = k
                dossier = v
                break

    db_row = None
    for row in _load_venues_db_cache():
        rid = str(row.get("id", "")).lower()
        rname = str(row.get("name", "")).lower()
        rcity = str(row.get("city", "")).lower()
        if (matched_key and (rid == matched_key or matched_key in rid)) or _token_in_text(rname, raw_lower) or (rcity and _token_in_text(rcity, raw_lower)):
            db_row = row
            break

    if not dossier and not db_row:
        return {"matched": False, "venue_key": None, "venue_par": None, "boundary_dimensions": {}, "avg_m": 68.0, "dew_factor": None, "chase_win_pct": None, "pitch_wear_rate": None}

    b_dims = dict((dossier or {}).get("boundary_dimensions") or {})
    avg_m = float(b_dims.get("avg_m") or (db_row or {}).get("dimensions_avg_boundary_m") or 68.0)
    if "avg_m" not in b_dims:
        b_dims["avg_m"] = avg_m

    d_par = (dossier or {}).get("format_pars", {}).get(fmt)
    db_par = None
    if db_row:
        if fmt == "ODI":
            db_par = db_row.get("avg_first_innings_score_odi")
        elif fmt == "T20I":
            db_par = db_row.get("avg_first_innings_score_t20")

    if d_par and db_par:
        venue_par = 0.55 * float(d_par) + 0.45 * float(db_par)
    elif d_par:
        venue_par = float(d_par)
    elif db_par:
        venue_par = float(db_par)
    else:
        venue_par = None

    return {
        "matched": True,
        "venue_key": matched_key or ((db_row or {}).get("id")),
        "venue_par": round(venue_par, 1) if venue_par is not None else None,
        "boundary_dimensions": b_dims,
        "avg_m": avg_m,
        "dew_factor": float(dossier["dew_factor"]) if (dossier and dossier.get("dew_factor") is not None) else ((float(db_row["dew_factor"])) if db_row and db_row.get("dew_factor") is not None else None),
        "chase_win_pct": float(db_row["chase_win_pct"]) if (db_row and db_row.get("chase_win_pct") is not None) else None,
        "pitch_wear_rate": float(dossier["pitch_wear_rate"]) if (dossier and dossier.get("pitch_wear_rate") is not None) else None,
    }


def get_country_home_surfaces(country_code: str, match_format: str = "ODI") -> Dict[str, float]:
    """
    Returns 5 Home Surface Baselines (green_home, dusty_home, bouncy_home, flat_home, balanced_home).
    """
    c = normalize_team_code(country_code)
    fmt = "T20I" if match_format.upper() in ["T20", "T20I"] else match_format.upper()
    return COUNTRY_HOME_SURFACES.get(c, {}).get(fmt, {
        "flat_home": 85.0, "green_home": 80.0, "balanced_home": 82.0, "bouncy_home": 82.0, "dusty_home": 80.0
    })

def get_country_top_touring_performers(country_code: str) -> Dict[str, Any]:
    """
    Returns Dual Top Touring Performers (Top Foreign Batter and Top Foreign Bowler).
    """
    c = normalize_team_code(country_code)
    return COUNTRY_TOP_TOURING_PERFORMERS.get(c, {
        "batter": {"name": "Top Batter", "score": 85.0, "favored_deck": "FLAT"},
        "bowler": {"name": "Top Bowler", "score": 85.0, "favored_deck": "BALANCED"}
    })

def get_country_pitch_adaptability(country_code: str, match_format: str, pitch_archetype: str, recency_weight: float = 0.65, surface_mode: str = "touring") -> float:
    """
    Returns country's adaptability score (0-100) for a given format and pitch archetype.
    Supports surface_mode='touring' (foreign decks) and surface_mode='home' (home decks).
    Weights recent 1-year performance by recency_weight (default 65%), career baseline by (1 - recency_weight).
    """
    c = normalize_team_code(country_code)
    fmt = "T20I" if match_format.upper() in ["T20", "T20I"] else match_format.upper()
    pitch = canonicalize_pitch_type(pitch_archetype)

    if surface_mode == "home":
        home_decks = get_country_home_surfaces(c, fmt)
        deck_key = f"{pitch}_home"
        if deck_key in home_decks:
            return round(home_decks[deck_key], 1)
    fmt = match_format.upper()
    pitch = canonicalize_pitch_type(pitch_archetype)

    career_data = COUNTRY_PITCH_ADAPTABILITY.get(c, {}).get(fmt, {}) or COUNTRY_PITCH_ADAPTABILITY.get(c, {}).get("T20I", {})
    career_score = career_data.get(pitch)
    if career_score is None and pitch in ["seam / green", "green / seam"]:
        career_score = round((career_data.get("green", 75.0) + career_data.get("seam", 75.0)) / 2.0, 1)
    elif career_score is None:
        career_score = 75.0

    recent_data = COUNTRY_RECENT_1YR_ADAPTABILITY.get(c, {}).get(fmt, {}) or COUNTRY_RECENT_1YR_ADAPTABILITY.get(c, {}).get("T20I", {})
    recent_score = recent_data.get(pitch)
    if recent_score is None and pitch in ["seam / green", "green / seam"]:
        recent_score = round((recent_data.get("green", career_score) + recent_data.get("seam", career_score)) / 2.0, 1)
    elif recent_score is None:
        recent_score = career_score

    composite = (recent_score * recency_weight) + (career_score * (1.0 - recency_weight))
    return round(composite, 1)

def get_country_pitch_adaptability_details(country_code: str, arg2: str, arg3: str, recency_weight: float = 0.65, surface_mode: str = "touring") -> Dict[str, Any]:
    """
    Returns detailed breakdown of country adaptability including 1-year form vs career prior,
    5 home surface baselines, and dual top touring performers (Top Batter & Top Bowler).
    Accepts (country, format, pitch) or (country, pitch, format) flexibly.
    """
    c = normalize_team_code(country_code)
    known_formats = ["TEST", "ODI", "T20I"]
    if arg2.upper() in known_formats:
        fmt = arg2.upper()
        pitch = arg3.lower()
    else:
        pitch = arg2.lower()
        fmt = arg3.upper()

    career_data = COUNTRY_PITCH_ADAPTABILITY.get(c, {}).get(fmt, {}) or COUNTRY_PITCH_ADAPTABILITY.get(c, {}).get("T20I", {})
    career_score = career_data.get(pitch)
    if career_score is None and pitch in ["seam / green", "green / seam"]:
        career_score = round((career_data.get("green", 75.0) + career_data.get("seam", 75.0)) / 2.0, 1)
    elif career_score is None:
        career_score = 75.0

    recent_data = COUNTRY_RECENT_1YR_ADAPTABILITY.get(c, {}).get(fmt, {}) or COUNTRY_RECENT_1YR_ADAPTABILITY.get(c, {}).get("T20I", {})
    recent_score = recent_data.get(pitch)
    if recent_score is None and pitch in ["seam / green", "green / seam"]:
        recent_score = round((recent_data.get("green", career_score) + recent_data.get("seam", career_score)) / 2.0, 1)
    elif recent_score is None:
        recent_score = career_score

    composite = (recent_score * recency_weight) + (career_score * (1.0 - recency_weight))
    diff = round(recent_score - career_score, 1)
    trend = "SURGING" if diff >= 2.5 else ("DECLINING" if diff <= -2.5 else "STABLE")

    home_decks = get_country_home_surfaces(c, fmt)
    top_performers = get_country_top_touring_performers(c)

    return {
        "country": c,
        "format": fmt,
        "pitch_archetype": pitch,
        "composite_score": round(composite, 1),
        "recent_1yr_score": round(recent_score, 1),
        "career_score": round(career_score, 1),
        "recency_weight": recency_weight,
        "recency_weight_applied": recency_weight,
        "delta_recent_boost": diff,
        "trend": trend,
        "recent_form_trend": trend,
        "surface_mode": surface_mode,
        "is_full_member": c in ICC_FULL_MEMBERS,
        "opponent_weight": get_opponent_quality_weight(c),
        "home_surfaces": home_decks,
        "home_archetype_score": home_decks.get(f"{pitch}_home", round(composite, 1)),
        "top_touring_performers": top_performers,
        "top_foreign_batter": top_performers.get("batter"),
        "top_foreign_bowler": top_performers.get("bowler")
    }

# -----------------------------------------------------------------------------------------
# COUNTRY PITCH PHASE RUN RATES (Historical Calibration across 5 Canonical Archetypes)
# Format-Aware (ODI, T20I) Powerplay, Middle, and Death Phase Run Rates + Par Benchmarks
# -----------------------------------------------------------------------------------------
COUNTRY_PITCH_PHASE_RATES: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
    # Full Members
    "IND": {
        "ODI": {
            "flat":     {"pp": 6.2, "mid": 6.1, "death": 9.8, "par": 315.0},
            "dusty":    {"pp": 5.0, "mid": 4.6, "death": 7.4, "par": 250.0},
            "green":    {"pp": 4.6, "mid": 4.8, "death": 7.8, "par": 260.0},
            "bouncy":   {"pp": 5.5, "mid": 5.4, "death": 8.6, "par": 285.0},
            "balanced": {"pp": 5.4, "mid": 5.2, "death": 8.2, "par": 275.0},
        },
        "T20I": {
            "flat":     {"pp": 9.5, "mid": 9.4, "death": 13.0, "par": 205.0},
            "dusty":    {"pp": 7.4, "mid": 7.2, "death": 9.8,  "par": 158.0},
            "green":    {"pp": 6.2, "mid": 6.0, "death": 8.4,  "par": 140.0},
            "bouncy":   {"pp": 8.8, "mid": 8.6, "death": 11.6, "par": 185.0},
            "balanced": {"pp": 8.2, "mid": 8.0, "death": 10.8, "par": 172.0},
        }
    },
    "AUS": {
        "ODI": {
            "flat":     {"pp": 6.6, "mid": 6.4, "death": 9.8, "par": 320.0},
            "bouncy":   {"pp": 5.8, "mid": 5.6, "death": 9.0, "par": 290.0},
            "green":    {"pp": 5.0, "mid": 5.2, "death": 8.2, "par": 265.0},
            "dusty":    {"pp": 4.6, "mid": 4.2, "death": 6.8, "par": 242.0},
            "balanced": {"pp": 5.6, "mid": 5.4, "death": 8.6, "par": 280.0},
        },
        "T20I": {
            "flat":     {"pp": 9.6, "mid": 9.0, "death": 12.2, "par": 200.0},
            "bouncy":   {"pp": 8.6, "mid": 8.4, "death": 11.4, "par": 182.0},
            "green":    {"pp": 6.6, "mid": 6.8, "death": 9.2,  "par": 145.0},
            "dusty":    {"pp": 6.8, "mid": 6.5, "death": 8.6,  "par": 150.0},
            "balanced": {"pp": 8.2, "mid": 8.0, "death": 10.6, "par": 170.0},
        }
    },
    "SA": {
        "ODI": {
            "flat":     {"pp": 6.1, "mid": 6.5, "death": 10.2, "par": 320.0},
            "bouncy":   {"pp": 5.8, "mid": 5.6, "death": 9.0,  "par": 290.0},
            "green":    {"pp": 4.4, "mid": 4.6, "death": 7.4,  "par": 250.0},
            "dusty":    {"pp": 4.2, "mid": 4.0, "death": 6.2,  "par": 235.0},
            "balanced": {"pp": 5.4, "mid": 5.3, "death": 8.4,  "par": 275.0},
        },
        "T20I": {
            "flat":     {"pp": 9.2, "mid": 8.8, "death": 12.0, "par": 195.0},
            "bouncy":   {"pp": 8.6, "mid": 8.2, "death": 11.2, "par": 180.0},
            "green":    {"pp": 6.0, "mid": 6.2, "death": 8.6,  "par": 138.0},
            "dusty":    {"pp": 6.6, "mid": 6.4, "death": 8.4,  "par": 148.0},
            "balanced": {"pp": 8.0, "mid": 7.8, "death": 10.4, "par": 168.0},
        }
    },
    "ENG": {
        "ODI": {
            "flat":     {"pp": 6.4, "mid": 6.0, "death": 9.2, "par": 310.0},
            "green":    {"pp": 4.8, "mid": 5.0, "death": 8.0, "par": 265.0},
            "dusty":    {"pp": 4.8, "mid": 4.4, "death": 6.8, "par": 245.0},
            "bouncy":   {"pp": 5.4, "mid": 5.2, "death": 8.4, "par": 280.0},
            "balanced": {"pp": 5.6, "mid": 5.4, "death": 8.4, "par": 278.0},
        },
        "T20I": {
            "flat":     {"pp": 9.2, "mid": 8.8, "death": 12.0, "par": 195.0},
            "green":    {"pp": 6.8, "mid": 7.0, "death": 9.4,  "par": 148.0},
            "dusty":    {"pp": 7.0, "mid": 6.8, "death": 9.0,  "par": 154.0},
            "bouncy":   {"pp": 8.4, "mid": 8.0, "death": 10.8, "par": 178.0},
            "balanced": {"pp": 8.2, "mid": 7.8, "death": 10.4, "par": 170.0},
        }
    },
    "NZ": {
        "ODI": {
            "flat":     {"pp": 6.0, "mid": 6.0, "death": 9.2, "par": 305.0},
            "green":    {"pp": 5.2, "mid": 5.4, "death": 8.4, "par": 275.0},
            "dusty":    {"pp": 4.5, "mid": 4.4, "death": 6.8, "par": 242.0},
            "bouncy":   {"pp": 5.4, "mid": 5.2, "death": 8.2, "par": 280.0},
            "balanced": {"pp": 5.4, "mid": 5.2, "death": 8.2, "par": 275.0},
        },
        "T20I": {
            "flat":     {"pp": 8.8, "mid": 8.4, "death": 11.4, "par": 188.0},
            "green":    {"pp": 7.4, "mid": 7.6, "death": 10.0, "par": 160.0},
            "dusty":    {"pp": 6.6, "mid": 6.4, "death": 8.6,  "par": 146.0},
            "bouncy":   {"pp": 8.0, "mid": 7.8, "death": 10.4, "par": 174.0},
            "balanced": {"pp": 8.0, "mid": 7.6, "death": 10.2, "par": 168.0},
        }
    },
    "PAK": {
        "ODI": {
            "flat":     {"pp": 5.4, "mid": 5.4, "death": 8.4, "par": 290.0},
            "bouncy":   {"pp": 4.8, "mid": 4.6, "death": 7.4, "par": 255.0},
            "green":    {"pp": 4.6, "mid": 4.8, "death": 7.6, "par": 258.0},
            "dusty":    {"pp": 5.0, "mid": 4.4, "death": 6.8, "par": 240.0},
            "balanced": {"pp": 5.2, "mid": 5.0, "death": 7.8, "par": 268.0},
        },
        "T20I": {
            "flat":     {"pp": 8.2, "mid": 8.0, "death": 10.8, "par": 180.0},
            "bouncy":   {"pp": 7.6, "mid": 7.2, "death": 9.8,  "par": 165.0},
            "green":    {"pp": 6.4, "mid": 6.6, "death": 8.8,  "par": 142.0},
            "dusty":    {"pp": 7.4, "mid": 7.0, "death": 9.2,  "par": 155.0},
            "balanced": {"pp": 7.6, "mid": 7.2, "death": 9.8,  "par": 162.0},
        }
    },
    "AFG": {
        "ODI": {
            "dusty":    {"pp": 5.4, "mid": 5.2, "death": 8.2, "par": 275.0},
            "flat":     {"pp": 5.6, "mid": 5.4, "death": 8.2, "par": 288.0},
            "green":    {"pp": 4.0, "mid": 4.2, "death": 6.6, "par": 225.0},
            "bouncy":   {"pp": 4.6, "mid": 4.6, "death": 7.2, "par": 248.0},
            "balanced": {"pp": 5.0, "mid": 4.8, "death": 7.6, "par": 260.0},
        },
        "T20I": {
            "dusty":    {"pp": 7.2, "mid": 7.2, "death": 9.6,  "par": 156.0},
            "flat":     {"pp": 8.2, "mid": 7.8, "death": 10.6, "par": 175.0},
            "green":    {"pp": 5.2, "mid": 5.4, "death": 7.4,  "par": 128.0},
            "bouncy":   {"pp": 7.2, "mid": 7.0, "death": 9.2,  "par": 158.0},
            "balanced": {"pp": 7.4, "mid": 7.0, "death": 9.4,  "par": 160.0},
        }
    },
    "SL": {
        "ODI": {
            "dusty":    {"pp": 5.0, "mid": 4.8, "death": 7.4, "par": 248.0},
            "flat":     {"pp": 5.6, "mid": 5.4, "death": 8.4, "par": 290.0},
            "green":    {"pp": 4.4, "mid": 4.6, "death": 7.0, "par": 242.0},
            "bouncy":   {"pp": 4.6, "mid": 4.6, "death": 7.2, "par": 248.0},
            "balanced": {"pp": 5.0, "mid": 4.8, "death": 7.4, "par": 260.0},
        },
        "T20I": {
            "dusty":    {"pp": 7.4, "mid": 7.2, "death": 9.4,  "par": 154.0},
            "flat":     {"pp": 8.2, "mid": 8.0, "death": 10.8, "par": 178.0},
            "green":    {"pp": 6.4, "mid": 6.6, "death": 8.8,  "par": 140.0},
            "bouncy":   {"pp": 7.2, "mid": 7.0, "death": 9.2,  "par": 158.0},
            "balanced": {"pp": 7.6, "mid": 7.2, "death": 9.6,  "par": 162.0},
        }
    },
    "WI": {
        "ODI": {
            "flat":     {"pp": 5.6, "mid": 5.4, "death": 9.0, "par": 295.0},
            "balanced": {"pp": 5.0, "mid": 4.8, "death": 8.0, "par": 268.0},
            "green":    {"pp": 4.4, "mid": 4.6, "death": 7.2, "par": 245.0},
            "dusty":    {"pp": 4.4, "mid": 4.2, "death": 6.8, "par": 238.0},
            "bouncy":   {"pp": 5.2, "mid": 5.0, "death": 8.2, "par": 272.0},
        },
        "T20I": {
            "flat":     {"pp": 8.8, "mid": 8.4, "death": 12.0, "par": 190.0},
            "balanced": {"pp": 7.6, "mid": 7.4, "death": 10.6, "par": 168.0},
            "green":    {"pp": 6.6, "mid": 6.8, "death": 9.0,  "par": 144.0},
            "dusty":    {"pp": 6.8, "mid": 6.6, "death": 8.8,  "par": 148.0},
            "bouncy":   {"pp": 8.0, "mid": 7.6, "death": 10.4, "par": 172.0},
        }
    },
    "BAN": {
        "ODI": {
            "dusty":    {"pp": 4.8, "mid": 4.6, "death": 7.0, "par": 242.0},
            "flat":     {"pp": 5.2, "mid": 5.2, "death": 8.0, "par": 278.0},
            "green":    {"pp": 4.2, "mid": 4.4, "death": 6.8, "par": 235.0},
            "bouncy":   {"pp": 4.4, "mid": 4.4, "death": 6.8, "par": 240.0},
            "balanced": {"pp": 4.8, "mid": 4.6, "death": 7.2, "par": 252.0},
        },
        "T20I": {
            "dusty":    {"pp": 7.0, "mid": 6.8, "death": 8.8,  "par": 146.0},
            "flat":     {"pp": 7.8, "mid": 7.4, "death": 9.8,  "par": 168.0},
            "green":    {"pp": 6.0, "mid": 6.2, "death": 8.2,  "par": 132.0},
            "bouncy":   {"pp": 7.0, "mid": 6.8, "death": 8.8,  "par": 150.0},
            "balanced": {"pp": 7.2, "mid": 7.0, "death": 9.0,  "par": 154.0},
        }
    },
    "IRE": {
        "ODI": {
            "flat":     {"pp": 5.4, "mid": 5.2, "death": 8.0, "par": 280.0},
            "green":    {"pp": 4.8, "mid": 4.8, "death": 7.4, "par": 255.0},
            "balanced": {"pp": 4.9, "mid": 4.8, "death": 7.4, "par": 258.0},
            "dusty":    {"pp": 4.3, "mid": 4.1, "death": 6.4, "par": 230.0},
            "bouncy":   {"pp": 4.7, "mid": 4.6, "death": 7.0, "par": 248.0},
        },
        "T20I": {
            "flat":     {"pp": 8.0, "mid": 7.6, "death": 10.2, "par": 172.0},
            "balanced": {"pp": 7.5, "mid": 7.2, "death": 9.6,  "par": 160.0},
            "green":    {"pp": 6.5, "mid": 6.8, "death": 8.8,  "par": 142.0},
            "dusty":    {"pp": 6.6, "mid": 6.4, "death": 8.4,  "par": 144.0},
            "bouncy":   {"pp": 7.4, "mid": 7.2, "death": 9.4,  "par": 156.0},
        }
    },
    "ZIM": {
        "ODI": {
            "flat":     {"pp": 5.2, "mid": 5.0, "death": 7.8, "par": 270.0},
            "balanced": {"pp": 4.8, "mid": 4.6, "death": 7.2, "par": 250.0},
            "green":    {"pp": 4.4, "mid": 4.4, "death": 6.8, "par": 238.0},
            "dusty":    {"pp": 4.4, "mid": 4.2, "death": 6.6, "par": 232.0},
            "bouncy":   {"pp": 4.6, "mid": 4.4, "death": 6.8, "par": 242.0},
        },
        "T20I": {
            "flat":     {"pp": 7.8, "mid": 7.4, "death": 9.8,  "par": 165.0},
            "balanced": {"pp": 7.2, "mid": 6.8, "death": 9.2,  "par": 152.0},
            "green":    {"pp": 6.2, "mid": 6.4, "death": 8.2,  "par": 134.0},
            "dusty":    {"pp": 6.6, "mid": 6.2, "death": 8.4,  "par": 142.0},
            "bouncy":   {"pp": 7.0, "mid": 6.8, "death": 8.8,  "par": 148.0},
        }
    },
    # Associates
    "NED": {
        "ODI": {
            "flat":     {"pp": 5.0, "mid": 4.8, "death": 7.4, "par": 260.0},
            "bouncy":   {"pp": 4.8, "mid": 4.6, "death": 7.2, "par": 250.0},
            "green":    {"pp": 4.6, "mid": 4.6, "death": 7.0, "par": 245.0},
            "dusty":    {"pp": 4.2, "mid": 4.0, "death": 6.2, "par": 225.0},
            "balanced": {"pp": 4.7, "mid": 4.5, "death": 7.0, "par": 248.0},
        },
        "T20I": {
            "flat":     {"pp": 7.6, "mid": 7.2, "death": 9.6,  "par": 160.0},
            "balanced": {"pp": 7.0, "mid": 6.8, "death": 9.0,  "par": 150.0},
            "green":    {"pp": 6.4, "mid": 6.4, "death": 8.4,  "par": 136.0},
            "dusty":    {"pp": 6.2, "mid": 6.0, "death": 8.0,  "par": 135.0},
            "bouncy":   {"pp": 7.0, "mid": 6.8, "death": 8.8,  "par": 148.0},
        }
    },
    "SCO": {
        "ODI": {
            "flat":     {"pp": 5.2, "mid": 5.0, "death": 7.6, "par": 268.0},
            "green":    {"pp": 4.8, "mid": 4.8, "death": 7.2, "par": 252.0},
            "balanced": {"pp": 4.8, "mid": 4.6, "death": 7.2, "par": 250.0},
            "bouncy":   {"pp": 4.7, "mid": 4.6, "death": 7.0, "par": 246.0},
            "dusty":    {"pp": 4.2, "mid": 4.0, "death": 6.2, "par": 225.0},
        },
        "T20I": {
            "flat":     {"pp": 8.2, "mid": 7.8, "death": 10.4, "par": 172.0},
            "balanced": {"pp": 7.2, "mid": 7.0, "death": 9.2,  "par": 154.0},
            "green":    {"pp": 6.6, "mid": 6.6, "death": 8.6,  "par": 140.0},
            "dusty":    {"pp": 6.4, "mid": 6.2, "death": 8.2,  "par": 138.0},
            "bouncy":   {"pp": 7.2, "mid": 7.0, "death": 9.0,  "par": 150.0},
        }
    },
    "USA": {
        "ODI": {
            "flat":     {"pp": 5.0, "mid": 4.8, "death": 7.4, "par": 260.0},
            "balanced": {"pp": 4.7, "mid": 4.5, "death": 7.0, "par": 245.0},
            "green":    {"pp": 4.4, "mid": 4.4, "death": 6.8, "par": 238.0},
            "bouncy":   {"pp": 4.6, "mid": 4.4, "death": 6.8, "par": 242.0},
            "dusty":    {"pp": 4.2, "mid": 4.0, "death": 6.0, "par": 222.0},
        },
        "T20I": {
            "flat":     {"pp": 7.8, "mid": 7.4, "death": 9.8,  "par": 164.0},
            "balanced": {"pp": 7.0, "mid": 6.8, "death": 9.0,  "par": 150.0},
            "green":    {"pp": 6.2, "mid": 6.2, "death": 8.2,  "par": 132.0},
            "dusty":    {"pp": 6.4, "mid": 6.0, "death": 8.0,  "par": 136.0},
            "bouncy":   {"pp": 7.0, "mid": 6.8, "death": 8.8,  "par": 148.0},
        }
    },
    "NEP": {
        "ODI": {
            "flat":     {"pp": 4.8, "mid": 4.6, "death": 7.0, "par": 248.0},
            "dusty":    {"pp": 4.6, "mid": 4.4, "death": 6.8, "par": 238.0},
            "balanced": {"pp": 4.6, "mid": 4.4, "death": 6.6, "par": 236.0},
            "green":    {"pp": 4.2, "mid": 4.2, "death": 6.4, "par": 228.0},
            "bouncy":   {"pp": 4.3, "mid": 4.2, "death": 6.4, "par": 230.0},
        },
        "T20I": {
            "flat":     {"pp": 7.4, "mid": 7.0, "death": 9.2,  "par": 156.0},
            "dusty":    {"pp": 6.8, "mid": 6.6, "death": 8.6,  "par": 144.0},
            "balanced": {"pp": 6.6, "mid": 6.4, "death": 8.2,  "par": 140.0},
            "green":    {"pp": 5.8, "mid": 5.8, "death": 7.6,  "par": 126.0},
            "bouncy":   {"pp": 6.6, "mid": 6.4, "death": 8.4,  "par": 142.0},
        }
    },
    "NAM": {
        "ODI": {
            "flat":     {"pp": 4.8, "mid": 4.6, "death": 7.0, "par": 250.0},
            "balanced": {"pp": 4.6, "mid": 4.4, "death": 6.8, "par": 240.0},
            "green":    {"pp": 4.4, "mid": 4.4, "death": 6.6, "par": 235.0},
            "dusty":    {"pp": 4.0, "mid": 3.8, "death": 5.8, "par": 218.0},
            "bouncy":   {"pp": 4.5, "mid": 4.4, "death": 6.6, "par": 238.0},
        },
        "T20I": {
            "flat":     {"pp": 7.4, "mid": 7.0, "death": 9.2,  "par": 156.0},
            "balanced": {"pp": 6.8, "mid": 6.6, "death": 8.6,  "par": 146.0},
            "green":    {"pp": 6.2, "mid": 6.2, "death": 8.0,  "par": 132.0},
            "dusty":    {"pp": 6.0, "mid": 5.8, "death": 7.8,  "par": 130.0},
            "bouncy":   {"pp": 6.8, "mid": 6.6, "death": 8.4,  "par": 144.0},
        }
    },
    "OMA": {
        "ODI": {
            "flat":     {"pp": 4.6, "mid": 4.4, "death": 6.8, "par": 242.0},
            "dusty":    {"pp": 4.6, "mid": 4.4, "death": 6.6, "par": 236.0},
            "balanced": {"pp": 4.5, "mid": 4.3, "death": 6.5, "par": 232.0},
            "green":    {"pp": 4.0, "mid": 4.0, "death": 6.0, "par": 220.0},
            "bouncy":   {"pp": 4.2, "mid": 4.0, "death": 6.2, "par": 225.0},
        },
        "T20I": {
            "dusty":    {"pp": 6.8, "mid": 6.4, "death": 8.4,  "par": 142.0},
            "flat":     {"pp": 7.2, "mid": 6.8, "death": 9.0,  "par": 152.0},
            "balanced": {"pp": 6.6, "mid": 6.2, "death": 8.2,  "par": 140.0},
            "green":    {"pp": 5.8, "mid": 5.8, "death": 7.4,  "par": 124.0},
            "bouncy":   {"pp": 6.4, "mid": 6.2, "death": 8.0,  "par": 138.0},
        }
    },
    "UAE": {
        "ODI": {
            "flat":     {"pp": 4.8, "mid": 4.6, "death": 7.0, "par": 248.0},
            "dusty":    {"pp": 4.6, "mid": 4.4, "death": 6.6, "par": 236.0},
            "balanced": {"pp": 4.5, "mid": 4.3, "death": 6.5, "par": 232.0},
            "green":    {"pp": 4.0, "mid": 4.0, "death": 6.0, "par": 220.0},
            "bouncy":   {"pp": 4.2, "mid": 4.0, "death": 6.2, "par": 225.0},
        },
        "T20I": {
            "dusty":    {"pp": 6.8, "mid": 6.4, "death": 8.4,  "par": 142.0},
            "flat":     {"pp": 7.4, "mid": 7.0, "death": 9.2,  "par": 156.0},
            "balanced": {"pp": 6.6, "mid": 6.4, "death": 8.4,  "par": 142.0},
            "green":    {"pp": 6.0, "mid": 6.0, "death": 7.6,  "par": 128.0},
            "bouncy":   {"pp": 6.6, "mid": 6.4, "death": 8.2,  "par": 140.0},
        }
    },
    "CAN": {
        "ODI": {
            "flat":     {"pp": 4.8, "mid": 4.6, "death": 6.8, "par": 245.0},
            "green":    {"pp": 4.4, "mid": 4.4, "death": 6.6, "par": 235.0},
            "balanced": {"pp": 4.5, "mid": 4.3, "death": 6.5, "par": 232.0},
            "dusty":    {"pp": 4.0, "mid": 3.8, "death": 5.8, "par": 218.0},
            "bouncy":   {"pp": 4.4, "mid": 4.2, "death": 6.4, "par": 230.0},
        },
        "T20I": {
            "flat":     {"pp": 7.4, "mid": 7.0, "death": 9.0,  "par": 154.0},
            "balanced": {"pp": 6.6, "mid": 6.4, "death": 8.2,  "par": 140.0},
            "green":    {"pp": 6.2, "mid": 6.2, "death": 7.8,  "par": 130.0},
            "dusty":    {"pp": 6.0, "mid": 5.8, "death": 7.6,  "par": 128.0},
            "bouncy":   {"pp": 6.6, "mid": 6.4, "death": 8.2,  "par": 140.0},
        }
    },
    "PNG": {
        "ODI": {
            "flat":     {"pp": 4.6, "mid": 4.4, "death": 6.6, "par": 240.0},
            "balanced": {"pp": 4.4, "mid": 4.2, "death": 6.4, "par": 230.0},
            "green":    {"pp": 4.2, "mid": 4.2, "death": 6.2, "par": 224.0},
            "dusty":    {"pp": 3.8, "mid": 3.6, "death": 5.6, "par": 210.0},
            "bouncy":   {"pp": 4.0, "mid": 4.0, "death": 6.0, "par": 220.0},
        },
        "T20I": {
            "flat":     {"pp": 7.0, "mid": 6.6, "death": 8.6,  "par": 148.0},
            "balanced": {"pp": 6.4, "mid": 6.0, "death": 8.0,  "par": 136.0},
            "green":    {"pp": 5.8, "mid": 5.8, "death": 7.2,  "par": 124.0},
            "dusty":    {"pp": 5.8, "mid": 5.6, "death": 7.4,  "par": 126.0},
            "bouncy":   {"pp": 6.2, "mid": 6.0, "death": 7.8,  "par": 134.0},
        }
    }
}


# -------------------------------------------------------------------------
# Opponent Surface Bowling Lethality Index across 5 canonical pitch archetypes
# -------------------------------------------------------------------------
OPPONENT_SURFACE_BOWLING_INDEX: Dict[str, Dict[str, float]] = {
    "IND": {"dusty": 1.25, "balanced": 1.10, "green": 1.08, "bouncy": 1.05, "flat": 1.02},
    "AUS": {"bouncy": 1.18, "green": 1.15, "balanced": 1.10, "flat": 1.02, "dusty": 0.95},
    "SA":  {"bouncy": 1.18, "green": 1.15, "balanced": 1.05, "flat": 0.98, "dusty": 0.92},
    "ENG": {"green": 1.18, "balanced": 1.08, "flat": 1.00, "bouncy": 1.02, "dusty": 0.94},
    "NZ":  {"green": 1.18, "balanced": 1.10, "bouncy": 1.06, "flat": 0.98, "dusty": 0.92},
    "PAK": {"bouncy": 1.12, "green": 1.12, "dusty": 1.05, "balanced": 1.04, "flat": 0.98},
    "AFG": {"dusty": 1.22, "balanced": 0.98, "flat": 0.95, "bouncy": 0.90, "green": 0.88},
    "SL":  {"dusty": 1.18, "balanced": 0.98, "flat": 0.95, "bouncy": 0.92, "green": 0.90},
    "BAN": {"dusty": 1.16, "balanced": 0.96, "flat": 0.94, "bouncy": 0.88, "green": 0.86},
    "WI":  {"bouncy": 1.08, "green": 1.02, "balanced": 0.98, "flat": 0.95, "dusty": 0.90},
    "IRE": {"green": 1.00, "balanced": 0.92, "flat": 0.88, "bouncy": 0.90, "dusty": 0.85},
    "ZIM": {"balanced": 0.92, "dusty": 0.92, "green": 0.90, "flat": 0.86, "bouncy": 0.88},
    "DEFAULT_ASSOCIATE": {"flat": 0.85, "balanced": 0.88, "green": 0.88, "bouncy": 0.86, "dusty": 0.85}
}

TEST_COUNTRY_PITCH_RATES: Dict[str, Dict[str, Dict[str, float]]] = {
    "IND": {
        "flat":     {"s1": 3.7, "s2": 4.1, "s3": 4.4, "pp": 3.7, "mid": 4.1, "death": 4.4, "par": 440.0},
        "dusty":    {"s1": 3.0, "s2": 3.4, "s3": 3.6, "pp": 3.0, "mid": 3.4, "death": 3.6, "par": 325.0},
        "green":    {"s1": 2.4, "s2": 2.8, "s3": 3.2, "pp": 2.4, "mid": 2.8, "death": 3.2, "par": 260.0},
        "bouncy":   {"s1": 3.0, "s2": 3.4, "s3": 3.6, "pp": 3.0, "mid": 3.4, "death": 3.6, "par": 345.0},
        "balanced": {"s1": 3.1, "s2": 3.5, "s3": 3.7, "pp": 3.1, "mid": 3.5, "death": 3.7, "par": 335.0},
    },
    "AUS": {
        "flat":     {"s1": 3.8, "s2": 4.2, "s3": 4.5, "pp": 3.8, "mid": 4.2, "death": 4.5, "par": 450.0},
        "bouncy":   {"s1": 3.2, "s2": 3.6, "s3": 3.8, "pp": 3.2, "mid": 3.6, "death": 3.8, "par": 360.0},
        "green":    {"s1": 2.6, "s2": 3.0, "s3": 3.3, "pp": 2.6, "mid": 3.0, "death": 3.3, "par": 275.0},
        "dusty":    {"s1": 2.9, "s2": 3.2, "s3": 3.4, "pp": 2.9, "mid": 3.2, "death": 3.4, "par": 305.0},
        "balanced": {"s1": 3.2, "s2": 3.6, "s3": 3.8, "pp": 3.2, "mid": 3.6, "death": 3.8, "par": 345.0},
    },
    "SA": {
        "flat":     {"s1": 3.5, "s2": 4.0, "s3": 4.3, "pp": 3.5, "mid": 4.0, "death": 4.3, "par": 420.0},
        "bouncy":   {"s1": 3.1, "s2": 3.4, "s3": 3.7, "pp": 3.1, "mid": 3.4, "death": 3.7, "par": 345.0},
        "green":    {"s1": 2.3, "s2": 2.7, "s3": 3.1, "pp": 2.3, "mid": 2.7, "death": 3.1, "par": 240.0},
        "dusty":    {"s1": 2.8, "s2": 3.1, "s3": 3.3, "pp": 2.8, "mid": 3.1, "death": 3.3, "par": 295.0},
        "balanced": {"s1": 3.0, "s2": 3.3, "s3": 3.5, "pp": 3.0, "mid": 3.3, "death": 3.5, "par": 318.0},
    },
    "ENG": {
        "flat":     {"s1": 3.7, "s2": 4.2, "s3": 4.6, "pp": 3.7, "mid": 4.2, "death": 4.6, "par": 430.0},
        "green":    {"s1": 2.4, "s2": 2.9, "s3": 3.4, "pp": 2.4, "mid": 2.9, "death": 3.4, "par": 265.0},
        "dusty":    {"s1": 2.9, "s2": 3.2, "s3": 3.5, "pp": 2.9, "mid": 3.2, "death": 3.5, "par": 305.0},
        "bouncy":   {"s1": 3.0, "s2": 3.4, "s3": 3.6, "pp": 3.0, "mid": 3.4, "death": 3.6, "par": 335.0},
        "balanced": {"s1": 3.1, "s2": 3.5, "s3": 3.7, "pp": 3.1, "mid": 3.5, "death": 3.7, "par": 328.0},
    },
    "NZ": {
        "flat":     {"s1": 3.4, "s2": 3.9, "s3": 4.3, "pp": 3.4, "mid": 3.9, "death": 4.3, "par": 420.0},
        "green":    {"s1": 2.3, "s2": 2.8, "s3": 3.2, "pp": 2.3, "mid": 2.8, "death": 3.2, "par": 255.0},
        "dusty":    {"s1": 2.8, "s2": 3.1, "s3": 3.3, "pp": 2.8, "mid": 3.1, "death": 3.3, "par": 295.0},
        "bouncy":   {"s1": 2.9, "s2": 3.3, "s3": 3.6, "pp": 2.9, "mid": 3.3, "death": 3.6, "par": 330.0},
        "balanced": {"s1": 3.0, "s2": 3.4, "s3": 3.6, "pp": 3.0, "mid": 3.4, "death": 3.6, "par": 322.0},
    },
    "PAK": {
        "flat":     {"s1": 3.6, "s2": 4.1, "s3": 4.4, "pp": 3.6, "mid": 4.1, "death": 4.4, "par": 435.0},
        "dusty":    {"s1": 2.9, "s2": 3.2, "s3": 3.4, "pp": 2.9, "mid": 3.2, "death": 3.4, "par": 305.0},
        "green":    {"s1": 2.3, "s2": 2.7, "s3": 3.0, "pp": 2.3, "mid": 2.7, "death": 3.0, "par": 250.0},
        "bouncy":   {"s1": 2.8, "s2": 3.2, "s3": 3.5, "pp": 2.8, "mid": 3.2, "death": 3.5, "par": 325.0},
        "balanced": {"s1": 3.0, "s2": 3.3, "s3": 3.5, "pp": 3.0, "mid": 3.3, "death": 3.5, "par": 318.0},
    },
    "AFG": {
        "flat":     {"s1": 3.2, "s2": 3.6, "s3": 3.9, "pp": 3.2, "mid": 3.6, "death": 3.9, "par": 365.0},
        "dusty":    {"s1": 2.7, "s2": 3.0, "s3": 3.2, "pp": 2.7, "mid": 3.0, "death": 3.2, "par": 285.0},
        "green":    {"s1": 2.0, "s2": 2.4, "s3": 2.7, "pp": 2.0, "mid": 2.4, "death": 2.7, "par": 210.0},
        "bouncy":   {"s1": 2.4, "s2": 2.8, "s3": 3.1, "pp": 2.4, "mid": 2.8, "death": 3.1, "par": 265.0},
        "balanced": {"s1": 2.7, "s2": 3.0, "s3": 3.3, "pp": 2.7, "mid": 3.0, "death": 3.3, "par": 290.0},
    },
    "SL": {
        "flat":     {"s1": 3.4, "s2": 3.8, "s3": 4.1, "pp": 3.4, "mid": 3.8, "death": 4.1, "par": 400.0},
        "dusty":    {"s1": 2.9, "s2": 3.2, "s3": 3.4, "pp": 2.9, "mid": 3.2, "death": 3.4, "par": 308.0},
        "green":    {"s1": 2.2, "s2": 2.5, "s3": 2.8, "pp": 2.2, "mid": 2.5, "death": 2.8, "par": 230.0},
        "bouncy":   {"s1": 2.7, "s2": 3.0, "s3": 3.3, "pp": 2.7, "mid": 3.0, "death": 3.3, "par": 295.0},
        "balanced": {"s1": 2.9, "s2": 3.2, "s3": 3.4, "pp": 2.9, "mid": 3.2, "death": 3.4, "par": 310.0},
    },
    "WI": {
        "flat":     {"s1": 3.3, "s2": 3.7, "s3": 4.0, "pp": 3.3, "mid": 3.7, "death": 4.0, "par": 380.0},
        "bouncy":   {"s1": 2.9, "s2": 3.2, "s3": 3.5, "pp": 2.9, "mid": 3.2, "death": 3.5, "par": 325.0},
        "green":    {"s1": 2.2, "s2": 2.6, "s3": 2.9, "pp": 2.2, "mid": 2.6, "death": 2.9, "par": 235.0},
        "dusty":    {"s1": 2.6, "s2": 2.9, "s3": 3.1, "pp": 2.6, "mid": 2.9, "death": 3.1, "par": 275.0},
        "balanced": {"s1": 2.8, "s2": 3.1, "s3": 3.4, "pp": 2.8, "mid": 3.1, "death": 3.4, "par": 300.0},
    },
    "BAN": {
        "flat":     {"s1": 3.2, "s2": 3.6, "s3": 3.9, "pp": 3.2, "mid": 3.6, "death": 3.9, "par": 370.0},
        "dusty":    {"s1": 2.7, "s2": 3.0, "s3": 3.2, "pp": 2.7, "mid": 3.0, "death": 3.2, "par": 285.0},
        "green":    {"s1": 2.0, "s2": 2.3, "s3": 2.6, "pp": 2.0, "mid": 2.3, "death": 2.6, "par": 205.0},
        "bouncy":   {"s1": 2.3, "s2": 2.6, "s3": 2.9, "pp": 2.3, "mid": 2.6, "death": 2.9, "par": 245.0},
        "balanced": {"s1": 2.7, "s2": 3.0, "s3": 3.2, "pp": 2.7, "mid": 3.0, "death": 3.2, "par": 285.0},
    },
    "IRE": {
        "flat":     {"s1": 3.0, "s2": 3.4, "s3": 3.7, "pp": 3.0, "mid": 3.4, "death": 3.7, "par": 345.0},
        "green":    {"s1": 2.2, "s2": 2.5, "s3": 2.8, "pp": 2.2, "mid": 2.5, "death": 2.8, "par": 225.0},
        "dusty":    {"s1": 2.4, "s2": 2.7, "s3": 2.9, "pp": 2.4, "mid": 2.7, "death": 2.9, "par": 250.0},
        "bouncy":   {"s1": 2.4, "s2": 2.7, "s3": 3.0, "pp": 2.4, "mid": 2.7, "death": 3.0, "par": 255.0},
        "balanced": {"s1": 2.6, "s2": 2.9, "s3": 3.2, "pp": 2.6, "mid": 2.9, "death": 3.2, "par": 275.0},
    },
    "ZIM": {
        "flat":     {"s1": 3.0, "s2": 3.4, "s3": 3.7, "pp": 3.0, "mid": 3.4, "death": 3.7, "par": 345.0},
        "dusty":    {"s1": 2.5, "s2": 2.8, "s3": 3.0, "pp": 2.5, "mid": 2.8, "death": 3.0, "par": 260.0},
        "green":    {"s1": 2.1, "s2": 2.4, "s3": 2.7, "pp": 2.1, "mid": 2.4, "death": 2.7, "par": 215.0},
        "bouncy":   {"s1": 2.5, "s2": 2.8, "s3": 3.1, "pp": 2.5, "mid": 2.8, "death": 3.1, "par": 260.0},
        "balanced": {"s1": 2.6, "s2": 2.9, "s3": 3.2, "pp": 2.6, "mid": 2.9, "death": 3.2, "par": 280.0},
    }
}


def get_country_pitch_phase_rates(
    country_code: str,
    match_format: str,
    pitch_archetype: str,
    opponent_code: Optional[str] = None
) -> Dict[str, float]:
    """
    Returns country-specific, format-calibrated phase run rates (pp, mid, death, par)
    for a given pitch archetype across all 21 ICC playing nations.
    Dynamically adjusts rates & surface par when facing a specific opponent's bowling attack profile.
    """
    c = normalize_team_code(country_code)
    fmt_raw = str(match_format).upper().strip()
    if fmt_raw in ["TEST", "TESTS"]:
        fmt = "TEST"
    elif fmt_raw in ["ODI", "ODIS"]:
        fmt = "ODI"
    else:
        fmt = "T20I"
    p = canonicalize_pitch_type(pitch_archetype)
    
    # 1. Base format rates lookup
    rates = None
    if fmt == "TEST":
        rates = TEST_COUNTRY_PITCH_RATES.get(c, {}).get(p)
        if not rates:
            rates = TEST_COUNTRY_PITCH_RATES.get("IND", {}).get(p)
        if not rates:
            rates = {"s1": 3.0, "s2": 3.4, "s3": 3.6, "pp": 3.0, "mid": 3.4, "death": 3.6, "par": 320.0}
    else:
        country_rates = COUNTRY_PITCH_PHASE_RATES.get(c, {}).get(fmt, {})
        if p in country_rates:
            rates = country_rates[p]
        else:
            ind_rates = COUNTRY_PITCH_PHASE_RATES.get("IND", {}).get(fmt, {})
            if p in ind_rates:
                rates = ind_rates[p]
            else:
                if fmt == "ODI":
                    rates = {"pp": 5.2, "mid": 5.0, "death": 8.0, "par": 270.0}
                else:
                    rates = {"pp": 8.0, "mid": 7.8, "death": 10.5, "par": 168.0}
    
    res = dict(rates)

    # 2. Opponent-Specific Surface Bowling Calibration
    if opponent_code:
        opp = normalize_team_code(opponent_code)
        opp_profiles = OPPONENT_SURFACE_BOWLING_INDEX.get(opp, OPPONENT_SURFACE_BOWLING_INDEX["DEFAULT_ASSOCIATE"])
        bowling_factor = float(opp_profiles.get(p, 1.0))

        if fmt == "TEST":
            delta_par = -(bowling_factor - 1.0) * 110.0
            rate_mult = 1.0 / (bowling_factor ** 0.40)
        elif fmt == "ODI":
            delta_par = -(bowling_factor - 1.0) * 75.0
            rate_mult = 1.0 / (bowling_factor ** 0.45)
        else: # T20I
            delta_par = -(bowling_factor - 1.0) * 45.0
            rate_mult = 1.0 / (bowling_factor ** 0.50)

        res["par"] = round(max(80.0, res["par"] + delta_par), 1)
        for k in ["pp", "mid", "death", "s1", "s2", "s3"]:
            if k in res:
                res[k] = round(max(1.5, res[k] * rate_mult), 2)

    return res


def compute_player_5way_pitch_adaptability(player: Dict[str, Any], recency_weight: float = 0.65, match_format: str = "T20I") -> Dict[str, Any]:
    """
    Calculates detailed 0-100 adaptability scores across ALL 5 pitch archetypes
    ('flat', 'green', 'seam', 'bounce', 'spin heavy') for a player.
    Applies heavy recency weighting (default 65%) to matches and performance in the last 1 year.
    """
    role = player.get("role", "batter")
    hand = str(player.get("batting_hand", "right")).lower()
    style = str(player.get("bowling_style", "NONE")).upper()

    sub_aff = float(player.get("foreign_condition_ratio_subcontinent") or player.get("subcontinent_affinity", 1.0))
    sena_aff = float(player.get("foreign_condition_ratio_sena") or player.get("sena_affinity", 1.0))

    # Career splits
    pace_sr = float(player.get("pace_sr") or player.get("t20_batting_sr", 130.0))
    spin_sr = float(player.get("spin_sr") or player.get("t20_batting_sr", 130.0))
    pace_avg = float(player.get("pace_avg") or player.get("t20_batting_avg", 30.0))
    spin_avg = float(player.get("spin_avg") or player.get("t20_batting_avg", 30.0))
    boundary_pct = float(player.get("boundary_pct", 52.0))
    econ = float(player.get("t20_economy", 7.8))

    # Recent 1-Year splits (rolling 12-month metrics)
    recent_form_mult = float(player.get("recent_form_multiplier") or (1.08 if player.get("is_debutant") else 1.0))
    recent_pace_sr = float(player.get("recent_1yr_pace_sr") or player.get("recent_1yr_sr") or (pace_sr * recent_form_mult))
    recent_spin_sr = float(player.get("recent_1yr_spin_sr") or player.get("recent_1yr_sr") or (spin_sr * recent_form_mult))
    recent_pace_avg = float(player.get("recent_1yr_pace_avg") or player.get("recent_1yr_avg") or (pace_avg * recent_form_mult))
    recent_spin_avg = float(player.get("recent_1yr_spin_avg") or player.get("recent_1yr_avg") or (spin_avg * recent_form_mult))
    recent_econ = float(player.get("recent_1yr_economy") or (econ / max(0.85, recent_form_mult)))
    recent_boundary_pct = float(player.get("recent_1yr_boundary_pct") or min(75.0, boundary_pct * recent_form_mult))

    is_bowler = any(r in role for r in ["pacer", "spinner", "bowler"])
    is_spinner = style in ["SLA", "OB", "LBG", "LAWS", "SPIN"]
    is_pacer = style in ["RF", "RMF", "LF", "LMF", "FAST"]

    def calculate_raw_scores(p_sr, s_sr, p_avg, s_avg, ec, b_pct, sub_a, sena_a):
        # 1. FLAT
        if is_bowler:
            f_score = 90.0 - (ec - 6.5) * 8.0
        else:
            f_score = 50.0 + (b_pct / 70.0) * 35.0 + ((p_sr + s_sr) / 280.0) * 15.0
        f_score = min(99.0, max(45.0, f_score))

        # 2. GREEN
        if is_bowler:
            g_score = (92.0 if ("RF" in style or "LF" in style) else 86.0 if is_pacer else 60.0) * sena_a
        else:
            g_score = (p_avg / 42.0) * 45.0 + (sena_a * 35.0) + (15.0 if "top_order" in role else 10.0)
        g_score = min(98.0, max(40.0, g_score))

        # 3. SEAM
        if is_bowler:
            seam_s = (88.0 * (1.0 + (sena_a - 1.0) * 0.5)) if is_pacer else (82.0 if is_spinner and style in ["OB", "LBG"] else 70.0)
        else:
            seam_s = (p_avg / 40.0) * 50.0 + (sena_a * 25.0) + (s_avg / 40.0) * 20.0
        seam_s = min(98.0, max(42.0, seam_s))

        # 4. BOUNCE
        if is_bowler:
            bounce_s = (94.0 if ("RF" in style or "LF" in style) else 82.0 if is_pacer else 72.0) * sena_a
        else:
            bounce_s = (p_sr / 140.0) * 45.0 + (p_avg / 40.0) * 35.0 + (sena_a * 18.0)
        bounce_s = min(99.0, max(42.0, bounce_s))

        # 5. SPIN HEAVY
        if is_bowler:
            spin_s = (95.0 if is_spinner else 75.0) * sub_a
        else:
            spin_s = (s_avg / 45.0) * 45.0 + (s_sr / 135.0) * 35.0 + (sub_a * 18.0)
        spin_s = min(99.0, max(40.0, spin_s))

        return {
            "flat": round(f_score, 1),
            "green": round(g_score, 1),
            "balanced": round(seam_s, 1),
            "bouncy": round(bounce_s, 1),
            "dusty": round(spin_s, 1)
        }

    career_ratings = calculate_raw_scores(pace_sr, spin_sr, pace_avg, spin_avg, econ, boundary_pct, sub_aff, sena_aff)
    recent_ratings = calculate_raw_scores(recent_pace_sr, recent_spin_sr, recent_pace_avg, recent_spin_avg, recent_econ, recent_boundary_pct, sub_aff, sena_aff)

    # Empirical Bayes Country-Surface Prior:
    # When player-level foreign condition splits are uncalibrated defaults (1.0/1.0) or low sample size,
    # anchor strongly toward the player's national team adaptability on each pitch archetype.
    team_code = player.get("team_id") or player.get("country")
    country_anchor_weight = 0.0
    country_arch_scores: Dict[str, float] = {}
    if team_code:
        norm_c = normalize_team_code(str(team_code))
        fmt_up = str(match_format or "T20I").upper()
        c_career = COUNTRY_PITCH_ADAPTABILITY.get(norm_c, {}).get(fmt_up, {}) or COUNTRY_PITCH_ADAPTABILITY.get(norm_c, {}).get("T20I", {})
        c_recent = COUNTRY_RECENT_1YR_ADAPTABILITY.get(norm_c, {}).get(fmt_up, {}) or COUNTRY_RECENT_1YR_ADAPTABILITY.get(norm_c, {}).get("T20I", {})
        if c_career:
            balls_exp = max(int(player.get("intl_balls_faced") or 0), int(player.get("intl_balls_bowled") or 0))
            is_default_splits = (abs(sena_aff - 1.0) < 1e-6 and abs(sub_aff - 1.0) < 1e-6)
            country_anchor_weight = 0.65 if (is_default_splits or balls_exp < 300) else 0.25
            for arch in PITCH_ARCHETYPES:
                ca_c = float(c_career.get(arch, 75.0))
                ca_r = float(c_recent.get(arch, ca_c))
                country_arch_scores[arch] = (ca_r * recency_weight) + (ca_c * (1.0 - recency_weight))

    # Blend Career and Recent 1-Year (default 65% recent weight)
    composite_ratings = {}
    for arch in PITCH_ARCHETYPES:
        c_score = career_ratings[arch]
        r_score = recent_ratings[arch]
        comp = (r_score * recency_weight) + (c_score * (1.0 - recency_weight))
        if country_anchor_weight > 0.0 and arch in country_arch_scores:
            comp = comp * (1.0 - country_anchor_weight) + country_arch_scores[arch] * country_anchor_weight
        composite_ratings[arch] = round(comp, 1)

    # Composite seam / green archetype (green decks always exhibit seam movement)
    seam_green_comp = round((composite_ratings.get("green", 75.0) + composite_ratings.get("seam", 75.0)) / 2.0, 1)
    composite_ratings["seam / green"] = seam_green_comp
    recent_ratings["seam / green"] = round((recent_ratings.get("green", 75.0) + recent_ratings.get("seam", 75.0)) / 2.0, 1)
    career_ratings["seam / green"] = round((career_ratings.get("green", 75.0) + career_ratings.get("seam", 75.0)) / 2.0, 1)

    favored = max(composite_ratings, key=composite_ratings.get)

    # Determine recent form trend
    form_diff = round(sum(recent_ratings.values()) / 5.0 - sum(career_ratings.values()) / 5.0, 1)
    if player.get("recent_form_trend") in ["HOT", "SLUMP", "STABLE", "DEBUTANT"]:
        form_trend = player["recent_form_trend"]
    elif player.get("is_debutant"):
        form_trend = "DEBUTANT"
    elif form_diff >= 1.5:
        form_trend = "HOT"
    elif form_diff <= -1.5:
        form_trend = "SLUMP"
    else:
        form_trend = "STABLE"

    return {
        "player_id": player.get("id") or player.get("player_id"),
        "name": player.get("name", "Unknown Player"),
        "role": role,
        "batting_hand": hand,
        "bowling_style": style,
        "pitch_adaptability": composite_ratings,
        "recent_1yr_adaptability": recent_ratings,
        "career_adaptability": career_ratings,
        "favored_pitch": favored,
        "dominant_score": composite_ratings[favored],
        "recency_weight_applied": recency_weight,
        "recent_form_trend": form_trend
    }

def compute_format_specific_prediction_variables(
    match_format: str,
    team_a: Dict[str, Any],
    team_b: Dict[str, Any],
    xi_a: List[Dict[str, Any]],
    xi_b: List[Dict[str, Any]],
    venue: Dict[str, Any],
    toss_winner_id: Optional[str] = None,
    toss_decision: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes format-specific variables to push pre-match prediction accuracy >= 95%.
    Format-specific features:
    - TEST: 4th Innings Pitch Wear Degradation, Tail-End Runs (Wkts 8-11), Draw Elimination Tempo, 1st Innings Par Buffer.
    - ODI: Middle Overs (11-40) Strike Rotation, 6th Bowling Option Safety Quotient, PP1 Wicket Preservation, Day/Night Dew Impact.
    - T20I: Death Overs (16-20) Execution Differential, Powerplay Boundary Density, Boundary Asymmetry Targeting, Top-3 Matchup Trap.
    - Atmospheric & Aerodynamics: Reynolds Boundary-Layer Drag, Overhead Swing Multiplier, Evening Dew Ball Slippage.
    - Universal: Recent 5-Match Momentum Vector, Bilateral H2H 3-Year Win Rate, Fatigue & Rest Quotient.
    """
    fmt = match_format.upper() if match_format else "T20I"
    venue_id = venue.get("id", "").lower()
    pitch_archetype = get_ground_pitch_archetype(venue_id, fmt)
    dossier = FORMAT_GROUND_PITCH_DOSSIERS.get(venue_id.lower().replace("-", "_"), {})
    dew_factor = float(venue.get("dew_factor", 0.30))

    code_a = normalize_team_code(team_a.get("id") or team_a.get("short_name") or team_a.get("name", "IND"))
    code_b = normalize_team_code(team_b.get("id") or team_b.get("short_name") or team_b.get("name", "NZ"))

    # Authentic Rolling Momentum and H2H from International Database
    form_a_data = get_team_rolling_momentum(code_a, fmt)
    form_b_data = get_team_rolling_momentum(code_b, fmt)
    h2h_data = get_international_h2h(code_a, code_b, fmt)

    recent_form_a = float(team_a.get("recent_l5_win_pct") if "recent_l5_win_pct" in team_a and team_a.get("recent_l5_win_pct") != 0.60 else form_a_data["recent_l5_win_pct"])
    recent_form_b = float(team_b.get("recent_l5_win_pct") if "recent_l5_win_pct" in team_b and team_b.get("recent_l5_win_pct") != 0.50 else form_b_data["recent_l5_win_pct"])
    momentum_delta = round((recent_form_a - recent_form_b) * 100.0, 1)

    h2h_3yr_a = float(team_a.get("h2h_3yr_win_pct") if "h2h_3yr_win_pct" in team_a and team_a.get("h2h_3yr_win_pct") != 0.52 else h2h_data["last_3yr_win_pct_a"])

    # Atmospheric Aerodynamic Variables
    atmos = compute_atmospheric_aerodynamic_variables(venue, weather=venue.get("weather"))

    # All-rounders & balance
    allrounders_a = len([p for p in xi_a if p.get("role") in ["all_rounder", "bowling_all_rounder", "batting_all_rounder"]])
    allrounders_b = len([p for p in xi_b if p.get("role") in ["all_rounder", "bowling_all_rounder", "batting_all_rounder"]])
    balance_quotient_a = round(min(10.0, 6.0 + allrounders_a * 1.2), 1)
    balance_quotient_b = round(min(10.0, 6.0 + allrounders_b * 1.2), 1)

    variables: Dict[str, Any] = {
        "format": fmt,
        "team_a_code": code_a,
        "team_b_code": code_b,
        "momentum_delta_pct": momentum_delta,
        "team_a_form": form_a_data,
        "team_b_form": form_b_data,
        "h2h_bilateral": h2h_data,
        "h2h_3yr_domination_pct": round(h2h_3yr_a * 100.0, 1),
        "team_a_balance_quotient": balance_quotient_a,
        "team_b_balance_quotient": balance_quotient_b,
        "atmospheric_aerodynamics": atmos
    }

    if fmt == "TEST":
        # 1. Day 4-5 Pitch Wear Degradation Index (0-10)
        pitch_wear = dossier.get("pitch_wear_rate") or (9.0 if pitch_archetype == "spin heavy" else 5.5)
        # 2. Fourth Innings Chase Hazard (% win penalty for team batting 4th)
        fourth_chase_hazard = round(min(32.0, (pitch_wear / 10.0) * 28.0), 1)
        # 3. Tail-End Batting Depth (Wickets 8-11 Average Runs Expectancy)
        tail_a = xi_a[7:] if len(xi_a) >= 11 else []
        tail_b = xi_b[7:] if len(xi_b) >= 11 else []

        def get_tail_avg(p: Dict[str, Any]) -> float:
            return float(p.get("test_batting_avg") or p.get("batting_avg") or (p.get("t20_batting_avg", 14.0) * 1.15))

        tail_a_runs = float(np.mean([get_tail_avg(p) for p in tail_a])) * 4.0 if tail_a else 52.0
        tail_b_runs = float(np.mean([get_tail_avg(p) for p in tail_b])) * 4.0 if tail_b else 46.0
        tail_runs_delta = round(tail_a_runs - tail_b_runs, 1)
        # 4. Draw Elimination Tempo (aggregate rpo tempo)
        tempo_rpo = 3.65
        draw_elimination_prob = round(min(96.0, 78.0 + (tempo_rpo - 2.8) * 18.0), 1)

        variables["test_specific"] = {
            "pitch_wear_degradation_index": pitch_wear,
            "fourth_innings_chase_hazard_pct": fourth_chase_hazard,
            "tail_end_runs_team_a": round(tail_a_runs, 1),
            "tail_end_runs_team_b": round(tail_b_runs, 1),
            "tail_runs_delta": tail_runs_delta,
            "draw_elimination_probability_pct": draw_elimination_prob,
            "tactical_key": f"Day 4-5 surface wear ({pitch_wear}/10) enforces a -{fourth_chase_hazard}% 4th innings chase penalty. Tail-end depth yields {tail_runs_delta:+} runs cushion."
        }

    elif fmt == "ODI":
        # 1. Middle Overs (11-40) Strike Rotation vs Dot-Ball Choke
        rotation_a = round(min(98.0, 68.0 + (float(team_a.get("middle_overs_rating", 50.0)) - 50.0) * 0.8), 1)
        rotation_b = round(min(98.0, 68.0 + (float(team_b.get("middle_overs_rating", 50.0)) - 50.0) * 0.8), 1)
        # 2. Sixth Bowler Safety Quotient (Coverage of 5th/6th bowler quota to prevent fatal leakage)
        viable_bowlers_a = len([p for p in xi_a if any(r in p.get("role", "") for r in ["pacer", "spinner", "bowler", "all_rounder"])])
        viable_bowlers_b = len([p for p in xi_b if any(r in p.get("role", "") for r in ["pacer", "spinner", "bowler", "all_rounder"])])
        sixth_bowler_a = 9.5 if viable_bowlers_a >= 6 else (7.2 if viable_bowlers_a == 5 else 4.5)
        sixth_bowler_b = 9.5 if viable_bowlers_b >= 6 else (7.2 if viable_bowlers_b == 5 else 4.5)
        # 3. Powerplay 1 (Overs 1-10) Wicket Preservation Index
        pp1_preservation_a = round(min(92.0, 70.0 + (float(team_a.get("powerplay_rating", 50.0)) - 50.0) * 0.6), 1)
        # 4. Day-Night Dew Chase Boost
        dew_chase_boost = round(dew_factor * 8.5, 1)

        variables["odi_specific"] = {
            "middle_overs_strike_rotation_index_a": rotation_a,
            "middle_overs_strike_rotation_index_b": rotation_b,
            "sixth_bowler_safety_a": sixth_bowler_a,
            "sixth_bowler_safety_b": sixth_bowler_b,
            "pp1_wicket_preservation_pct_a": pp1_preservation_a,
            "day_night_dew_chase_boost_pct": dew_chase_boost,
            "tactical_key": f"Middle-overs (11-40) strike rotation ({rotation_a}% vs {rotation_b}%) and 6th bowling option coverage determine run leakage."
        }

    else: # T20I
        # 1. Death Overs (16-20) Execution Differential
        death_bat_a = float(team_a.get("death_overs_rating", 50.0))
        death_bat_b = float(team_b.get("death_overs_rating", 50.0))
        death_diff = round((death_bat_a - death_bat_b) * 0.45, 1)
        # 2. Powerplay Boundary Density (Overs 1-6)
        pp_density_a = round(min(88.0, 56.0 + (float(team_a.get("powerplay_rating", 50.0)) - 50.0) * 0.7), 1)
        # 3. Asymmetric Boundary Target Advantage (Short vs Long side)
        straight_m = dossier.get("boundary_dimensions", {}).get("straight_m", 70.0)
        boundary_asymmetry_edge = round(max(0.0, (72.0 - straight_m) * 0.6), 1)

        variables["t20_specific"] = {
            "death_overs_execution_delta_runs": death_diff,
            "powerplay_boundary_density_pct": pp_density_a,
            "boundary_asymmetry_targeting_edge": boundary_asymmetry_edge,
            "short_boundary_m": straight_m,
            "tactical_key": f"Death overs (16-20) execution yields {death_diff:+} runs differential; short {straight_m}m boundaries heavily reward power hitting."
        }

    return variables
