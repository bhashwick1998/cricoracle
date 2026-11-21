"""
CricOracle Player Strengths, Favored Pitch Conditions & Matchup Tensor
Models:
- Detailed player vectors: Pace SR vs Spin SR, Pace Avg vs Spin Avg, Boundary % vs Dot %
- Favored pitch condition affinities for each player (e.g. subcontinent turn vs SENA bounce)
- Country-level historical condition ratings (Home, Away Subcontinent, Away SENA)
- 11x6 Pairwise Matchup Matrix with Empirical Bayes shrinkage
"""

from typing import Dict, Any, List, Tuple, Optional
import math
from .debut_priors import compute_empirical_bayes_shrinkage

# 1. Historical Player Database with Pace/Spin Splits & Favored Pitch Affinities
HISTORICAL_PLAYER_DOSSIERS: Dict[str, Dict[str, Any]] = {
    # India
    "vkohli_ind": {
        "id": "vkohli_ind",
        "name": "Virat Kohli",
        "team_id": "IND",
        "role": "top_order_batter",
        "batting_hand": "right",
        "bowling_style": "RMF",
        "overall_sr": 138.2,
        "overall_avg": 51.8,
        "pace_sr": 136.5,
        "spin_sr": 142.0,
        "pace_avg": 49.2,
        "spin_avg": 56.4,
        "boundary_pct": 54.2,
        "dot_pct": 32.5,
        "favored_pitch": "balanced",
        "sena_affinity": 1.05,
        "subcontinent_affinity": 1.10,
        "pp_sr": 128.0,
        "mid_sr": 136.0,
        "death_sr": 188.0
    },
    "rsharma_ind": {
        "id": "rsharma_ind",
        "name": "Rohit Sharma",
        "team_id": "IND",
        "role": "top_order_batter",
        "batting_hand": "right",
        "bowling_style": "OB",
        "overall_sr": 140.5,
        "overall_avg": 32.5,
        "pace_sr": 144.0,
        "spin_sr": 134.5,
        "pace_avg": 34.0,
        "spin_avg": 30.5,
        "boundary_pct": 62.0,
        "dot_pct": 36.0,
        "favored_pitch": "batting_paradise",
        "sena_affinity": 0.98,
        "subcontinent_affinity": 1.08,
        "pp_sr": 152.0,
        "mid_sr": 132.0,
        "death_sr": 175.0
    },
    "skyadav_ind": {
        "id": "skyadav_ind",
        "name": "Suryakumar Yadav",
        "team_id": "IND",
        "role": "middle_order_batter",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 168.0,
        "overall_avg": 42.0,
        "pace_sr": 172.5,
        "spin_sr": 161.0,
        "pace_avg": 43.5,
        "spin_avg": 40.0,
        "boundary_pct": 68.4,
        "dot_pct": 28.0,
        "favored_pitch": "batting_paradise",
        "sena_affinity": 1.02,
        "subcontinent_affinity": 1.06,
        "pp_sr": 145.0,
        "mid_sr": 164.0,
        "death_sr": 210.0
    },
    "hardik_ind": {
        "id": "hardik_ind",
        "name": "Hardik Pandya",
        "team_id": "IND",
        "role": "all_rounder",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_sr": 142.0,
        "overall_avg": 27.5,
        "pace_sr": 146.0,
        "spin_sr": 135.0,
        "pace_avg": 28.0,
        "spin_avg": 26.5,
        "boundary_pct": 60.5,
        "dot_pct": 34.0,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.04,
        "subcontinent_affinity": 0.98,
        "pp_sr": 125.0,
        "mid_sr": 138.0,
        "death_sr": 182.0,
        "bowling_econ": 8.4,
        "wicket_hazard": 0.048
    },
    "jbumrah_ind": {
        "id": "jbumrah_ind",
        "name": "Jasprit Bumrah",
        "team_id": "IND",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 6.20,
        "bowling_sr": 14.2,
        "wicket_hazard": 0.062,
        "pp_econ": 5.4,
        "mid_econ": 6.1,
        "death_econ": 7.2,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.12,
        "subcontinent_affinity": 1.10
    },
    "k_yadav_ind": {
        "id": "k_yadav_ind",
        "name": "Kuldeep Yadav",
        "team_id": "IND",
        "role": "frontline_spinner",
        "batting_hand": "left",
        "bowling_style": "LAWS",
        "overall_econ": 6.70,
        "bowling_sr": 13.8,
        "wicket_hazard": 0.064,
        "pp_econ": 7.2,
        "mid_econ": 6.2,
        "death_econ": 8.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.94,
        "subcontinent_affinity": 1.22
    },

    # New Zealand
    "k_williamson_nz": {
        "id": "k_williamson_nz",
        "name": "Kane Williamson",
        "team_id": "NZ",
        "role": "top_order_batter",
        "batting_hand": "right",
        "bowling_style": "OB",
        "overall_sr": 126.0,
        "overall_avg": 33.5,
        "pace_sr": 124.0,
        "spin_sr": 130.0,
        "pace_avg": 34.0,
        "spin_avg": 32.5,
        "boundary_pct": 46.0,
        "dot_pct": 33.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.08,
        "subcontinent_affinity": 1.02,
        "pp_sr": 118.0,
        "mid_sr": 128.0,
        "death_sr": 160.0
    },
    "d_conway_nz": {
        "id": "d_conway_nz",
        "name": "Devon Conway",
        "team_id": "NZ",
        "role": "top_order_batter",
        "batting_hand": "left",
        "bowling_style": "NONE",
        "overall_sr": 133.0,
        "overall_avg": 39.5,
        "pace_sr": 130.0,
        "spin_sr": 138.0,
        "pace_avg": 38.0,
        "spin_avg": 42.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.06,
        "subcontinent_affinity": 1.02
    },
    "g_phillips_nz": {
        "id": "g_phillips_nz",
        "name": "Glenn Phillips",
        "team_id": "NZ",
        "role": "middle_order_batter",
        "batting_hand": "right",
        "bowling_style": "OB",
        "overall_sr": 146.0,
        "overall_avg": 32.5,
        "pace_sr": 142.0,
        "spin_sr": 150.0,
        "favored_pitch": "batting_paradise",
        "sena_affinity": 1.04,
        "subcontinent_affinity": 1.05
    },
    "r_ravindra_nz": {
        "id": "r_ravindra_nz",
        "name": "Rachin Ravindra",
        "team_id": "NZ",
        "role": "all_rounder",
        "batting_hand": "left",
        "bowling_style": "SLA",
        "overall_sr": 138.0,
        "overall_avg": 35.0,
        "pace_sr": 135.0,
        "spin_sr": 142.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 1.02,
        "subcontinent_affinity": 1.15
    },
    "l_ferguson_nz": {
        "id": "l_ferguson_nz",
        "name": "Lockie Ferguson",
        "team_id": "NZ",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 7.50,
        "bowling_sr": 14.2,
        "wicket_hazard": 0.055,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.14,
        "subcontinent_affinity": 0.92
    },
    "rpant_ind": {
        "id": "rpant_ind",
        "name": "Rishabh Pant",
        "team_id": "IND",
        "role": "wicket_keeper",
        "batting_hand": "left",
        "bowling_style": "NONE",
        "overall_sr": 134.0,
        "overall_avg": 34.0,
        "pace_sr": 138.0,
        "spin_sr": 130.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.08,
        "subcontinent_affinity": 1.02
    },
    "rjadeja_ind": {
        "id": "rjadeja_ind",
        "name": "Ravindra Jadeja",
        "team_id": "IND",
        "role": "all_rounder",
        "batting_hand": "left",
        "bowling_style": "SLA",
        "overall_econ": 6.80,
        "bowling_sr": 21.0,
        "wicket_hazard": 0.045,
        "overall_sr": 128.0,
        "overall_avg": 32.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.96,
        "subcontinent_affinity": 1.20
    },
    "akshar_ind": {
        "id": "akshar_ind",
        "name": "Axar Patel",
        "team_id": "IND",
        "role": "all_rounder",
        "batting_hand": "left",
        "bowling_style": "SLA",
        "overall_econ": 7.10,
        "bowling_sr": 20.5,
        "wicket_hazard": 0.046,
        "overall_sr": 138.0,
        "overall_avg": 26.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.98,
        "subcontinent_affinity": 1.18
    },
    "asir_ind": {
        "id": "asir_ind",
        "name": "Mohammed Siraj",
        "team_id": "IND",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 7.80,
        "bowling_sr": 16.5,
        "wicket_hazard": 0.052,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.12,
        "subcontinent_affinity": 0.98
    },
    "arsheep_ind": {
        "id": "arsheep_ind",
        "name": "Arshdeep Singh",
        "team_id": "IND",
        "role": "frontline_pacer",
        "batting_hand": "left",
        "bowling_style": "RF",
        "overall_econ": 8.10,
        "bowling_sr": 13.5,
        "wicket_hazard": 0.060,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.10,
        "subcontinent_affinity": 1.02
    },
    "d_mitchell_nz": {
        "id": "d_mitchell_nz",
        "name": "Daryl Mitchell",
        "team_id": "NZ",
        "role": "middle_order_batter",
        "batting_hand": "right",
        "bowling_style": "RMF",
        "overall_sr": 141.0,
        "overall_avg": 32.0,
        "pace_sr": 138.0,
        "spin_sr": 146.0,
        "pace_avg": 31.0,
        "spin_avg": 35.0,
        "boundary_pct": 58.0,
        "dot_pct": 30.0,
        "favored_pitch": "batting_paradise",
        "sena_affinity": 1.05,
        "subcontinent_affinity": 1.04,
        "pp_sr": 122.0,
        "mid_sr": 140.0,
        "death_sr": 178.0
    },
    "t_boult_nz": {
        "id": "t_boult_nz",
        "name": "Trent Boult",
        "team_id": "NZ",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 7.80,
        "bowling_sr": 15.0,
        "wicket_hazard": 0.058,
        "pp_econ": 6.6,
        "mid_econ": 7.8,
        "death_econ": 9.2,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.15,
        "subcontinent_affinity": 0.95
    },
    "m_santner_nz": {
        "id": "m_santner_nz",
        "name": "Mitchell Santner",
        "team_id": "NZ",
        "role": "all_rounder",
        "batting_hand": "left",
        "bowling_style": "SLA",
        "overall_econ": 6.90,
        "bowling_sr": 19.5,
        "wicket_hazard": 0.046,
        "pp_econ": 6.8,
        "mid_econ": 6.4,
        "death_econ": 8.6,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.98,
        "subcontinent_affinity": 1.14
    },

    # Australia
    "t_head_aus": {
        "id": "t_head_aus",
        "name": "Travis Head",
        "team_id": "AUS",
        "role": "top_order_batter",
        "batting_hand": "left",
        "bowling_style": "OB",
        "overall_sr": 158.0,
        "overall_avg": 34.0,
        "pace_sr": 168.0,
        "spin_sr": 132.0,
        "pace_avg": 38.0,
        "spin_avg": 25.5,
        "boundary_pct": 72.0,
        "dot_pct": 34.0,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.14,
        "subcontinent_affinity": 0.88,
        "pp_sr": 178.0,
        "mid_sr": 142.0,
        "death_sr": 165.0
    },
    "p_cummins_aus": {
        "id": "p_cummins_aus",
        "name": "Pat Cummins",
        "team_id": "AUS",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 7.30,
        "bowling_sr": 18.0,
        "wicket_hazard": 0.052,
        "pp_econ": 7.0,
        "mid_econ": 7.1,
        "death_econ": 8.4,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.12,
        "subcontinent_affinity": 0.96
    },
    "a_zampa_aus": {
        "id": "a_zampa_aus",
        "name": "Adam Zampa",
        "team_id": "AUS",
        "role": "frontline_spinner",
        "batting_hand": "right",
        "bowling_style": "LBG",
        "overall_econ": 7.20,
        "bowling_sr": 17.0,
        "wicket_hazard": 0.055,
        "pp_econ": 7.6,
        "mid_econ": 6.8,
        "death_econ": 8.6,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 1.02,
        "subcontinent_affinity": 1.12
    },

    # England
    "j_buttler_eng": {
        "id": "j_buttler_eng",
        "name": "Jos Buttler",
        "team_id": "ENG",
        "role": "wicket_keeper",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 145.0,
        "overall_avg": 35.0,
        "pace_sr": 148.0,
        "spin_sr": 140.0,
        "pace_avg": 36.0,
        "spin_avg": 33.5,
        "boundary_pct": 64.0,
        "dot_pct": 32.0,
        "favored_pitch": "batting_paradise",
        "sena_affinity": 1.04,
        "subcontinent_affinity": 1.02,
        "pp_sr": 148.0,
        "mid_sr": 136.0,
        "death_sr": 195.0
    },
    "j_archer_eng": {
        "id": "j_archer_eng",
        "name": "Jofra Archer",
        "team_id": "ENG",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 7.40,
        "bowling_sr": 15.0,
        "wicket_hazard": 0.058,
        "pp_econ": 6.5,
        "mid_econ": 7.2,
        "death_econ": 8.6,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.10,
        "subcontinent_affinity": 0.94
    },

    # Afghanistan
    "r_khan_afg": {
        "id": "r_khan_afg",
        "name": "Rashid Khan",
        "team_id": "AFG",
        "role": "all_rounder",
        "batting_hand": "right",
        "bowling_style": "LBG",
        "overall_econ": 6.20,
        "bowling_sr": 13.5,
        "wicket_hazard": 0.068,
        "pp_econ": 6.4,
        "mid_econ": 5.8,
        "death_econ": 7.4,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.98,
        "subcontinent_affinity": 1.25
    },

    # Zimbabwe
    "s_raza_zim": {
        "id": "s_raza_zim",
        "name": "Sikandar Raza",
        "team_id": "ZIM",
        "role": "all_rounder",
        "batting_hand": "right",
        "bowling_style": "OB",
        "overall_sr": 135.0,
        "overall_avg": 26.0,
        "pace_sr": 132.0,
        "spin_sr": 140.0,
        "pace_avg": 25.0,
        "spin_avg": 28.0,
        "boundary_pct": 58.0,
        "dot_pct": 36.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.92,
        "subcontinent_affinity": 1.08,
        "bowling_econ": 7.2,
        "wicket_hazard": 0.044
    },

    # India ODI additions
    "sgill_ind": {
        "id": "sgill_ind",
        "name": "Shubman Gill",
        "team_id": "IND",
        "role": "top_order_batter",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 102.5,
        "overall_avg": 58.2,
        "pace_sr": 105.0,
        "spin_sr": 98.0,
        "pace_avg": 56.0,
        "spin_avg": 62.0,
        "boundary_pct": 54.0,
        "dot_pct": 42.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.08,
        "subcontinent_affinity": 1.10,
        "pp_sr": 92.0,
        "mid_sr": 104.0,
        "death_sr": 165.0
    },
    "siyer_ind": {
        "id": "siyer_ind",
        "name": "Shreyas Iyer",
        "team_id": "IND",
        "role": "middle_order_batter",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 101.0,
        "overall_avg": 47.5,
        "pace_sr": 92.0,
        "spin_sr": 128.0,
        "pace_avg": 40.0,
        "spin_avg": 64.0,
        "boundary_pct": 56.0,
        "dot_pct": 38.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.94,
        "subcontinent_affinity": 1.18,
        "pp_sr": 85.0,
        "mid_sr": 108.0,
        "death_sr": 155.0
    },
    "klrahul_ind": {
        "id": "klrahul_ind",
        "name": "KL Rahul",
        "team_id": "IND",
        "role": "wicket_keeper",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 88.0,
        "overall_avg": 50.2,
        "pace_sr": 92.0,
        "spin_sr": 85.0,
        "pace_avg": 52.0,
        "spin_avg": 48.0,
        "boundary_pct": 48.0,
        "dot_pct": 44.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.05,
        "subcontinent_affinity": 1.04,
        "pp_sr": 82.0,
        "mid_sr": 88.0,
        "death_sr": 160.0
    },
    "wsundar_ind": {
        "id": "wsundar_ind",
        "name": "Washington Sundar",
        "team_id": "IND",
        "role": "all_rounder",
        "batting_hand": "left",
        "bowling_style": "OB",
        "overall_econ": 4.85,
        "bowling_sr": 28.0,
        "wicket_hazard": 0.038,
        "overall_sr": 86.0,
        "overall_avg": 29.5,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 1.02,
        "subcontinent_affinity": 1.12
    },
    "hrana_ind": {
        "id": "hrana_ind",
        "name": "Harshit Rana",
        "team_id": "IND",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 5.80,
        "bowling_sr": 22.0,
        "wicket_hazard": 0.048,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.06,
        "subcontinent_affinity": 0.98
    },

    # New Zealand ODI additions
    "t_latham_nz": {
        "id": "t_latham_nz",
        "name": "Tom Latham",
        "team_id": "NZ",
        "role": "wicket_keeper",
        "batting_hand": "left",
        "bowling_style": "NONE",
        "overall_sr": 86.0,
        "overall_avg": 35.8,
        "pace_sr": 84.0,
        "spin_sr": 94.0,
        "pace_avg": 34.0,
        "spin_avg": 46.0,
        "boundary_pct": 46.0,
        "dot_pct": 48.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.04,
        "subcontinent_affinity": 1.10
    },
    "w_young_nz": {
        "id": "w_young_nz",
        "name": "Will Young",
        "team_id": "NZ",
        "role": "top_order_batter",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 89.0,
        "overall_avg": 42.0,
        "pace_sr": 92.0,
        "spin_sr": 84.0,
        "pace_avg": 45.0,
        "spin_avg": 38.0,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.08,
        "subcontinent_affinity": 0.95
    },
    "m_henry_nz": {
        "id": "m_henry_nz",
        "name": "Matt Henry",
        "team_id": "NZ",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 5.15,
        "bowling_sr": 24.5,
        "wicket_hazard": 0.052,
        "pp_econ": 4.60,
        "mid_econ": 5.30,
        "death_econ": 7.10,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.18,
        "subcontinent_affinity": 0.92
    },
    "j_duffy_nz": {
        "id": "j_duffy_nz",
        "name": "Jacob Duffy",
        "team_id": "NZ",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RMF",
        "overall_econ": 5.60,
        "bowling_sr": 26.0,
        "wicket_hazard": 0.045,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.06,
        "subcontinent_affinity": 0.90
    },
    "i_sodhi_nz": {
        "id": "i_sodhi_nz",
        "name": "Ish Sodhi",
        "team_id": "NZ",
        "role": "frontline_spinner",
        "batting_hand": "right",
        "bowling_style": "LBG",
        "overall_econ": 5.50,
        "bowling_sr": 27.0,
        "wicket_hazard": 0.046,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.95,
        "subcontinent_affinity": 1.15
    },
    "m_bracewell_nz": {
        "id": "m_bracewell_nz",
        "name": "Michael Bracewell",
        "team_id": "NZ",
        "role": "all_rounder",
        "batting_hand": "left",
        "bowling_style": "OB",
        "overall_sr": 118.0,
        "overall_avg": 38.0,
        "overall_econ": 5.30,
        "favored_pitch": "balanced",
        "sena_affinity": 1.04,
        "subcontinent_affinity": 1.06
    },

    # Live Match: West Indies Playing XI
    "b_king_wi": {
        "id": "b_king_wi",
        "name": "Brandon King",
        "team_id": "WI",
        "role": "top_order_batter",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 134.0,
        "overall_avg": 32.5,
        "pace_sr": 138.0,
        "spin_sr": 126.0,
        "pace_avg": 34.0,
        "spin_avg": 29.0,
        "boundary_pct": 62.0,
        "dot_pct": 36.0,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.02,
        "subcontinent_affinity": 0.92
    },
    "j_dasilva_wi": {
        "id": "j_dasilva_wi",
        "name": "Joshua Da Silva",
        "team_id": "WI",
        "role": "wicket_keeper",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 62.0,
        "overall_avg": 28.5,
        "pace_sr": 65.0,
        "spin_sr": 58.0,
        "pace_avg": 30.0,
        "spin_avg": 26.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.00,
        "subcontinent_affinity": 0.95
    },
    "k_wickham_wi": {
        "id": "k_wickham_wi",
        "name": "Kevin Wickham",
        "team_id": "WI",
        "role": "top_order_batter",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 75.0,
        "overall_avg": 31.0,
        "pace_sr": 78.0,
        "spin_sr": 70.0,
        "favored_pitch": "balanced",
        "sena_affinity": 0.98,
        "subcontinent_affinity": 0.96
    },
    "k_hodge_wi": {
        "id": "k_hodge_wi",
        "name": "Kavem Hodge",
        "team_id": "WI",
        "role": "middle_order_batter",
        "batting_hand": "right",
        "bowling_style": "SLA",
        "overall_sr": 72.0,
        "overall_avg": 34.0,
        "pace_sr": 74.0,
        "spin_sr": 68.0,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.05,
        "subcontinent_affinity": 0.90
    },
    "a_jangoo_wi": {
        "id": "a_jangoo_wi",
        "name": "Amir Jangoo",
        "team_id": "WI",
        "role": "wicket_keeper",
        "batting_hand": "left",
        "bowling_style": "NONE",
        "overall_sr": 78.0,
        "overall_avg": 27.0,
        "pace_sr": 82.0,
        "spin_sr": 72.0,
        "favored_pitch": "balanced",
        "sena_affinity": 0.96,
        "subcontinent_affinity": 0.94
    },
    "j_greaves_wi": {
        "id": "j_greaves_wi",
        "name": "Justin Greaves",
        "team_id": "WI",
        "role": "all_rounder",
        "batting_hand": "right",
        "bowling_style": "RMF",
        "overall_sr": 76.0,
        "overall_avg": 30.5,
        "overall_econ": 4.10,
        "bowling_sr": 32.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.00,
        "subcontinent_affinity": 0.95
    },
    "r_chase_wi": {
        "id": "r_chase_wi",
        "name": "Roston Chase",
        "team_id": "WI",
        "role": "all_rounder",
        "batting_hand": "right",
        "bowling_style": "OB",
        "overall_sr": 75.0,
        "overall_avg": 28.5,
        "pace_sr": 70.0,
        "spin_sr": 82.0,
        "overall_econ": 3.45,
        "bowling_sr": 35.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.95,
        "subcontinent_affinity": 1.10
    },
    "s_joseph_wi": {
        "id": "s_joseph_wi",
        "name": "Shamar Joseph",
        "team_id": "WI",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 3.85,
        "bowling_sr": 18.5,
        "wicket_hazard": 0.055,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.15,
        "subcontinent_affinity": 0.95
    },
    "j_bishop_wi": {
        "id": "j_bishop_wi",
        "name": "Joshua Bishop",
        "team_id": "WI",
        "role": "frontline_spinner",
        "batting_hand": "left",
        "bowling_style": "SLA",
        "overall_econ": 3.60,
        "bowling_sr": 34.0,
        "wicket_hazard": 0.040,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.90,
        "subcontinent_affinity": 1.12
    },
    "j_seales_wi": {
        "id": "j_seales_wi",
        "name": "Jayden Seales",
        "team_id": "WI",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 3.50,
        "bowling_sr": 21.0,
        "wicket_hazard": 0.050,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.12,
        "subcontinent_affinity": 0.96
    },
    "j_warrican_wi": {
        "id": "j_warrican_wi",
        "name": "Jomel Warrican",
        "team_id": "WI",
        "role": "frontline_spinner",
        "batting_hand": "left",
        "bowling_style": "SLA",
        "overall_econ": 3.20,
        "bowling_sr": 30.0,
        "wicket_hazard": 0.044,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.88,
        "subcontinent_affinity": 1.18
    },

    # Live Match: Bangladesh Playing XI
    "s_islam_ban": {
        "id": "s_islam_ban",
        "name": "Shadman Islam",
        "team_id": "BAN",
        "role": "top_order_batter",
        "batting_hand": "left",
        "bowling_style": "NONE",
        "overall_sr": 48.0,
        "overall_avg": 30.5,
        "pace_sr": 46.0,
        "spin_sr": 52.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.88,
        "subcontinent_affinity": 1.08
    },
    "t_hasan_ban": {
        "id": "t_hasan_ban",
        "name": "Tanzid Hasan",
        "team_id": "BAN",
        "role": "top_order_batter",
        "batting_hand": "left",
        "bowling_style": "NONE",
        "overall_sr": 82.0,
        "overall_avg": 28.0,
        "pace_sr": 85.0,
        "spin_sr": 78.0,
        "favored_pitch": "balanced",
        "sena_affinity": 0.95,
        "subcontinent_affinity": 1.04
    },
    "n_shanto_ban": {
        "id": "n_shanto_ban",
        "name": "Najmul Hossain Shanto",
        "team_id": "BAN",
        "role": "top_order_batter",
        "batting_hand": "left",
        "bowling_style": "OB",
        "overall_sr": 68.0,
        "overall_avg": 35.0,
        "pace_sr": 65.0,
        "spin_sr": 72.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.92,
        "subcontinent_affinity": 1.12
    },
    "t_hridoy_ban": {
        "id": "t_hridoy_ban",
        "name": "Towhid Hridoy",
        "team_id": "BAN",
        "role": "middle_order_batter",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 88.0,
        "overall_avg": 38.0,
        "pace_sr": 86.0,
        "spin_sr": 92.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.95,
        "subcontinent_affinity": 1.14
    },
    "m_rahim_ban": {
        "id": "m_rahim_ban",
        "name": "Mushfiqur Rahim",
        "team_id": "BAN",
        "role": "middle_order_batter",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 72.0,
        "overall_avg": 38.5,
        "pace_sr": 68.0,
        "spin_sr": 78.0,
        "pace_avg": 36.0,
        "spin_avg": 42.0,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.94,
        "subcontinent_affinity": 1.16
    },
    "l_das_ban": {
        "id": "l_das_ban",
        "name": "Litton Das",
        "team_id": "BAN",
        "role": "wicket_keeper",
        "batting_hand": "right",
        "bowling_style": "NONE",
        "overall_sr": 78.0,
        "overall_avg": 36.0,
        "pace_sr": 80.0,
        "spin_sr": 75.0,
        "favored_pitch": "balanced",
        "sena_affinity": 0.98,
        "subcontinent_affinity": 1.12
    },
    "m_hasan_miraz_ban": {
        "id": "m_hasan_miraz_ban",
        "name": "Mehidy Hasan Miraz",
        "team_id": "BAN",
        "role": "all_rounder",
        "batting_hand": "right",
        "bowling_style": "OB",
        "overall_sr": 76.0,
        "overall_avg": 26.5,
        "overall_econ": 3.10,
        "bowling_sr": 32.0,
        "wicket_hazard": 0.048,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.88,
        "subcontinent_affinity": 1.25
    },
    "n_hasan_ban": {
        "id": "n_hasan_ban",
        "name": "Nayeem Hasan",
        "team_id": "BAN",
        "role": "frontline_spinner",
        "batting_hand": "right",
        "bowling_style": "OB",
        "overall_econ": 3.15,
        "bowling_sr": 35.0,
        "wicket_hazard": 0.045,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.85,
        "subcontinent_affinity": 1.20
    },
    "t_ahmed_ban": {
        "id": "t_ahmed_ban",
        "name": "Taskin Ahmed",
        "team_id": "BAN",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 3.65,
        "bowling_sr": 25.0,
        "wicket_hazard": 0.052,
        "favored_pitch": "pace_friendly",
        "sena_affinity": 1.08,
        "subcontinent_affinity": 1.02
    },
    "t_islam_ban": {
        "id": "t_islam_ban",
        "name": "Taijul Islam",
        "team_id": "BAN",
        "role": "frontline_spinner",
        "batting_hand": "left",
        "bowling_style": "SLA",
        "overall_econ": 2.95,
        "bowling_sr": 28.0,
        "wicket_hazard": 0.056,
        "favored_pitch": "spin_friendly",
        "sena_affinity": 0.85,
        "subcontinent_affinity": 1.28
    },
    "h_mahmud_ban": {
        "id": "h_mahmud_ban",
        "name": "Hasan Mahmud",
        "team_id": "BAN",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RMF",
        "overall_econ": 3.40,
        "bowling_sr": 26.5,
        "wicket_hazard": 0.048,
        "favored_pitch": "balanced",
        "sena_affinity": 1.04,
        "subcontinent_affinity": 1.02
    },
    "b_azam_pak": {
        "id": "b_azam_pak",
        "name": "Babar Azam",
        "team_id": "PAK",
        "role": "top_order_batter",
        "batting_hand": "right",
        "bowling_style": "OB",
        "overall_sr": 132.5,
        "overall_avg": 48.6,
        "pace_sr": 134.0,
        "spin_sr": 131.0,
        "pace_avg": 47.0,
        "spin_avg": 52.0,
        "boundary_pct": 53.0,
        "dot_pct": 34.0,
        "favored_pitch": "balanced",
        "sena_affinity": 0.96,
        "subcontinent_affinity": 1.10,
        "pp_sr": 122.0,
        "mid_sr": 134.0,
        "death_sr": 175.0
    },
    "s_shakeel_pak": {
        "id": "s_shakeel_pak",
        "name": "Saud Shakeel",
        "team_id": "PAK",
        "role": "middle_order_batter",
        "batting_hand": "left",
        "bowling_style": "SLA",
        "overall_sr": 125.0,
        "overall_avg": 54.2,
        "pace_sr": 122.0,
        "spin_sr": 138.0,
        "pace_avg": 48.0,
        "spin_avg": 62.0,
        "boundary_pct": 48.0,
        "dot_pct": 36.0,
        "favored_pitch": "dusty",
        "sena_affinity": 0.92,
        "subcontinent_affinity": 1.18
    },
    "s_masood_pak": {
        "id": "s_masood_pak",
        "name": "Shan Masood",
        "team_id": "PAK",
        "role": "top_order_batter",
        "batting_hand": "left",
        "bowling_style": "RM",
        "overall_sr": 130.0,
        "overall_avg": 34.5,
        "pace_sr": 132.0,
        "spin_sr": 126.0,
        "pace_avg": 35.0,
        "spin_avg": 33.0,
        "boundary_pct": 51.0,
        "dot_pct": 38.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.02,
        "subcontinent_affinity": 1.00
    },
    "s_ayub_pak": {
        "id": "s_ayub_pak",
        "name": "Saim Ayub",
        "team_id": "PAK",
        "role": "top_order_batter",
        "batting_hand": "left",
        "bowling_style": "OB",
        "overall_sr": 136.0,
        "overall_avg": 32.0,
        "pace_sr": 138.0,
        "spin_sr": 134.0,
        "pace_avg": 30.0,
        "spin_avg": 35.0,
        "boundary_pct": 56.0,
        "dot_pct": 39.0,
        "favored_pitch": "flat",
        "sena_affinity": 0.94,
        "subcontinent_affinity": 1.08
    },
    "p_jayasuriya_sl": {
        "id": "p_jayasuriya_sl",
        "name": "Prabath Jayasuriya",
        "team_id": "SL",
        "role": "frontline_spinner",
        "batting_hand": "right",
        "bowling_style": "SLA",
        "overall_sr": 55.0,
        "overall_avg": 11.2,
        "pace_sr": 50.0,
        "spin_sr": 60.0,
        "pace_avg": 10.0,
        "spin_avg": 12.0,
        "boundary_pct": 20.0,
        "dot_pct": 58.0,
        "favored_pitch": "dusty",
        "sena_affinity": 0.85,
        "subcontinent_affinity": 1.25,
        "overall_econ": 2.95,
        "bowling_sr": 24.0,
        "wicket_hazard": 0.058
    },
    "a_fernando_sl": {
        "id": "a_fernando_sl",
        "name": "Asitha Fernando",
        "team_id": "SL",
        "role": "frontline_pacer",
        "batting_hand": "right",
        "bowling_style": "RF",
        "overall_econ": 3.35,
        "bowling_sr": 25.0,
        "wicket_hazard": 0.052,
        "favored_pitch": "green",
        "sena_affinity": 1.08,
        "subcontinent_affinity": 1.04
    },
    "d_karunaratne_sl": {
        "id": "d_karunaratne_sl",
        "name": "Dimuth Karunaratne",
        "team_id": "SL",
        "role": "top_order_batter",
        "batting_hand": "left",
        "bowling_style": "RM",
        "overall_sr": 118.0,
        "overall_avg": 40.5,
        "pace_sr": 115.0,
        "spin_sr": 125.0,
        "pace_avg": 38.0,
        "spin_avg": 44.0,
        "boundary_pct": 46.0,
        "dot_pct": 42.0,
        "favored_pitch": "balanced",
        "sena_affinity": 0.98,
        "subcontinent_affinity": 1.12
    }
}

def get_player_dossier(player_id: str, default_role: str = "batter") -> Dict[str, Any]:
    """
    Returns the comprehensive player dossier or synthetically derives one.
    """
    pid = player_id.lower()
    if pid in HISTORICAL_PLAYER_DOSSIERS:
        return HISTORICAL_PLAYER_DOSSIERS[pid]

    # Partial match lookup
    for k, v in HISTORICAL_PLAYER_DOSSIERS.items():
        if k in pid or pid in k:
            return v

    # Baseline international profile
    is_bowler = "pacer" in default_role or "spinner" in default_role
    return {
        "id": player_id,
        "name": player_id.replace("_", " ").title(),
        "role": default_role,
        "batting_hand": "right",
        "bowling_style": "RF" if "pacer" in default_role else ("SLA" if "spinner" in default_role else "NONE"),
        "overall_sr": 130.0,
        "overall_avg": 28.0,
        "pace_sr": 130.0,
        "spin_sr": 130.0,
        "pace_avg": 28.0,
        "spin_avg": 28.0,
        "boundary_pct": 52.0,
        "dot_pct": 35.0,
        "favored_pitch": "balanced",
        "sena_affinity": 1.0,
        "subcontinent_affinity": 1.0,
        "overall_econ": 7.80,
        "bowling_sr": 18.0,
        "wicket_hazard": 0.045
    }

def calculate_player_venue_affinity(
    player: Dict[str, Any],
    venue_profile: Dict[str, Any],
    live_pitch_type: Optional[str] = None
) -> Tuple[float, str]:
    """
    Calculates Player-to-Venue Affinity Multiplier (Phi_{P,V}) and Dynamic Pitch Fit.
    Evaluates:
    1. Historical 5-Way Surface Mastery on detected live pitch archetype (65% recency weighted).
    2. Subcontinent vs SENA regional condition affinity.
    3. Live Match Scorecard Performance in-play (runs, balls faced, wickets taken, economy).
    Returns (affinity_multiplier, explanation).
    """
    pid_str = str(player.get("player_id") or player.get("id") or "").lower()
    dossier = get_player_dossier(pid_str, player.get("role", "batter"))
    region = (venue_profile.get("region_cluster") or "subcontinent").lower()
    pitch_type = (live_pitch_type or venue_profile.get("pitch_type") or "balanced").lower()
    spin_ratio = float(venue_profile.get("spin_wicket_ratio") or 0.40)
    role = str(player.get("role") or dossier.get("role", "")).lower()

    # 1. 5-Way Historical Surface Mastery
    from .cricsheet_pitch_intelligence import (
        compute_player_5way_pitch_adaptability,
        canonicalize_pitch_type
    )
    pitch_arch = canonicalize_pitch_type(pitch_type)

    if "pitch_adaptability" in player and isinstance(player["pitch_adaptability"], dict):
        adapt_scores = player["pitch_adaptability"]
    else:
        adapt_data = compute_player_5way_pitch_adaptability(player)
        adapt_scores = adapt_data["pitch_adaptability"]

    surface_score = float(adapt_scores.get(pitch_arch, 80.0))
    # Neutral benchmark is 82.0 (scores from 40 to 99)
    base_ratio = surface_score / 82.0

    # 2. Regional / Foreign Condition Adjustment
    sub_ratio = float(player.get("foreign_condition_ratio_subcontinent") or dossier.get("subcontinent_affinity", 1.0))
    sena_ratio = float(player.get("foreign_condition_ratio_sena") or dossier.get("sena_affinity", 1.0))
    reg_mult = sub_ratio if region == "subcontinent" else sena_ratio
    reg_adj = 1.0 + (reg_mult - 1.0) * 0.35

    notes = [f"Deck: {pitch_arch.upper()} ({round(surface_score)}/100)"]

    # 3. Live Match Scorecard Performance Modulation
    live_mod = 0.0

    # Batter in-play metrics
    runs = int(player.get("current_match_runs") or 0)
    balls = int(player.get("balls_faced") or 0)
    is_out = bool(player.get("is_out", 0))

    if balls >= 25:
        # Batter has spent significant time in middle: demonstrated pitch mastery
        settled_bonus = min(0.12, (runs / 150.0) * 0.12)
        if not is_out:
            settled_bonus += 0.03
        live_mod += settled_bonus
        notes.append(f"Crease Mastery: {runs}*({balls}b) live")
    elif is_out and runs < 15 and balls > 0:
        live_mod -= 0.04
        notes.append(f"Dismissed early ({runs} off {balls}b)")

    # Bowler in-play metrics
    wkts = int(player.get("wickets_taken") or 0)
    overs = float(player.get("overs_bowled") or 0.0)
    econ = float(player.get("bowling_economy") or 0.0)

    if overs >= 3.0 or wkts > 0:
        wkt_bonus = min(0.14, wkts * 0.05)
        live_mod += wkt_bonus
        if wkts > 0:
            notes.append(f"Bowling Impact: {wkts}w live")

    # Combine
    multiplier = base_ratio * reg_adj * (1.0 + live_mod)
    multiplier = max(0.65, min(1.35, round(multiplier, 3)))

    explanation = " • ".join(notes)
    return multiplier, explanation

def compute_11x6_archetype_tensor(
    batters: List[Dict[str, Any]],
    bowlers: List[Dict[str, Any]],
    venue_profile: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Calculates detailed pairwise matchup tensor between 11 batters and 6 frontline bowlers.
    Incorporates:
    - Handedness advantage (LHB vs SLA/OB; RHB vs LBG/SLA)
    - Pace vs Spin historical strike rate & economy
    - Venue pitch condition multiplier
    - Empirical Bayes shrinkage for unobserved sample size
    """
    matrix = []
    biggest_adv = {"diff": 0.0, "description": "Evenly matched across all phases"}

    spin_ratio = venue_profile.get("spin_wicket_ratio", 0.40)

    for b in batters:
        b_pid = str(b.get("player_id") or b.get("id") or "")
        b_dossier = get_player_dossier(b_pid, b.get("role", "batter"))
        b_name = b.get("name", "Unknown")
        b_hand = b.get("batting_hand", "right").lower()
        b_affinity, b_reason = calculate_player_venue_affinity(b, venue_profile)

        row = {
            "batter_id": b_pid,
            "batter_name": b_name,
            "batting_hand": b_hand,
            "pace_sr": b_dossier.get("pace_sr", 130.0),
            "spin_sr": b_dossier.get("spin_sr", 130.0),
            "venue_affinity": b_affinity,
            "venue_affinity_notes": b_reason,
            "matchups": []
        }

        for bw in bowlers:
            bw_pid = str(bw.get("player_id") or bw.get("id") or "")
            bw_dossier = get_player_dossier(bw_pid, bw.get("role", "pacer"))
            bw_name = bw.get("name", "Unknown")
            bw_style = bw.get("bowling_style", "RF")
            bw_affinity, bw_reason = calculate_player_venue_affinity(bw, venue_profile)

            is_spin = bw_style in ["SLA", "LBG", "OB", "LAWS"]
            target_sr = b_dossier.get("spin_sr", 130.0) if is_spin else b_dossier.get("pace_sr", 130.0)
            target_econ = bw_dossier.get("overall_econ", 7.8)

            # Matchup differential:
            # Expected runs per ball = (target_sr / 100.0) * (b_affinity / bw_affinity)
            run_expectancy_ratio = (target_sr / (target_econ * 12.5)) * (b_affinity / bw_affinity)
            shrunk_ratio = compute_empirical_bayes_shrinkage(run_expectancy_ratio, 250, 1.0, 100)
            net_pct = round((shrunk_ratio - 1.0) * 100.0, 1)

            entry = {
                "bowler_id": bw.get("id"),
                "bowler_name": bw_name,
                "bowler_style": bw_style,
                "is_spin": is_spin,
                "net_advantage_pct": net_pct,
                "favour": "BATTER" if net_pct > 0 else "BOWLER"
            }
            row["matchups"].append(entry)

            if abs(net_pct) > biggest_adv["diff"]:
                biggest_adv["diff"] = abs(net_pct)
                if net_pct > 0:
                    biggest_adv["description"] = f"{b_name} targets {bw_name} ({bw_style}) with +{net_pct}% run expectancy ({target_sr} SR vs {'spin' if is_spin else 'pace'})"
                else:
                    biggest_adv["description"] = f"{bw_name} ({bw_style}) restricts {b_name} with {abs(net_pct)}% wicket/dot ball pressure"

        matrix.append(row)

    return matrix, biggest_adv

def calculate_player_pitch_adaptability(player: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes a player's adaptability ratings across 4 core cricket pitch archetypes
    using historical pace/spin averages, strike rates, and regional condition ratios.
    """
    pid = str(player.get("player_id") or player.get("id") or "")
    role = player.get("role", "batter")
    dossier = get_player_dossier(pid, role)
    name = player.get("name") or dossier.get("name", "Unknown Player")
    hand = player.get("batting_hand") or dossier.get("batting_hand", "right")
    style = player.get("bowling_style") or dossier.get("bowling_style", "NONE")

    sub_aff = float(player.get("foreign_condition_ratio_subcontinent") or dossier.get("subcontinent_affinity", 1.0))
    sena_aff = float(player.get("foreign_condition_ratio_sena") or dossier.get("sena_affinity", 1.0))
    pace_sr = float(dossier.get("pace_sr", 130.0))
    spin_sr = float(dossier.get("spin_sr", 130.0))
    pace_avg = float(dossier.get("pace_avg", 30.0))
    spin_avg = float(dossier.get("spin_avg", 30.0))

    is_bowler = any(r in role for r in ["pacer", "spinner", "bowler"])
    is_spinner = style in ["SLA", "OB", "LBG", "LAWS"]
    is_pacer = style in ["RF", "RMF", "LF", "LMF"]

    # 1. FLAT Pitch Adaptability (True batting paradise, high scoring)
    if is_bowler:
        flat_rating = 90.0 - (float(dossier.get("t20_economy", 7.8)) - 6.5) * 8.0
    else:
        boundary_pct = float(dossier.get("boundary_pct", 52.0))
        flat_rating = 52.0 + (boundary_pct / 70.0) * 35.0 + ((pace_sr + spin_sr) / 280.0) * 15.0
    flat_rating = round(min(99.0, max(45.0, flat_rating)), 1)

    # 2. GREEN Pitch Adaptability (Lush grass, early lateral swing & seam)
    if is_bowler:
        green_rating = (92.0 if is_pacer else 60.0) * sena_aff
    else:
        green_rating = (pace_avg / 42.0) * 45.0 + (sena_aff * 35.0) + (15.0 if "top_order" in role else 10.0)
    green_rating = round(min(98.0, max(40.0, green_rating)), 1)

    # 3. SEAM Pitch Adaptability (Abrasive deck, cutters, wobble seam)
    if is_bowler:
        seam_rating = 88.0 * (1.0 + (sena_aff - 1.0) * 0.5) if is_pacer else 72.0
    else:
        seam_rating = (pace_avg / 40.0) * 50.0 + (sena_aff * 25.0) + (spin_avg / 40.0) * 20.0
    seam_rating = round(min(98.0, max(42.0, seam_rating)), 1)

    # 4. BOUNCE Pitch Adaptability (Steep vertical lift, express pace)
    if is_bowler:
        bounce_rating = (94.0 if ("RF" in style or "LF" in style) else 82.0) * sena_aff if is_pacer else 74.0
    else:
        bounce_rating = (pace_sr / 140.0) * 45.0 + (pace_avg / 40.0) * 35.0 + (sena_aff * 18.0)
    bounce_rating = round(min(99.0, max(42.0, bounce_rating)), 1)

    # 5. SPIN HEAVY Pitch Adaptability (Dry clay, sharp turn, low carry)
    if is_bowler:
        spin_rating = (95.0 * sub_aff) if is_spinner else (75.0 * sub_aff)
    else:
        spin_rating = (spin_avg / 45.0) * 45.0 + (spin_sr / 135.0) * 35.0 + (sub_aff * 18.0)
    spin_rating = round(min(99.0, max(40.0, spin_rating)), 1)

    # Determine highest affinity across 5 archetypes
    ratings = {
        "flat": flat_rating,
        "green": green_rating,
        "seam": seam_rating,
        "bounce": bounce_rating,
        "spin heavy": spin_rating,
        # Backward compatibility aliases
        "spin_friendly": spin_rating,
        "pace_friendly": max(green_rating, seam_rating, bounce_rating),
        "batting_paradise": flat_rating,
        "balanced": seam_rating
    }
    favored = max(["flat", "green", "seam", "bounce", "spin heavy"], key=lambda k: ratings[k])

    return {
        "player_id": pid,
        "name": name,
        "role": role,
        "batting_hand": hand,
        "bowling_style": style,
        "pitch_adaptability": ratings,
        "favored_pitch": favored,
        "splits": {
            "pace_avg": pace_avg,
            "pace_sr": pace_sr,
            "spin_avg": spin_avg,
            "spin_sr": spin_sr,
            "subcontinent_affinity": sub_aff,
            "sena_affinity": sena_aff
        }
    }

def get_active_crease_matchups(
    striker_id: str,
    non_striker_id: str,
    bowler_id: str,
    pitch_type: str = "spin_friendly"
) -> Dict[str, Any]:
    """
    Computes precise head-to-head metrics for the two active batters against the active bowler
    on the currently classified pitch condition.
    """
    st_dossier = get_player_dossier(striker_id, "batter")
    nst_dossier = get_player_dossier(non_striker_id, "batter")
    bw_dossier = get_player_dossier(bowler_id, "bowler")

    bw_style = bw_dossier.get("bowling_style", "SLA")
    is_spin = bw_style in ["SLA", "OB", "LBG", "LAWS"]

    def evaluate_pair(b_dossier, b_name):
        b_hand = b_dossier.get("batting_hand", "right").lower()
        sr = b_dossier.get("spin_sr", 130.0) if is_spin else b_dossier.get("pace_sr", 130.0)
        avg = b_dossier.get("spin_avg", 32.0) if is_spin else b_dossier.get("pace_avg", 32.0)

        # Tactical angle & archetype threat
        hazard = 1.0
        threat_desc = "Neutral matchup"

        if is_spin:
            if b_hand == "left" and bw_style == "OB":
                hazard = 1.35
                threat_desc = "High hazard: Off-spinner spinning into LHB, LBW & bat-pad risk"
            elif b_hand == "right" and bw_style == "SLA":
                hazard = 1.30
                threat_desc = "High hazard: Left-arm orthodox turning away from RHB, outside edge to slips"
            elif b_hand == "left" and bw_style == "SLA":
                hazard = 0.85
                threat_desc = "Batter advantage: Turning into the pads, easily worked to leg side"
            elif b_hand == "right" and bw_style == "OB":
                hazard = 0.90
                threat_desc = "Batter advantage: Off-break spinning into RHB's natural arc"
        else:
            if bw_style in ["RF", "LF"]:
                hazard = 1.25
                threat_desc = "High pace hazard: 140kph+ seam movement testing channel outside off"

        # Pitch modification across the 5 archetypes
        p_norm = pitch_type.lower()
        if p_norm in ["spin heavy", "spin_friendly"] and is_spin:
            hazard *= 1.30
        elif p_norm in ["green", "seam", "bounce", "pace_friendly"] and not is_spin:
            hazard *= 1.25
        elif p_norm in ["flat", "batting_paradise"]:
            hazard *= 0.85

        run_expectancy_delta = round(((sr / 125.0) - hazard) * 35.0, 1)

        return {
            "batter_name": b_name,
            "batter_hand": b_hand,
            "bowler_name": bw_dossier.get("name", "Bowler"),
            "bowler_style": bw_style,
            "is_spin": is_spin,
            "threat_description": threat_desc,
            "dismissal_hazard_multiplier": round(hazard, 2),
            "run_expectancy_delta_pct": run_expectancy_delta,
            "favors": "BATTER" if run_expectancy_delta > 0 else "BOWLER"
        }

    striker_matchup = evaluate_pair(st_dossier, st_dossier.get("name", "Striker"))
    non_striker_matchup = evaluate_pair(nst_dossier, nst_dossier.get("name", "Non-Striker"))

    return {
        "pitch_type": pitch_type,
        "striker_matchup": striker_matchup,
        "non_striker_matchup": non_striker_matchup
    }
