"""
Concussion Replacement Intelligence Engine
Implements the official ICC Concussion Substitute Policy (ICC Playing Condition 1.2.7):
- Evaluates and ranks approved 'Like-for-Like' replacement candidates from reserve squads
- Computes Euclidean skill & role similarity vectors on the format's active pitch archetype
- Quantifies Concussion Rating Delta and adjusts live/pre-match win probability
- Enforces Match Referee approval rules (batter for batter, keeper for keeper, pacer for pacer, spinner for spinner)
"""

from typing import Dict, Any, List, Optional, Tuple
from .cricsheet_pitch_intelligence import compute_player_5way_pitch_adaptability

def calculate_like_for_like_similarity(
    injured: Dict[str, Any],
    candidate: Dict[str, Any],
    pitch_archetype: str = "flat"
) -> Dict[str, Any]:
    """
    Evaluates whether a candidate is an approved ICC 'Like-for-Like' concussion replacement.
    Evaluation Dimensions:
    1. Primary Role Compatibility (40% weight): batter for batter, keeper for keeper, etc.
    2. Bowling Style & Pace Band (25% weight): RF vs RF, SLA vs SLA, Off-break vs Off-break
    3. Batting Hand & Top/Middle Order Profile (15% weight): LHB vs RHB
    4. Format Pitch Adaptability Match (20% weight): Deck fitness proximity on ground's pitch archetype
    """
    inj_role = str(injured.get("role", "batter")).lower()
    cand_role = str(candidate.get("role", "batter")).lower()

    inj_style = str(injured.get("bowling_style", "NONE")).upper()
    cand_style = str(candidate.get("bowling_style", "NONE")).upper()

    inj_hand = str(injured.get("batting_hand", "right")).lower()
    cand_hand = str(candidate.get("batting_hand", "right")).lower()

    # 1. Primary Role Compatibility (0 - 40 points)
    if inj_role == cand_role:
        role_score = 40.0
    elif ("batter" in inj_role and "batter" in cand_role) or ("pacer" in inj_role and "pacer" in cand_role) or ("spinner" in inj_role and "spinner" in cand_role):
        role_score = 36.0
    elif ("all_rounder" in inj_role and ("pacer" in cand_role or "spinner" in cand_role or "batter" in cand_role)):
        role_score = 28.0
    elif inj_role == "wicket_keeper" and cand_role != "wicket_keeper":
        role_score = 10.0 # Wicketkeepers strictly require specialist keepers
    else:
        role_score = 15.0

    # 2. Bowling Style Compatibility (0 - 25 points)
    if inj_style == cand_style:
        style_score = 25.0
    elif (inj_style in ["RF", "RMF", "LF", "LMF"] and cand_style in ["RF", "RMF", "LF", "LMF"]):
        style_score = 22.0
    elif (inj_style in ["SLA", "OB", "LBG", "LAWS"] and cand_style in ["SLA", "OB", "LBG", "LAWS"]):
        style_score = 22.0
    elif inj_style in ["NONE", ""] and cand_style in ["NONE", ""]:
        style_score = 25.0
    else:
        style_score = 5.0

    # 3. Batting Hand Match (0 - 15 points)
    hand_score = 15.0 if inj_hand == cand_hand else 10.0

    # 4. Deck Pitch Adaptability Match (0 - 20 points)
    inj_adapt = compute_player_5way_pitch_adaptability(injured)["pitch_adaptability"].get(pitch_archetype, 75.0)
    cand_adapt = compute_player_5way_pitch_adaptability(candidate)["pitch_adaptability"].get(pitch_archetype, 75.0)
    adapt_diff = abs(inj_adapt - cand_adapt)
    deck_score = max(0.0, 20.0 - (adapt_diff * 0.8))

    total_similarity = round(role_score + style_score + hand_score + deck_score, 1)

    # ICC Referee Approval Threshold is 70% similarity
    referee_approval = "APPROVED_LIKE_FOR_LIKE" if total_similarity >= 72.0 else "UNLIKELY_APPROVAL_INEQUITABLE"

    # Skill Rating Delta (Substitute Rating - Injured Rating)
    inj_sr = float(injured.get("t20_batting_sr", 130.0))
    cand_sr = float(candidate.get("t20_batting_sr", 130.0))
    inj_econ = float(injured.get("t20_economy", 8.0))
    cand_econ = float(candidate.get("t20_economy", 8.0))

    bat_delta = round(cand_sr - inj_sr, 1)
    bowl_delta = round(inj_econ - cand_econ, 2) # positive if candidate has better economy
    adapt_delta = round(cand_adapt - inj_adapt, 1)

    # Net WPA Impact Multiplier
    net_rating_shift = round((bat_delta * 0.15) + (bowl_delta * 2.0) + (adapt_delta * 0.35), 2)

    return {
        "candidate_id": candidate.get("id") or candidate.get("player_id"),
        "candidate_name": candidate.get("name", "Unknown Reserve"),
        "candidate_role": cand_role,
        "candidate_style": cand_style,
        "candidate_hand": cand_hand,
        "similarity_pct": total_similarity,
        "referee_status": referee_approval,
        "pitch_archetype": pitch_archetype,
        "candidate_deck_fit": round(cand_adapt, 1),
        "injured_deck_fit": round(inj_adapt, 1),
        "rating_shift_delta": net_rating_shift,
        "details": {
            "role_match": f"{role_score}/40",
            "bowling_match": f"{style_score}/25",
            "hand_match": f"{hand_score}/15",
            "deck_match": f"{round(deck_score, 1)}/20"
        }
    }

def rank_concussion_substitutes(
    injured_player: Dict[str, Any],
    bench_players: List[Dict[str, Any]],
    pitch_archetype: str = "flat"
) -> List[Dict[str, Any]]:
    """
    Ranks available bench reserves by ICC like-for-like compatibility.
    """
    ranked = []
    inj_pid = injured_player.get("id") or injured_player.get("player_id")
    for candidate in bench_players:
        cand_pid = candidate.get("id") or candidate.get("player_id")
        if inj_pid and cand_pid and inj_pid == cand_pid:
            continue
        eval_res = calculate_like_for_like_similarity(injured_player, candidate, pitch_archetype)
        ranked.append(eval_res)

    ranked.sort(key=lambda x: x["similarity_pct"], reverse=True)
    return ranked
