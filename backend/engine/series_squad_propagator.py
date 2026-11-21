"""
CricOracle Series Squad Propagation Engine
When an official 15-player squad is announced for a tour or bilateral series,
this engine automatically propagates the squad across all remaining matches of the same tour and format.
It updates match states to SQUAD_ANNOUNCED, computes venue-specific pitch adaptability for each ground,
evaluates probable par scores, and identifies the expected highest run scorer and highest wicket taker.
"""

import re
import logging
from typing import Dict, Any, List, Optional
import aiosqlite

from .pitch_history_model import get_historical_venue_profile
from .cricsheet_pitch_intelligence import (
    get_ground_pitch_archetype,
    get_country_pitch_adaptability,
    compute_player_5way_pitch_adaptability
)

logger = logging.getLogger("cricoracle.series_squad")

def extract_tour_series_key(series_name: str, match_format: str, team_a_id: str, team_b_id: str) -> str:
    """
    Extracts canonical tour series prefix.
    Examples:
    'India Tour of New Zealand 2026 - 2nd ODI' -> 'india_tour_of_new_zealand_2026_odi_ind_nz'
    'India Tour of New Zealand 2026 - 3rd ODI' -> 'india_tour_of_new_zealand_2026_odi_ind_nz'
    'Afghanistan vs Zimbabwe T20I Series 2026 - 1st T20I' -> 'afghanistan_vs_zimbabwe_t20i_series_2026_t20i_afg_zim'
    """
    cleaned = re.sub(r"\s*-\s*\d+(?:st|nd|rd|th)?\s*(?:ODI|T20I|Test|T20).*$", "", series_name, flags=re.IGNORECASE).strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    fmt = (match_format or "T20I").lower()
    teams = "_".join(sorted([team_a_id.lower(), team_b_id.lower()]))
    return f"{cleaned}_{fmt}_{teams}"

async def propagate_series_squads_async(conn: aiosqlite.Connection) -> List[Dict[str, Any]]:
    """
    Scans all matches in the database. When any match in a tour series has an announced squad (>=11 players),
    propagates that squad to all future matches in the same tour with the same format.
    Advances target match state from UPCOMING_NO_SQUAD to SQUAD_ANNOUNCED.
    """
    conn.row_factory = aiosqlite.Row
    propagated_matches: List[Dict[str, Any]] = []

    # 1. Fetch all matches
    async with conn.execute("""
        SELECT id, series_name, format, team_a_id, team_b_id, venue_id, state, scheduled_start,
               p1_macro, par_score_projected
        FROM matches
        ORDER BY scheduled_start ASC
    """) as cur:
        matches = [dict(r) for r in await cur.fetchall()]

    # 2. Fetch all match_squads entries
    async with conn.execute("""
        SELECT ms.*, p.name, p.role, p.batting_hand, p.bowling_style
        FROM match_squads ms
        JOIN players p ON ms.player_id = p.id
    """) as cur_s:
        all_squad_rows = [dict(r) for r in await cur_s.fetchall()]

    squads_by_match: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_squad_rows:
        squads_by_match.setdefault(row["match_id"], []).append(row)

    # 3. Group matches by series key
    matches_by_series: Dict[str, List[Dict[str, Any]]] = {}
    for m in matches:
        s_key = extract_tour_series_key(m["series_name"], m["format"], m["team_a_id"], m["team_b_id"])
        matches_by_series.setdefault(s_key, []).append(m)

    # 4. Propagate squads within each series group
    for s_key, s_matches in matches_by_series.items():
        # Find canonical source match with full announced squad
        source_squad: Optional[List[Dict[str, Any]]] = None
        source_match: Optional[Dict[str, Any]] = None

        for m in s_matches:
            sq = squads_by_match.get(m["id"], [])
            if len(sq) >= 30:
                source_squad = sq
                source_match = m
                break
            elif len(sq) >= 15 or (len(sq) >= 11 and source_squad is None):
                source_squad = sq
                source_match = m

        if not source_squad or not source_match:
            continue

        # Distribute canonical squad to any future matches in the series lacking a squad or having incomplete squad
        for target_m in s_matches:
            if (
                target_m["id"] == source_match["id"]
                or target_m.get("state") in ("COMPLETED", "LIVE_BALL_BY_BALL")
                or target_m["id"] in ("m_aus_eng_2026_t20_2", "m_nz_ind_2026_odi_3")
            ):
                continue

            target_sq = squads_by_match.get(target_m["id"], [])
            # Propagate if target match has fewer players than source or fewer than 30 (15 per team)
            if len(target_sq) < len(source_squad) or len(target_sq) < 30:
                target_id = target_m["id"]
                logger.info(f"Propagating {len(source_squad)} squad players from {source_match['id']} to {target_id} ({target_m['series_name']})")

                # Insert squad rows for the target match
                for p in source_squad:
                    await conn.execute("""
                        INSERT INTO match_squads (
                            match_id, team_id, player_id, is_in_squad, is_probable_xi, is_confirmed_xi,
                            batting_order, pitch_fit_rating, current_match_runs, balls_faced, fours, sixes,
                            strike_rate, is_out, dismissal, overs_bowled, maidens_bowled, runs_conceded,
                            wickets_taken, bowling_economy
                        ) VALUES (
                            ?, ?, ?, ?, ?, 0,
                            ?, ?, 0, 0, 0, 0,
                            0.0, 0, 'Yet to Bat', 0.0, 0, 0,
                            0, 0.0
                        )
                        ON CONFLICT(match_id, player_id) DO UPDATE SET
                            is_in_squad = excluded.is_in_squad,
                            is_probable_xi = excluded.is_probable_xi,
                            batting_order = excluded.batting_order,
                            pitch_fit_rating = excluded.pitch_fit_rating
                    """, (
                        target_id,
                        p["team_id"],
                        p["player_id"],
                        p.get("is_in_squad", 1),
                        p.get("is_probable_xi", 1),
                        p.get("batting_order", 99),
                        p.get("pitch_fit_rating", 80.0)
                    ))

                # Update match state to SQUAD_ANNOUNCED if it was UPCOMING_NO_SQUAD
                new_state = "SQUAD_ANNOUNCED" if target_m["state"] == "UPCOMING_NO_SQUAD" else target_m["state"]

                # Recalculate venue-specific par score & Win probability for target venue
                v_profile = get_historical_venue_profile(target_m.get("venue_id", ""), target_m["format"])
                g_arch = get_ground_pitch_archetype(target_m.get("venue_id", ""), target_m["format"])
                if target_m["format"] == "TEST":
                    base_par = v_profile.get("avg_first_innings_score_test", 355)
                elif target_m["format"] == "ODI":
                    base_par = v_profile.get("avg_first_innings_score_odi", 275)
                else:
                    base_par = v_profile.get("avg_first_innings_score_t20", 168)

                # Compute squad pitch adaptability for target ground
                team_a_players = [p for p in source_squad if p["team_id"] == target_m["team_a_id"]]
                team_b_players = [p for p in source_squad if p["team_id"] == target_m["team_b_id"]]

                def avg_pitch_fit(pl_list):
                    if not pl_list:
                        return 75.0
                    total = 0.0
                    for p_item in pl_list:
                        adapt = compute_player_5way_pitch_adaptability(p_item)
                        total += adapt["pitch_adaptability"].get(g_arch, 75.0)
                    return total / len(pl_list)

                a_fit = avg_pitch_fit(team_a_players)
                b_fit = avg_pitch_fit(team_b_players)
                fit_diff = (a_fit - b_fit) * 0.15

                # Adjust pre-match win %
                p_macro = target_m.get("p1_macro") or 50.0
                p_squad_a = max(20.0, min(80.0, round(p_macro + fit_diff, 1)))
                p_squad_b = round(100.0 - p_squad_a, 1)

                await conn.execute("""
                    UPDATE matches
                    SET state = ?,
                        p_team_a = ?,
                        p_team_b = ?,
                        p2_squad = ?,
                        par_score_projected = ?,
                        delta_squad_shift = ?
                    WHERE id = ?
                """, (
                    new_state,
                    p_squad_a,
                    p_squad_b,
                    p_squad_a,
                    base_par,
                    round(fit_diff, 2),
                    target_id
                ))

                propagated_matches.append({
                    "match_id": target_id,
                    "series_name": target_m["series_name"],
                    "format": target_m["format"],
                    "venue_id": target_m["venue_id"],
                    "ground_archetype": g_arch,
                    "state": new_state,
                    "squad_size": len(source_squad),
                    "source_match_id": source_match["id"],
                    "p_team_a": p_squad_a,
                    "p_team_b": p_squad_b,
                    "par_score": base_par
                })

    await conn.commit()
    logger.info(f"Series Squad Propagator completed: {len(propagated_matches)} future matches updated.")
    return propagated_matches
