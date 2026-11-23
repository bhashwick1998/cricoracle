"""
Cricket Analytics Matcher & Early Playing XI Synchronization Engine
Integrates with https://cricket-analytics-ghpv.onrender.com/ and local backend to:
1. Synchronize authentic player profiles (full names, roles, batting hand, bowling style).
2. Match scraped player abbreviations and aliases across CricAPI, Cricbuzz, and backend databases.
3. Guarantee early, complete 11-player Playing XI for both teams from Ball 1.
"""

import json
import logging
import os
import urllib.request
from typing import Dict, Any, List, Optional
import aiosqlite

logger = logging.getLogger("cricoracle.cricket_analytics_matcher")

ANALYTICS_API_URL = "https://cricket-analytics-ghpv.onrender.com/api/players"
LOCAL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "cricket_analytics_players_cache.json")

# Country ID mapping from CricPulse AI to CricOracle canonical
COUNTRY_CODE_MAP = {
    "RSA": "SA",
    "NZL": "NZ",
    "SRI": "SL",
    "WIN": "WI",
    "BAN": "BAN",
    "IND": "IND",
    "AUS": "AUS",
    "ENG": "ENG",
    "PAK": "PAK",
    "AFG": "AFG",
    "ZIM": "ZIM",
    "IRE": "IRE",
    "SCO": "SCO",
    "NED": "NED",
    "NEP": "NEP",
    "USA": "USA",
    "UAE": "UAE",
    "OMA": "OMA",
    "NAM": "NAM",
    "CAN": "CAN"
}

def load_cricket_analytics_players() -> List[Dict[str, Any]]:
    """Loads player catalog from local cache with remote fallback."""
    if os.path.exists(LOCAL_CACHE_PATH):
        try:
            with open(LOCAL_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if cached and len(cached) > 100:
                    return cached
        except Exception as e:
            logger.warning(f"Error reading local cache: {e}")

    try:
        req = urllib.request.Request(ANALYTICS_API_URL, headers={"User-Agent": "Mozilla/5.0 (CricOracle Engine)"})
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read())
            players = data.get("players", [])
            if players:
                try:
                    with open(LOCAL_CACHE_PATH, "w", encoding="utf-8") as f:
                        json.dump(players, f, indent=2)
                except Exception:
                    pass
                return players
    except Exception as e:
        logger.warning(f"Failed to fetch live players from {ANALYTICS_API_URL}: {e}")

    return []

def map_analytics_role(role_raw: Optional[str], bowling_style_raw: Optional[str]) -> str:
    r = str(role_raw or "").upper()
    bw = str(bowling_style_raw or "").lower()
    if "WICKET" in r or "WK" in r:
        return "wicket_keeper"
    if "ALL" in r:
        return "all_rounder"
    if "BOWL" in r:
        if any(k in bw for k in ["fast", "pace", "seam", "medium"]):
            return "frontline_pacer"
        return "frontline_spinner"
    return "top_order_batter"

def map_analytics_batting_hand(batting_raw: Optional[str]) -> str:
    b = str(batting_raw or "").lower()
    return "left" if "left" in b else "right"

def map_analytics_bowling_style(bowling_raw: Optional[str]) -> str:
    bw = str(bowling_raw or "").lower()
    if not bw or bw == "none":
        return "NONE"
    if "slow left-arm" in bw or "orthodox" in bw:
        return "SLA"
    if "offbreak" in bw or "off break" in bw:
        return "OB"
    if "legbreak" in bw or "leg break" in bw:
        return "LBG"
    if "left-arm fast" in bw:
        return "LF"
    if "left-arm medium" in bw:
        return "LMF"
    if "fast-medium" in bw or "medium-fast" in bw or "medium" in bw:
        return "RMF"
    if "fast" in bw:
        return "RF"
    return "NONE"

def generate_canonical_id(name: str, team_id: str) -> str:
    clean = "".join(c for c in name.lower() if c.isalnum() or c == " ").strip()
    parts = clean.split()
    if len(parts) == 1:
        return f"{parts[0]}_{team_id.lower()}"
    return f"{parts[0][0]}_{parts[-1]}_{team_id.lower()}"

async def sync_cricket_analytics_players_to_db(conn: aiosqlite.Connection) -> int:
    """Upserts all players from CricPulse AI into local database and alias tables."""
    raw_players = load_cricket_analytics_players()
    if not raw_players:
        return 0

    inserted_count = 0
    for p in raw_players:
        c_code = p.get("country_id")
        our_team = COUNTRY_CODE_MAP.get(c_code)
        if not our_team:
            continue

        raw_name = p.get("name", "").strip()
        if not raw_name:
            continue

        p_id = generate_canonical_id(raw_name, our_team)
        role = map_analytics_role(p.get("player_role"), p.get("bowling_style"))
        bat_hand = map_analytics_batting_hand(p.get("batting_style"))
        bowl_style = map_analytics_bowling_style(p.get("bowling_style"))

        await conn.execute("""
            INSERT INTO players (id, name, team_id, role, batting_hand, bowling_style, league_experience_tier)
            VALUES (?, ?, ?, ?, ?, ?, 'tier_1_ipl')
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                team_id = excluded.team_id,
                role = excluded.role,
                batting_hand = excluded.batting_hand,
                bowling_style = excluded.bowling_style
        """, (p_id, raw_name, our_team, role, bat_hand, bowl_style))

        # Register alias variations
        aliases = [
            raw_name.lower(),
            raw_name.lower().replace(".", " ").replace("  ", " ").strip(),
            p.get("name", "").lower()
        ]
        # Initial + Last name (e.g. 'Aiden Markram' -> 'a markram', 'k rabada')
        parts = raw_name.split()
        if len(parts) >= 2:
            aliases.append(f"{parts[0][0].lower()} {parts[-1].lower()}")
            aliases.append(f"{parts[0][:2].lower()} {parts[-1].lower()}")

        for al in set(aliases):
            await conn.execute("""
                INSERT OR IGNORE INTO player_aliases (cricsheet_id, alias, country)
                VALUES (?, ?, ?)
            """, (p_id, al, our_team))

        inserted_count += 1

    await conn.commit()
    logger.info(f"Synced {inserted_count} players from Cricket Analytics into database.")
    return inserted_count

async def ensure_full_playing_xi_for_match(conn: aiosqlite.Connection, match_id: str) -> Dict[str, Any]:
    """
    Guarantees that both teams have EXACTLY 11 CONFIRMED PLAYERS (is_confirmed_xi = 1).
    - Preserves all players who have batted (runs, balls) or bowled (overs).
    - Fills remaining slots up to 11 using official squad / CricPulse AI player roster.
    - Demotes any remaining squad members to BENCH_RESERVE (is_confirmed_xi = 0).
    - Assigns canonical 1 to 11 batting order.
    """
    conn.row_factory = aiosqlite.Row
    async with conn.execute("SELECT id, team_a_id, team_b_id, format, current_innings FROM matches WHERE id = ?", (match_id,)) as m_cur:
        m_row = await m_cur.fetchone()
        if not m_row:
            return {"status": "error", "reason": f"Match {match_id} not found"}
        m = dict(m_row)

    team_a = m["team_a_id"]
    team_b = m["team_b_id"]

    # Delete corrupt or dummy player entries (e.g. sa_striker, sa_non_striker)
    await conn.execute("""
        DELETE FROM match_squads
        WHERE match_id = ? AND (player_id LIKE 'sa_striker%' OR player_id LIKE 'sa_non_striker%' OR player_id LIKE '%dummy%')
    """, (match_id,))

    # Fix cross-team misallocations (e.g. Mahfijul Islam who is BAN assigned to SA)
    if match_id == "m_ban_sa_2026_test_1":
        await conn.execute("""
            UPDATE match_squads SET team_id = 'BAN'
            WHERE match_id = ? AND player_id LIKE '%mahfijul%'
        """, (match_id,))

    # Canonicalize uncapped_ player IDs into genuine international player IDs from players table
    async with conn.execute("""
        SELECT ms.player_id, p.name, ms.team_id
        FROM match_squads ms
        JOIN players p ON ms.player_id = p.id
        WHERE ms.match_id = ? AND ms.player_id LIKE 'uncapped_%'
    """, (match_id,)) as unc_cur:
        unc_rows = [dict(r) for r in await unc_cur.fetchall()]

    for unc in unc_rows:
        raw_nm = unc["name"]
        tm = unc["team_id"]
        # Find genuine player in players table with the same or matching name
        async with conn.execute("""
            SELECT id, name FROM players
            WHERE team_id = ? AND id NOT LIKE 'uncapped_%' AND LOWER(name) = LOWER(?)
            LIMIT 1
        """, (tm, raw_nm)) as p_find:
            found = await p_find.fetchone()
            if not found:
                # Try fuzzy/token match
                async with conn.execute("""
                    SELECT id, name FROM players
                    WHERE team_id = ? AND id NOT LIKE 'uncapped_%'
                """, (tm,)) as all_tm_p:
                    for cand in await all_tm_p.fetchall():
                        cand_parts = cand["name"].lower().split()
                        raw_parts = raw_nm.lower().split()
                        if cand_parts and raw_parts and cand_parts[-1] == raw_parts[-1]:
                            found = cand
                            break

            if found:
                canon_id = found["id"]
                # Check if canon_id is already in match_squads for this match
                async with conn.execute("SELECT 1 FROM match_squads WHERE match_id = ? AND player_id = ?", (match_id, canon_id)) as exist_cur:
                    already_in = await exist_cur.fetchone()
                if not already_in:
                    await conn.execute("""
                        UPDATE match_squads SET player_id = ?
                        WHERE match_id = ? AND player_id = ?
                    """, (canon_id, match_id, unc["player_id"]))
                else:
                    # Merge statistics into existing row and remove duplicate uncapped entry
                    await conn.execute("""
                        UPDATE match_squads SET
                            current_match_runs = MAX(current_match_runs, (SELECT current_match_runs FROM match_squads WHERE match_id = ? AND player_id = ?)),
                            balls_faced = MAX(balls_faced, (SELECT balls_faced FROM match_squads WHERE match_id = ? AND player_id = ?)),
                            overs_bowled = MAX(overs_bowled, (SELECT overs_bowled FROM match_squads WHERE match_id = ? AND player_id = ?)),
                            wickets_taken = MAX(wickets_taken, (SELECT wickets_taken FROM match_squads WHERE match_id = ? AND player_id = ?)),
                            is_confirmed_xi = 1
                        WHERE match_id = ? AND player_id = ?
                    """, (match_id, unc["player_id"], match_id, unc["player_id"], match_id, unc["player_id"], match_id, unc["player_id"], match_id, canon_id))
                    await conn.execute("DELETE FROM match_squads WHERE match_id = ? AND player_id = ?", (match_id, unc["player_id"]))

    updated_teams = {}
    for tid in [team_a, team_b]:
        # 1. Fetch current squad rows for this team in this match
        async with conn.execute("""
            SELECT ms.*, p.name, p.role, p.batting_hand, p.bowling_style
            FROM match_squads ms
            JOIN players p ON ms.player_id = p.id
            WHERE ms.match_id = ? AND ms.team_id = ?
        """, (match_id, tid)) as s_cur:
            existing_squad = [dict(r) for r in await s_cur.fetchall()]

        # Partition into: active players who must be in XI vs candidates vs bench
        confirmed_players = []
        candidates = []

        for p in existing_squad:
            has_played = (
                (p.get("balls_faced") or 0) > 0
                or (p.get("current_match_runs") or 0) > 0
                or bool(p.get("is_out"))
                or (p.get("overs_bowled") or 0) > 0
                or (p.get("wickets_taken") or 0) > 0
            )
            if has_played or p.get("is_confirmed_xi") == 1:
                confirmed_players.append(p)
            else:
                candidates.append(p)

        # If candidates are needed to make 11:
        if len(confirmed_players) > 11:
            # Keep top 11 by played priority, then order
            confirmed_players.sort(
                key=lambda x: (
                    1 if ((x.get("balls_faced") or 0) > 0 or (x.get("overs_bowled") or 0) > 0) else 0,
                    x.get("batting_order") or 99
                ),
                reverse=True
            )
            candidates.extend(confirmed_players[11:])
            confirmed_players = confirmed_players[:11]
        elif len(confirmed_players) < 11:
            needed = 11 - len(confirmed_players)
            # Pick from candidates
            confirmed_players.extend(candidates[:needed])
            candidates = candidates[needed:]

        # If total players in squad was less than 11, pull standard international players for this team from players table
        if len(confirmed_players) < 11:
            existing_ids = {p["player_id"] for p in (confirmed_players + candidates)}
            async with conn.execute("""
                SELECT id as player_id, name, team_id, role, batting_hand, bowling_style
                FROM players
                WHERE team_id = ? AND id NOT IN ({})
                LIMIT ?
            """.format(",".join("?" for _ in existing_ids) if existing_ids else "''"),
            ([tid] + list(existing_ids) + [11 - len(confirmed_players)]) if existing_ids else [tid, 11 - len(confirmed_players)]) as p_cur:
                pool = [dict(r) for r in await p_cur.fetchall()]
                for p in pool:
                    p["match_id"] = match_id
                    p["is_confirmed_xi"] = 1
                    p["sub_status"] = "CONFIRMED_XI"
                    p["batting_order"] = len(confirmed_players) + 1
                    confirmed_players.append(p)
                    await conn.execute("""
                        INSERT OR IGNORE INTO match_squads (match_id, player_id, team_id, is_confirmed_xi, sub_status, batting_order)
                        VALUES (?, ?, ?, 1, 'CONFIRMED_XI', ?)
                    """, (match_id, p["player_id"], tid, p["batting_order"]))

        # 2. Update all confirmed XI players with is_confirmed_xi = 1, sub_status = 'CONFIRMED_XI'
        for idx, p in enumerate(confirmed_players[:11], start=1):
            b_ord = p.get("batting_order")
            if not b_ord or b_ord >= 90:
                b_ord = idx
            await conn.execute("""
                UPDATE match_squads SET
                    is_confirmed_xi = 1,
                    sub_status = 'CONFIRMED_XI',
                    batting_order = ?
                WHERE match_id = ? AND player_id = ?
            """, (b_ord, match_id, p["player_id"]))

        # 3. Update candidates/reserves with is_confirmed_xi = 0, sub_status = 'BENCH_RESERVE'
        for p in candidates:
            # Only set to reserve if they haven't batted or bowled
            has_played = ((p.get("balls_faced") or 0) > 0 or (p.get("overs_bowled") or 0) > 0)
            if not has_played:
                await conn.execute("""
                    UPDATE match_squads SET
                        is_confirmed_xi = 0,
                        sub_status = 'BENCH_RESERVE',
                        batting_order = 99
                    WHERE match_id = ? AND player_id = ?
                """, (match_id, p["player_id"]))

        updated_teams[tid] = {
            "confirmed_xi_count": min(11, len(confirmed_players)),
            "confirmed_players": [p["player_id"] for p in confirmed_players[:11]],
            "reserves_count": len(candidates)
        }

    await conn.commit()
    logger.info(f"Enforced full 11 Playing XI for match {match_id}: {updated_teams}")
    return updated_teams
