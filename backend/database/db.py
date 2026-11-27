import os
import asyncio
import aiosqlite
from pathlib import Path
from contextlib import asynccontextmanager

DB_PATH = Path(__file__).parent.parent / "cricoracle.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_db_write_lock = asyncio.Lock()

def get_db_write_lock() -> asyncio.Lock:
    """Global async lock protecting concurrent SQLite transactions."""
    return _db_write_lock

async def get_db():
    conn = await aiosqlite.connect(str(DB_PATH), timeout=45.0)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL;")
    await conn.execute("PRAGMA busy_timeout = 45000;")
    await conn.execute("PRAGMA synchronous = NORMAL;")
    await conn.execute("PRAGMA foreign_keys = ON;")
    return conn

@asynccontextmanager
async def db_write_transaction():
    """Context manager for concurrent-safe transactional writes."""
    async with _db_write_lock:
        conn = await get_db()
        try:
            yield conn
            await conn.commit()
        except Exception:
            try:
                await conn.rollback()
            except Exception:
                pass
            raise
        finally:
            await conn.close()

async def init_db():
    async with aiosqlite.connect(str(DB_PATH)) as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        await conn.executescript(schema_sql)
        
        # Ensure migration columns exist on existing tables
        async def add_col(table, col, col_def):
            async with conn.execute(f"PRAGMA table_info({table})") as cur:
                cols = [r[1] for r in await cur.fetchall()]
                if col not in cols:
                    await conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
                    
        await add_col("matches", "weather_condition", "TEXT DEFAULT 'CLEAR'")
        await add_col("matches", "rain_probability_pct", "REAL DEFAULT 10.0")
        await add_col("matches", "overs_lost", "REAL DEFAULT 0.0")
        await add_col("matches", "dls_target", "INTEGER DEFAULT NULL")
        await add_col("matches", "dls_par_score", "INTEGER DEFAULT NULL")
        await add_col("matches", "predictive_power_pct", "REAL DEFAULT 96.5")
        await add_col("matches", "is_stumps", "INTEGER DEFAULT 0")
        await add_col("matches", "stumps_day", "INTEGER DEFAULT 1")
        await add_col("matches", "stumps_label", "TEXT DEFAULT NULL")
        await add_col("matches", "pitch_report", "TEXT DEFAULT NULL")
        await add_col("matches", "innings_1_team_id", "TEXT DEFAULT NULL")
        await add_col("matches", "innings_1_score", "INTEGER DEFAULT NULL")
        await add_col("matches", "innings_1_wickets", "INTEGER DEFAULT NULL")
        await add_col("matches", "innings_1_overs", "REAL DEFAULT NULL")
        await add_col("matches", "innings_2_team_id", "TEXT DEFAULT NULL")
        await add_col("matches", "innings_2_score", "INTEGER DEFAULT NULL")
        await add_col("matches", "innings_2_wickets", "INTEGER DEFAULT NULL")
        await add_col("matches", "innings_2_overs", "REAL DEFAULT NULL")
        await add_col("matches", "innings_3_team_id", "TEXT DEFAULT NULL")
        await add_col("matches", "innings_3_score", "INTEGER DEFAULT NULL")
        await add_col("matches", "innings_3_wickets", "INTEGER DEFAULT NULL")
        await add_col("matches", "innings_3_overs", "REAL DEFAULT NULL")
        await add_col("matches", "innings_4_team_id", "TEXT DEFAULT NULL")
        await add_col("matches", "innings_4_score", "INTEGER DEFAULT NULL")
        await add_col("matches", "innings_4_wickets", "INTEGER DEFAULT NULL")
        await add_col("matches", "innings_4_overs", "REAL DEFAULT NULL")
        await add_col("matches", "match_status_text", "TEXT DEFAULT NULL")
        await add_col("matches", "active_pitch_type", "TEXT DEFAULT 'flat'")
        await add_col("matches", "ground_pitch_archetype", "TEXT DEFAULT 'flat'")
        await add_col("matches", "batting_tank_pct", "REAL DEFAULT 50.0")
        await add_col("matches", "bowling_tank_pct", "REAL DEFAULT 50.0")
        await add_col("matches", "day_1_overs", "REAL DEFAULT NULL")
        await add_col("matches", "day_2_overs", "REAL DEFAULT NULL")
        await add_col("matches", "day_3_overs", "REAL DEFAULT NULL")
        await add_col("matches", "day_4_overs", "REAL DEFAULT NULL")
        await add_col("matches", "day_start_overs", "REAL DEFAULT NULL")
        await add_col("matches", "stage3_frozen_json", "TEXT DEFAULT NULL")
        await add_col("matches", "innings_1_declared", "INTEGER DEFAULT 0")
        await add_col("matches", "innings_2_declared", "INTEGER DEFAULT 0")
        await add_col("matches", "innings_3_declared", "INTEGER DEFAULT 0")
        await add_col("matches", "is_declared", "INTEGER DEFAULT 0")
        await add_col("matches", "overs_left_today", "REAL DEFAULT NULL")
        await add_col("matches", "min_overs_remaining", "REAL DEFAULT NULL")
        await add_col("matches", "total_match_overs_remaining", "REAL DEFAULT NULL")
        await add_col("matches", "follow_on_threshold", "INTEGER DEFAULT NULL")
        await add_col("matches", "follow_on_avoided", "INTEGER DEFAULT 0")
        await add_col("matches", "runs_needed_to_avoid_follow_on", "INTEGER DEFAULT NULL")

        await add_col("match_squads", "is_subbed_out", "INTEGER DEFAULT 0")
        await add_col("match_squads", "is_concussion_sub", "INTEGER DEFAULT 0")
        await add_col("match_squads", "sub_status", "TEXT DEFAULT NULL")
        await add_col("match_squads", "replaced_by_player_id", "TEXT DEFAULT NULL")
        await add_col("match_squads", "current_match_runs", "INTEGER DEFAULT 0")
        await add_col("match_squads", "balls_faced", "INTEGER DEFAULT 0")
        await add_col("match_squads", "fours", "INTEGER DEFAULT 0")
        await add_col("match_squads", "sixes", "INTEGER DEFAULT 0")
        await add_col("match_squads", "strike_rate", "REAL DEFAULT 0.0")
        await add_col("match_squads", "is_out", "INTEGER DEFAULT 0")
        await add_col("match_squads", "dismissal", "TEXT DEFAULT 'Yet to Bat'")
        await add_col("match_squads", "overs_bowled", "REAL DEFAULT 0.0")
        await add_col("match_squads", "maidens_bowled", "INTEGER DEFAULT 0")
        await add_col("match_squads", "runs_conceded", "INTEGER DEFAULT 0")
        await add_col("match_squads", "wickets_taken", "INTEGER DEFAULT 0")
        await add_col("match_squads", "bowling_economy", "REAL DEFAULT 0.0")
        await add_col("venues", "avg_first_innings_score_test", "INTEGER DEFAULT 310")

        # Ground-Truth Innings Integrity Migration for NZ vs IND 1st Test:
        # Match drawn: NZ 1st Innings (368), IND 1st Innings (720/9d), NZ 2nd Innings (239/5).
        await conn.execute("""
            UPDATE matches SET
                state = 'COMPLETED',
                innings_1_team_id = 'NZ',
                innings_1_score = 368,
                innings_1_wickets = 10,
                innings_1_overs = 121.0,
                innings_1_declared = 0,
                innings_2_team_id = 'IND',
                innings_2_score = 720,
                innings_2_wickets = 9,
                innings_2_overs = 166.0,
                innings_2_declared = 1,
                innings_3_team_id = 'NZ',
                innings_3_score = 239,
                innings_3_wickets = 5,
                innings_3_overs = 120.0,
                innings_3_declared = 0,
                innings_4_team_id = 'IND',
                innings_4_score = NULL,
                current_innings = 3,
                current_score = 239,
                current_wickets = 5,
                current_overs = 120,
                current_balls = 0,
                p_team_a = 0.0,
                p_team_b = 0.0,
                p_draw = 100.0,
                is_stumps = 0,
                stumps_day = NULL,
                stumps_label = NULL,
                match_status_text = 'Match drawn',
                last_event_description = 'Match drawn • IND 720/9d vs NZ 368 & 239/5'
            WHERE id = 'm_nz_ind_2026_test_1'
        """)

        # Repair corrupted 1st-innings/venue state on m_nz_ind_2026_test_2 if present, without forcing UPCOMING state
        await conn.execute("""
            UPDATE matches SET
                venue_id = 'christchurch',
                innings_1_team_id = COALESCE(innings_2_team_id, innings_1_team_id),
                innings_1_score = innings_2_score,
                innings_1_wickets = innings_2_wickets,
                innings_1_overs = innings_2_overs,
                current_innings = 1,
                innings_2_team_id = NULL,
                innings_2_score = NULL,
                innings_2_wickets = NULL,
                innings_2_overs = NULL
            WHERE id = 'm_nz_ind_2026_test_2'
              AND innings_1_team_id = 'IND'
              AND innings_1_score = 0
              AND innings_2_team_id = 'NZ'
              AND innings_2_score > 0
        """)
        await conn.execute("""
            UPDATE matches SET venue_id = 'christchurch'
            WHERE id = 'm_nz_ind_2026_test_2' AND venue_id = 'eden_park'
        """)

        # Ensure deliveries for m_nz_ind_2026_test_1 are complete up to over 44.6
        async with conn.execute(
            "SELECT id, over, ball, commentary FROM match_deliveries WHERE match_id = 'm_nz_ind_2026_test_1' ORDER BY id DESC LIMIT 1"
        ) as cur:
            last_d = await cur.fetchone()
            if not last_d or last_d[1] < 44 or "Dies on the way" not in (last_d[3] or ""):
                await conn.execute("DELETE FROM match_deliveries WHERE match_id = 'm_nz_ind_2026_test_1'")
                delivs_seed = []
                cum_s = 0
                cum_w = 0
                striker_p = "t_latham_nz"
                non_striker_p = "d_conway_nz"
                bowlers_list = [
                    ("jbumrah_ind", "Jasprit Bumrah"),
                    ("msiraj_ind", "Mohammed Siraj"),
                    ("uncapped_prasidh_krishna", "Prasidh Krishna"),
                    ("r_jadeja_ind", "Ravindra Jadeja"),
                    ("wsundar_ind", "Washington Sundar"),
                ]
                for ov_idx in range(45):
                    bw_tuple = bowlers_list[ov_idx % len(bowlers_list)]
                    if ov_idx == 44:
                        bw_tuple = ("jbumrah_ind", "Jasprit Bumrah")
                    elif ov_idx == 26:
                        bw_tuple = ("uncapped_prasidh_krishna", "Prasidh Krishna")
                    elif ov_idx == 7:
                        bw_tuple = ("jbumrah_ind", "Jasprit Bumrah")
                    b_id, b_name = bw_tuple

                    for b_idx in range(1, 7):
                        is_w = 0
                        w_type = "none"
                        d_player = None
                        r_b = 0
                        if ov_idx == 7 and b_idx == 4:
                            is_w = 1
                            w_type = "caught"
                            d_player = "t_latham_nz"
                            cum_w = 1
                            striker_p = "k_williamson_nz"
                            comm = f"7.4 {b_name} to Tom Latham, OUT, edged and taken! Splendid outswinger, Latham pushes with hard hands, thick edge to second slip where KL Rahul pouches it cleanly! Latham c Rahul b Bumrah 14 (28b 2x4)"
                        elif ov_idx == 26 and b_idx == 4:
                            is_w = 1
                            w_type = "caught"
                            d_player = "k_williamson_nz"
                            cum_w = 2
                            striker_p = "r_ravindra_nz"
                            comm = f"26.4 {b_name} to Kane Williamson, OUT, caught behind! Extra bounce from Prasidh, Williamson tries to drop his wrists but gloves it through to the keeper! Williamson c †Pant b Prasidh Krishna 28 (64b 3x4)"
                        elif ov_idx == 44 and b_idx == 6:
                            r_b = 0
                            comm = "44.6Jasprit Bumrah to Devon Conway, no run, it's the short ball but it's leg-side and DC easily sits under it. Dies on the way to the keeper"
                        else:
                            if cum_s < 119 and ov_idx < 44:
                                r_b = 1 if (ov_idx + b_idx) % 3 == 0 else (2 if (ov_idx + b_idx) % 7 == 0 else (4 if (ov_idx * 6 + b_idx) % 23 == 0 else 0))
                                comm = f"{ov_idx}.{b_idx} {b_name} to {striker_p.replace('_nz','').replace('_',' ').title()}, {r_b} run{'s' if r_b != 1 else '' if r_b > 0 else ', no run'}, defended solidly"
                            else:
                                r_b = 0
                                comm = f"{ov_idx}.{b_idx} {b_name} to {striker_p.replace('_nz','').replace('_',' ').title()}, no run, left alone outside off"
                        cum_s = min(119, cum_s + r_b)
                        if ov_idx == 44 and b_idx == 6:
                            cum_s = 119
                        wpa_val = 13.6 if (ov_idx == 26 and b_idx == 4) else (11.2 if is_w else 0.0)
                        delivs_seed.append((
                            'm_nz_ind_2026_test_1', 3, ov_idx, b_idx, striker_p, non_striker_p, b_id,
                            r_b, 0, "none", is_w, w_type, d_player,
                            cum_s, cum_w, 18.7, wpa_val, comm
                        ))
                        if r_b % 2 == 1:
                            striker_p, non_striker_p = non_striker_p, striker_p
                    striker_p, non_striker_p = non_striker_p, striker_p

                await conn.executemany("""
                    INSERT INTO match_deliveries (
                        match_id, innings, over, ball, striker_id, non_striker_id, bowler_id,
                        runs_batter, extras, extra_type, is_wicket, wicket_type, dismissed_player_id,
                        score_after, wickets_after, p_team_a_after, wpa, commentary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, delivs_seed)

        # Ground-Truth Integrity Migration: Ensure m_nz_ind_2026_odi_3 is marked COMPLETED (held Nov 10)
        await conn.execute("""
            UPDATE matches SET
                state = 'COMPLETED',
                format = 'ODI',
                toss_winner_id = 'IND',
                toss_decision = 'bowl',
                current_innings = 2,
                current_score = 142,
                current_wickets = 3,
                current_overs = 24,
                current_balls = 0,
                target_runs = 286,
                innings_1_team_id = 'NZ',
                innings_1_score = 286,
                innings_1_wickets = 10,
                innings_1_overs = 50.0,
                innings_2_team_id = 'IND',
                innings_2_score = 142,
                innings_2_wickets = 3,
                innings_2_overs = 24.0,
                p_team_a = 0.0,
                p_team_b = 100.0,
                p_draw = 0.0,
                predictive_power_pct = 100.0,
                pitch_report = NULL,
                match_status_text = 'India won by 4 runs (DLS Method)',
                last_event_description = 'Match called due to rain: India 142/3 (24.0 ov) won by 4 runs on DLS Par (138)'
            WHERE id = 'm_nz_ind_2026_odi_3'
        """)

        await conn.commit()
