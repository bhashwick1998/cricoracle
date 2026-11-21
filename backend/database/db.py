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

        await conn.commit()
