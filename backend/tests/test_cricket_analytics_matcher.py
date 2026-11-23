import unittest
import asyncio
import aiosqlite
from backend.engine.cricket_analytics_matcher import (
    load_cricket_analytics_players,
    sync_cricket_analytics_players_to_db,
    ensure_full_playing_xi_for_match,
    generate_canonical_id,
    map_analytics_role
)
from backend.database.db import get_db

class TestCricketAnalyticsMatcher(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.conn = await aiosqlite.connect(":memory:")
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("""
            CREATE TABLE players (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                team_id TEXT NOT NULL,
                role TEXT NOT NULL,
                batting_hand TEXT DEFAULT 'right',
                bowling_style TEXT DEFAULT 'NONE',
                league_experience_tier TEXT DEFAULT 'tier_1_ipl'
            )
        """)
        await self.conn.execute("""
            CREATE TABLE player_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cricsheet_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                country TEXT
            )
        """)
        await self.conn.execute("""
            CREATE TABLE matches (
                id TEXT PRIMARY KEY,
                series_name TEXT,
                format TEXT,
                team_a_id TEXT,
                team_b_id TEXT,
                venue_id TEXT,
                state TEXT,
                current_innings INTEGER DEFAULT 1
            )
        """)
        await self.conn.execute("""
            CREATE TABLE match_squads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                team_id TEXT NOT NULL,
                is_confirmed_xi INTEGER DEFAULT 0,
                is_in_squad INTEGER DEFAULT 1,
                is_probable_xi INTEGER DEFAULT 0,
                sub_status TEXT DEFAULT 'BENCH_RESERVE',
                batting_order INTEGER DEFAULT 99,
                current_match_runs INTEGER DEFAULT 0,
                balls_faced INTEGER DEFAULT 0,
                fours INTEGER DEFAULT 0,
                sixes INTEGER DEFAULT 0,
                strike_rate REAL DEFAULT 0.0,
                is_out INTEGER DEFAULT 0,
                dismissal TEXT,
                overs_bowled REAL DEFAULT 0.0,
                maidens_bowled INTEGER DEFAULT 0,
                runs_conceded INTEGER DEFAULT 0,
                wickets_taken INTEGER DEFAULT 0,
                bowling_economy REAL DEFAULT 0.0,
                UNIQUE(match_id, player_id)
            )
        """)
        await self.conn.commit()

    async def asyncTearDown(self):
        await self.conn.close()

    def test_canonical_id_generator(self):
        self.assertEqual(generate_canonical_id("Aiden Markram", "SA"), "a_markram_sa")
        self.assertEqual(generate_canonical_id("Kagiso Rabada", "SA"), "k_rabada_sa")
        self.assertEqual(generate_canonical_id("Najmul Hossain Shanto", "BAN"), "n_shanto_ban")

    def test_role_mapping(self):
        self.assertEqual(map_analytics_role("BATTER", "None"), "top_order_batter")
        self.assertEqual(map_analytics_role("BOWLER", "right-arm fast"), "frontline_pacer")
        self.assertEqual(map_analytics_role("BOWLER", "slow left-arm orthodox"), "frontline_spinner")
        self.assertEqual(map_analytics_role("ALL_ROUNDER", "fast-medium"), "all_rounder")
        self.assertEqual(map_analytics_role("WICKETKEEPER", "None"), "wicket_keeper")

    async def test_ensure_full_playing_xi_guarantees_11_players(self):
        # Insert a live match
        await self.conn.execute("""
            INSERT INTO matches (id, series_name, format, team_a_id, team_b_id, venue_id, state)
            VALUES ('m_ban_sa_test', 'SA vs BAN - 2nd Test', 'TEST', 'SA', 'BAN', 'centurion', 'LIVE_BALL_BY_BALL')
        """)
        
        # Populate 15 SA players and 15 BAN players in players table
        for i in range(1, 16):
            await self.conn.execute("""
                INSERT INTO players (id, name, team_id, role)
                VALUES (?, ?, 'SA', 'top_order_batter')
            """, (f"sa_player_{i}", f"SA Player {i}"))
            await self.conn.execute("""
                INSERT INTO players (id, name, team_id, role)
                VALUES (?, ?, 'BAN', 'top_order_batter')
            """, (f"ban_player_{i}", f"BAN Player {i}"))

        # Initially, only 3 batters and 2 bowlers have played (partial XI)
        for i in range(1, 4):
            await self.conn.execute("""
                INSERT INTO match_squads (match_id, player_id, team_id, is_confirmed_xi, balls_faced, current_match_runs)
                VALUES ('m_ban_sa_test', ?, 'BAN', 1, 25, 15)
            """, (f"ban_player_{i}",))
        for i in range(1, 3):
            await self.conn.execute("""
                INSERT INTO match_squads (match_id, player_id, team_id, is_confirmed_xi, overs_bowled, wickets_taken)
                VALUES ('m_ban_sa_test', ?, 'SA', 1, 4.0, 1)
            """, (f"sa_player_{i}",))

        # Also add a few bench players
        for i in range(4, 8):
            await self.conn.execute("""
                INSERT INTO match_squads (match_id, player_id, team_id, is_confirmed_xi)
                VALUES ('m_ban_sa_test', ?, 'BAN', 0)
            """, (f"ban_player_{i}",))
        for i in range(3, 7):
            await self.conn.execute("""
                INSERT INTO match_squads (match_id, player_id, team_id, is_confirmed_xi)
                VALUES ('m_ban_sa_test', ?, 'SA', 0)
            """, (f"sa_player_{i}",))

        await self.conn.commit()

        # Run ensure_full_playing_xi_for_match
        res = await ensure_full_playing_xi_for_match(self.conn, 'm_ban_sa_test')
        self.assertEqual(res["SA"]["confirmed_xi_count"], 11)
        self.assertEqual(res["BAN"]["confirmed_xi_count"], 11)

        # Verify exact counts in match_squads
        async with self.conn.execute("SELECT COUNT(*) FROM match_squads WHERE match_id = 'm_ban_sa_test' AND team_id = 'SA' AND is_confirmed_xi = 1") as c:
            sa_count = (await c.fetchone())[0]
            self.assertEqual(sa_count, 11)

        async with self.conn.execute("SELECT COUNT(*) FROM match_squads WHERE match_id = 'm_ban_sa_test' AND team_id = 'BAN' AND is_confirmed_xi = 1") as c:
            ban_count = (await c.fetchone())[0]
            self.assertEqual(ban_count, 11)

if __name__ == "__main__":
    unittest.main()
