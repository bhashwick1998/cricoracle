import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import aiosqlite

from backend.database.db import init_db
from backend.scraper.providers.cricapi_provider import CricApiSquadProvider


class _MockResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _RecordingClient:
    def __init__(self, calls: list, responses: list):
        self._calls = calls
        self._responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url: str, *args, **kwargs):
        self._calls.append(url)
        idx = min(len(self._calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


class TestCricApiPaidLiveSync(unittest.TestCase):
    def test_live_scores_always_use_paid_key_never_fallback_free_key(self):
        paid_key = "paid-test-key-999999"
        free_key = "free-test-key-111111"
        provider = CricApiSquadProvider(api_key=paid_key, fallback_key=free_key)

        calls = []
        responses = [
            _MockResponse(200, {"status": "failure", "reason": "hits today exceeded hits limit"}),
        ]

        with patch(
            "backend.scraper.providers.cricapi_provider.httpx.Client",
            side_effect=lambda *a, **kw: _RecordingClient(calls, responses),
        ):
            res = provider.fetch_live_senior_scores(force=True)

        self.assertEqual(res, [])
        self.assertEqual(len(calls), 1)
        self.assertIn(paid_key, calls[0])
        self.assertNotIn(free_key, calls[0])

    def test_empty_live_scores_are_cached_to_protect_paid_quota(self):
        paid_key = "paid-test-key-888888"
        provider = CricApiSquadProvider(api_key=paid_key, fallback_key="free-key")

        calls = []
        responses = [
            _MockResponse(200, {
                "status": "success",
                "info": {"hitsToday": 10, "hitsLimit": 2000},
                "data": [
                    {
                        "id": "match-done-1",
                        "series": "Bangladesh tour of South Africa 2026",
                        "matchType": "test",
                        "ms": "result",
                        "status": "South Africa won by an innings and 219 runs",
                        "t1": "South Africa [RSA]",
                        "t2": "Bangladesh [BAN]",
                        "t1s": "494/7 (117)",
                        "t2s": "91 (33.3) & 184 (49.4)",
                    }
                ],
            })
        ]

        with patch(
            "backend.scraper.providers.cricapi_provider.httpx.Client",
            side_effect=lambda *a, **kw: _RecordingClient(calls, responses),
        ):
            first = provider.fetch_live_senior_scores(force=False)
            second = provider.fetch_live_senior_scores(force=False)

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        # Second call within 90s idle TTL must hit cache cleanly instead of burning another paid hit
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(provider._cached_finished_scores), 1)

    def test_sync_auto_inserts_new_live_match_and_completes_finished_match(self):
        paid_key = "paid-test-key-777777"
        provider = CricApiSquadProvider(api_key=paid_key, fallback_key="free-key")
        schema_path = Path(__file__).resolve().parent.parent / "database" / "schema.sql"
        schema_sql = schema_path.read_text(encoding="utf-8")

        async def _scenario():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "test.db"
                with patch("backend.database.db.DB_PATH", db_path):
                    await init_db()
                conn = await aiosqlite.connect(str(db_path))
                conn.row_factory = aiosqlite.Row
                try:
                    # 1. Seed a stale live SA vs BAN match + report new live IND vs AUS ODI from CricAPI paid key
                    await conn.execute("""
                        INSERT INTO teams (id, name, short_name, color) VALUES
                        ('SA', 'South Africa', 'SA', '#16A34A'),
                        ('BAN', 'Bangladesh', 'BAN', '#047857')
                    """)
                    await conn.execute("""
                        INSERT INTO venues (id, name, city, country, region_cluster, pitch_type)
                        VALUES ('centurion', 'SuperSport Park', 'Centurion', 'South Africa', 'sena', 'bouncy')
                    """)
                    await conn.execute("""
                        INSERT INTO matches (
                            id, series_name, format, team_a_id, team_b_id, venue_id,
                            scheduled_start, state, current_innings, current_score, current_wickets
                        ) VALUES (
                            'm_ban_sa_2026_test_1', 'BAN vs SA 2nd Test', 'TEST', 'SA', 'BAN', 'centurion',
                            '2026-11-23T08:00:00Z', 'LIVE_BALL_BY_BALL', 3, 184, 0
                        )
                    """)
                    await conn.commit()

                    calls = []
                    live_payload = _MockResponse(200, {
                        "status": "success",
                        "info": {"hitsToday": 25, "hitsLimit": 2000},
                        "data": [
                            {
                                "id": "cricapi-ind-aus-1",
                                "series": "Australia tour of India 2026",
                                "matchType": "odi",
                                "ms": "live",
                                "status": "India opt to bat",
                                "dateTimeGMT": "2026-11-26T08:00:00Z",
                                "t1": "India [IND]",
                                "t2": "Australia [AUS]",
                                "t1s": "142/2 (22.4)",
                                "t2s": "",
                            },
                            {
                                "id": "cricapi-ban-sa-2",
                                "series": "Bangladesh tour of South Africa 2026",
                                "matchType": "test",
                                "ms": "result",
                                "status": "South Africa won by an innings and 219 runs",
                                "dateTimeGMT": "2026-11-23T08:00:00Z",
                                "t1": "South Africa [RSA]",
                                "t2": "Bangladesh [BAN]",
                                "t1s": "494/7d",
                                "t2s": "91 & 184",
                            },
                        ],
                    })

                    with patch(
                        "backend.scraper.providers.cricapi_provider.httpx.Client",
                        side_effect=lambda *a, **kw: _RecordingClient(calls, [live_payload]),
                    ):
                        updated = await provider.sync_live_senior_mens_matches(conn)

                    async with conn.execute("SELECT * FROM matches WHERE team_a_id = 'IND' AND team_b_id = 'AUS'") as cur:
                        new_match = dict(await cur.fetchone())
                    async with conn.execute("SELECT * FROM matches WHERE id = 'm_ban_sa_2026_test_1'") as cur:
                        sa_match = dict(await cur.fetchone())

                    return updated, new_match, sa_match, calls
                finally:
                    await conn.close()

        updated, new_match, sa_match, calls = asyncio.run(_scenario())
        self.assertEqual(len(calls), 1)
        self.assertIn(paid_key, calls[0])
        self.assertEqual(new_match["state"], "LIVE_BALL_BY_BALL")
        self.assertEqual(new_match["format"], "ODI")
        self.assertEqual(new_match["current_score"], 142)
        self.assertEqual(new_match["current_wickets"], 2)
        self.assertEqual(new_match["current_overs"], 22)
        self.assertEqual(new_match["current_balls"], 4)
        self.assertEqual(sa_match["state"], "COMPLETED")
        self.assertIn("South Africa won by an innings and 219 runs", sa_match["match_status_text"])
        self.assertEqual(sa_match["p_team_a"], 100.0)
        self.assertEqual(sa_match["p_team_b"], 0.0)
        self.assertIn("m_ban_sa_2026_test_1", updated)
        self.assertIn(new_match["id"], updated)


if __name__ == "__main__":
    unittest.main()
