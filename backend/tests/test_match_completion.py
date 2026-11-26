import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend.engine.cricket_rules import evaluate_match_completion
from backend.scraper.providers.cricbuzz_score_api import CricbuzzScoreApi


class TestMatchCompletionDetection(unittest.TestCase):
    def test_sa_ban_take_the_game_by_an_innings_phrase(self):
        match = {
            "id": "m_ban_sa_2026_test_1",
            "format": "TEST",
            "state": "LIVE_BALL_BY_BALL",
            "team_a_id": "SA",
            "team_a_name": "South Africa",
            "team_b_id": "BAN",
            "team_b_name": "Bangladesh",
            "innings_1_team_id": "BAN",
            "innings_1_score": 91,
            "innings_2_team_id": "SA",
            "innings_2_score": 494,
            "innings_3_team_id": "BAN",
            "innings_3_score": 184,
            "current_innings": 3,
            "current_score": 184,
            "current_wickets": 0,
            "match_status_text": "SA lead by 219 runs (BAN trail by 219 runs)",
            "last_event_description": (
                "49.4WMarco Jansen to Khaled Ahmed, out Caught by Wiaan Mulder!! "
                "Mulder takes it at third man. Had to be him. SA take the game by an innings and"
            ),
        }
        res = evaluate_match_completion(match)
        self.assertTrue(res["is_completed"])
        self.assertEqual(res["winning_team_id"], "SA")
        self.assertEqual(res["winning_team_name"], "South Africa")
        self.assertIn("an innings and 219 runs", res["result_description"])

    def test_third_innings_defeat_without_follow_on_all_out(self):
        match = {
            "id": "m_ban_sa_2026_test_1",
            "format": "TEST",
            "state": "LIVE_BALL_BY_BALL",
            "team_a_id": "SA",
            "team_a_name": "South Africa",
            "team_b_id": "BAN",
            "team_b_name": "Bangladesh",
            "innings_1_team_id": "BAN",
            "innings_1_score": 91,
            "innings_2_team_id": "SA",
            "innings_2_score": 494,
            "innings_3_team_id": "BAN",
            "innings_3_score": 184,
            "batting_team_id": "BAN",
            "current_innings": 3,
            "current_score": 184,
            "current_wickets": 10,
            "match_status_text": "Bangladesh all out",
            "last_event_description": "Innings closed",
        }
        res = evaluate_match_completion(match)
        self.assertTrue(res["is_completed"])
        self.assertEqual(res["winning_team_id"], "SA")
        self.assertEqual(res["result_description"], "South Africa won by an innings and 219 runs")

    def test_cricbuzz_concluded_compound_innings_parses_10_wickets(self):
        html = """
        <html><head><title>RSA vs BAN, 2nd Test | Cricbuzz</title></head><body>
          <div class="text-cbTextLink">South Africa won by an innings and 219 runs</div>
          <div>RSA 494/7d • BAN 91 &amp; 184 won by an innings and 219 runs</div>
        </body></html>
        """

        class _FakeResp:
            status_code = 200
            text = html

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, *args, **kwargs):
                return _FakeResp()

        with patch("backend.scraper.providers.cricbuzz_score_api.httpx.AsyncClient", return_value=_FakeClient()):
            scraped = asyncio.run(CricbuzzScoreApi().get_live_score("147964"))

        self.assertEqual(scraped["match_status"], "South Africa won by an innings and 219 runs")
        self.assertEqual(len(scraped["all_innings"]), 3)
        self.assertEqual(scraped["all_innings"][-1]["wickets"], 10)


if __name__ == "__main__":
    unittest.main()
