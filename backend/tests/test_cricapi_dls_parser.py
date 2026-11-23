"""
Unit Tests for CricAPI (CricketData.org) Resilient Match-State & DLS Parser Module.
Covers:
(a) normal T20/ODI chase,
(b) rain delay pause,
(c) DLS mid-chase target & ball revision,
(d) reduced 12-over T20 match with updated bowler max-over limits,
(e) abandoned match,
(f) ICC fractional bowler quota distribution.
"""

import unittest
from backend.engine.cricapi_match_parser import (
    CricApiMatchParser,
    ICCReducedOversEngine,
    parse_cricket_overs_to_balls,
    balls_to_cricket_overs,
    MatchStateContext
)


class TestCricApiMatchParser(unittest.TestCase):

    def test_overs_to_balls_conversion(self):
        """Test accurate cricket fractional overs to deliveries conversion."""
        self.assertEqual(parse_cricket_overs_to_balls(0.0), 0)
        self.assertEqual(parse_cricket_overs_to_balls(0.1), 1)
        self.assertEqual(parse_cricket_overs_to_balls(0.5), 5)
        self.assertEqual(parse_cricket_overs_to_balls(1.0), 6)
        self.assertEqual(parse_cricket_overs_to_balls(12.4), 76)  # 12 * 6 + 4 = 76
        self.assertEqual(parse_cricket_overs_to_balls(19.5), 119) # 19 * 6 + 5 = 119
        self.assertEqual(parse_cricket_overs_to_balls(20.0), 120)

        # Reverse conversion
        self.assertEqual(balls_to_cricket_overs(76), 12.4)
        self.assertEqual(balls_to_cricket_overs(119), 19.5)
        self.assertEqual(balls_to_cricket_overs(120), 20.0)

    def test_case_a_normal_t20_chase(self):
        """(a) Test normal T20 chase without weather interruptions."""
        payload = {
            "id": "match_t20_normal",
            "matchType": "t20",
            "status": "Pakistan need 42 runs in 24 balls",
            "score": [
                {"r": 160, "w": 6, "o": 20.0, "inning": "New Zealand Inning 1"},
                {"r": 119, "w": 4, "o": 16.0, "inning": "Pakistan Inning 2"}
            ],
            "scorecard": {
                "bowling": [
                    {"name": "Trent Boult", "o": 3.0, "m": 0, "r": 20, "w": 1},
                    {"name": "Tim Southee", "o": 4.0, "m": 0, "r": 32, "w": 2},
                    {"name": "Mitchell Santner", "o": 4.0, "m": 0, "r": 24, "w": 1},
                    {"name": "Lockie Ferguson", "o": 3.0, "m": 0, "r": 26, "w": 0},
                    {"name": "Ish Sodhi", "o": 2.0, "m": 0, "r": 17, "w": 0}
                ]
            }
        }
        ctx: MatchStateContext = CricApiMatchParser.parse_match_data(payload)

        self.assertEqual(ctx.format, "T20I")
        self.assertFalse(ctx.is_interrupted)
        self.assertFalse(ctx.is_abandoned)
        self.assertFalse(ctx.is_reduced_overs)
        self.assertEqual(ctx.standard_max_overs, 20.0)
        self.assertEqual(ctx.revised_max_overs, 20.0)
        self.assertEqual(ctx.current_score, 119)
        self.assertEqual(ctx.current_wickets, 4)
        self.assertEqual(ctx.overs_bowled, 16.0)
        self.assertEqual(ctx.balls_bowled, 96)
        self.assertEqual(ctx.runs_needed, 42)
        self.assertEqual(ctx.balls_remaining, 24)
        self.assertEqual(ctx.revised_target, 161)  # 119 + 42 = 161
        self.assertEqual(ctx.max_overs_per_bowler, 4)
        self.assertFalse(ctx.is_powerplay)  # 16 overs bowled > 6 powerplay overs
        self.assertEqual(ctx.ml_features["effective_run_rate_required"], 10.5)  # 42 runs in 4.0 overs = 10.5 RPO

    def test_case_a_normal_odi_chase(self):
        """(a) Test normal ODI chase without weather interruptions."""
        payload = {
            "id": "match_odi_normal",
            "matchType": "odi",
            "status": "India need 65 runs in 48 balls",
            "score": [
                {"r": 280, "w": 10, "o": 49.2, "inning": "Australia Inning 1"},
                {"r": 216, "w": 5, "o": 42.0, "inning": "India Inning 2"}
            ]
        }
        ctx = CricApiMatchParser.parse_match_data(payload)

        self.assertEqual(ctx.format, "ODI")
        self.assertFalse(ctx.is_interrupted)
        self.assertEqual(ctx.standard_max_overs, 50.0)
        self.assertEqual(ctx.revised_max_overs, 50.0)
        self.assertEqual(ctx.total_balls_innings, 300)
        self.assertEqual(ctx.runs_needed, 65)
        self.assertEqual(ctx.balls_remaining, 48)
        self.assertEqual(ctx.revised_target, 281)  # 216 + 65
        self.assertEqual(ctx.max_overs_per_bowler, 10)
        self.assertAlmostEqual(ctx.ml_features["effective_run_rate_required"], 8.12, places=1)

    def test_case_b_rain_delay_pause(self):
        """(b) Test rain delay and play suspension detection."""
        payload = {
            "id": "match_rain_delay",
            "matchType": "t20",
            "status": "Rain stops play. Covers are on due to persistent drizzle.",
            "score": [
                {"r": 85, "w": 2, "o": 10.3, "inning": "South Africa Inning 1"}
            ]
        }
        ctx = CricApiMatchParser.parse_match_data(payload)

        self.assertTrue(ctx.is_interrupted)
        self.assertFalse(ctx.is_abandoned)
        self.assertIn("Rain delay", ctx.interruption_reason)
        self.assertEqual(ctx.current_score, 85)
        self.assertEqual(ctx.current_wickets, 2)
        self.assertEqual(ctx.overs_bowled, 10.3)
        self.assertEqual(ctx.balls_bowled, 63)  # 10 * 6 + 3 = 63
        self.assertEqual(ctx.ml_features["is_interrupted"], 1.0)
        self.assertEqual(ctx.ml_features["is_abandoned"], 0.0)

    def test_case_c_dls_mid_chase_target_and_ball_revision(self):
        """(c) Test DLS revised target and ball remaining extractor."""
        payload = {
            "id": "match_dls_mid_chase",
            "matchType": "t20",
            "status": "India need 94 runs in 60 balls (DLS method)",
            "score": [
                {"r": 175, "w": 7, "o": 20.0, "inning": "England Inning 1"},
                {"r": 48, "w": 1, "o": 5.0, "inning": "India Inning 2"}
            ]
        }
        ctx = CricApiMatchParser.parse_match_data(payload)

        self.assertTrue(ctx.is_dls_applied)
        self.assertTrue(ctx.dls_method_used)
        self.assertEqual(ctx.runs_needed, 94)
        self.assertEqual(ctx.balls_remaining, 60)
        self.assertEqual(ctx.revised_target, 142)  # 48 + 94 = 142 DLS target
        self.assertAlmostEqual(ctx.ml_features["effective_run_rate_required"], 9.40, places=2)  # 94 / 10 ov = 9.40 RPO
        self.assertEqual(ctx.ml_features["is_dls_applied"], 1.0)

    def test_case_d_reduced_12_over_t20_with_bowler_quotas(self):
        """(d) Test reduced 12-over T20 match with ICC bowler max-over limits."""
        payload = {
            "id": "match_reduced_12_ov",
            "matchType": "t20",
            "status": "Match reduced to 12 overs per side due to rain. Target is 115 in 12 overs (DLS)",
            "score": [
                {"r": 32, "w": 1, "o": 3.4, "inning": "West Indies Inning 2"}
            ],
            "scorecard": {
                "bowling": [
                    {"name": "Mitchell Starc", "o": 2.0, "m": 0, "r": 14, "w": 1},
                    {"name": "Josh Hazlewood", "o": 1.4, "m": 0, "r": 18, "w": 0}
                ]
            }
        }
        ctx = CricApiMatchParser.parse_match_data(payload)

        self.assertTrue(ctx.is_interrupted)
        self.assertTrue(ctx.is_reduced_overs)
        self.assertEqual(ctx.revised_max_overs, 12.0)
        self.assertEqual(ctx.total_balls_innings, 72)
        self.assertEqual(ctx.revised_target, 115)
        self.assertEqual(ctx.runs_needed, 83)  # 115 - 32 = 83

        # ICC Clause 13.8 for 12-over match:
        # 12 / 5 = 2 remainder 2 -> 2 bowlers may bowl 3 overs, 3 bowlers may bowl 2 overs.
        # Strict ceiling for any bowler is ceil(12 / 5) = 3 overs.
        self.assertEqual(ctx.max_overs_per_bowler, 3)
        self.assertIn("2 bowlers x 3 ov", ctx.bowler_quota_summary)
        self.assertIn("3 bowlers x 2 ov", ctx.bowler_quota_summary)

        # Powerplay for 12-over match: round(12 * 0.3) = 4 overs
        self.assertEqual(ctx.powerplay_overs, 4)
        # Current overs 3.4 is within 4 overs
        self.assertTrue(ctx.is_powerplay)

        # Check bowler restriction tracking:
        # Starc bowled 2.0 overs -> 1.0 over remaining under 3-over cap
        starc = next(b for b in ctx.bowler_restrictions if "Starc" in b.name)
        self.assertEqual(starc.overs_bowled, 2.0)
        self.assertEqual(starc.overs_remaining, 1.0)
        self.assertFalse(starc.is_bowled_out)

        # Hazlewood bowled 1.4 overs (10 balls) -> 3.0 ov (18 balls) - 10 balls = 8 balls = 1.2 overs remaining
        hazlewood = next(b for b in ctx.bowler_restrictions if "Hazlewood" in b.name)
        self.assertEqual(hazlewood.balls_bowled, 10)
        self.assertEqual(hazlewood.balls_remaining, 8)
        self.assertEqual(hazlewood.overs_remaining, 1.2)
        self.assertFalse(hazlewood.is_bowled_out)

    def test_case_e_abandoned_match(self):
        """(e) Test abandoned match detection."""
        payload = {
            "id": "match_abandoned",
            "matchType": "t20",
            "status": "Match abandoned without a ball bowled (No result - rain)"
        }
        ctx = CricApiMatchParser.parse_match_data(payload)

        self.assertTrue(ctx.is_interrupted)
        self.assertTrue(ctx.is_abandoned)
        self.assertIn("abandoned", ctx.interruption_reason.lower())
        self.assertEqual(ctx.ml_features["is_abandoned"], 1.0)
        self.assertEqual(ctx.ml_features["is_interrupted"], 1.0)

    def test_case_f_icc_fractional_bowler_quotas(self):
        """(f) Verify exact ICC fractional bowler distribution across multiple curtailed overs."""
        # 16-over T20: 16 / 5 = 3 rem 1 -> 1 bowler x 4 ov, 4 bowlers x 3 ov (Total = 1*4 + 4*3 = 16)
        q16 = ICCReducedOversEngine.calculate_bowler_quotas(16.0, "T20")
        self.assertEqual(q16["max_overs_per_bowler"], 4)
        self.assertEqual(q16["base_overs_per_bowler"], 3)
        self.assertEqual(q16["bowlers_with_extra_over"], 1)
        self.assertEqual(q16["bowlers_with_base_over"], 4)

        # 15-over T20: 15 / 5 = 3 rem 0 -> 5 bowlers x 3 ov (Total = 15)
        q15 = ICCReducedOversEngine.calculate_bowler_quotas(15.0, "T20")
        self.assertEqual(q15["max_overs_per_bowler"], 3)
        self.assertEqual(q15["base_overs_per_bowler"], 3)
        self.assertEqual(q15["bowlers_with_extra_over"], 0)
        self.assertEqual(q15["bowlers_with_base_over"], 5)

        # 43-over ODI: 43 / 5 = 8 rem 3 -> 3 bowlers x 9 ov, 2 bowlers x 8 ov (Total = 3*9 + 2*8 = 43)
        q43 = ICCReducedOversEngine.calculate_bowler_quotas(43.0, "ODI")
        self.assertEqual(q43["max_overs_per_bowler"], 9)
        self.assertEqual(q43["base_overs_per_bowler"], 8)
        self.assertEqual(q43["bowlers_with_extra_over"], 3)
        self.assertEqual(q43["bowlers_with_base_over"], 2)


if __name__ == "__main__":
    unittest.main()
