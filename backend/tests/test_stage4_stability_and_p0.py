import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.engine.live_pitch_classifier import LivePitchClassifier
from backend.engine.player_projections import (
    STAGE3_FROZEN_MODEL_VERSION,
    build_frozen_stage3_snapshot,
)
from backend.engine.scorecard_delivery_reconstructor import build_deliveries_from_scorecard
from backend.engine.stage3_ground_pitch_model import compute_stage3_dual_first_innings_projections
from backend.engine.stage3_xi_locked import calculate_stage3_xi_locked
from backend.engine.stage4_live import calculate_stage4_live, resolve_break_frozen_probs


def _make_xi(team_prefix: str, bat_avg: float, sr: float, bowl_avg: float, econ: float, adapt: float):
    players = []
    for idx in range(11):
        role = "BATTER" if idx < 5 else ("ALLROUNDER" if idx < 7 else "BOWLER")
        players.append(
            {
                "id": f"{team_prefix}_p{idx}",
                "name": f"{team_prefix} Player {idx}",
                "role": role,
                "batting_avg": bat_avg if idx < 7 else max(10.0, bat_avg * 0.35),
                "strike_rate": sr if idx < 7 else max(55.0, sr * 0.7),
                "bowling_avg": bowl_avg if idx >= 5 else 99.0,
                "economy": econ if idx >= 5 else 9.0,
                "bowling_style": "Right-arm fast-medium" if idx % 2 == 0 else "Slow left-arm orthodox",
                "is_playing_xi": True,
                "country_pitch_adaptability": {
                    "IND": adapt,
                    "NZ": adapt,
                    "AUS": adapt,
                    "ENG": adapt,
                },
            }
        )
    return players


class TestStage4StabilityAndStage3P0(unittest.TestCase):
    def test_stage3_calibrated_p0_not_stuck_at_50_50_on_mismatched_xis(self):
        xi_strong = _make_xi("IND", bat_avg=49.0, sr=146.0, bowl_avg=20.0, econ=6.6, adapt=88.0)
        xi_weak = _make_xi("NAM", bat_avg=24.0, sr=112.0, bowl_avg=36.0, econ=9.2, adapt=48.0)

        for fmt in ("T20I", "ODI", "TEST"):
            dual = compute_stage3_dual_first_innings_projections(
                fmt=fmt,
                team_a_id="IND",
                team_b_id="NAM",
                xi_a=xi_strong,
                xi_b=xi_weak,
                venue_name="Wankhede Stadium",
                city="Mumbai",
                venue_id="wankhede",
                toss_winner_id="IND",
                toss_decision="bat",
            )
            self.assertIn("p0_team_a", dual)
            self.assertIn("p0_team_b", dual)
            self.assertIn("p0_draw", dual)
            self.assertGreater(dual["p0_team_a"], dual["p0_team_b"] + 5.0, f"Expected P0 separation in {fmt}")

            xi_res = calculate_stage3_xi_locked(
                confirmed_xi_a=xi_strong,
                confirmed_xi_b=xi_weak,
                toss_winner_id="IND",
                toss_decision="bat",
                venue={"id": "wankhede", "name": "Wankhede Stadium", "city": "Mumbai", "country": "India", "pitch_type": "balanced"},
                team_a_id="IND",
                team_b_id="NAM",
                p2_squad_prob_a=50.0,
                match_format=fmt,
                use_ground_pitch_model=True,
            )
            self.assertGreater(xi_res["p_team_a"], 55.0, f"Stale 50.0 p2 must not pin {fmt} P0 to 50%")
            self.assertAlmostEqual(
                xi_res["p_team_a"] + xi_res["p_team_b"] + xi_res.get("p_draw", 0.0),
                100.0,
                delta=0.25,
            )

        snapshot = build_frozen_stage3_snapshot(
            match_data={
                "id": "m_test_snapshot_p0",
                "match_format": "ODI",
                "team_a_id": "IND",
                "team_b_id": "NAM",
                "venue_id": "wankhede",
                "venue_name": "Wankhede Stadium",
                "venue_city": "Mumbai",
                "toss_winner_id": "IND",
                "toss_decision": "bat",
            },
            team_a_squad=xi_strong,
            team_b_squad=xi_weak,
            venue_profile={"id": "wankhede", "name": "Wankhede Stadium", "city": "Mumbai", "country": "India", "pitch_type": "balanced"},
        )
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["model_version"], STAGE3_FROZEN_MODEL_VERSION)
        self.assertEqual(STAGE3_FROZEN_MODEL_VERSION, "stage3_6signal_v4")
        self.assertIn("p0_team_a", snapshot)
        self.assertGreater(snapshot["p0_team_a"], 55.0)

    def test_live_pitch_classifier_80_over_phase_and_kalman_hysteresis(self):
        lpc = LivePitchClassifier()
        r_over82 = lpc.classify_live_pitch(
            venue_id="hagley_oval",
            match_format="TEST",
            current_overs=82,
            current_balls=3,
            current_score=280,
            current_wickets=6,
        )
        self.assertEqual(r_over82["phase_id"], "P3")
        self.assertIn("pitch_state", r_over82)
        prior_state = r_over82["pitch_state"]

        # Single noisy seam/green poll after stable balanced state must not instantaneously flip pitch_type
        noisy_deliveries = [
            {"over": 83, "ball": i + 1, "runs_off_bat": 0, "extras": 0, "is_wicket": (i == 2), "commentary": "edge and seam movement"}
            for i in range(6)
        ]
        r_noisy = lpc.classify_live_pitch(
            venue_id="hagley_oval",
            match_format="TEST",
            current_overs=83,
            current_balls=6,
            current_score=280,
            current_wickets=7,
            deliveries=noisy_deliveries,
            previous_pitch_state=prior_state,
        )
        self.assertEqual(
            r_noisy["classified_pitch_type"],
            prior_state["pitch"],
            "Hysteresis + Kalman blend should resist single-poll noisy flip",
        )

    def test_scorecard_delivery_reconstructor_caps_and_clean_text(self):
        batters = [{"id": "b1", "name": "Batter One", "runs": 120, "balls": 65, "fours": 12, "sixes": 6}]
        bowlers = [{"id": "bw1", "name": "Bowler One", "overs": 4.0, "runs": 32, "wickets": 1}]

        t20_rows = build_deliveries_from_scorecard(
            match_id="m_t20_cap",
            innings_num=1,
            batters=batters,
            bowlers=bowlers,
            total_overs=120.5,
            total_runs=220,
            total_wickets=5,
            team_bat="AUS",
            team_bowl="ENG",
            match_format="T20I",
        )
        self.assertTrue(len(t20_rows) <= 120)
        self.assertTrue(all(0 <= d["over"] < 20 for d in t20_rows), "T20I reconstructed overs must remain < 20")
        self.assertTrue(all(1 <= d["ball"] <= 6 for d in t20_rows))
        self.assertFalse(
            any("beaten outside off" in (d.get("commentary") or "").lower() for d in t20_rows),
            "Clean synthetic commentary must not emit 'beaten outside off'",
        )

        odi_rows = build_deliveries_from_scorecard(
            match_id="m_odi_cap",
            innings_num=1,
            batters=batters,
            bowlers=bowlers,
            total_overs=115.2,
            total_runs=310,
            total_wickets=6,
            team_bat="AUS",
            team_bowl="ENG",
            match_format="ODI",
        )
        self.assertTrue(len(odi_rows) <= 300)
        self.assertTrue(all(0 <= d["over"] < 50 for d in odi_rows), "ODI reconstructed overs must remain < 50")

    def test_stage4_live_wpa_stumps_progress_counterfactual_and_determinism(self):
        striker = {"id": "s1", "name": "Striker", "runs": 45, "balls": 55, "batting_avg": 44.0, "strike_rate": 85.0}
        non_striker = {"id": "s2", "name": "NonStriker", "runs": 28, "balls": 40, "batting_avg": 40.0, "strike_rate": 80.0}
        bowler = {"id": "b1", "name": "Bowler", "bowling_avg": 26.0, "economy": 3.8, "bowling_style": "Right-arm fast"}
        xi_bat = _make_xi("NZ", bat_avg=41.0, sr=85.0, bowl_avg=28.0, econ=4.8, adapt=78.0)
        xi_bowl = _make_xi("IND", bat_avg=45.0, sr=92.0, bowl_avg=25.0, econ=4.5, adapt=82.0)

        for fmt, overs, max_dot_swing in (("TEST", 84, 1.85), ("ODI", 24, 3.55), ("T20I", 12, 5.05)):
            base_kwargs = dict(
                innings=1,
                score=260 if fmt != "T20I" else 110,
                wickets=4 if fmt != "T20I" else 2,
                overs=overs,
                balls=3,
                target=0,
                striker=striker,
                non_striker=non_striker,
                bowler=bowler,
                team_a_batting=True,
                p0_pre_match=48.0,
                previous_p_team_a=50.5,
                dugout_players=xi_bat[:5],
                bowlers_list=xi_bowl[5:],
                team_a="NZ",
                team_b="IND",
                match_format=fmt,
                live_pitch_type="balanced",
                prev_innings=1,
                prev_score=260 if fmt != "T20I" else 110,
                prev_wickets=4 if fmt != "T20I" else 2,
                prev_overs=overs,
                prev_balls=2,
                is_progress_step=True,
            )

            res1 = calculate_stage4_live(**base_kwargs)
            res2 = calculate_stage4_live(**base_kwargs)
            self.assertEqual(res1["p_team_a"], res2["p_team_a"], f"Deterministic SHA-256 seed failed for {fmt}")
            self.assertEqual(res1["wpa"], res2["wpa"], f"Deterministic WPA failed for {fmt}")
            self.assertLessEqual(
                abs(res1["wpa"]),
                max_dot_swing,
                f"Single non-wicket dot ball WPA swing exceeded counterfactual bound in {fmt}: {res1['wpa']}",
            )

            # Stumps / break must always produce 0.0 WPA
            stumps_kwargs = dict(base_kwargs)
            stumps_kwargs["is_stumps"] = True
            res_stumps = calculate_stage4_live(**stumps_kwargs)
            self.assertEqual(res_stumps["wpa"], 0.0, f"Stumps WPA must be 0.0 in {fmt}")

            # Non-progress / duplicate poll must always produce 0.0 WPA
            noop_kwargs = dict(base_kwargs)
            noop_kwargs["prev_balls"] = 3
            noop_kwargs["is_progress_step"] = False
            res_noop = calculate_stage4_live(**noop_kwargs)
            self.assertEqual(res_noop["wpa"], 0.0, f"Non-progress WPA must be 0.0 in {fmt}")

    def test_resolve_break_frozen_probs_rejects_unnormalized_test_draw_stacking(self):
        corrupt_row = {"p_team_a": 50.5, "p_team_b": 49.5, "p_draw": 39.4}
        clean_res = {"p_team_a": 41.3, "p_team_b": 40.5, "p_draw": 18.2}
        pa, pb, pd = resolve_break_frozen_probs(corrupt_row, clean_res, "TEST")
        self.assertAlmostEqual(pa + pb + pd, 100.0, delta=0.1)
        self.assertEqual((pa, pb, pd), (41.3, 40.5, 18.2))

        valid_row = {"p_team_a": 41.3, "p_team_b": 40.5, "p_draw": 18.2}
        pa2, pb2, pd2 = resolve_break_frozen_probs(valid_row, {"p_team_a": 60.0, "p_team_b": 30.0, "p_draw": 10.0}, "TEST")
        self.assertEqual((pa2, pb2, pd2), (41.3, 40.5, 18.2))


if __name__ == "__main__":
    unittest.main()
