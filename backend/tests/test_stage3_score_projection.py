import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aiosqlite

from backend.engine.player_projections import (
    calculate_asymmetric_1st_innings_par,
    compute_match_scenario_score_projections,
)
from backend.engine.stage3_ground_pitch_model import (
    CANONICAL_PITCHES,
    compute_stage3_dual_first_innings_projections,
    load_stage3_calibrated_priors,
    resolve_stage3_pitch_distribution_and_pars,
)
from backend.engine.stage3_xi_locked import calculate_stage3_xi_locked
from backend.scraper.ingestion_manager import IngestionManager


def _sample_xi(prefix: str, team: str):
    roles = [
        ("top_order_batter", "right", ""),
        ("top_order_batter", "left", ""),
        ("top_order_batter", "right", ""),
        ("middle_order_batter", "right", ""),
        ("wicketkeeper", "right", ""),
        ("all_rounder", "left", "Slow left-arm orthodox"),
        ("all_rounder", "right", "Right-arm offbreak"),
        ("fast_bowler", "right", "Right-arm fast"),
        ("fast_bowler", "left", "Left-arm fast-medium"),
        ("spinner", "left", "Slow left-arm orthodox"),
        ("fast_bowler", "right", "Right-arm fast-medium"),
    ]
    xi = []
    for i, (role, bat_h, bowl_s) in enumerate(roles, start=1):
        xi.append({
            "id": f"{prefix}_{i}",
            "name": f"{prefix} Player {i}",
            "country": team,
            "team_id": team,
            "batting_order": i,
            "is_confirmed_xi": 1,
            "role": role,
            "batting_hand": bat_h,
            "bowling_style": bowl_s,
            "batting_avg": 42.0 if i <= 6 else 16.0,
            "strike_rate": 136.0 if i <= 6 else 105.0,
            "bowling_econ": 7.4 if i >= 6 else 8.8,
        })
    return xi


class TestStage3ScoreProjection(unittest.TestCase):
    def test_calibrated_priors_loaded_and_constrained(self):
        priors = load_stage3_calibrated_priors()
        for fmt in ("ODI", "T20I", "TEST"):
            self.assertIn(fmt, priors)
            gp = priors[fmt]["global_pitch_pars"]
            for pitch in CANONICAL_PITCHES:
                self.assertIn(pitch, gp)
            self.assertGreater(gp["flat"], gp["green"])
            if fmt != "TEST":
                self.assertGreater(gp["flat"], gp["dusty"])
            else:
                self.assertGreater(gp["dusty"], gp["green"])

            w = priors[fmt]["score_model"]["weights"]
            self.assertGreater(w["same_ground_pitch_par"], 0.0)
            self.assertGreaterEqual(w["xi_bat_adapt_diff"], 0.0)
            self.assertLessEqual(w["xi_bowl_adapt_diff"], 0.0)
            self.assertGreaterEqual(w["duel_run_delta"], 0.0)

    def test_resolve_ground_pitch_distribution_and_pars(self):
        for fmt in ("ODI", "T20I"):
            info = resolve_stage3_pitch_distribution_and_pars(
                fmt=fmt,
                venue_name="Wankhede Stadium",
                city="Mumbai",
            )
            self.assertEqual(info["lookup_source"], "venue_exact")
            self.assertAlmostEqual(sum(info["pitch_probs"].values()), 1.0, places=2)
            self.assertIn(info["dominant_pitch"], CANONICAL_PITCHES)

            flat_info = resolve_stage3_pitch_distribution_and_pars(
                fmt=fmt,
                venue_name="Wankhede Stadium",
                city="Mumbai",
                pitch_override="flat",
            )
            dusty_info = resolve_stage3_pitch_distribution_and_pars(
                fmt=fmt,
                venue_name="Wankhede Stadium",
                city="Mumbai",
                pitch_override="dusty",
            )
            self.assertEqual(flat_info["dominant_pitch"], "flat")
            self.assertEqual(dusty_info["dominant_pitch"], "dusty")
            self.assertGreater(
                flat_info["same_ground_pitch_weighted_par"],
                dusty_info["same_ground_pitch_weighted_par"],
            )

    def test_stage3_xi_locked_populates_6_signal_score_model(self):
        xi_ind = _sample_xi("ind", "IND")
        xi_aus = _sample_xi("aus", "AUS")
        # Inject real stars from committed Render cache
        xi_ind[0]["name"] = "Rohit Sharma"
        xi_ind[2]["name"] = "Virat Kohli"
        xi_ind[7]["name"] = "Jasprit Bumrah"
        venue = {
            "id": "wankhede",
            "name": "Wankhede Stadium",
            "city": "Mumbai",
            "country": "India",
            "dimensions_avg_boundary_m": 67.0,
            "dew_factor": 0.40,
            "avg_first_innings_score_odi": 288,
            "avg_first_innings_score_t20": 182,
        }

        for fmt in ("ODI", "T20I"):
            res = calculate_stage3_xi_locked(
                confirmed_xi_a=xi_ind,
                confirmed_xi_b=xi_aus,
                toss_winner_id="IND",
                toss_decision="bat",
                venue=venue,
                team_a_id="IND",
                team_b_id="AUS",
                p2_squad_prob_a=52.0,
                match_format=fmt,
            )
            feats = res["details"].get("score_model_features")
            self.assertIsNotNone(feats)
            self.assertIn("same_ground_pitch_par", feats)
            self.assertIn("global_pitch_par", feats)
            self.assertIn("duel_run_delta", feats)
            self.assertIn("xi_bat_adapt", feats)
            self.assertIn("xi_bowl_adapt", feats)
            self.assertLess(res["par_score_min"], res["par_score_projected"])
            self.assertLess(res["par_score_projected"], res["par_score_max"])

            res_flat = calculate_stage3_xi_locked(
                confirmed_xi_a=xi_ind,
                confirmed_xi_b=xi_aus,
                toss_winner_id="IND",
                toss_decision="bat",
                venue=venue,
                team_a_id="IND",
                team_b_id="AUS",
                p2_squad_prob_a=52.0,
                match_format=fmt,
                pitch_report="Flat batting paradise road, zero grass",
            )
            res_green = calculate_stage3_xi_locked(
                confirmed_xi_a=xi_ind,
                confirmed_xi_b=xi_aus,
                toss_winner_id="IND",
                toss_decision="bat",
                venue=venue,
                team_a_id="IND",
                team_b_id="AUS",
                p2_squad_prob_a=52.0,
                match_format=fmt,
                pitch_report="Green tinge seam movement pitch under overcast cloud",
            )
            self.assertGreater(res_flat["par_score_projected"], res_green["par_score_projected"])

    def test_asymmetric_par_canonical_surfaces(self):
        xi_ind = _sample_xi("ind", "IND")
        xi_aus = _sample_xi("aus", "AUS")
        venue = {"id": "wankhede", "name": "Wankhede Stadium", "city": "Mumbai"}
        par_flat, _ = calculate_asymmetric_1st_innings_par(
            batting_team_id="IND",
            bowling_team_id="AUS",
            fmt="T20I",
            ground_arch="Flat Road",
            base_par=180,
            bat_squad=xi_ind,
            bowl_squad=xi_aus,
            venue_profile=venue,
            match_data={"team_a_id": "IND", "is_pitch_override": True},
        )
        par_dusty, _ = calculate_asymmetric_1st_innings_par(
            batting_team_id="IND",
            bowling_team_id="AUS",
            fmt="T20I",
            ground_arch="Dusty Spinner",
            base_par=180,
            bat_squad=xi_ind,
            bowl_squad=xi_aus,
            venue_profile=venue,
            match_data={"team_a_id": "IND", "is_pitch_override": True},
        )
        self.assertGreater(par_flat, par_dusty)

    def test_dual_first_innings_projections_pre_toss(self):
        xi_ind = _sample_xi("ind", "IND")
        xi_zim = _sample_xi("zim", "ZIM")
        venue = {
            "id": "wankhede",
            "name": "Wankhede Stadium",
            "city": "Mumbai",
            "country": "India",
            "dimensions_avg_boundary_m": 67.0,
            "dew_factor": 0.40,
            "avg_first_innings_score_odi": 288,
            "avg_first_innings_score_t20": 182,
        }
        res = calculate_stage3_xi_locked(
            confirmed_xi_a=xi_ind,
            confirmed_xi_b=xi_zim,
            toss_winner_id="",
            toss_decision="",
            venue=venue,
            team_a_id="IND",
            team_b_id="ZIM",
            p2_squad_prob_a=65.0,
            match_format="T20I",
        )
        dual = res["dual_first_innings_scenarios"]
        self.assertIsNotNone(dual)
        self.assertFalse(dual["toss_confirmed"])
        a_1st = dual["team_a_bats_first"]
        b_1st = dual["team_b_bats_first"]
        self.assertGreater(a_1st["projected_score"], b_1st["projected_score"])
        self.assertIn("same_ground_pitch_par", a_1st["signals"])
        self.assertIn("duel_run_delta", b_1st["signals"])

    def test_match_scenario_projections_uses_stage3_before_toss_when_xi_confirmed(self):
        xi_ind = _sample_xi("ind", "IND")
        xi_aus = _sample_xi("aus", "AUS")
        venue = {
            "id": "wankhede",
            "name": "Wankhede Stadium",
            "city": "Mumbai",
            "country": "India",
            "dimensions_avg_boundary_m": 67.0,
            "dew_factor": 0.35,
        }
        match_data = {
            "id": "m_test_xi_pretoss",
            "format": "T20I",
            "state": "SQUAD_ANNOUNCED",
            "team_a_id": "IND",
            "team_b_id": "AUS",
            "venue_id": "wankhede",
            "venue_name": "Wankhede Stadium",
            "venue_city": "Mumbai",
            "scheduled_start": "2026-11-28T14:00:00Z",
        }
        out = compute_match_scenario_score_projections(
            match_data=match_data,
            team_a_squad=xi_ind,
            team_b_squad=xi_aus,
            venue_profile=venue,
        )
        self.assertEqual(out["scenario_team_1_bats_first"]["methodology"], "stage3_6signal_calibrated")
        self.assertEqual(out["scenario_team_2_bats_first"]["methodology"], "stage3_6signal_calibrated")
        s3 = out["four_stage_progression"]["stage_3_xi_toss"]
        self.assertFalse(s3["toss_confirmed"])
        self.assertIsNotNone(s3["signals_a"])
        self.assertIsNotNone(s3["signals_b"])
        self.assertGreater(s3["team_a_par"], 120)
        self.assertGreater(s3["team_b_par"], 120)

    def test_cricapi_schedule_xi_and_venue_sync_triggers_stage3(self):
        schema_sql = (Path(__file__).resolve().parent.parent / "database" / "schema.sql").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test_cricapi_s3.db"
            conn_init = sqlite3.connect(db_file)
            conn_init.executescript(schema_sql)
            conn_init.commit()
            conn_init.close()

            async def _db_factory():
                c = await aiosqlite.connect(str(db_file))
                c.row_factory = aiosqlite.Row
                return c

            fake_fixtures = [{
                "cricapi_id": "ca-12345",
                "series_name": "IND vs AUS 2026 T20I Series",
                "team_a_code": "IND",
                "team_b_code": "AUS",
                "team_a_name": "India [IND]",
                "team_b_name": "Australia [AUS]",
                "format": "T20I",
                "scheduled_start": "2026-11-28T14:00:00Z",
                "is_live": False,
            }]
            fake_toss_venue = {
                "status": "success",
                "toss_winner": None,
                "toss_choice": "",
                "venue_name": "Wankhede Stadium, Mumbai",
                "venue_id": "wankhede",
                "teams": ["India", "Australia"],
            }
            fake_xi = {
                "status": "success",
                "match_id": "ca-12345",
                "teams": [
                    {
                        "team_name": "India",
                        "playing_xi": [{"order": i, "name": f"IND Star {i}", "role": "Batter" if i <= 6 else "Bowler", "bowling_style": "Right-arm offbreak" if i >= 7 else ""} for i in range(1, 12)],
                    },
                    {
                        "team_name": "Australia",
                        "playing_xi": [{"order": i, "name": f"AUS Star {i}", "role": "Batter" if i <= 6 else "Bowler", "bowling_style": "Right-arm fast" if i >= 7 else ""} for i in range(1, 12)],
                    },
                ],
            }

            async def _run():
                mgr = IngestionManager(_db_factory)
                with (
                    patch("backend.scraper.providers.cricapi_provider.cricapi_provider.sync_senior_international_fixtures", return_value=fake_fixtures),
                    patch("backend.scraper.providers.cricapi_provider.cricapi_provider.fetch_match_toss_and_venue", return_value=fake_toss_venue),
                    patch("backend.scraper.providers.cricapi_provider.cricapi_provider.fetch_playing_xi_for_match", return_value=fake_xi),
                ):
                    return await mgr.sync_cricapi_schedule_xi_and_venue(force=True)

            updated = asyncio.run(_run())
            self.assertEqual(len(updated), 1)

            conn_chk = sqlite3.connect(db_file)
            conn_chk.row_factory = sqlite3.Row
            m_row = dict(conn_chk.execute("SELECT * FROM matches WHERE id = ?", (updated[0],)).fetchone())
            xi_cnt = conn_chk.execute("SELECT count(*) FROM match_squads WHERE match_id = ? AND is_confirmed_xi = 1", (updated[0],)).fetchone()[0]
            conn_chk.close()
            self.assertEqual(m_row["state"], "XI_AND_TOSS_LOCKED")
            self.assertEqual(m_row["venue_id"], "wankhede")
            self.assertEqual(xi_cnt, 22)
            self.assertGreater(m_row["par_score_projected"], 120)
            self.assertIsNotNone(m_row["stage3_frozen_json"])
            self.assertIn("stage3_6signal_v3", m_row["stage3_frozen_json"])

    def test_test_format_6_signal_stage3_and_stage4_freeze_invariance(self):
        import json

        xi_nz = _sample_xi("nz", "NZ")
        xi_ind = _sample_xi("ind", "IND")
        xi_nz[2]["name"] = "Kane Williamson"
        xi_nz[7]["name"] = "Tim Southee"
        xi_ind[2]["name"] = "Virat Kohli"
        xi_ind[7]["name"] = "Jasprit Bumrah"

        venue = {
            "id": "hagley",
            "name": "Hagley Oval",
            "city": "Christchurch",
            "country": "New Zealand",
            "pitch_type": "green",
            "dimensions_avg_boundary_m": 69.0,
            "avg_first_innings_score_test": 318,
        }
        pre_match = {
            "id": "m_nz_ind_test_freeze",
            "format": "TEST",
            "state": "XI_AND_TOSS_LOCKED",
            "team_a_id": "NZ",
            "team_b_id": "IND",
            "venue_id": "hagley",
            "venue_name": "Hagley Oval",
            "venue_city": "Christchurch",
            "toss_winner_id": "IND",
            "toss_decision": "field",
            "historic_ground_archetype": "green",
            "scheduled_start": "2026-11-27T00:00:00Z",
        }
        out_pre = compute_match_scenario_score_projections(
            match_data=pre_match,
            team_a_squad=xi_nz,
            team_b_squad=xi_ind,
            venue_profile=venue,
            live_pitch_type="green",
        )
        s3_pre = out_pre["four_stage_progression"]["stage_3_xi_toss"]
        frozen = out_pre["stage3_frozen"]
        self.assertIsNotNone(frozen)
        self.assertGreater(s3_pre["team_a_par"], 220)
        self.assertLess(s3_pre["team_a_par"], 480)
        self.assertIsNotNone(s3_pre["signals_a"])
        self.assertIsNotNone(s3_pre["signals_b"])
        self.assertIsNotNone(s3_pre["four_innings_projection"])
        self.assertEqual(out_pre["scenario_team_1_bats_first"]["methodology"], "stage3_6signal_calibrated")

        # Simulate heavy Stage 4 live match drift (score collapses, pitch mutates to dusty, rain/players mutated)
        xi_nz_live = [dict(p) for p in xi_nz]
        xi_ind_live = [dict(p) for p in xi_ind]
        xi_nz_live[2]["current_match_runs"] = 6
        xi_nz_live[2]["is_out"] = 1
        xi_ind_live[7]["wickets_taken"] = 4
        live_match = {
            **pre_match,
            "state": "LIVE_BALL_BY_BALL",
            "current_innings": 1,
            "current_score": 48,
            "current_wickets": 5,
            "current_overs": 14,
            "current_balls": 2,
            "ground_pitch_archetype": "dusty",
            "stage3_frozen_json": json.dumps(frozen),
        }
        out_live = compute_match_scenario_score_projections(
            match_data=live_match,
            team_a_squad=xi_nz_live,
            team_b_squad=xi_ind_live,
            venue_profile=venue,
            weather_report={"condition": "Heavy Rain", "overs_lost": 12},
            live_pitch_type="dusty",
        )
        from backend.engine.player_projections import parse_frozen_stage3_snapshot

        parsed_roundtrip = parse_frozen_stage3_snapshot(json.dumps(frozen))
        self.assertIsNotNone(parsed_roundtrip)
        self.assertEqual(parsed_roundtrip["frozen_at"], frozen["frozen_at"])
        self.assertEqual(out_live["stage3_frozen"]["frozen_at"], frozen["frozen_at"])

        s3_live = out_live["four_stage_progression"]["stage_3_xi_toss"]
        self.assertEqual(s3_live["team_a_par"], s3_pre["team_a_par"])
        self.assertEqual(s3_live["team_b_par"], s3_pre["team_b_par"])
        self.assertEqual(s3_live["team_a_range"], s3_pre["team_a_range"])
        self.assertEqual(s3_live["team_b_range"], s3_pre["team_b_range"])
        self.assertEqual(s3_live["signals_a"], s3_pre["signals_a"])
        self.assertEqual(s3_live["signals_b"], s3_pre["signals_b"])
        self.assertEqual(s3_live["pitch_type"], s3_pre["pitch_type"])
        self.assertEqual(s3_live["four_innings_projection"], s3_pre["four_innings_projection"])
        self.assertEqual(s3_live["top_performers"], s3_pre["top_performers"])
        self.assertNotEqual(out_live["four_stage_progression"]["stage_4_live"]["active_team_projected_final"], s3_live["team_a_par"])


if __name__ == "__main__":
    unittest.main()
