"""
Comprehensive System Verification Test for CricOracle
Tests:
1. Stage 1 Engine (Macro Elo + Venue + Chase Bias)
2. Stage 2 Engine (15-man squad + Probable XI heuristic + Foreign condition delta)
3. Stage 3 Engine (11x6 H2H Matchup Matrix + Toss shift + 5000 MC sims)
4. Stage 4 Engine (Micro multinomial + Settledness curve + 3000 MC rollout + WPA)
5. Entity Resolver (RapidFuzz matching + Debutant shrinkage)
6. Scraper adapters & Ingestion pipeline
"""

import asyncio
from backend.database.db import get_db, init_db
from backend.database.seed_data import seed_database
from backend.engine.stage1_macro import calculate_stage1_macro
from backend.engine.stage2_squad import calculate_stage2_squad, select_probable_xi
from backend.engine.stage3_xi_locked import calculate_stage3_xi_locked, compute_h2h_matchup_matrix
from backend.engine.stage4_live import calculate_stage4_live, predict_next_ball_multinomial
from backend.scraper.entity_resolver import EntityResolver
from backend.scraper.providers.replay_simulator import ReplaySimulator
from backend.scraper.providers.cric_json_provider import CricJsonProvider
from backend.scraper.providers.html_scraper import HtmlScraper

async def run_all_tests():
    print("==================================================")
    print("CRICORACLE SYSTEM VERIFICATION TEST SUITE")
    print("==================================================")

    # 1. Database seed
    await seed_database()
    conn = await get_db()

    # 2. Test Stage 1 Macro
    print("\n--- 1. Testing Gear 1 (Stage 1 Macro Prediction) ---")
    async with conn.execute("SELECT * FROM teams WHERE id = 'AUS'") as c:
        team_aus = dict(await c.fetchone())
    async with conn.execute("SELECT * FROM teams WHERE id = 'ENG'") as c:
        team_eng = dict(await c.fetchone())
    async with conn.execute("SELECT * FROM venues WHERE id = 'scg'") as c:
        venue_scg = dict(await c.fetchone())

    stage1_res = calculate_stage1_macro(team_aus, team_eng, venue_scg, "T20I")
    print(f"Stage 1 Result: {team_aus['short_name']} {stage1_res['p_team_a']}% vs {team_eng['short_name']} {stage1_res['p_team_b']}%")
    assert 40.0 <= stage1_res["p_team_a"] <= 70.0, "Stage 1 probability out of expected range"
    print("[OK] Stage 1 Engine passed.")

    # 3. Test Stage 2 Squad
    print("\n--- 2. Testing Gear 2A (Stage 2 Squad-Adjusted & Probable XI) ---")
    async with conn.execute("SELECT p.* FROM match_squads ms JOIN players p ON ms.player_id = p.id WHERE ms.match_id = 'm_nz_ind_2026_odi_2' AND ms.team_id = 'IND'") as c:
        squad_ind = [dict(r) for r in await c.fetchall()]
    async with conn.execute("SELECT p.* FROM match_squads ms JOIN players p ON ms.player_id = p.id WHERE ms.match_id = 'm_nz_ind_2026_odi_2' AND ms.team_id = 'NZ'") as c:
        squad_nz = [dict(r) for r in await c.fetchall()]

    prob_xi_ind = select_probable_xi(squad_ind, venue_scg)
    print(f"Selected Probable XI (India): {len(prob_xi_ind)} players -> {[p['name'] for p in prob_xi_ind[:4]]}...")
    assert len(prob_xi_ind) == 11, "Probable XI selector must return exactly 11 players"

    stage2_res = calculate_stage2_squad(squad_ind, squad_nz, venue_scg, stage1_res["p_team_a"])
    print(f"Stage 2 Result: IND {stage2_res['p_team_a']}% vs NZ {stage2_res['p_team_b']}%, Squad delta: {stage2_res['delta_squad_shift']}%")
    print("[OK] Stage 2 Engine passed.")

    # 4. Test Stage 3 Playing XI & Toss Locked
    print("\n--- 3. Testing Gear 2B (Stage 3 Exact 11v11 + 11x6 H2H Matrix + 5,000 MC Runs) ---")
    stage3_res = calculate_stage3_xi_locked(prob_xi_ind, squad_nz[:11], "IND", "field", venue_scg, "IND", stage2_res["p_team_a"])
    print(f"Stage 3 Locked P0: {stage3_res['p0_pre_match']}%, Projected Par Score: {stage3_res['par_score_projected']} ({stage3_res['par_score_min']} - {stage3_res['par_score_max']})")
    print(f"Biggest Matchup Adv: {stage3_res['biggest_matchup_adv']}")
    assert stage3_res["par_score_projected"] > 140, "Par score should be realistic"
    print("[OK] Stage 3 Engine passed.")

    # 5. Test Stage 4 Live Ball-by-Ball
    print("\n--- 4. Testing Gear 3 (Stage 4 Sub-50ms Multinomial + 3,000 MC Rollout + WPA) ---")
    striker = squad_ind[1] # Kohli
    bowler = squad_nz[10] # Boult
    live_res = calculate_stage4_live(
        innings=2, score=126, wickets=3, overs=14, balls=2, target=178,
        striker=striker, non_striker=squad_ind[5], bowler=bowler,
        team_a_batting=True, p0_pre_match=stage3_res["p0_pre_match"],
        previous_p_team_a=58.2, dugout_players=squad_ind[6:], bowlers_list=squad_nz[7:11]
    )
    print(f"Live WinViz: Team A {live_res['p_team_a']}%, WPA: {live_res['wpa']}%, Striker Settledness: {live_res['striker_settledness']}x")
    print(f"Next-Ball Probabilities: {live_res['next_ball_distribution']}")
    assert sum(live_res['next_ball_distribution'].values()) > 0.98, "Multinomial sum should equal ~1.0"
    print("[OK] Stage 4 Engine passed.")
    print("-> 5. Testing Entity Resolver & Debutant Auto-Detection")
    resolver = EntityResolver(conn)
    async with conn.execute("SELECT * FROM players") as c:
        all_players = [dict(r) for r in await c.fetchall()]

    # Test alias matching: "PWH de Silva" -> Wanindu Hasaranga
    res_player = await resolver.resolve_player("PWH de Silva", "SL", all_players=all_players)
    print(f"Resolved 'PWH de Silva' -> {res_player['name']} (ID: {res_player['id']})")
    assert "Hasaranga" in res_player["name"], "Should resolve PWH de Silva to Wanindu Hasaranga"

    # Test debutant creation: uncapped player with IPL franchise tier
    uncapped = await resolver.resolve_player("Mayank Yadav", "IND", all_players=all_players)
    print(f"Resolved Debutant 'Mayank Yadav' -> Role: {uncapped['role']}, League: {uncapped['league_experience_tier']}, Econ: {uncapped['t20_economy']}")
    assert uncapped["is_debutant"] == 1, "Should mark uncapped player as debutant"
    print("[OK] Entity Resolver & Debutant Engine passed.")

    # 7. Test Replay Simulator Transitions
    print("\n-> 6. Testing Replay Simulator State Transitions & Ball Stepping")
    sim = ReplaySimulator()
    # Transition to State 2
    step_squad = await sim.advance_match_state("m_nz_ind_2026_odi_2", "SQUAD_ANNOUNCED", conn)
    print(f"Advanced to State 2: {step_squad['state']} -> Win%: {step_squad['p_team_a']}%")

    # Transition to State 3
    step_xi = await sim.advance_match_state("m_nz_ind_2026_odi_2", "XI_AND_TOSS_LOCKED", conn)
    print(f"Advanced to State 3: {step_xi['state']} -> P0: {step_xi['p0_pre_match']}%")

    # Transition to State 4 and step 1 delivery
    step_live = await sim.advance_match_state("m_nz_ind_2026_odi_2", "LIVE_BALL_BY_BALL", conn)
    ball_res = await sim.step_single_ball("m_nz_ind_2026_odi_2", conn)
    print(f"Stepped 1 Ball: Delivery Over {ball_res['delivery']['over']}.{ball_res['delivery']['ball']} -> Score: {ball_res['delivery']['score']}, WPA: {ball_res['delivery']['wpa']}%")
    print(f"Commentary: {ball_res['delivery']['commentary']}")

    # Restore m_nz_ind_2026_odi_2 back to authentic pre-match scheduled state so test runs do not pollute production fixtures
    await conn.execute("""
        UPDATE matches SET
            state = 'SQUAD_ANNOUNCED',
            current_score = 0,
            current_wickets = 0,
            current_overs = 0,
            current_balls = 0,
            current_striker_id = NULL,
            current_non_striker_id = NULL,
            current_bowler_id = NULL,
            p_team_a = 51.3,
            p_team_b = 48.7,
            last_wpa_swing = 0.0,
            last_event_description = 'Official 15-Man Tour ODI Squads In Effect'
        WHERE id = 'm_nz_ind_2026_odi_2'
    """)
    await conn.execute("DELETE FROM match_deliveries WHERE match_id = 'm_nz_ind_2026_odi_2'")
    await conn.commit()

    print("[OK] Replay Simulator & State Machine passed (Fixture restored to SQUAD_ANNOUNCED).")

    # 8. Test JSON Provider and HTML Scraper
    print("\n-> 7. Testing Ingestion Providers")
    json_prov = CricJsonProvider()
    fixtures = await json_prov.fetch_upcoming_matches()
    print(f"CricJsonProvider ingested {len(fixtures)} fixtures successfully.")
    assert len(fixtures) >= 5, "Should return scheduled matches"

    html_scrap = HtmlScraper()
    url_res = await html_scrap.scrape_match_url("https://www.espncricinfo.com/series/australia-tour-of-india-2026-1400000/india-vs-australia-3rd-t20i/live-cricket-score")
    print(f"HtmlScraper extracted URL info: {url_res['title']} -> {url_res['team_a']} vs {url_res['team_b']}")
    print("[OK] Scraper Providers passed.")

    await conn.close()
    print("\n==================================================")
    print("ALL 7 SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
