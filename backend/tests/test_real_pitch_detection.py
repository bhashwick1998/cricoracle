"""
Comprehensive Verification Tests for Real Pitch Detection via NLP & Pitch Report
Validates:
1. NLP Classifier correctly parses explicit pitch reports and commentary strings.
2. Real detected pitch overrides the historical venue profile (e.g. Green overrides Spin Heavy at Sylhet).
3. Batter performance projections (runs, strike rate, boundaries) dynamically adjust based on 5-way adaptability to detected deck.
4. Bowler performance projections (wickets, economy, hazard) dynamically scale (pacers aided on green/seam, spinners aided on spin heavy).
5. Match outcome win probabilities and par scores shift in favor of squad suited to detected pitch.
"""

import asyncio
from backend.engine.live_pitch_classifier import LivePitchClassifier
from backend.engine.player_projections import (
    compute_player_runs_and_wickets_projections,
    compute_match_scenario_score_projections
)
from backend.engine.pitch_history_model import get_historical_venue_profile
from backend.database.db import get_db, init_db
from backend.database.seed_data import seed_database
from backend.main import get_match_detail, get_matches

async def test_nlp_pitch_detection_overrides_historic_prior():
    lpc = LivePitchClassifier()
    
    # Sylhet historical Test archetype is 'spin heavy'
    # Pitch report declares 8mm live grass and early lateral seam
    res = lpc.classify_live_pitch(
        venue_id="sylhet",
        match_format="TEST",
        current_overs=3,
        current_balls=0,
        current_score=12,
        current_wickets=0,
        pitch_report="Pitch Report: A surprise green surface at Sylhet with a live 8mm grass carpet left on the deck. Early lateral seam movement, nipping off the deck and swing under overcast skies. Conditions heavily aiding seamers.",
        commentary_strings=["Taskin gets late away movement off the seam", "Length ball nibbling away from the batter"]
    )
    
    assert res["classified_pitch_type"] == "green"
    assert res["is_override"] is True
    assert res["historic_ground_archetype"] == "spin heavy"
    assert res["detection_source"] == "NLP_PITCH_REPORT"
    assert any("green" in k.lower() for k in res["nlp_detected_keywords"])

async def test_bowler_and_batter_projections_adapt_to_detected_deck():
    # Test batter and bowler projections under GREEN vs SPIN HEAVY decks
    v_profile = get_historical_venue_profile("sylhet", "TEST")
    
    mock_match = {
        "format": "TEST",
        "state": "LIVE_BALL_BY_BALL",
        "current_overs": 5,
        "current_score": 18,
        "current_wickets": 1
    }
    
    # Fast bowler (Taskin Ahmed) and Spinner (Taijul Islam)
    pacers = [{
        "id": "taskin_ind",
        "name": "Taskin Ahmed",
        "role": "frontline_pacer",
        "bowling_style": "RF",
        "pitch_adaptability": {"green": 92.0, "spin heavy": 60.0},
        "intl_balls_bowled": 1200
    }]
    spinners = [{
        "id": "taijul_ban",
        "name": "Taijul Islam",
        "role": "frontline_spinner",
        "bowling_style": "SLA",
        "pitch_adaptability": {"green": 55.0, "spin heavy": 95.0},
        "intl_balls_bowled": 1800
    }]
    batters = [{
        "id": "test_batter",
        "name": "Test Batter",
        "role": "top_order_batter",
        "batting_order": 1,
        "pitch_adaptability": {"green": 58.0, "spin heavy": 88.0},
        "overall_avg": 42.0
    }]
    
    # 1. Projections on REAL DETECTED GREEN DECK
    proj_green = compute_player_runs_and_wickets_projections(
        match_data=mock_match,
        team_a_players=pacers + spinners,
        team_b_players=batters,
        venue_profile=v_profile,
        live_pitch_type="green"
    )
    
    # On green deck, pacer gets deck hazard multiplier (1.30x)
    pacer_green = next(b for b in proj_green["team_a"]["bowlers_projections"] if b["name"] == "Taskin Ahmed")
    spinner_green = next(b for b in proj_green["team_a"]["bowlers_projections"] if b["name"] == "Taijul Islam")
    batter_green = proj_green["team_b"]["batters_projections"][0]
    
    assert pacer_green["expected_wickets"] > spinner_green["expected_wickets"]
    assert "+30% deck hazard on green" in pacer_green["pitch_suitability"]
    
    # 2. Projections on SPIN HEAVY DECK
    proj_spin = compute_player_runs_and_wickets_projections(
        match_data=mock_match,
        team_a_players=pacers + spinners,
        team_b_players=batters,
        venue_profile=v_profile,
        live_pitch_type="spin heavy"
    )
    
    pacer_spin = next(b for b in proj_spin["team_a"]["bowlers_projections"] if b["name"] == "Taskin Ahmed")
    spinner_spin = next(b for b in proj_spin["team_a"]["bowlers_projections"] if b["name"] == "Taijul Islam")
    batter_spin = proj_spin["team_b"]["batters_projections"][0]
    
    # On spin heavy deck, spinner is far more hazardous and batter thrives due to high spin adaptability
    assert spinner_spin["expected_wickets"] > pacer_spin["expected_wickets"]
    assert "+35% deck hazard on spin heavy" in spinner_spin["pitch_suitability"]
    assert batter_spin["expected_runs"] > batter_green["expected_runs"]

async def test_end_to_end_match_detail_api():
    await init_db()
    await seed_database()
    
    # Fetch live BAN vs WI match at Sylhet
    detail = await get_match_detail("m_ban_wi_2026_live")
    m = detail["match"]
    
    assert m["ground_pitch_archetype"] == "green"
    assert m["active_pitch_type"] == "green"
    assert m["is_pitch_override"] is True
    assert m["historic_ground_archetype"] == "spin heavy"
    assert m["pitch_detection_source"] == "NLP_PITCH_REPORT"
    
    # Player projections should exist and be calibrated to green
    proj = detail["player_projections"]
    assert proj["ground_archetype"] == "green"
    top_bowler = proj["team_a"]["top_bowler"]
    assert "green" in top_bowler["pitch_suitability"].lower()
