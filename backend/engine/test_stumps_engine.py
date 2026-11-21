"""
CricOracle Test Cricket Stumps & Multi-Day Intelligence Engine
Models authentic close-of-play (Stumps) detection, overnight crease persistence,
under-covers condensation seam dynamics, second new-ball countdown,
and Day 2 morning session probabilistic forecasts.
"""

from typing import Dict, Any, List, Optional
import math

def compute_test_stumps_intelligence(
    match: Dict[str, Any],
    squad_a: List[Dict[str, Any]],
    squad_b: List[Dict[str, Any]],
    venue_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes authentic Test cricket Day Close (Stumps) analysis.
    
    Test Cricket Domain Rules:
    1. A standard Test day consists of 90 scheduled overs across 3 sessions (Morning, Afternoon, Evening).
    2. Play concludes at STUMPS (either scheduled close, bad light, rain, or overs completed).
    3. Trapped condensation under tarpaulin covers overnight creates high morning seam movement (+15% - 22%).
    4. The 2nd New Ball becomes available to the bowling side strictly after 80.0 overs of an innings.
    """
    fmt = match.get("format", "TEST")
    series_name = str(match.get("series_name") or "").lower()
    mid_low = str(match.get("id") or "").lower()
    if "test" in series_name or "test" in mid_low:
        fmt = "TEST"

    if fmt != "TEST" or match.get("state") == "COMPLETED" or bool(match.get("is_completed")):
        return {
            "is_stumps": False,
            "stumps_day": None,
            "stumps_label": None,
            "morning_session_forecast": None
        }

    # If match started more than 5.2 days ago, the 5-day match duration has fully elapsed
    if match.get("scheduled_start"):
        try:
            from datetime import datetime, timezone
            sch_str = str(match["scheduled_start"]).replace("Z", "+00:00")
            sch_dt = datetime.fromisoformat(sch_str)
            if sch_dt.tzinfo is None:
                sch_dt = sch_dt.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - sch_dt).total_seconds() >= 5.2 * 86400:
                return {
                    "is_stumps": False,
                    "stumps_day": None,
                    "stumps_label": None,
                    "morning_session_forecast": None
                }
        except Exception:
            pass

    # Authentic Test Cricket Stumps Detection:
    # Rule 1: A standard Test day consists of 90.0 scheduled overs across 3 sessions.
    # Overs in the 60s, 70s, or 80s are Session 3 (Post-Tea to Stumps) and are actively in-play!
    # Rule 2: Play ONLY ceases for STUMPS when the umpires officially signal the close of play.
    cur_overs = float(match.get("current_overs") or 0.0)
    last_event = str(match.get("last_event_description") or "").lower()
    explicit_is_stumps = match.get("is_stumps")
    match_status = str(match.get("match_status_text") or match.get("match_status") or "").lower()
    
    # Determine Day number first
    import re
    combined_status = f"{last_event} {match_status}"
    day_m = re.search(r"day\s*([1-5])", combined_status) or re.search(r"([1-5])(?:st|nd|rd|th)\s*day", combined_status)
    if day_m:
        stumps_day = int(day_m.group(1))
    elif match.get("stumps_day"):
        stumps_day = int(match.get("stumps_day"))
    else:
        inn = int(match.get("current_innings") or 1)
        stumps_day = 1 if inn == 1 else (2 if inn <= 2 else 3)

    # Active live in-play indicators vs genuine close of play
    is_session_active = any(k in combined_status for k in [
        "session 1", "session 2", "session 3", "live in-play", "in progress",
        "post-tea", "post-lunch", "over ", "needs ", "need ", "lead by", "trail by"
    ])
    has_explicit_stumps_kw = bool(re.search(r"\bstumps\b|\bclose of play\b|\bday close\b|\bovernight\b", combined_status))

    # CRITICAL TEST CRICKET LAW:
    # 1. Day 5 is the final day of a 5-day Test match. There is NEVER "Day 5 Stumps".
    #    On Day 5, play is either actively LIVE or the match concludes as a Draw / Result.
    # 2. If an active session is in play (e.g. Session 1, Session 2, Session 3), it is NOT stumps.
    # 3. Dismissal term "stumped" must NEVER trigger Stumps.
    if stumps_day >= 5:
        stumps_detected = False
    elif is_session_active and not has_explicit_stumps_kw:
        stumps_detected = False
    elif has_explicit_stumps_kw:
        stumps_detected = True
    elif (explicit_is_stumps == 1 or explicit_is_stumps is True) and not is_session_active:
        stumps_detected = True
    else:
        stumps_detected = False

    # Test cricket stoppage evaluation (stoppages during the day)
    is_stopped = False
    stoppage_reason = None
    stoppage_label = None

    if not stumps_detected:
        if any(k in last_event for k in ["rain stopped play", "play stopped due to rain", "covers are coming on", "rain delay", "rain interruption"]):
            is_stopped = True
            stoppage_reason = "RAIN"
            stoppage_label = "🌧️ PLAY STOPPED • RAIN"
        elif any(k in last_event for k in ["bad light", "umpires offer light", "light stops play"]):
            is_stopped = True
            stoppage_reason = "BAD_LIGHT"
            stoppage_label = "⚠️ PLAY STOPPED • BAD LIGHT"
        elif "lunch" in last_event and not is_live_active:
            is_stopped = True
            stoppage_reason = "LUNCH"
            stoppage_label = f"🥪 LUNCH BREAK • DAY {stumps_day}"
        elif "tea" in last_event and not is_live_active:
            is_stopped = True
            stoppage_reason = "TEA"
            stoppage_label = f"☕ TEA BREAK • DAY {stumps_day}"
        elif "drinks" in last_event and not is_live_active:
            is_stopped = True
            stoppage_reason = "DRINKS"
            stoppage_label = "🥤 DRINKS BREAK"
    
    # Determine whether the current innings has concluded (all 10 wickets down)
    cur_score = int(match.get("current_score") or 0)
    cur_wkts = int(match.get("current_wickets") or 0)
    cur_overs = float(match.get("current_overs") or 0.0)
    cur_innings = int(match.get("current_innings") or 1)
    is_innings_concluded = (cur_wkts >= 10)

    team_a_id = match.get("team_a_id")
    team_b_id = match.get("team_b_id")
    team_a_name = match.get("team_a_name") or match.get("team_a_short") or team_a_id
    team_b_name = match.get("team_b_name") or match.get("team_b_short") or team_b_id

    inn_1_team = match.get("innings_1_team_id")
    inn_2_team = match.get("innings_2_team_id")
    s1 = int(match.get("innings_1_score") or 0)
    s2 = int(match.get("innings_2_score") or (cur_score if cur_innings == 2 else 0))

    # Resolve batting and bowling squads dynamically with multi-innings fallbacks
    batting_team_id = match.get("batting_team_id")
    if not batting_team_id:
        if cur_innings == 2 and inn_2_team:
            batting_team_id = inn_2_team
        elif cur_innings == 1 and inn_1_team:
            batting_team_id = inn_1_team
        elif match.get("toss_winner_id"):
            first_bat = match["toss_winner_id"] if match.get("toss_decision") == "bat" else (team_b_id if match["toss_winner_id"] == team_a_id else team_a_id)
            second_bat = team_b_id if first_bat == team_a_id else team_a_id
            batting_team_id = first_bat if cur_innings in [1, 3] else second_bat
        else:
            batting_team_id = team_a_id

    bowling_team_id = match.get("bowling_team_id") or (team_b_id if batting_team_id == team_a_id else team_a_id)
    
    batting_squad = squad_a if team_a_id == batting_team_id else squad_b
    bowling_squad = squad_b if batting_squad == squad_a else squad_a

    batting_team_name = match.get("batting_team_name") or (team_a_name if batting_team_id == team_a_id else team_b_name)
    bowling_team_name = match.get("bowling_team_name") or (team_b_name if batting_team_id == team_a_id else team_a_name)

    # Identify overnight crease batters (unbeaten batters who will resume tomorrow)
    # Cricket Law & Multi-Innings Integrity:
    # 1. At any point in play (including at stumps), exactly AT MOST TWO batters can be at the crease.
    # 2. When W wickets have fallen in an innings, ONLY batters who have walked to the crease (batting order <= W + 2) can have batted.
    # 3. Any tailender or batter whose batting order > W + 2 has strictly YET TO BAT and CANNOT be at the crease.
    # 4. If striker_id and non_striker_id are identified in the match record, THOSE TWO are the authoritative crease batters.
    striker_id = match.get("current_striker_id")
    non_striker_id = match.get("current_non_striker_id")
    cur_wkts = int(match.get("current_wickets") or 0)
    
    overnight_batters = []
    if not is_innings_concluded and cur_wkts < 10:
        active_candidates = []
        for p in batting_squad:
            pid = p.get("player_id") or p.get("id")
            dis = str(p.get("dismissal") or "").lower()
            order = int(p.get("batting_order") or 99)
            is_out = bool(p.get("is_out", 0))

            # Exclude dismissed players and players who haven't batted yet
            if is_out or "yet to bat" in dis:
                continue

            # Authoritative check: player is striker, non-striker, or marked batting
            is_crease = (pid in [striker_id, non_striker_id])
            is_batting_status = ("batting" in dis or "not out" in dis)
            
            # Legitimate order check: cannot be a tailender yet to walk out
            order_valid = (order <= cur_wkts + 2) or is_crease

            if order_valid and (is_crease or is_batting_status):
                runs = int(p.get("current_match_runs") or 0)
                balls = int(p.get("balls_faced") or 0)
                sr = round((runs / max(1, balls)) * 100.0, 1) if balls > 0 else 0.0
                active_candidates.append({
                    "player_id": pid,
                    "name": p.get("name", "Batter"),
                    "runs": runs,
                    "balls": balls,
                    "fours": int(p.get("fours") or 0),
                    "sixes": int(p.get("sixes") or 0),
                    "strike_rate": sr,
                    "is_striker": (pid == striker_id),
                    "order": order,
                    "tomorrow_morning_outlook": f"Resumes on {runs}* ({balls}b) • Crucial first 45 mins survival required against covers moisture"
                })

        # Sort: striker first, then batting order, then runs
        active_candidates.sort(key=lambda b: (not b["is_striker"], b["order"], -b["runs"]))
        # In cricket, exactly AT MOST TWO batters can be at the crease overnight
        overnight_batters = active_candidates[:2]
    
    # Identify key bowling figures at stumps
    stumps_bowlers = []
    for bw in bowling_squad:
        ov = float(bw.get("overs_bowled") or 0.0)
        if ov > 0.0 or bw.get("wickets_taken", 0) > 0:
            stumps_bowlers.append({
                "player_id": bw.get("player_id") or bw.get("id"),
                "name": bw.get("name", "Bowler"),
                "overs": ov,
                "maidens": int(bw.get("maidens_bowled") or 0),
                "runs": int(bw.get("runs_conceded") or 0),
                "wickets": int(bw.get("wickets_taken") or 0),
                "economy": round(bw.get("bowling_economy") or (bw.get("runs_conceded", 0) / max(0.1, ov)), 2)
            })
    stumps_bowlers.sort(key=lambda bw: (bw["wickets"], -bw["economy"]), reverse=True)

    bowling_team_name = bowling_team_name or match.get("bowling_team_name") or match.get("bowling_team_short") or "Bowling side"
    batting_team_name = batting_team_name or match.get("batting_team_name") or match.get("batting_team_short") or "Batting side"

    # Pitch Archetype & Morning Session Condensation Seam Dynamics
    arch = (match.get("active_pitch_type") or match.get("ground_pitch_archetype") or venue_profile.get("pitch_type", "balanced")).lower()
    v_name = venue_profile.get("name", "Venue")
    seam_hazard_pct = 18.5 if arch in ["pace_friendly", "green", "seam", "seam / green"] else 14.0

    # Morning Session Par runs & expected wickets calibrated by pitch archetype & surface wear
    if arch == "flat":
        session_par_runs = 94 if stumps_day <= 2 else (88 if stumps_day == 3 else 78)
        session_exp_wkts = 2
    elif arch in ["green", "seam"]:
        session_par_runs = 76 if stumps_day <= 2 else (84 if stumps_day == 3 else 80)
        session_exp_wkts = 3 if stumps_day <= 2 else 2
    elif arch == "dusty":
        session_par_runs = 82 if stumps_day <= 2 else (74 if stumps_day == 3 else 64)
        session_exp_wkts = 3 if stumps_day >= 3 else 2
    else:  # balanced
        session_par_runs = 86 if stumps_day <= 2 else 80
        session_exp_wkts = 2

    if is_innings_concluded:
        # The team batting tomorrow morning is the NEXT team in the multi-innings rotation
        if cur_innings == 1:
            next_bat_id = inn_2_team or (team_b_id if inn_1_team == team_a_id else team_a_id)
            next_bowl_id = inn_1_team or (team_a_id if next_bat_id == team_b_id else team_b_id)
            next_inn_label = "2nd Innings"
            lead_diff = -s1
        elif cur_innings == 2:
            next_bat_id = inn_1_team or (team_a_id if inn_2_team == team_b_id else team_b_id)
            next_bowl_id = inn_2_team or (team_b_id if next_bat_id == team_a_id else team_a_id)
            next_inn_label = "3rd Innings"
            lead_diff = s1 - s2
        else:
            next_bat_id = inn_2_team or (team_b_id if inn_1_team == team_a_id else team_a_id)
            next_bowl_id = inn_1_team or (team_a_id if next_bat_id == team_b_id else team_b_id)
            next_inn_label = "4th Innings"
            lead_diff = 0

        next_bat_name = team_a_name if next_bat_id == team_a_id else team_b_name
        next_bowl_name = team_a_name if next_bowl_id == team_a_id else team_b_name

        # Upcoming team begins at 0/0 (0.0 ov); Lunch target is their session par
        proj_lunch_runs = session_par_runs
        proj_lunch_wkts = session_exp_wkts
        if cur_innings == 2 and lead_diff > 0:
            lunch_score_target = f"~{proj_lunch_runs}/{proj_lunch_wkts} ({next_bat_name} • Lead: ~{lead_diff + proj_lunch_runs})"
        elif cur_innings == 2 and lead_diff < 0:
            lunch_score_target = f"~{proj_lunch_runs}/{proj_lunch_wkts} ({next_bat_name} • Trail: ~{abs(lead_diff) - proj_lunch_runs})"
        else:
            lunch_score_target = f"~{proj_lunch_runs}/{proj_lunch_wkts} ({next_bat_name} {next_inn_label})"

        new_ball_available = True
        overs_until_new_ball = 0.0
        new_ball_desc = f"Brand New Ball • {next_bowl_name} frontline attack will take fresh new ball from Ball 1 of {next_bat_name}'s {next_inn_label}"

        tactical_keys = [
            f"{next_bat_name} (Batting): Openers must negotiate early morning nip under covers moisture against {next_bowl_name}'s fresh new ball" + (f" to expand the {lead_diff}-run lead" if lead_diff > 0 else ""),
            f"{next_bowl_name} (Bowling): Frontline pacers must strike early with brand new ball during opening 45-60 mins to break {next_bat_name}'s top order before pitch flattens under sun",
            "Pitch Drift: Clay surface dries rapidly under afternoon heat; spinners will gain sharper turn from Session 2 onwards"
        ]

        lead_text = f" • {next_bat_name} leads by {lead_diff} runs" if lead_diff > 0 else (f" • {next_bat_name} trails by {abs(lead_diff)} runs" if lead_diff < 0 else "")
        score_headline = f"Day {stumps_day} Stumps: {batting_team_name} {cur_score}/{cur_wkts} (All Out){lead_text}"
    else:
        # Innings is actively in-play overnight (resuming tomorrow, e.g. AFG 402/7 or WI 7/0)
        proj_session_total = cur_score + session_par_runs
        proj_session_wkts = min(10, cur_wkts + session_exp_wkts)
        lunch_score_target = f"~{proj_session_total}/{proj_session_wkts}"

        deficit = s1 - cur_score if cur_innings == 2 else 0
        overs_to_2nd_ball = max(0.0, round(80.0 - cur_overs, 1))
        if cur_overs >= 80.0:
            overs_with_2nd_ball = round(cur_overs - 80.0, 1)
            overs_to_3rd_ball = max(0.0, round(160.0 - cur_overs, 1))
            new_ball_available = False
            overs_until_new_ball = overs_to_3rd_ball
            new_ball_desc = f"2nd New Ball is {overs_with_2nd_ball} ov old (taken at 80.0 ov) • Ball is softened and scuffed, aiding reverse swing and rough spin. 3rd New Ball due in {overs_to_3rd_ball} ov"
        elif overs_to_2nd_ball <= 12.0:
            overs_until_new_ball = overs_to_2nd_ball
            new_ball_available = False
            new_ball_desc = f"Due in {overs_until_new_ball} overs (Over {cur_overs}/80.0) • {bowling_team_name} frontline pacers will attack early in Day {stumps_day + 1} Morning Session"
        else:
            overs_until_new_ball = overs_to_2nd_ball
            new_ball_available = False
            new_ball_desc = f"{overs_until_new_ball} overs until 2nd New Ball (due at 80.0 ov)"

        active_batters_str = " & ".join(f"{b['name']} ({b['runs']}*)" for b in overnight_batters[:2]) if overnight_batters else "Lower order"
        tactical_keys = [
            f"{batting_team_name} (Batting): {active_batters_str} must negotiate opening hour moisture" + (f" to erase the {deficit}-run deficit" if deficit > 0 else " before scoring accelerates"),
            f"{bowling_team_name} (Bowling): Attack with reverse swing and spinners targeting rough outside off stump to claim the remaining {10 - cur_wkts} wickets before Lunch",
            "Pitch Drift: Clay surface dries rapidly under afternoon heat; spinners will gain sharper turn from Session 2 onwards"
        ]

        score_headline = f"Day {stumps_day} Stumps: {batting_team_name} {cur_score}/{cur_wkts} ({cur_overs} ov)"
        if cur_innings == 2 and s1 > 0:
            if cur_score > s1:
                score_headline += f" • {batting_team_name} lead by {cur_score - s1} runs"
            elif cur_score < s1:
                score_headline += f" • {batting_team_name} trail by {s1 - cur_score} runs"
            else:
                score_headline += " • Scores Level"
        elif overnight_batters:
            bat_names = " & ".join(f"{b['name']} {b['runs']}*" for b in overnight_batters[:2])
            score_headline += f" • {bat_names}"

    if stumps_detected and stumps_day < 5:
        stumps_label = f"🌙 STUMPS • DAY {stumps_day}"
        morning_forecast = {
            "session_title": f"Day {stumps_day + 1} Session 1 (Morning)",
            "start_time_local": "09:30 AM Local",
            "weather": "Clear sunny morning • 26°C • Trapped condensation under tarpaulin covers",
            "covers_condensation_impact": f"+{seam_hazard_pct}% lateral seam deviation during opening 45-60 mins before pitch dries out",
            "projected_session_runs": f"{session_par_runs} runs ({session_par_runs - 15} - {session_par_runs + 18})",
            "projected_session_wickets": f"{session_exp_wkts} wickets",
            "lunch_score_target": lunch_score_target,
            "tactical_keys": tactical_keys
        }
    else:
        score_headline = f"Day {stumps_day} Live In-Play: {batting_team_name} {cur_score}/{cur_wkts} ({cur_overs} ov)"
        if overnight_batters:
            bat_names = " & ".join(f"{b['name']} {b['runs']}*" for b in overnight_batters[:2])
            score_headline += f" • {bat_names} at crease"
        stumps_label = None
        morning_forecast = None

    return {
        "is_stumps": stumps_detected,
        "stumps_day": stumps_day,
        "stumps_label": stumps_label,
        "is_stopped": is_stopped,
        "stoppage_reason": stoppage_reason,
        "stoppage_label": stoppage_label,
        "score_headline": score_headline,
        "batting_team_short": match.get("batting_team_short"),
        "bowling_team_short": match.get("bowling_team_short"),
        "overnight_score": f"{cur_score}/{cur_wkts}",
        "overnight_overs": cur_overs,
        "overnight_batters": overnight_batters,
        "stumps_bowlers": stumps_bowlers[:4],
        "second_new_ball": {
            "is_available": new_ball_available,
            "overs_remaining": overs_until_new_ball,
            "threshold_overs": 80.0,
            "description": new_ball_desc
        },
        "morning_session_forecast": morning_forecast
    }
