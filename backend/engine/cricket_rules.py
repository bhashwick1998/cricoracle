"""
Cricket Comprehensive Rules & Format Regulations Engine
Defines the official ICC rules and regulations across TEST, ODI, and T20I formats:
- Test: 5-day format, 2-innings per side, follow-on thresholds, declarations, NO DLS (draw probability curve instead)
- ODI: 50-over format, 1-innings, bowler 10-over quota, 3-phase powerplays, 20-over min for DLS
- T20I: 20-over format, 1-innings, bowler 4-over quota, 1-phase powerplay, 5-over min for DLS, Super Over
"""

import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

FORMAT_RULES: Dict[str, Dict[str, Any]] = {
    "TEST": {
        "format_name": "Test Cricket",
        "duration_days": 5,
        "scheduled_overs_per_day": 90,
        "total_scheduled_overs": 450,
        "innings_per_team": 2,
        "max_innings_match": 4,
        "bowler_over_quota": None,  # No restrictions on bowler overs in Test cricket
        "dls_applicable": False,    # STRICT ICC RULE: DLS NEVER APPLIED TO TEST MATCHES
        "rain_interruption_outcome": "Lost overs increase DRAW probability curve P(Draw); match drawn if 4 innings not completed in time",
        "follow_on_rules": {
            "5_day_match_lead_threshold": 200,
            "3_or_4_day_match_lead_threshold": 150,
            "2_day_match_lead_threshold": 100,
            "1_day_match_lead_threshold": 75,
            "description": "Team batting 1st leading by >=200 runs (in 5-day Test) can enforce follow-on, compelling opponent to bat 2nd innings consecutively (ICC Law 14)"
        },
        "declarations_allowed": True,
        "new_ball_available_after_overs": 80,
        "powerplay_restrictions": None,  # No mandatory fielding circles / powerplay restrictions in Tests
        "result_types": ["WIN", "LOSS", "DRAW", "TIE (Both innings scores identical upon completion, 2 in history)"]
    },
    "ODI": {
        "format_name": "One Day International (ODI)",
        "duration_days": 1,
        "scheduled_overs_per_side": 50,
        "innings_per_team": 1,
        "max_innings_match": 2,
        "bowler_over_quota": 10,   # Max 10 overs per bowler in 50-over match
        "dls_applicable": True,    # ICC Duckworth-Lewis-Stern Method applies on interruptions
        "min_overs_for_official_result": 20,  # Minimum 20 overs per side for DLS result
        "powerplay_phases": {
            "P1": {"overs": "1 - 10", "max_fielders_outside_circle": 2, "name": "Mandatory Powerplay 1"},
            "P2": {"overs": "11 - 40", "max_fielders_outside_circle": 4, "name": "Middle Overs Powerplay 2"},
            "P3": {"overs": "41 - 50", "max_fielders_outside_circle": 5, "name": "Death Overs Powerplay 3"}
        },
        "free_hit_on_no_ball": True,
        "follow_on_rules": None,
        "declarations_allowed": False
    },
    "T20I": {
        "format_name": "Twenty20 International (T20I)",
        "duration_days": 1,
        "scheduled_overs_per_side": 20,
        "innings_per_team": 1,
        "max_innings_match": 2,
        "bowler_over_quota": 4,    # Max 4 overs per bowler in 20-over match
        "dls_applicable": True,    # ICC Duckworth-Lewis-Stern Method applies
        "min_overs_for_official_result": 5,   # Minimum 5 overs per side for DLS result
        "powerplay_phases": {
            "P1": {"overs": "1 - 6", "max_fielders_outside_circle": 2, "name": "Mandatory Powerplay"},
            "P2": {"overs": "7 - 15", "max_fielders_outside_circle": 4, "name": "Middle Overs Restriction"},
            "P3": {"overs": "16 - 20", "max_fielders_outside_circle": 5, "name": "Death Overs Restriction"}
        },
        "free_hit_on_no_ball": True,
        "tie_breaker": "Super Over (1 over per side, 2 wickets maximum)",
        "in_match_over_rate_penalty": "If bowling team fails to bowl 1st ball of 20th over by cut-off time, 1 additional fielder must enter the 30-yard circle (max 4 outside)",
        "follow_on_rules": None,
        "declarations_allowed": False
    }
}

def compute_curtailed_powerplay_breakdown(format_code: str, scheduled_overs: float) -> Dict[str, Any]:
    """
    ICC Playing Conditions Clause 28:
    Dynamic Powerplay and Fielding Restriction Calculator for rain-reduced matches.
    Calculates exact ball cutoffs, over ranges, and fielders permitted outside the 30-yard circle.
    """
    fmt = str(format_code or "T20I").upper()
    n = float(scheduled_overs)
    if fmt == "ODI":
        # Standard: P1 (1-10, 2 outside), P2 (11-40, 4 outside), P3 (41-50, 5 outside)
        p1_end = max(1, int(round(n * 0.20)))
        p2_end = max(p1_end + 1, int(round(n * 0.80)))
        p3_end = int(n)
        return {
            "format": "ODI",
            "scheduled_overs": n,
            "p1": {"start": 1, "end": p1_end, "overs": f"1 - {p1_end}", "max_outside": 2, "name": "Mandatory Powerplay 1"},
            "p2": {"start": p1_end + 1, "end": p2_end, "overs": f"{p1_end + 1} - {p2_end}", "max_outside": 4, "name": "Middle Overs Powerplay 2"},
            "p3": {"start": p2_end + 1, "end": p3_end, "overs": f"{p2_end + 1} - {p3_end}", "max_outside": 5, "name": "Death Overs Powerplay 3"} if p3_end > p2_end else None,
            "summary": f"PP1 (1-{p1_end} ov, 2 out) • PP2 ({p1_end + 1}-{p2_end} ov, 4 out)" + (f" • PP3 ({p2_end + 1}-{p3_end} ov, 5 out)" if p3_end > p2_end else "")
        }
    elif "T20" in fmt:
        # Standard: P1 (1-6, 2 outside), P2 (7-15, 4 outside), P3 (16-20, 5 outside)
        p1_end = max(1, int(round(n * 0.30)))
        p2_end = max(p1_end + 1, int(round(n * 0.75)))
        p3_end = int(n)
        return {
            "format": "T20I",
            "scheduled_overs": n,
            "p1": {"start": 1, "end": p1_end, "overs": f"1 - {p1_end}", "max_outside": 2, "name": "Mandatory Powerplay"},
            "p2": {"start": p1_end + 1, "end": p2_end, "overs": f"{p1_end + 1} - {p2_end}", "max_outside": 4, "name": "Middle Overs Restriction"},
            "p3": {"start": p2_end + 1, "end": p3_end, "overs": f"{p2_end + 1} - {p3_end}", "max_outside": 5, "name": "Death Overs Restriction"} if p3_end > p2_end else None,
            "summary": f"PP (1-{p1_end} ov, 2 out) • Middle ({p1_end + 1}-{p2_end} ov, 4 out)" + (f" • Death ({p2_end + 1}-{p3_end} ov, 5 out)" if p3_end > p2_end else "")
        }
    else:  # TEST
        return {
            "format": "TEST",
            "scheduled_overs": n,
            "p1": None,
            "p2": None,
            "p3": None,
            "summary": "No mandatory fielding circles in Test cricket"
        }

def get_format_rules(format_code: str, scheduled_overs: Optional[float] = None) -> Dict[str, Any]:
    fmt = str(format_code or "T20I").upper()
    rules = dict(FORMAT_RULES.get(fmt, FORMAT_RULES["T20I"]))
    if scheduled_overs is not None and scheduled_overs > 0 and fmt in ["ODI", "T20I", "T20"]:
        pp = compute_curtailed_powerplay_breakdown(fmt, scheduled_overs)
        rules["scheduled_overs_per_side"] = scheduled_overs
        phases = {}
        if pp.get("p1"): phases["P1"] = {"overs": pp["p1"]["overs"], "max_fielders_outside_circle": pp["p1"]["max_outside"], "name": pp["p1"]["name"]}
        if pp.get("p2"): phases["P2"] = {"overs": pp["p2"]["overs"], "max_fielders_outside_circle": pp["p2"]["max_outside"], "name": pp["p2"]["name"]}
        if pp.get("p3"): phases["P3"] = {"overs": pp["p3"]["overs"], "max_fielders_outside_circle": pp["p3"]["max_outside"], "name": pp["p3"]["name"]}
        rules["powerplay_phases"] = phases
        rules["powerplay_summary"] = pp.get("summary")
    return rules

_CONCLUSION_PHRASES = (
    "won by", "won the match", "win by", "take the game by", "takes the game by",
    "match drawn", "match tied", "abandoned", "no result", "called off",
)

_TEAM_ALIASES = {
    "SA": ("south africa", "sa", "rsa", "proteas"),
    "BAN": ("bangladesh", "ban", "tigers"),
    "IND": ("india", "ind", "bharat"),
    "AUS": ("australia", "aus"),
    "ENG": ("england", "eng"),
    "NZ": ("new zealand", "nz", "blackcaps"),
    "PAK": ("pakistan", "pak"),
    "SL": ("sri lanka", "sl"),
    "WI": ("west indies", "wi"),
    "AFG": ("afghanistan", "afg"),
    "ZIM": ("zimbabwe", "zim"),
    "IRE": ("ireland", "ire"),
}


def _resolve_winning_team_from_text(
    text: str, t_a_id: str, t_a_name: str, t_b_id: str, t_b_name: str
) -> tuple[Optional[str], Optional[str]]:
    """Resolves winning team from result phrasing before falling back to isolated team tokens."""
    txt_low = (text or "").lower()
    win_subject = re.search(
        r"(?:^|[.!]\s*|\b)([a-z\s]{2,30}?)\s+(?:won\s+by|won\s+the\s+match|win\s+by|wins\s+by|takes?\s+the\s+game)",
        txt_low,
    )
    probe = win_subject.group(1).strip() if win_subject else txt_low

    a_tokens = tuple(dict.fromkeys([t_a_name.lower(), t_a_id.lower(), *_TEAM_ALIASES.get(str(t_a_id).upper(), ())]))
    b_tokens = tuple(dict.fromkeys([t_b_name.lower(), t_b_id.lower(), *_TEAM_ALIASES.get(str(t_b_id).upper(), ())]))

    for target_text in (probe, txt_low):
        a_hit = any(re.search(rf"\b{re.escape(tok)}\b", target_text) for tok in a_tokens if tok)
        b_hit = any(re.search(rf"\b{re.escape(tok)}\b", target_text) for tok in b_tokens if tok)
        if a_hit and not b_hit:
            return t_a_id, t_a_name
        if b_hit and not a_hit:
            return t_b_id, t_b_name
        if a_hit and b_hit and win_subject:
            a_pos = min((target_text.find(tok) for tok in a_tokens if tok and tok in target_text), default=9999)
            b_pos = min((target_text.find(tok) for tok in b_tokens if tok and tok in target_text), default=9999)
            return (t_a_id, t_a_name) if a_pos <= b_pos else (t_b_id, t_b_name)
    return None, None


def evaluate_match_completion(match: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates whether a match is officially completed according to official ICC rules.
    Returns:
    {
        "is_completed": bool,
        "result_description": Optional[str],
        "winning_team_id": Optional[str],
        "winning_team_name": Optional[str],
        "margin": Optional[str],
        "is_tie": bool,
        "is_draw": bool
    }
    """
    state = match.get("state")
    last_event = str(match.get("last_event_description") or "").lower()
    match_status = str(match.get("match_status_text") or "").lower()
    combined_status = f"{match_status} {last_event}"
    series_name = str(match.get("series_name") or "").lower()
    mid_low = str(match.get("id") or "").lower()

    has_explicit_conclusion = any(k in combined_status for k in _CONCLUSION_PHRASES)

    # Guard -1: If match is already marked COMPLETED in state or status explicitly declares winner/result
    if state == "COMPLETED" or has_explicit_conclusion:
        is_tie = "tied" in combined_status or "tie" in combined_status
        is_draw = "drawn" in combined_status or "draw" in combined_status
        t_a_id = match.get("team_a_id", "Team A")
        t_b_id = match.get("team_b_id", "Team B")
        t_a_name = match.get("team_a_name") or match.get("team_a_short") or t_a_id
        t_b_name = match.get("team_b_name") or match.get("team_b_short") or t_b_id
        win_id = match.get("winning_team_id")
        win_name = match.get("winning_team_name")
        if not win_id and not (is_tie or is_draw):
            win_id, win_name = _resolve_winning_team_from_text(combined_status, t_a_id, t_a_name, t_b_id, t_b_name)
        desc = match.get("result_description")
        if not desc:
            if any(k in match_status for k in _CONCLUSION_PHRASES):
                desc = match.get("match_status_text")
            elif win_name and "by an innings" in combined_status and "won by" not in combined_status:
                inn1 = match.get("innings_1_score")
                inn2 = match.get("innings_2_score")
                inn3 = match.get("innings_3_score")
                if inn1 is not None and inn2 is not None and inn3 is not None:
                    margin = abs(max(int(inn1), int(inn2)) - (min(int(inn1), int(inn2)) + int(inn3)))
                    desc = f"{win_name} won by an innings and {margin} runs"
                else:
                    desc = f"{win_name} won by an innings"
            else:
                desc = match.get("match_status_text") or (f"{win_name} won the match" if win_name else "Match Completed")
        return {
            "is_completed": True,
            "result_description": desc,
            "winning_team_id": None if (is_tie or is_draw) else win_id,
            "winning_team_name": None if (is_tie or is_draw) else win_name,
            "margin": match.get("margin"),
            "is_tie": is_tie,
            "is_draw": is_draw
        }

    # Check for active Stumps / Day Close indicator
    is_stumps_indicator = bool(match.get("is_stumps")) or bool(re.search(r"\bstumps\b", combined_status)) or any(
        k in combined_status for k in ["close of play", "day close", "overnight"]
    )

    # Compute elapsed days from scheduled start for multi-day Test match rules
    test_days_elapsed = 0.0
    if match.get("scheduled_start"):
        try:
            sch_str = str(match["scheduled_start"]).replace("Z", "+00:00")
            sch_dt = datetime.fromisoformat(sch_str)
            if sch_dt.tzinfo is None:
                sch_dt = sch_dt.replace(tzinfo=timezone.utc)
            test_days_elapsed = (datetime.now(timezone.utc) - sch_dt).total_seconds() / 86400.0
        except Exception:
            pass

    # Guard 0: An in-progress Test match at Stumps (Day Close) is NEVER completed unless 5 days have elapsed!
    if is_stumps_indicator and not has_explicit_conclusion:
        if test_days_elapsed >= 5.2:
            return {
                "is_completed": True,
                "result_description": match.get("match_status_text") if "drawn" in match_status else "Match Drawn (5-day match duration elapsed)",
                "winning_team_id": None,
                "winning_team_name": None,
                "margin": "Draw",
                "is_tie": False,
                "is_draw": True
            }
        day_m = re.search(r"day\s*([1-5])", combined_status) or re.search(r"([1-5])(?:st|nd|rd|th)\s*day", combined_status)
        day_num = int(day_m.group(1)) if day_m else int(match.get("stumps_day") or 3)
        return {
            "is_completed": False,
            "is_stopped": True,
            "stoppage_label": match.get("stumps_label") or f"🌙 STUMPS • DAY {day_num}",
            "result_description": match.get("match_status_text") or f"Day {day_num}: Stumps - In Progress",
            "winning_team_id": None,
            "winning_team_name": None,
            "margin": None,
            "is_tie": False,
            "is_draw": False
        }

    # Format self-healing: If series_name, match_id, status, or overs indicates Test, enforce TEST format
    fmt = str(match.get("format", "T20I")).upper()
    if fmt == "TEST" or (
        "test" in series_name
        or "test" in mid_low
        or is_stumps_indicator
        or any(k in combined_status for k in ["trail by", "lead by", "follow-on"])
        or (bool(re.search(r"\bday\s*[1-5]\b", combined_status)) and any(k in combined_status for k in ["session", "stumps", "trail", "lead", "innings", "test"]))
        or float(match.get("current_overs") or 0.0) > 50.0
        or float(match.get("innings_1_overs") or 0.0) > 50.0
    ):
        fmt = "TEST"
    cur_inn = int(match.get("current_innings") or 1)
    cur_score = int(match.get("current_score") or 0)
    cur_wkts = int(match.get("current_wickets") or 0)
    cur_overs = float(match.get("current_overs") or 0.0)
    target_runs = match.get("target_runs")
    overs_lost = float(match.get("overs_lost") or 0.0)
    last_event = str(match.get("last_event_description") or "").lower()
    match_status = str(match.get("match_status_text") or "").lower()
    combined_status = f"{match_status} {last_event}"

    t_a_id = match.get("team_a_id", "Team A")
    t_b_id = match.get("team_b_id", "Team B")
    t_a_name = match.get("team_a_name") or match.get("team_a_short") or t_a_id
    t_b_name = match.get("team_b_name") or match.get("team_b_short") or t_b_id

    # Check for active rain stoppage / weather delay BEFORE evaluating completion
    rain_stoppage_keywords = [
        "rain stops play", "delayed by rain", "rain delay", "match delayed",
        "bad light", "covers on", "covers are on", "wet outfield", "play halted",
        "play stopped", "interrupted by rain", "rain interruption", "drizzle"
    ]
    is_rain_stopped = any(k in combined_status for k in rain_stoppage_keywords)
    is_officially_called_off = any(k in combined_status for k in [
        "match called off", "match abandoned", "abandoned without a ball", "no result",
        "match over", "match concluded", "dls method", "won by"
    ])

    min_dls_overs = 20.0 if fmt == "ODI" else 5.0

    # Minimum Result Cutoffs & Official Abandonment Rule:
    # If match is abandoned / called off / no result before reaching minimum overs cutoff,
    # it CANNOT be awarded via DLS or score lead — it is officially a NO RESULT (ICC Law 16.4).
    is_explicit_abandonment = any(k in combined_status for k in [
        "match called off", "match abandoned", "abandoned without a ball", "no result", "abandoned"
    ]) and not any(k in combined_status for k in ["won by", "won the match"])

    if is_explicit_abandonment:
        # Check if 2nd innings completed minimum cutoff for a valid DLS result
        if fmt in ["ODI", "T20I"] and cur_inn >= 2 and cur_overs >= min_dls_overs and "dls" in combined_status:
            pass  # Fall through to DLS evaluation below
        else:
            abandon_desc = match.get("match_status_text") or match.get("last_event_description") or f"Match Abandoned - No Result (Minimum {int(min_dls_overs)} overs cutoff not met)"
            return {
                "is_completed": True,
                "result_description": abandon_desc,
                "winning_team_id": None,
                "winning_team_name": None,
                "margin": "No Result",
                "is_tie": False,
                "is_draw": False,
                "is_abandoned": True
            }

    # Check for explicit official completion keywords in status or event text
    if any(k in combined_status for k in ["won by", "won the match", "match tied", "match drawn", "dls method"]):
        is_tie = "tied" in combined_status
        is_draw = "drawn" in combined_status
        # Resolve winning team accurately
        win_id = None
        win_name = None
        if not (is_tie or is_draw):
            if any(k in combined_status for k in [t_a_name.lower(), t_a_id.lower(), f"{t_a_id.lower()} won", f"{t_a_name.lower()} won", f"{t_a_name.lower()} win"]):
                win_id = t_a_id
                win_name = t_a_name
            elif any(k in combined_status for k in [t_b_name.lower(), t_b_id.lower(), f"{t_b_id.lower()} won", f"{t_b_name.lower()} won", f"{t_b_name.lower()} win"]):
                win_id = t_b_id
                win_name = t_b_name
            elif match.get("p_team_a", 0) > 50:
                win_id = t_a_id
                win_name = t_a_name
            else:
                win_id = t_b_id
                win_name = t_b_name

        official_desc = match.get("match_status_text") if any(k in match_status for k in ["won by", "drawn", "tied", "called off", "dls method"]) else match.get("last_event_description")
        return {
            "is_completed": True,
            "result_description": official_desc or ("Match Drawn" if is_draw else ("Match Tied" if is_tie else f"{win_name} won the match")),
            "winning_team_id": None if (is_tie or is_draw) else win_id,
            "winning_team_name": None if (is_tie or is_draw) else win_name,
            "margin": None,
            "is_tie": is_tie,
            "is_draw": is_draw
        }

    # Minimum overs DLS rule (ODI: >= 20.0 ov, T20I: >= 5.0 ov) when rain stops play permanently
    min_dls_overs = 20.0 if fmt == "ODI" else 5.0
    if fmt in ["ODI", "T20I"] and cur_inn >= 2 and cur_overs >= min_dls_overs and (is_rain_stopped and (overs_lost >= 15.0 or "dls" in combined_status)):
        par_val = match.get("dls_par_score")
        if par_val is not None:
            diff = cur_score - int(par_val)
            chasing_id = match.get("innings_2_team_id") or t_b_id
            defending_id = match.get("innings_1_team_id") or t_a_id
            chasing_name = t_b_name if chasing_id == t_b_id else t_a_name
            defending_name = t_a_name if defending_id == t_a_id else t_b_name

            if diff > 0:
                return {
                    "is_completed": True,
                    "result_description": f"{chasing_name} won by {diff} runs (DLS method)",
                    "winning_team_id": chasing_id,
                    "winning_team_name": chasing_name,
                    "margin": f"{diff} runs",
                    "is_tie": False,
                    "is_draw": False
                }
            elif diff < 0:
                return {
                    "is_completed": True,
                    "result_description": f"{defending_name} won by {abs(diff)} runs (DLS method)",
                    "winning_team_id": defending_id,
                    "winning_team_name": defending_name,
                    "margin": f"{abs(diff)} runs",
                    "is_tie": False,
                    "is_draw": False
                }
            else:
                return {
                    "is_completed": True,
                    "result_description": f"Match Tied (DLS method • Par: {par_val})",
                    "winning_team_id": None,
                    "winning_team_name": None,
                    "margin": "Tied",
                    "is_tie": True,
                    "is_draw": False
                }

    if is_rain_stopped and not is_officially_called_off:
        return {
            "is_completed": False,
            "is_stopped": True,
            "stoppage_label": "RAIN DELAY",
            "result_description": match.get("match_status_text") or match.get("last_event_description") or "Rain stops play",
            "winning_team_id": None,
            "winning_team_name": None,
            "margin": None,
            "is_tie": False,
            "is_draw": False
        }

    # Only in-play matches can conclude via score threshold
    if state != "LIVE_BALL_BY_BALL":
        return {
            "is_completed": False,
            "result_description": None,
            "winning_team_id": None,
            "winning_team_name": None,
            "margin": None,
            "is_tie": False,
            "is_draw": False
        }

    # 1. Limited Overs Regulations (ODI & T20I)
    if fmt in ["ODI", "T20I"]:
        max_overs = 50.0 if fmt == "ODI" else 20.0
        scheduled_overs = max(5.0 if fmt == "T20I" else 20.0, max_overs - overs_lost)

        # Match only ends in 2nd innings
        if cur_inn >= 2 and target_runs is not None:
            target = int(target_runs)
            # Determine batting (chasing) team and bowling (defending) team
            batting_team_id = match.get("batting_team_id") or t_b_id
            bowling_team_id = t_a_id if batting_team_id == t_b_id else t_b_id
            batting_team_name = t_b_name if batting_team_id == t_b_id else t_a_name
            bowling_team_name = t_a_name if bowling_team_id == t_a_id else t_b_name

            # Condition A: Chasing team reaches target -> Chasing team wins by wickets
            if cur_score >= target:
                wkts_left = max(0, 10 - cur_wkts)
                w_str = f"{wkts_left} wicket" if wkts_left == 1 else f"{wkts_left} wickets"
                res_desc = f"{batting_team_name} won by {w_str} (Target {target} reached in {cur_overs} ov)"
                return {
                    "is_completed": True,
                    "result_description": res_desc,
                    "winning_team_id": batting_team_id,
                    "winning_team_name": batting_team_name,
                    "margin": w_str,
                    "is_tie": False,
                    "is_draw": False
                }

            # Condition B: Chasing team is bowled out (10 wickets) short of target -> Defending team wins by runs
            if cur_wkts >= 10:
                deficit = (target - 1) - cur_score
                if deficit > 0:
                    r_str = f"{deficit} run" if deficit == 1 else f"{deficit} runs"
                    res_desc = f"{bowling_team_name} won by {r_str} (All out {cur_score} in {cur_overs} ov)"
                    return {
                        "is_completed": True,
                        "result_description": res_desc,
                        "winning_team_id": bowling_team_id,
                        "winning_team_name": bowling_team_name,
                        "margin": r_str,
                        "is_tie": False,
                        "is_draw": False
                    }
                elif deficit == 0:
                    return {
                        "is_completed": True,
                        "result_description": f"Match Tied (Target {target}, Scores Level at {cur_score})",
                        "winning_team_id": None,
                        "winning_team_name": None,
                        "margin": "Tied",
                        "is_tie": True,
                        "is_draw": False
                    }

            # Condition C: Scheduled overs exhausted in 2nd innings
            if cur_overs >= scheduled_overs:
                if cur_score > target - 1:
                    wkts_left = max(0, 10 - cur_wkts)
                    res_desc = f"{batting_team_name} won by {wkts_left} wickets"
                    return {
                        "is_completed": True,
                        "result_description": res_desc,
                        "winning_team_id": batting_team_id,
                        "winning_team_name": batting_team_name,
                        "margin": f"{wkts_left} wickets",
                        "is_tie": False,
                        "is_draw": False
                    }
                elif cur_score < target - 1:
                    deficit = (target - 1) - cur_score
                    res_desc = f"{bowling_team_name} won by {deficit} runs ({scheduled_overs} ov completed)"
                    return {
                        "is_completed": True,
                        "result_description": res_desc,
                        "winning_team_id": bowling_team_id,
                        "winning_team_name": bowling_team_name,
                        "margin": f"{deficit} runs",
                        "is_tie": False,
                        "is_draw": False
                    }
                else:
                    return {
                        "is_completed": True,
                        "result_description": f"Match Tied (Scores Level at {cur_score} after {scheduled_overs} ov)",
                        "winning_team_id": None,
                        "winning_team_name": None,
                        "margin": "Tied",
                        "is_tie": True,
                        "is_draw": False
                    }

        # Condition D: Limited-Overs Time Allocation & NLP Completion
        # ODIs and T20s conclude on the same calendar day.
        # T20 scheduled duration ~3.5h (cutoff: 4.5h), ODI scheduled duration ~8h (cutoff: 9.0h).
        if match.get("scheduled_start"):
            try:
                sch_str = str(match["scheduled_start"]).replace("Z", "+00:00")
                sch_dt = datetime.fromisoformat(sch_str)
                if sch_dt.tzinfo is None:
                    sch_dt = sch_dt.replace(tzinfo=timezone.utc)
                now_dt = datetime.now(timezone.utc)
                elapsed_hrs = (now_dt - sch_dt).total_seconds() / 3600.0
                cutoff_hrs = 4.5 if fmt in ["T20I", "T20"] else 9.0

                if elapsed_hrs >= cutoff_hrs:
                    # Time allocation has passed for this limited-overs match
                    # Check for NLP result statement in match_status_text or last_event_description
                    nlp_win_m = re.search(r"([A-Za-z\s]+)\s+won\s+by\s+(\d+\s+(?:runs?|wickets?))", combined_status, re.IGNORECASE)
                    if nlp_win_m:
                        c_team = nlp_win_m.group(1).strip()
                        c_margin = nlp_win_m.group(2).strip()
                        resolved_team_id = t_a_id if (t_a_name.lower() in c_team.lower() or t_a_id.lower() in c_team.lower()) else t_b_id
                        resolved_team_name = t_a_name if resolved_team_id == t_a_id else t_b_name
                        return {
                            "is_completed": True,
                            "result_description": f"{resolved_team_name} won by {c_margin}",
                            "winning_team_id": resolved_team_id,
                            "winning_team_name": resolved_team_name,
                            "margin": c_margin,
                            "is_tie": False,
                            "is_draw": False
                        }

                    # If 2nd innings reached target
                    if cur_inn >= 2 and target_runs is not None:
                        if cur_score >= int(target_runs):
                            w_left = max(0, 10 - cur_wkts)
                            b_id = match.get("batting_team_id") or t_b_id
                            b_name = t_b_name if b_id == t_b_id else t_a_name
                            return {
                                "is_completed": True,
                                "result_description": f"{b_name} won by {w_left} wickets (Target reached)",
                                "winning_team_id": b_id,
                                "winning_team_name": b_name,
                                "margin": f"{w_left} wickets",
                                "is_tie": False,
                                "is_draw": False
                            }
                        elif cur_wkts >= 10 or cur_overs >= scheduled_overs:
                            d_id = t_a_id if (match.get("batting_team_id") or t_b_id) == t_b_id else t_b_id
                            d_name = t_a_name if d_id == t_a_id else t_b_name
                            d_runs = max(1, int(target_runs) - 1 - cur_score)
                            return {
                                "is_completed": True,
                                "result_description": f"{d_name} won by {d_runs} runs",
                                "winning_team_id": d_id,
                                "winning_team_name": d_name,
                                "margin": f"{d_runs} runs",
                                "is_tie": False,
                                "is_draw": False
                            }
                        elif cur_overs >= min_dls_overs and match.get("dls_par_score") is not None:
                            dls_diff = cur_score - int(match["dls_par_score"])
                            b_id = match.get("batting_team_id") or t_b_id
                            d_id = t_a_id if b_id == t_b_id else t_b_id
                            if dls_diff > 0:
                                return {
                                    "is_completed": True,
                                    "result_description": f"{t_b_name if b_id == t_b_id else t_a_name} won by {dls_diff} runs (DLS method)",
                                    "winning_team_id": b_id,
                                    "winning_team_name": t_b_name if b_id == t_b_id else t_a_name,
                                    "margin": f"{dls_diff} runs (DLS)",
                                    "is_tie": False,
                                    "is_draw": False
                                }
                            elif dls_diff < 0:
                                return {
                                    "is_completed": True,
                                    "result_description": f"{t_a_name if d_id == t_a_id else t_b_name} won by {abs(dls_diff)} runs (DLS method)",
                                    "winning_team_id": d_id,
                                    "winning_team_name": t_a_name if d_id == t_a_id else t_b_name,
                                    "margin": f"{abs(dls_diff)} runs (DLS)",
                                    "is_tie": False,
                                    "is_draw": False
                                }
                        elif cur_overs < min_dls_overs:
                            return {
                                "is_completed": True,
                                "result_description": "Match Abandoned - No Result (Minimum overs not met within time window)",
                                "winning_team_id": None,
                                "winning_team_name": None,
                                "margin": "No Result",
                                "is_tie": False,
                                "is_draw": False,
                                "is_abandoned": True
                            }
                    elif elapsed_hrs >= (cutoff_hrs + 2.0):
                        # Expired without 2nd innings
                        return {
                            "is_completed": True,
                            "result_description": "Match Abandoned - No Result (Match time window elapsed)",
                            "winning_team_id": None,
                            "winning_team_name": None,
                            "margin": "No Result",
                            "is_tie": False,
                            "is_draw": False,
                            "is_abandoned": True
                        }
            except Exception:
                pass

    # 2. Test Match Completion Regulations
    elif fmt == "TEST":
        # Case A: Test match concludes in 3rd innings via an innings defeat (follow-on OR 1st-innings heavy lead)
        if cur_inn == 3 and cur_wkts >= 10:
            inn1_score = match.get("innings_1_score")
            inn2_score = match.get("innings_2_score")
            inn1_team_id = match.get("innings_1_team_id") or t_a_id
            inn2_team_id = match.get("innings_2_team_id") or t_b_id
            batting_team_id = match.get("batting_team_id") or match.get("innings_3_team_id") or inn2_team_id

            if inn1_score is not None and inn2_score is not None:
                deficit = 0
                bowling_team_id = None
                if batting_team_id == inn2_team_id and (int(inn2_score) + cur_score) < int(inn1_score):
                    # Follow-on enforced: team 2 all out behind team 1's 1st innings
                    deficit = int(inn1_score) - (int(inn2_score) + cur_score)
                    bowling_team_id = inn1_team_id
                elif batting_team_id == inn1_team_id and (int(inn1_score) + cur_score) < int(inn2_score):
                    # Team 1 bowled out in 3rd innings still behind team 2's 1st innings lead
                    deficit = int(inn2_score) - (int(inn1_score) + cur_score)
                    bowling_team_id = inn2_team_id

                if deficit > 0 and bowling_team_id:
                    bowling_team_name = t_a_name if bowling_team_id == t_a_id else t_b_name
                    m_str = f"an innings and {deficit} runs" if deficit != 1 else "an innings and 1 run"
                    return {
                        "is_completed": True,
                        "result_description": f"{bowling_team_name} won by {m_str}",
                        "winning_team_id": bowling_team_id,
                        "winning_team_name": bowling_team_name,
                        "margin": m_str,
                        "is_tie": False,
                        "is_draw": False
                    }

        # Case B: Test match concludes in 4th innings upon target reached or all out
        if cur_inn == 4 and target_runs is not None:
            target = int(target_runs)
            batting_team_id = match.get("batting_team_id") or t_b_id
            bowling_team_id = t_a_id if batting_team_id == t_b_id else t_b_id
            batting_team_name = t_b_name if batting_team_id == t_b_id else t_a_name
            bowling_team_name = t_a_name if bowling_team_id == t_a_id else t_b_name

            if cur_score >= target:
                wkts_left = max(0, 10 - cur_wkts)
                w_str = f"{wkts_left} wicket" if wkts_left == 1 else f"{wkts_left} wickets"
                return {
                    "is_completed": True,
                    "result_description": f"{batting_team_name} won by {w_str} (Target {target} reached in 4th innings)",
                    "winning_team_id": batting_team_id,
                    "winning_team_name": batting_team_name,
                    "margin": w_str,
                    "is_tie": False,
                    "is_draw": False
                }
            if cur_wkts >= 10:
                deficit = (target - 1) - cur_score
                if deficit == 0:
                    return {
                        "is_completed": True,
                        "result_description": f"Match Tied (Scores Level at {cur_score} in 4th Innings - All Out)",
                        "winning_team_id": None,
                        "winning_team_name": None,
                        "margin": "Tied",
                        "is_tie": True,
                        "is_draw": False
                    }
                elif deficit > 0:
                    r_str = f"{deficit} run" if deficit == 1 else f"{deficit} runs"
                    return {
                        "is_completed": True,
                        "result_description": f"{bowling_team_name} won by {r_str} (Defended target {target} in 4th innings)",
                        "winning_team_id": bowling_team_id,
                        "winning_team_name": bowling_team_name,
                        "margin": r_str,
                        "is_tie": False,
                        "is_draw": False
                    }

        # Case C: Test match maximum duration (5 days) elapsed
        if test_days_elapsed >= 5.2:
            nlp_win_m = re.search(r"([A-Za-z\s]+)\s+won\s+by\s+((?:an\s+innings\s+and\s+)?\d+\s+(?:runs?|wickets?))", combined_status, re.IGNORECASE)
            if nlp_win_m:
                c_team = nlp_win_m.group(1).strip()
                c_margin = nlp_win_m.group(2).strip()
                resolved_team_id = t_a_id if (t_a_name.lower() in c_team.lower() or t_a_id.lower() in c_team.lower()) else t_b_id
                resolved_team_name = t_a_name if resolved_team_id == t_a_id else t_b_name
                return {
                    "is_completed": True,
                    "result_description": f"{resolved_team_name} won by {c_margin}",
                    "winning_team_id": resolved_team_id,
                    "winning_team_name": resolved_team_name,
                    "margin": c_margin,
                    "is_tie": False,
                    "is_draw": False
                }
            drawn_desc = match.get("match_status_text") if "drawn" in match_status else "Match Drawn (5-day match duration elapsed)"
            return {
                "is_completed": True,
                "result_description": drawn_desc,
                "winning_team_id": None,
                "winning_team_name": None,
                "margin": "Draw",
                "is_tie": False,
                "is_draw": True
            }

    return {
        "is_completed": False,
        "result_description": None,
        "winning_team_id": None,
        "winning_team_name": None,
        "margin": None,
        "is_tie": False,
        "is_draw": False
    }

def evaluate_test_follow_on(
    inn_1_score: Optional[int],
    inn_2_score: Optional[int],
    inn_2_wickets: Optional[int] = 0,
    inn_2_team_id: Optional[str] = "SL",
    inn_1_team_id: Optional[str] = "PAK",
    inn_2_team_name: Optional[str] = "Sri Lanka",
    inn_1_team_name: Optional[str] = "Pakistan",
    test_days: int = 5,
    overs_lost: float = 0.0,
    days_lost: int = 0
) -> Dict[str, Any]:
    """
    ICC Law 14 (The Follow-On):
    In a standard 5-day Test match:
    - Margin: 200 runs (team batting first leads by >= 200 runs).
    - Threshold score to avoid follow-on: inn_1_score - 200 + 1.
    - If Innings 1 score is 434: Threshold = 434 - 200 + 1 = 235 runs.
    - Runs needed to avoid: max(0, threshold - inn_2_score).
    - If inn_2 is complete and inn_2_score < threshold, follow-on is enforceable.

    ICC Law 14.1.2 (Follow-On Decay in Rain-Curtailed Matches):
    If play is lost on Day 1 or multiple days lost to rain:
    - 5 days (0 days lost): 200 runs lead
    - 3 or 4 days (1-2 days lost or >=90 overs lost): 150 runs lead
    - 2 days (3 days lost or >=180 overs lost): 100 runs lead
    - 1 day (4 days lost or >=270 overs lost): 75 runs lead
    """
    eff_days_lost = max(days_lost, int(overs_lost // 90.0)) if (overs_lost > 0 or days_lost > 0) else 0
    effective_days = max(1, test_days - eff_days_lost)

    if effective_days >= 5:
        margin = 200
    elif effective_days in [3, 4]:
        margin = 150
    elif effective_days == 2:
        margin = 100
    else:
        margin = 75

    if inn_1_score is None or inn_1_score <= 0:
        return {
            "applicable": False,
            "margin": margin,
            "threshold": None,
            "deficit_to_avoid": None,
            "is_avoided": False,
            "is_enforceable": False,
            "effective_test_days": effective_days,
            "days_lost": eff_days_lost,
            "overs_lost": overs_lost,
            "status_label": "Innings 1 not completed"
        }

    threshold = max(0, inn_1_score - margin + 1)
    current_score = inn_2_score or 0
    deficit_to_avoid = max(0, threshold - current_score)
    is_avoided = (current_score >= threshold)
    is_inn_2_complete = (inn_2_wickets is not None and inn_2_wickets >= 10)
    enforceable = (is_inn_2_complete and not is_avoided)

    decay_note = f" (Decayed to {margin} runs under ICC Law 14.1.2 due to {int(overs_lost)} overs lost)" if eff_days_lost > 0 else ""

    if is_avoided:
        status_label = f"{inn_2_team_name or inn_2_team_id} have avoided the follow-on ({current_score} runs, threshold {threshold} reached){decay_note}"
    else:
        status_label = f"{inn_2_team_name or inn_2_team_id} need {deficit_to_avoid} more runs to avoid the follow-on (Target to avoid: {threshold}){decay_note}"

    return {
        "applicable": True,
        "inn_1_score": inn_1_score,
        "inn_2_score": current_score,
        "margin": margin,
        "threshold": threshold,
        "deficit_to_avoid": deficit_to_avoid,
        "is_avoided": is_avoided,
        "is_enforceable": enforceable,
        "status_label": status_label,
        "enforcing_team_id": inn_1_team_id,
        "enforcing_team_name": inn_1_team_name,
        "batting_team_id": inn_2_team_id,
        "batting_team_name": inn_2_team_name,
        "effective_test_days": effective_days,
        "days_lost": eff_days_lost,
        "overs_lost": overs_lost
    }

def declare_test_innings(
    match_dict: Dict[str, Any],
    innings_num: int,
    declaring_team_id: str,
    score: int,
    wickets: int,
    overs: float
) -> Dict[str, Any]:
    """
    ICC Law 15: Declaration & Forfeiture
    The captain of the batting side may declare an innings closed at any time during a match.
    Updates the innings score, marks it declared ('d'), and transitions to the subsequent innings.
    """
    updated = dict(match_dict)
    if innings_num == 1:
        updated["innings_1_score"] = score
        updated["innings_1_wickets"] = wickets
        updated["innings_1_overs"] = overs
        updated["innings_1_team_id"] = declaring_team_id
        updated["innings_1_declared"] = 1
        updated["is_declared"] = 1
        updated["current_innings"] = 2
    elif innings_num == 2:
        updated["innings_2_score"] = score
        updated["innings_2_wickets"] = wickets
        updated["innings_2_overs"] = overs
        updated["innings_2_team_id"] = declaring_team_id
        updated["innings_2_declared"] = 1
        updated["is_declared"] = 1
        updated["current_innings"] = 3
    elif innings_num == 3:
        updated["innings_3_score"] = score
        updated["innings_3_wickets"] = wickets
        updated["innings_3_overs"] = overs
        updated["innings_3_team_id"] = declaring_team_id
        updated["innings_3_declared"] = 1
        updated["is_declared"] = 1
        updated["current_innings"] = 4
    return updated
