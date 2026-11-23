"""
CricAPI (CricketData.org) Resilient Match-State, Interruption, and DLS / Reduced-Overs Parser.
Implements official ICC Reduced-Overs Playing Conditions (Clauses 12.4, 13.8, 13.9):
- Dynamic rain delay & match suspension detection from free-text CricAPI status.
- DLS revised target & remaining balls extraction via resilient regex pattern matching.
- ICC standard bowler quota reduction & fractional bowler distribution.
- Powerplay adjustment across curtailed T20 & ODI formats.
- Remaining death / key bowler quota tracking.
- Output typed Pydantic & Dataclass model (MatchStateContext) ready for ML win/score prediction pipelines.
"""

from __future__ import annotations
import math
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


def parse_cricket_overs_to_balls(overs_val: float | int | str) -> int:
    """
    Accurately converts cricket overs notation (e.g., 12.4 overs) into total legal deliveries.
    In cricket, 12.4 means 12 complete overs (72 balls) + 4 balls = 76 balls.
    Handles float, int, or string representations.
    """
    if overs_val is None:
        return 0
    try:
        f_ov = float(overs_val)
    except (ValueError, TypeError):
        return 0

    overs_int = int(f_ov)
    # Cricket fractional part is .1 through .5
    balls_part = int(round((f_ov - overs_int) * 10))
    balls_part = min(5, max(0, balls_part))
    return overs_int * 6 + balls_part


def balls_to_cricket_overs(total_balls: int) -> float:
    """
    Converts legal delivery count back into cricket overs notation (e.g. 76 balls -> 12.4 ov).
    """
    if total_balls <= 0:
        return 0.0
    overs = total_balls // 6
    balls = total_balls % 6
    return round(overs + balls * 0.1, 1)


class BowlerRestriction(BaseModel):
    name: str
    overs_bowled: float = 0.0
    balls_bowled: int = 0
    max_quota_overs: float = 4.0
    overs_remaining: float = 4.0
    balls_remaining: int = 24
    is_bowled_out: bool = False
    wickets: int = 0
    runs_conceded: int = 0
    economy: float = 0.0


class ICCReducedOversEngine:
    """
    Encapsulates ICC Playing Conditions for Reduced Overs Matches (T20 Clause 13.8, ODI Clause 13.9):
    - Maximum overs per bowler & fractional bowler allowance.
    - Powerplay duration adjustments for curtailed matches.
    """

    @staticmethod
    def calculate_bowler_quotas(revised_overs: float, match_format: str = "T20") -> Dict[str, Any]:
        """
        Calculates ICC bowler quotas:
        In an uninterrupted match of N overs, the maximum number of overs per bowler is:
        - Base: floor(N / 5)
        - Remainder: N % 5 bowlers may bowl floor(N / 5) + 1 overs.
        - Absolute ceiling for any bowler: ceil(N / 5)
        """
        fmt = match_format.upper()
        standard_overs = 50.0 if "ODI" in fmt else (90.0 if "TEST" in fmt else 20.0)
        eff_overs = float(revised_overs) if revised_overs > 0 else standard_overs

        if "TEST" in fmt:
            return {
                "standard_overs": standard_overs,
                "revised_overs": eff_overs,
                "is_reduced": False,
                "max_overs_per_bowler": 999,
                "base_overs_per_bowler": 999,
                "bowlers_with_extra_over": 0,
                "summary": "Unlimited overs per bowler in Test match"
            }

        base_quota = math.floor(eff_overs / 5.0)
        remainder = int(round(eff_overs % 5))
        max_quota = math.ceil(eff_overs / 5.0)

        if remainder > 0:
            bowlers_with_extra = remainder
            bowlers_with_base = 5 - remainder
            summary = (
                f"{bowlers_with_extra} bowler{'s' if bowlers_with_extra > 1 else ''} x {max_quota} ov, "
                f"{bowlers_with_base} bowler{'s' if bowlers_with_base > 1 else ''} x {base_quota} ov"
            )
        else:
            bowlers_with_extra = 0
            bowlers_with_base = 5
            summary = f"5 bowlers x {base_quota} ov"

        return {
            "standard_overs": standard_overs,
            "revised_overs": eff_overs,
            "is_reduced": eff_overs < standard_overs,
            "max_overs_per_bowler": max_quota,
            "base_overs_per_bowler": base_quota,
            "bowlers_with_extra_over": bowlers_with_extra,
            "bowlers_with_base_over": bowlers_with_base,
            "summary": summary
        }

    @staticmethod
    def calculate_powerplay_overs(revised_overs: float, match_format: str = "T20") -> Dict[str, Any]:
        """
        Calculates adjusted Powerplay overs in reduced innings according to ICC Playing Conditions:
        - T20: Standard is 6 overs (30%). For curtailed matches: round(revised_overs * 0.30).
          Minimum 1 over if match >= 5 overs.
        - ODI: Standard 10 / 30 / 10 split (P1: 20%, P2: 60%, P3: 20%).
        """
        fmt = match_format.upper()
        if "TEST" in fmt:
            return {
                "powerplay_overs": 0,
                "p1_overs": 0,
                "p2_overs": 0,
                "p3_overs": 0,
                "description": "No field restriction powerplay in Test match"
            }

        if "ODI" in fmt:
            p1 = max(2, int(round(revised_overs * 0.20)))
            p2 = max(6, int(round(revised_overs * 0.60)))
            p3 = max(0, int(round(revised_overs)) - p1 - p2)
            return {
                "powerplay_overs": p1,
                "p1_overs": p1,
                "p2_overs": p2,
                "p3_overs": p3,
                "description": f"ODI Powerplays: P1 ({p1} ov), P2 ({p2} ov), P3 ({p3} ov)"
            }

        # T20 calculation
        pp_overs = max(1, int(round(revised_overs * 0.30)))
        return {
            "powerplay_overs": pp_overs,
            "p1_overs": pp_overs,
            "p2_overs": 0,
            "p3_overs": 0,
            "description": f"T20 Mandatory Powerplay: {pp_overs} overs"
        }


class MatchStateContext(BaseModel):
    """
    Clean, strongly-typed model feeding directly into the CricOracle ML and WinWiz feature pipeline.
    """
    match_id: str
    format: str = "T20I"
    raw_status: str = ""

    # Interruption & Stoppage
    is_interrupted: bool = False
    is_abandoned: bool = False
    interruption_reason: Optional[str] = None

    # Overs & Target
    standard_max_overs: float = 20.0
    revised_max_overs: float = 20.0
    is_reduced_overs: bool = False
    total_balls_innings: int = 120
    overs_bowled: float = 0.0
    balls_bowled: int = 0
    balls_remaining: int = 120

    # Scores & Target
    current_innings: int = 1
    current_score: int = 0
    current_wickets: int = 0
    runs_needed: Optional[int] = None
    target_runs: Optional[int] = None
    revised_target: Optional[int] = None
    is_dls_applied: bool = False
    dls_method_used: bool = False

    # ICC Reduced Restrictions
    is_powerplay: bool = False
    powerplay_overs: int = 6
    powerplay_overs_remaining: float = 6.0
    max_overs_per_bowler: int = 4
    bowler_quota_summary: str = "5 bowlers x 4 ov"
    bowler_restrictions: List[BowlerRestriction] = Field(default_factory=list)
    death_bowler_overs_remaining: float = 0.0

    # ML Pipeline Feature Vector
    ml_features: Dict[str, float] = Field(default_factory=dict)


class CricApiMatchParser:
    """
    Robust match-state, interruption, DLS, and revised conditions parser for CricketData.org (CricAPI).
    Extracts structured metrics from free-text `status`, `score`, and `scorecard.bowling`.
    """

    INTERRUPTION_KEYWORDS = [
        "rain", "delayed", "wet outfield", "bad light", "interrupted",
        "abandoned", "no result", "play stopped", "drizzle", "covers on"
    ]

    ABANDONED_KEYWORDS = [
        "abandoned", "no result", "called off", "match washed out", "washed out"
    ]

    # Pattern A: Chasing in-play (e.g. "need 94 runs in 60 balls (DLS method)")
    PATTERN_CHASE = re.compile(
        r"(?:need|needs|require|requires)\s+(\d+)\s+runs?\s+(?:in|from|off)\s+(\d+)\s+balls?",
        re.IGNORECASE
    )

    # Pattern B: Reduced target / revised overs announcement (e.g. "Target reduced to 142 in 16 overs", "Match reduced to 15 overs per side")
    PATTERN_TARGET_REDUCED = re.compile(
        r"target\s+(?:reduced|revised|set)?\s*(?:to|of)?\s*(\d+)\s*(?:runs?)?\s*(?:in|from|off)?\s*(\d+(?:\.\d+)?)\s*(?:overs|ov)",
        re.IGNORECASE
    )

    PATTERN_MATCH_REDUCED = re.compile(
        r"match\s+(?:reduced|curtailed|shortened)\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*(?:overs|ov)",
        re.IGNORECASE
    )

    PATTERN_TARGET_SIMPLE = re.compile(
        r"target\s+(?:is\s+)?(\d+)(?:\s+in\s+(\d+(?:\.\d+)?)\s*(?:overs|ov))?",
        re.IGNORECASE
    )

    @classmethod
    def detect_interruption(cls, status_text: str) -> tuple[bool, bool, Optional[str]]:
        """
        Inspects free-text status for interruption keywords and classifies interruption reason.
        Returns: (is_interrupted, is_abandoned, interruption_reason)
        """
        if not status_text:
            return False, False, None

        st_low = status_text.lower()
        is_abandoned = any(k in st_low for k in cls.ABANDONED_KEYWORDS)
        is_interrupted = is_abandoned or any(k in st_low for k in cls.INTERRUPTION_KEYWORDS)

        reason: Optional[str] = None
        if is_abandoned:
            if "rain" in st_low or "wet" in st_low or "washed out" in st_low:
                reason = "Match abandoned due to persistent rain and wet outfield"
            else:
                reason = "Match abandoned (No Result)"
        elif is_interrupted:
            if "wet outfield" in st_low:
                reason = "Play delayed due to wet outfield"
            elif "bad light" in st_low:
                reason = "Play suspended due to bad light"
            elif "rain" in st_low or "drizzle" in st_low or "covers" in st_low:
                reason = "Rain delay (covers on, play suspended)"
            else:
                reason = status_text.strip()

        return is_interrupted, is_abandoned, reason

    @classmethod
    def parse_dls_and_reduced_overs(
        cls,
        status_text: str,
        match_format: str,
        current_score: int = 0,
        current_overs_bowled: float = 0.0
    ) -> Dict[str, Any]:
        """
        Extracts revised overs, DLS target, runs needed, and remaining balls using resilient regex cascades.
        """
        fmt = match_format.upper()
        standard_overs = 50.0 if "ODI" in fmt else (90.0 if "TEST" in fmt else 20.0)

        runs_needed: Optional[int] = None
        balls_remaining: Optional[int] = None
        revised_target: Optional[int] = None
        revised_max_overs: float = standard_overs
        dls_applied = False

        if status_text:
            st_low = status_text.lower()
            if "dls" in st_low or "duckworth" in st_low:
                dls_applied = True

            # 1. Check Pattern A: In-Play Chasing (e.g. "need 94 runs in 60 balls (DLS method)")
            chase_match = cls.PATTERN_CHASE.search(status_text)
            if chase_match:
                runs_needed = int(chase_match.group(1))
                balls_remaining = int(chase_match.group(2))
                revised_target = current_score + runs_needed

            # 2. Check Pattern B: Target Announcement (e.g. "Target reduced to 142 in 16 overs")
            target_match = cls.PATTERN_TARGET_REDUCED.search(status_text)
            if target_match:
                revised_target = int(target_match.group(1))
                revised_max_overs = float(target_match.group(2))
                if runs_needed is None and revised_target > current_score:
                    runs_needed = revised_target - current_score

            # 3. Check Match Reduced to N overs per side
            reduced_match = cls.PATTERN_MATCH_REDUCED.search(status_text)
            if reduced_match:
                revised_max_overs = float(reduced_match.group(1))

            # 4. Check simple Target pattern if revised_target still not found
            if revised_target is None:
                simple_match = cls.PATTERN_TARGET_SIMPLE.search(status_text)
                if simple_match:
                    revised_target = int(simple_match.group(1))
                    if simple_match.group(2):
                        revised_max_overs = float(simple_match.group(2))
                    if runs_needed is None and revised_target > current_score:
                        runs_needed = max(0, revised_target - current_score)

        # Standard bounds validation
        if revised_max_overs <= 0 or revised_max_overs > standard_overs:
            revised_max_overs = standard_overs

        total_balls_innings = int(revised_max_overs * 6)
        balls_bowled = parse_cricket_overs_to_balls(current_overs_bowled)

        if balls_remaining is None:
            balls_remaining = max(0, total_balls_innings - balls_bowled)

        is_reduced = revised_max_overs < standard_overs

        return {
            "standard_max_overs": standard_overs,
            "revised_max_overs": revised_max_overs,
            "is_reduced_overs": is_reduced,
            "total_balls_innings": total_balls_innings,
            "balls_bowled": balls_bowled,
            "balls_remaining": balls_remaining,
            "runs_needed": runs_needed,
            "revised_target": revised_target,
            "is_dls_applied": dls_applied or is_reduced,
            "dls_method_used": dls_applied
        }

    @classmethod
    def parse_bowler_restrictions(
        cls,
        bowlers_data: List[Dict[str, Any]],
        max_overs_per_bowler: int
    ) -> tuple[List[BowlerRestriction], float]:
        """
        Parses scorecard bowling array and evaluates remaining overs for each bowler under revised cap.
        Returns: (bowler_restrictions, death_bowler_overs_remaining)
        """
        restrictions: List[BowlerRestriction] = []
        death_overs_pool = 0.0

        for b in bowlers_data:
            # Handle name nested in {"bowler": {"name": ...}} or top-level "name"
            b_name = b.get("name") or (b.get("bowler", {}).get("name") if isinstance(b.get("bowler"), dict) else None) or "Bowler"
            o_raw = b.get("o") or b.get("overs") or 0.0
            try:
                o_val = float(o_raw)
            except (ValueError, TypeError):
                o_val = 0.0

            balls_b = parse_cricket_overs_to_balls(o_val)
            max_balls = int(max_overs_per_bowler * 6)
            rem_balls = max(0, max_balls - balls_b)
            rem_overs = balls_to_cricket_overs(rem_balls)
            is_done = (rem_balls == 0)

            w = int(b.get("w") or b.get("wickets") or 0)
            r = int(b.get("r") or b.get("runs") or 0)
            econ = round((r / (balls_b / 6.0)), 2) if balls_b > 0 else 0.0

            res = BowlerRestriction(
                name=b_name,
                overs_bowled=o_val,
                balls_bowled=balls_b,
                max_quota_overs=float(max_overs_per_bowler),
                overs_remaining=rem_overs,
                balls_remaining=rem_balls,
                is_bowled_out=is_done,
                wickets=w,
                runs_conceded=r,
                economy=econ
            )
            restrictions.append(res)
            # Accumulate death overs pool for active bowlers with economy under 10.0 or wickets >= 1
            if not is_done:
                death_overs_pool += rem_overs

        return restrictions, round(death_overs_pool, 1)

    @classmethod
    def parse_match_data(
        cls,
        data: Dict[str, Any],
        match_id: Optional[str] = None,
        override_format: Optional[str] = None
    ) -> MatchStateContext:
        """
        Main entrypoint: parses raw CricAPI match dict into strongly-typed MatchStateContext.
        """
        mid = match_id or data.get("id") or "match_cricapi"
        raw_status = str(data.get("status") or "")

        # Format resolution
        fmt_in = override_format or data.get("matchType") or data.get("format") or "t20"
        fmt_str = str(fmt_in).upper()
        if "TEST" in fmt_str:
            fmt = "TEST"
        elif "ODI" in fmt_str:
            fmt = "ODI"
        else:
            fmt = "T20I"

        # 1. Rain & Interruption Detection
        is_interrupted, is_abandoned, reason = cls.detect_interruption(raw_status)

        # 2. Extract score and current overs bowled from CricAPI score structure
        score_list = data.get("score") or []
        current_score = 0
        current_wickets = 0
        current_overs_bowled = 0.0
        current_innings = 1

        if isinstance(score_list, list) and len(score_list) > 0:
            current_innings = len(score_list)
            last_inn = score_list[-1]
            current_score = int(last_inn.get("r") or last_inn.get("runs") or 0)
            current_wickets = int(last_inn.get("w") or last_inn.get("wickets") or 0)
            current_overs_bowled = float(last_inn.get("o") or last_inn.get("overs") or 0.0)
        elif "r" in data or "runs" in data:
            current_score = int(data.get("r") or data.get("runs") or 0)
            current_wickets = int(data.get("w") or data.get("wickets") or 0)
            current_overs_bowled = float(data.get("o") or data.get("overs") or 0.0)

        # 3. DLS & Reduced Overs Parsing
        dls_info = cls.parse_dls_and_reduced_overs(
            status_text=raw_status,
            match_format=fmt,
            current_score=current_score,
            current_overs_bowled=current_overs_bowled
        )

        revised_overs = dls_info["revised_max_overs"]
        balls_bowled = dls_info["balls_bowled"]
        balls_remaining = dls_info["balls_remaining"]
        runs_needed = dls_info["runs_needed"]
        revised_target = dls_info["revised_target"]

        # 4. ICC Reduced-Overs Rule Engine
        bowler_quotas = ICCReducedOversEngine.calculate_bowler_quotas(revised_overs, fmt)
        pp_info = ICCReducedOversEngine.calculate_powerplay_overs(revised_overs, fmt)

        pp_overs = pp_info["powerplay_overs"]
        is_pp = (balls_bowled < pp_overs * 6) and (fmt != "TEST")
        pp_balls_rem = max(0, pp_overs * 6 - balls_bowled)
        pp_overs_rem = balls_to_cricket_overs(pp_balls_rem)

        # 5. Bowler Restrictions
        bowling_data = []
        scorecard = data.get("scorecard") or {}
        if isinstance(scorecard, dict):
            bowling_data = scorecard.get("bowling") or []
        elif isinstance(data.get("bowling"), list):
            bowling_data = data.get("bowling")

        bowler_restrictions, death_overs_pool = cls.parse_bowler_restrictions(
            bowlers_data=bowling_data,
            max_overs_per_bowler=bowler_quotas["max_overs_per_bowler"]
        )

        # 6. ML Feature Pipeline Vector
        eff_rrr = 0.0
        if runs_needed is not None and balls_remaining > 0:
            eff_rrr = round((runs_needed / (balls_remaining / 6.0)), 2)
        elif revised_target is not None and balls_remaining > 0:
            needed = max(0, revised_target - current_score)
            eff_rrr = round((needed / (balls_remaining / 6.0)), 2)

        crr = round((current_score / (balls_bowled / 6.0)), 2) if balls_bowled > 0 else 0.0
        balls_left_ratio = round((balls_remaining / max(1, dls_info["total_balls_innings"])), 3)
        overs_reduction_factor = round((revised_overs / dls_info["standard_max_overs"]), 3)

        ml_features = {
            "effective_run_rate_required": eff_rrr,
            "current_run_rate": crr,
            "balls_left_ratio": balls_left_ratio,
            "wickets_in_hand": float(10 - current_wickets),
            "is_powerplay": 1.0 if is_pp else 0.0,
            "powerplay_overs_remaining": pp_overs_rem,
            "overs_reduction_factor": overs_reduction_factor,
            "death_bowler_overs_remaining": death_overs_pool,
            "is_interrupted": 1.0 if is_interrupted else 0.0,
            "is_abandoned": 1.0 if is_abandoned else 0.0,
            "is_dls_applied": 1.0 if dls_info["is_dls_applied"] else 0.0
        }

        return MatchStateContext(
            match_id=mid,
            format=fmt,
            raw_status=raw_status,
            is_interrupted=is_interrupted,
            is_abandoned=is_abandoned,
            interruption_reason=reason,
            standard_max_overs=dls_info["standard_max_overs"],
            revised_max_overs=revised_overs,
            is_reduced_overs=dls_info["is_reduced_overs"],
            total_balls_innings=dls_info["total_balls_innings"],
            overs_bowled=current_overs_bowled,
            balls_bowled=balls_bowled,
            balls_remaining=balls_remaining,
            current_innings=current_innings,
            current_score=current_score,
            current_wickets=current_wickets,
            runs_needed=runs_needed,
            target_runs=revised_target,
            revised_target=revised_target,
            is_dls_applied=dls_info["is_dls_applied"],
            dls_method_used=dls_info["dls_method_used"],
            is_powerplay=is_pp,
            powerplay_overs=pp_overs,
            powerplay_overs_remaining=pp_overs_rem,
            max_overs_per_bowler=bowler_quotas["max_overs_per_bowler"],
            bowler_quota_summary=bowler_quotas["summary"],
            bowler_restrictions=bowler_restrictions,
            death_bowler_overs_remaining=death_overs_pool,
            ml_features=ml_features
        )


# Convenience function for ML feature pipeline and ingestion manager
def parse_cricapi_match_state(
    data: Dict[str, Any],
    match_id: Optional[str] = None,
    match_format: Optional[str] = None
) -> MatchStateContext:
    """
    Parses CricAPI JSON match payload into typed MatchStateContext.
    """
    return CricApiMatchParser.parse_match_data(data, match_id=match_id, override_format=match_format)
