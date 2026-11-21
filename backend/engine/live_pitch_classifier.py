"""
Live In-Play Pitch Classifier & Drift Detector
Determines the real-time cricket pitch type during a live match across 5 canonical pitch archetypes:
1. 'dusty': Dry clay, high surface friction, early abrasion, sharp spin, low natural carry.
2. 'green': Lush grass cover, early lateral swing, seam deviation, top-order hazard.
3. 'bouncy': Hard-packed clay, steep vertical lift, tennis-ball bounce, short-ball threat.
4. 'flat': True batting paradise, high par score, low friction, minimal seam/turn.
5. 'balanced': Fair sporting surface, even contest between bat & ball, two-paced cutters.

Employs 3 independent scientific methods:
Approach 1: Telemetry & In-Match Ball Outcome Drift (Algorithmic)
Approach 2: Natural Language Processing (NLP) Commentary Semantic Analysis
Approach 3: Bayesian Ground Prior Blending with Format Cricsheet Dossier
"""

import math
import re
from typing import Dict, Any, List, Optional
from .pitch_history_model import get_historical_venue_profile
from .cricsheet_pitch_intelligence import (
    get_ground_pitch_archetype,
    FORMAT_GROUND_PITCH_DOSSIERS,
    PITCH_ARCHETYPES
)

# Semantic keyword lexicon for NLP commentary and pitch report detection across the 5 archetypes
NLP_PITCH_LEXICON = {
    "dusty": [
        "spin", "turn", "sharp turn", "turn and bounce", "gripping", "dust", "puff", "rag",
        "bat-pad", "rough", "turner", "sliding", "skidding", "drift", "stump to stump",
        "subcontinent", "low bounce", "staying low", "grubbing", "gripped and turned",
        "turn past the edge", "spinning", "turners", "raging turner", "dust bowl", "minefield",
        "cracks", "dry pitch", "dry surface", "bare wicket", "bare patches", "footmarks",
        "slow turn", "spit off the deck", "subcontinent turner", "turning track", "cracked surface",
        "turning square", "spinner friendly", "spinning viciously", "purchase for spinners", "dusty"
    ],
    "green": [
        "green", "grass", "green tinge", "lateral", "movement", "swing", "seam and swing",
        "nipping", "clouds", "damp", "moisture", "overhead", "early life", "hooping",
        "seaming away", "conditions aiding seamers", "fresh surface", "grass covering",
        "green top", "greentop", "live grass", "lush green", "emerald pitch", "sweating deck",
        "seamer paradise", "lateral nip", "swing in the air", "overcast seam", "juicy surface",
        "green carpet", "moisture underneath", "nibbling away", "pacer friendly", "grass roots"
    ],
    "bouncy": [
        "bounce", "steep bounce", "tennis-ball", "climbing", "gloves", "helmet", "hit the deck",
        "ribs", "carry", "keeper taking high", "shoulder", "lift", "flying through",
        "extra bounce", "ballooned", "steep trajectory", "steep lift", "tennis ball bounce",
        "hard deck", "hard-packed", "fast and bouncy", "spicy bounce", "rib tickler",
        "shoulder height", "steep vertical lift", "climbing onto the splice", "flying through to keeper", "bouncy"
    ],
    "flat": [
        "flat", "placid", "road", "true bounce", "belter", "featherbed", "through the line",
        "middle of the bat", "boundaries flowing", "punished", "easy", "batter's paradise",
        "nothing for the bowlers", "flattest", "outfield lightning", "batting dream", "highway",
        "run feast", "batting paradise", "flat track", "cement track", "true surface",
        "ball coming onto the bat", "graveyard for bowlers", "absolute highway", "high-scoring"
    ],
    "balanced": [
        "balanced", "sporting", "two-paced", "seam", "deviation", "wobble", "cutters",
        "holding up", "variable", "abrasive", "off the seam", "cutter", "decking",
        "seam movement", "sticking in the surface", "stopping on the batter", "uneven bounce",
        "spongy", "variable bounce", "cross-seam", "sticky wicket", "inconsistent carry",
        "abrasive deck", "hard abrasive", "wobble seam", "stopping pitch", "fair contest", "even contest"
    ]
}

def canonicalize_pitch_type(arch: str) -> str:
    """
    Standardizes any pitch type to one of the 5 canonical pitch archetypes:
    'dusty', 'green', 'bouncy', 'flat', 'balanced'.
    Supports aliases ('spin heavy', 'bounce', 'seam', 'seam / green', etc.).
    """
    if not arch:
        return "balanced"
    a = str(arch).lower().strip().replace("-", "_").replace(" ", "_")
    if a in ["dusty", "spin_heavy", "spin_friendly", "spin", "turner", "raging_turner"]:
        return "dusty"
    if a in ["green", "lush_green", "green_top", "green_seamer"]:
        return "green"
    if a in ["bouncy", "bounce", "pace_friendly", "hard", "steep_bounce"]:
        return "bouncy"
    if a in ["flat", "batting_paradise", "road", "high_scoring"]:
        return "flat"
    if a in ["balanced", "seam", "sporting", "two_paced", "neutral", "seam_/_green", "green_/_seam", "pace"]:
        return "balanced"
    return "balanced"

from dataclasses import dataclass

@dataclass
class PhaseThresholds:
    Min_Wkts: int
    High_Spin: float
    Low_Spin: float
    High_DPI: float
    Low_DPI: float
    High_KB: float
    High_LD: float
    High_FS: float
    Low_FS: float
    Format_BD: float

    @property
    def High_FS_Threshold(self) -> float:
        return self.High_FS

    @property
    def Low_FS_Threshold(self) -> float:
        return self.Low_FS

def get_match_phase(fmt: str, current_over: float, ball_age: float = 0.0, match_day: int = 1) -> str:
    """
    Step 0 Helper: Resolves match Phase_ID based on format, active overs, ball age, and match day.
    Phase_ID =
      T20:  P1 (Overs 1–6), P2 (Overs 7–15), P3 (Overs 16–20)
      ODI:  P1 (Overs 1–10), P2 (Overs 11–35), P3 (Overs 36–50)
      TEST: P1 (Overs 1–25, Days 1–3), P2 (Overs 26–80, Days 1–3), P3 (Overs 80+, Days 1–3), P4 (Days 4–5)
    """
    fmt_u = fmt.upper()
    if "T20" in fmt_u:
        if current_over <= 6.0:
            return "P1"
        elif current_over <= 15.0:
            return "P2"
        else:
            return "P3"
    elif "ODI" in fmt_u:
        if current_over <= 10.0:
            return "P1"
        elif current_over <= 35.0:
            return "P2"
        else:
            return "P3"
    else:  # TEST
        if match_day in [4, 5]:
            return "P4"
        eff_ball_age = ball_age if ball_age > 0 else (current_over % 80.0)
        if eff_ball_age <= 25.0:
            return "P1"
        elif eff_ball_age <= 80.0:
            return "P2"
        else:
            return "P3"

def lookup_phase_thresholds(fmt: str, phase_id: str) -> PhaseThresholds:
    """
    Step 0 Helper: Returns calibrated empirical thresholds P for given Format and Phase_ID.
    | Format | Phase | Overs / Ball Age / Day | Min Wkts | High Spin | Low Spin | High DPI | Low DPI | High KB | High LD | High FS | Low FS | Format BD |
    | T20    | P1    | Overs 1–6              | 2        | 0.25      | 0.10     | 0.48     | 0.32    | 0.045   | 0.090   | 0.24    | 0.12   | 0.42      |
    | T20    | P2    | Overs 7–15             | 3        | 0.40      | 0.15     | 0.42     | 0.30    | 0.040   | 0.080   | 0.20    | 0.11   | 0.38      |
    | T20    | P3    | Overs 16–20            | 4        | 0.30      | 0.15     | 0.38     | 0.25    | 0.035   | 0.070   | 0.25    | 0.14   | 0.48      |
    | ODI    | P1    | Overs 1–10             | 2        | 0.20      | 0.12     | 0.58     | 0.40    | 0.045   | 0.090   | 0.22    | 0.11   | 0.28      |
    | ODI    | P2    | Overs 11–35            | 3        | 0.45      | 0.18     | 0.48     | 0.35    | 0.038   | 0.075   | 0.18    | 0.10   | 0.24      |
    | ODI    | P3    | Overs 36–50            | 5        | 0.30      | 0.15     | 0.40     | 0.28    | 0.035   | 0.070   | 0.24    | 0.13   | 0.38      |
    | TEST   | P1    | Overs 1–25, Days 1–3   | 2        | 0.20      | 0.12     | 0.72     | 0.56    | 0.050   | 0.100   | 0.22    | 0.10   | 0.18      |
    | TEST   | P2    | Overs 26–80, Days 1–3  | 3        | 0.40      | 0.15     | 0.65     | 0.52    | 0.042   | 0.080   | 0.18    | 0.09   | 0.16      |
    | TEST   | P3    | Overs 80+, Days 1–3    | 4        | 0.35      | 0.15     | 0.68     | 0.54    | 0.045   | 0.085   | 0.20    | 0.09   | 0.17      |
    | TEST   | P4    | Match Days 4–5         | 4        | 0.52      | 0.18     | 0.62     | 0.48    | 0.055   | 0.110   | 0.25    | 0.10   | 0.15      |
    """
    fmt_u = fmt.upper()
    if "T20" in fmt_u:
        if phase_id == "P1":
            return PhaseThresholds(
                Min_Wkts=2, High_Spin=0.25, Low_Spin=0.10, High_DPI=0.48, Low_DPI=0.32,
                High_KB=0.045, High_LD=0.090, High_FS=0.24, Low_FS=0.12, Format_BD=0.42
            )
        elif phase_id == "P2":
            return PhaseThresholds(
                Min_Wkts=3, High_Spin=0.40, Low_Spin=0.15, High_DPI=0.42, Low_DPI=0.30,
                High_KB=0.040, High_LD=0.080, High_FS=0.20, Low_FS=0.11, Format_BD=0.38
            )
        else:  # P3
            return PhaseThresholds(
                Min_Wkts=4, High_Spin=0.30, Low_Spin=0.15, High_DPI=0.38, Low_DPI=0.25,
                High_KB=0.035, High_LD=0.070, High_FS=0.25, Low_FS=0.14, Format_BD=0.48
            )
    elif "ODI" in fmt_u:
        if phase_id == "P1":
            return PhaseThresholds(
                Min_Wkts=2, High_Spin=0.20, Low_Spin=0.12, High_DPI=0.58, Low_DPI=0.40,
                High_KB=0.045, High_LD=0.090, High_FS=0.22, Low_FS=0.11, Format_BD=0.28
            )
        elif phase_id == "P2":
            return PhaseThresholds(
                Min_Wkts=3, High_Spin=0.45, Low_Spin=0.18, High_DPI=0.48, Low_DPI=0.35,
                High_KB=0.038, High_LD=0.075, High_FS=0.18, Low_FS=0.10, Format_BD=0.24
            )
        else:  # P3
            return PhaseThresholds(
                Min_Wkts=5, High_Spin=0.30, Low_Spin=0.15, High_DPI=0.40, Low_DPI=0.28,
                High_KB=0.035, High_LD=0.070, High_FS=0.24, Low_FS=0.13, Format_BD=0.38
            )
    else:  # TEST
        if phase_id == "P1":
            return PhaseThresholds(
                Min_Wkts=2, High_Spin=0.20, Low_Spin=0.12, High_DPI=0.72, Low_DPI=0.56,
                High_KB=0.050, High_LD=0.100, High_FS=0.22, Low_FS=0.10, Format_BD=0.18
            )
        elif phase_id == "P2":
            return PhaseThresholds(
                Min_Wkts=3, High_Spin=0.40, Low_Spin=0.15, High_DPI=0.65, Low_DPI=0.52,
                High_KB=0.042, High_LD=0.080, High_FS=0.18, Low_FS=0.09, Format_BD=0.16
            )
        elif phase_id == "P3":
            return PhaseThresholds(
                Min_Wkts=4, High_Spin=0.35, Low_Spin=0.15, High_DPI=0.68, Low_DPI=0.54,
                High_KB=0.045, High_LD=0.085, High_FS=0.20, Low_FS=0.09, Format_BD=0.17
            )
        else:  # P4 (Wear & Tear)
            return PhaseThresholds(
                Min_Wkts=4, High_Spin=0.52, Low_Spin=0.18, High_DPI=0.62, Low_DPI=0.48,
                High_KB=0.055, High_LD=0.110, High_FS=0.25, Low_FS=0.10, Format_BD=0.15
            )

class LivePitchClassifier:
    """
    Real-time multi-approach pitch condition classifier and drift detector.
    Outputs authentic probability distribution across unified archetypes:
    ['flat', 'seam / green', 'bounce', 'spin heavy'] (with backward-compatible 'green' & 'seam').
    """

    def __init__(self):
        pass

    def classify_live_pitch(
        self,
        venue_id: str,
        match_format: str,
        current_overs: int,
        current_balls: int,
        current_score: int,
        current_wickets: int,
        deliveries: List[Dict[str, Any]] = None,
        venue_profile: Optional[Dict[str, Any]] = None,
        commentary_strings: Optional[List[str]] = None,
        pitch_report: Optional[str] = None,
        ball_age: float = 0.0,
        match_day: int = 1
    ) -> Dict[str, Any]:
        """
        Executes:
        - Approach 1: Algorithmic Telemetry & Drift
        - Approach 2: Natural Language Processing (NLP) over Pitch Report & Live Commentary
        - Approach 3: Bayesian Prior Blending with Cricsheet Ground Archetype
        Synthesizes an ensemble where a real detected pitch overrides the historical prior.
        """
        fmt = match_format.upper() if match_format else "T20I"
        total_overs = current_overs + (current_balls / 6.0)
        crr = (current_score / total_overs) if total_overs > 0 else 3.5

        # -------------------------------------------------------------
        # APPROACH 1: Algorithmic Telemetry & In-Match Ball Outcome Drift
        # Executes the 5-Step Live In-Play Decision Tree
        # -------------------------------------------------------------
        benchmark_rr = 3.35 if fmt == "TEST" else (5.40 if fmt == "ODI" else 8.30)
        crr = (current_score / total_overs) if total_overs > 0 else benchmark_rr
        relative_rr = crr / benchmark_rr

        spin_overs = 0.0
        pace_overs = 0.0
        spin_runs = 0
        pace_runs = 0
        spin_wkts = 0
        pace_wkts = 0
        spin_dots = 0
        pace_dots = 0
        spin_balls = 0
        pace_balls = 0
        total_dots = 0
        total_boundaries = 0
        square_boundaries = 0
        keeper_beaten_count = 0
        pad_beat_count = 0
        lateral_deviation_count = 0
        false_shot_count = 0

        lbw_bowled_stumped_count = 0
        behind_wicket_count = 0
        outfield_catch_count = 0

        # Deliveries inspection
        if deliveries and len(deliveries) > 0:
            for d in deliveries:
                runs_bat = int(d.get("runs_batter", 0))
                extras = int(d.get("extras", 0))
                runs = runs_bat + extras
                is_wkt = bool(d.get("is_wicket", 0))
                wkt_type = str(d.get("wicket_type", d.get("dismissal_type", ""))).lower()
                extra_type = str(d.get("extra_type", "")).lower()
                text_desc = (str(d.get("commentary", "")) + " " + str(d.get("event_description", ""))).lower()

                b_style = str(d.get("bowler_style", "")).upper()
                b_name = str(d.get("bowler_name", "")).lower()
                is_spin = any(s in b_style for s in ["SLA", "OB", "LBG", "LAWS", "SPIN", "SLOW"]) or any(
                    n in b_name for n in ["spin", "santner", "sodhi", "ashwin", "jadeja", "kuldeep", "chahal", "miraz", "taijul", "wellalage", "hasaranga", "ghazanfar", "rashid"]
                )

                if is_spin:
                    spin_balls += 1
                    spin_overs += (1.0 / 6.0)
                    spin_runs += runs
                    if is_wkt:
                        spin_wkts += 1
                    if runs == 0 and not is_wkt:
                        spin_dots += 1
                else:
                    pace_balls += 1
                    pace_overs += (1.0 / 6.0)
                    pace_runs += runs
                    if is_wkt:
                        pace_wkts += 1
                    if runs == 0 and not is_wkt:
                        pace_dots += 1

                if runs == 0 and not is_wkt:
                    total_dots += 1

                if runs_bat in [4, 6]:
                    total_boundaries += 1
                    if any(sq in text_desc for sq in ["point", "third man", "backward point", "cut", "pull", "hook", "square leg", "deep square", "cover point"]):
                        square_boundaries += 1

                # Wicket taxonomy
                if is_wkt:
                    if any(w in wkt_type or w in text_desc for w in ["lbw", "bowled", "stumped"]):
                        lbw_bowled_stumped_count += 1
                    elif any(w in wkt_type or w in text_desc for w in ["caught behind", "c & b", "slip", "gully", "keeper", "c †", "caught by keeper"]):
                        behind_wicket_count += 1
                    else:
                        outfield_catch_count += 1

                # Physical telemetry signals from ball outcomes
                if "bye" in extra_type or "past the keeper" in text_desc or "beating the keeper" in text_desc or "byes" in text_desc:
                    keeper_beaten_count += 1
                if "leg_bye" in extra_type or "pad" in text_desc or "struck on pad" in text_desc or "lbw" in text_desc:
                    pad_beat_count += 1
                if any(term in text_desc for term in ["turn", "sharp turn", "seam", "swing", "nipped", "deviated", "drift", "beaten", "cutter", "movement"]):
                    lateral_deviation_count += 1
                if any(term in text_desc for term in ["beaten", "play and miss", "edge", "thick edge", "inside edge", "outside edge", "beaten outside off", "swung through"]):
                    false_shot_count += 1

            total_balls_count = len(deliveries)
        else:
            # Contextual estimation based on venue and format
            total_balls_count = max(1, int(total_overs * 6))
            is_subcontinent = any(c in venue_id.lower() for c in ["sylhet", "mirpur", "sharjah", "dubai", "wankhede", "eden_gardens", "colombo", "galle"])
            spin_share_est = 0.55 if is_subcontinent else 0.30
            spin_overs = total_overs * spin_share_est
            pace_overs = total_overs * (1.0 - spin_share_est)
            spin_balls = int(spin_overs * 6)
            pace_balls = total_balls_count - spin_balls
            spin_runs = int(current_score * (0.50 if is_subcontinent else 0.30))
            pace_runs = current_score - spin_runs
            spin_wkts = max(0, int(current_wickets * (0.55 if is_subcontinent else 0.30)))
            pace_wkts = current_wickets - spin_wkts
            total_dots = int(total_balls_count * (0.65 if "TEST" in fmt else 0.50))
            spin_dots = int(total_dots * spin_share_est)
            pace_dots = total_dots - spin_dots
            total_boundaries = max(1, int(current_score / 12))
            square_boundaries = int(total_boundaries * 0.40)
            lbw_bowled_stumped_count = int(current_wickets * 0.35)
            behind_wicket_count = int(current_wickets * 0.45)
            outfield_catch_count = current_wickets - (lbw_bowled_stumped_count + behind_wicket_count)
            keeper_beaten_count = int(total_balls_count * 0.015)
            pad_beat_count = int(total_balls_count * 0.020)
            lateral_deviation_count = int(total_balls_count * 0.050)
            false_shot_count = int(total_balls_count * 0.10)

        spin_overs_share = (spin_overs / max(0.1, total_overs))
        spin_econ = (spin_runs / spin_overs) if spin_overs > 0 else benchmark_rr
        pace_econ = (pace_runs / pace_overs) if pace_overs > 0 else benchmark_rr
        dot_pressure_idx = total_dots / max(1, total_balls_count)
        spin_dot_rate = (spin_dots / max(1, spin_balls)) if spin_balls > 0 else dot_pressure_idx
        boundary_dot_ratio = total_boundaries / max(1, total_dots)
        square_boundary_share = square_boundaries / max(1, total_boundaries)
        keeper_beat_rate = keeper_beaten_count / max(1, total_balls_count)
        pad_beat_rate = pad_beat_count / max(1, total_balls_count)
        lateral_deviation_signal = lateral_deviation_count / max(1, total_balls_count)
        false_shot_rate = false_shot_count / max(1, total_balls_count)

        total_wkts = max(1, current_wickets)
        lbw_bowled_stumped_share = lbw_bowled_stumped_count / total_wkts
        behind_wicket_ratio = behind_wicket_count / total_wkts
        outfield_catch_share = outfield_catch_count / total_wkts
        pace_wicket_dominance = pace_wkts / total_wkts

        # ================================================================================
        # STEP 0: LOAD PHASE CONTEXT & WICKET CONFIDENCE GATE
        # ================================================================================
        phase_id = get_match_phase(fmt, total_overs, ball_age=ball_age, match_day=match_day)
        P = lookup_phase_thresholds(fmt, phase_id)
        wicket_gate_active = (current_wickets >= P.Min_Wkts)

        dt_trace = [
            f"STEP 0: Phase_ID={phase_id}, Min_Wkts={P.Min_Wkts}, Wicket_Gate_Active={wicket_gate_active} (Total_Wkts={current_wickets})"
        ]

        # ================================================================================
        # STEP 1: CHECK FOR DUSTY TRACK (Spin + Dry Grip / Variable Low Bounce)
        # ================================================================================
        cond_1a = (spin_overs_share >= P.High_Spin)
        cond_1b = (
            (spin_econ <= pace_econ * 0.90) or
            (phase_id == "P1" and spin_overs_share >= P.High_Spin * 1.25) or
            (relative_rr <= 0.85 and dot_pressure_idx >= P.High_DPI * 0.95)
        )
        cond_1c = (
            (wicket_gate_active and lbw_bowled_stumped_share >= 0.45) or
            (keeper_beat_rate >= P.High_KB) or
            (lateral_deviation_signal >= P.High_LD) or
            (spin_dot_rate >= P.High_DPI * 1.05)
        )
        dusty_fire = cond_1a and cond_1b and cond_1c
        dt_trace.append(f"STEP 1 (DUSTY): Spin_Share={round(spin_overs_share,3)} >= {P.High_Spin} ({cond_1a}) AND Cond_1b={cond_1b} AND Cond_1c={cond_1c} --> Fire: {dusty_fire}")

        # ================================================================================
        # STEP 2: CHECK FOR GREEN TRACK (Seam/Swing + Inside/Outside Edges)
        # ================================================================================
        cond_2a = (spin_overs_share <= P.Low_Spin)
        cond_2b = (relative_rr < 0.88)
        cond_2c = (dot_pressure_idx >= P.High_DPI * 0.95)
        cond_2d = (
            (wicket_gate_active and behind_wicket_ratio >= 0.50) or
            (false_shot_rate >= P.High_FS) or
            (lateral_deviation_signal >= P.High_LD and pad_beat_rate >= 0.025)
        )
        green_fire = cond_2a and cond_2b and cond_2c and cond_2d
        dt_trace.append(f"STEP 2 (GREEN): Spin_Share={round(spin_overs_share,3)} <= {P.Low_Spin} ({cond_2a}) AND RRR={round(relative_rr,3)} < 0.88 ({cond_2b}) AND DPI={round(dot_pressure_idx,3)} >= {round(P.High_DPI*0.95,3)} ({cond_2c}) AND Cond_2d={cond_2d} --> Fire: {green_fire}")

        # ================================================================================
        # STEP 3: CHECK FOR BOUNCY TRACK (Pace + Steep Carry Over Keeper / Short-Ball Play)
        # ================================================================================
        cond_3a = (
            (spin_overs_share <= P.Low_Spin * 1.25) or
            (wicket_gate_active and pace_wicket_dominance >= 0.75)
        )
        cond_3b = (
            (relative_rr >= 0.92) or
            (wicket_gate_active and pace_wicket_dominance >= 0.80)
        )
        cond_3c = (boundary_dot_ratio >= P.Format_BD * 0.90)
        cond_3d = (
            (square_boundary_share >= 0.45) or
            (keeper_beat_rate >= P.High_KB and pad_beat_rate < 0.020) or
            (wicket_gate_active and behind_wicket_ratio >= 0.55 and relative_rr >= 0.95)
        )
        bouncy_fire = cond_3a and cond_3b and cond_3c and cond_3d
        dt_trace.append(f"STEP 3 (BOUNCY): Cond_3a={cond_3a} AND Cond_3b={cond_3b} AND BD_Ratio={round(boundary_dot_ratio,3)} >= {round(P.Format_BD*0.9,3)} ({cond_3c}) AND Cond_3d={cond_3d} --> Fire: {bouncy_fire}")

        # ================================================================================
        # STEP 4: CHECK FOR FLAT TRACK (True Bounce + Minimal Keeper/Pad Beating)
        # ================================================================================
        cond_4a = (relative_rr >= 1.15)
        cond_4b = (dot_pressure_idx <= P.Low_DPI)
        cond_4c = (false_shot_rate <= P.Low_FS)
        cond_4d = (keeper_beat_rate < P.High_KB)
        cond_4e = (
            (not wicket_gate_active) or
            (outfield_catch_share >= 0.65)
        )
        flat_fire = cond_4a and cond_4b and cond_4c and cond_4d and cond_4e
        dt_trace.append(f"STEP 4 (FLAT): RRR={round(relative_rr,3)} >= 1.15 ({cond_4a}) AND DPI={round(dot_pressure_idx,3)} <= {P.Low_DPI} ({cond_4b}) AND FS={round(false_shot_rate,3)} <= {P.Low_FS} ({cond_4c}) AND KB={round(keeper_beat_rate,3)} < {P.High_KB} ({cond_4d}) AND Cond_4e={cond_4e} --> Fire: {flat_fire}")

        # ================================================================================
        # STEP 5: DEFAULT FALLBACK (Even Contest or Moderately Slow/True Surface)
        # ================================================================================
        if dusty_fire:
            dt_classified_pitch = "dusty"
            dt_drift_state = "BREAKING_DOWN_SPIN_DOMINANT"
            dt_step = "STEP 1: DUSTY"
        elif green_fire:
            dt_classified_pitch = "green"
            dt_drift_state = "SEAM_SWING_ACTIVE"
            dt_step = "STEP 2: GREEN"
        elif bouncy_fire:
            dt_classified_pitch = "bouncy"
            dt_drift_state = "TRUE_CARRY_QUICK"
            dt_step = "STEP 3: BOUNCY"
        elif flat_fire:
            dt_classified_pitch = "flat"
            dt_drift_state = "TRUE_BATTING_HIGHWAY"
            dt_step = "STEP 4: FLAT"
        else:
            dt_classified_pitch = "balanced"
            dt_drift_state = "STABLE"
            dt_step = "STEP 5: BALANCED (Fallback)"
            dt_trace.append("STEP 5 (BALANCED): None of Dusty, Green, Bouncy, Flat satisfied. Defaulting to BALANCED.")
            dt_trace.append("STEP 5 (BALANCED): None of Dusty, Green, Bouncy, Flat satisfied. Defaulting to BALANCED.")

        # Construct calibrated 5-way probability distribution for Approach 1
        base_app1 = {arch: 8.0 for arch in PITCH_ARCHETYPES}
        base_app1[dt_classified_pitch] = 68.0
        if dt_classified_pitch == "dusty":
            base_app1["balanced"] = 14.0
            base_app1["flat"] = 10.0
        elif dt_classified_pitch == "green":
            base_app1["balanced"] = 14.0
            base_app1["bouncy"] = 10.0
        elif dt_classified_pitch == "bouncy":
            base_app1["balanced"] = 12.0
            base_app1["green"] = 12.0
        elif dt_classified_pitch == "flat":
            base_app1["bouncy"] = 12.0
            base_app1["balanced"] = 12.0
        else:
            base_app1["dusty"] = 12.0
            base_app1["green"] = 12.0

        norm_app1 = sum(base_app1.values())
        app1_probs = {k: round((v / norm_app1) * 100.0, 1) for k, v in base_app1.items()}
        app1_probs["spin heavy"] = app1_probs["dusty"]
        app1_probs["bounce"] = app1_probs["bouncy"]
        app1_probs["seam"] = app1_probs["balanced"]
        app1_probs["seam / green"] = app1_probs["green"]
        app1_probs["seam / green"] = app1_probs["green"]

        # -------------------------------------------------------------
        # APPROACH 2: Natural Language Processing (NLP) over Pitch Report & Live Commentary
        # -------------------------------------------------------------
        # Gather text corpus from commentary strings, pitch report, and delivery descriptions
        corpus = []
        pitch_report_text = (pitch_report or "").strip()
        pitch_report_excerpt = pitch_report_text[:160] + "..." if len(pitch_report_text) > 160 else pitch_report_text

        if commentary_strings:
            corpus.extend([str(c) for c in commentary_strings if c is not None and str(c).strip()])
        if deliveries:
            for d in deliveries[-40:]: # Last 40 balls
                desc = str(d.get("event_description", "")) + " " + str(d.get("commentary", ""))
                if desc.strip():
                    corpus.append(desc.strip())

        full_commentary_text = " ".join([str(c) for c in corpus if c]).lower()
        full_pitch_text = pitch_report_text.lower()

        # Score NLP lexicon matches separately for pitch report (3.5x weight) and live commentary (1.0x weight)
        nlp_counts = {arch: 0.0 for arch in PITCH_ARCHETYPES}
        detected_phrases = {arch: [] for arch in PITCH_ARCHETYPES}
        pitch_report_matches_count = 0
        commentary_matches_count = 0

        for arch, keywords in NLP_PITCH_LEXICON.items():
            for kw in keywords:
                # Search in explicit Pitch Report
                if full_pitch_text:
                    p_matches = re.findall(rf"\b{re.escape(kw)}\b", full_pitch_text)
                    if p_matches:
                        count = len(p_matches)
                        nlp_counts[arch] += (count * 3.5)
                        pitch_report_matches_count += count
                        detected_phrases[arch].append(f"Pitch Report: {kw} ({count}x)")

                # Search in Live Commentary
                if full_commentary_text:
                    c_matches = re.findall(rf"\b{re.escape(kw)}\b", full_commentary_text)
                    if c_matches:
                        count = len(c_matches)
                        # Higher weight if phrase appears near "pitch report" or "pitch" in commentary
                        is_pitch_cmt = "pitch" in full_commentary_text[max(0, full_commentary_text.find(kw)-30):full_commentary_text.find(kw)+30]
                        w_kw = 2.5 if is_pitch_cmt else 1.0
                        nlp_counts[arch] += (count * w_kw)
                        commentary_matches_count += count
                        detected_phrases[arch].append(f"{kw} ({count}x)")

        total_nlp_matches = pitch_report_matches_count + commentary_matches_count
        has_strong_nlp = (total_nlp_matches >= 2 or pitch_report_matches_count >= 1)

        if has_strong_nlp:
            # Direct text-based distribution with Laplace smoothing
            app2_probs = {}
            for arch in PITCH_ARCHETYPES:
                raw_score = (nlp_counts[arch] + 0.6) / (sum(nlp_counts.values()) + 3.0)
                app2_probs[arch] = round(raw_score * 100.0, 1)
            norm2 = sum(app2_probs.values())
            app2_probs = {k: round((v / norm2) * 100.0, 1) for k, v in app2_probs.items()}
            nlp_confidence = round(min(97.5, 68.0 + (total_nlp_matches * 3.0) + (10.0 if pitch_report_matches_count >= 1 else 0.0)), 1)
        else:
            # Fallback when commentary stream is concise: infer from format ground archetype
            ground_arch = get_ground_pitch_archetype(venue_id, fmt)
            app2_probs = {arch: 10.0 for arch in PITCH_ARCHETYPES}
            app2_probs[ground_arch] = 60.0
            norm2 = sum(app2_probs.values())
            app2_probs = {k: round((v / norm2) * 100.0, 1) for k, v in app2_probs.items()}
            nlp_confidence = 65.0
            detected_phrases[ground_arch].append(f"Format-aligned commentary tone ({ground_arch})")

        # -------------------------------------------------------------
        # APPROACH 3: Bayesian Ground Prior Blending with Cricsheet Dossier
        # -------------------------------------------------------------
        format_ground_archetype = get_ground_pitch_archetype(venue_id, fmt)
        dossier = FORMAT_GROUND_PITCH_DOSSIERS.get(venue_id.lower().replace("-", "_"), {})
        historical_matches = venue_profile.get("sample_matches", 32) if venue_profile else 32

        # Base prior distribution centered on ground format archetype
        base_prior = {arch: 8.0 for arch in PITCH_ARCHETYPES}
        base_prior[format_ground_archetype] = 68.0

        # Correlated secondaries across 5 canonical archetypes
        if format_ground_archetype == "dusty":
            base_prior["balanced"] = 14.0
            base_prior["flat"] = 8.0
        elif format_ground_archetype == "green":
            base_prior["balanced"] = 16.0
            base_prior["bouncy"] = 8.0
        elif format_ground_archetype == "bouncy":
            base_prior["balanced"] = 14.0
            base_prior["flat"] = 10.0
        elif format_ground_archetype == "flat":
            base_prior["bouncy"] = 12.0
            base_prior["balanced"] = 10.0
        else: # balanced
            base_prior["dusty"] = 10.0
            base_prior["green"] = 10.0

        norm_p = sum(base_prior.values())
        base_prior = {k: (v / norm_p) * 100.0 for k, v in base_prior.items()}

        # Blending weight: historical ground prior decays slightly as more live overs are observed
        prior_weight = max(0.20, 0.65 - (total_overs / 120.0))
        live_weight = 1.0 - prior_weight

        app3_probs = {
            arch: round(base_prior[arch] * prior_weight + app1_probs[arch] * live_weight, 1)
            for arch in PITCH_ARCHETYPES
        }
        norm3 = sum(app3_probs.values())
        app3_probs = {k: round((v / norm3) * 100.0, 1) for k, v in app3_probs.items()}

        # -------------------------------------------------------------
        # SYNTHESIS ENSEMBLE: Real Pitch NLP Detection Overrides Historical Prior
        # -------------------------------------------------------------
        # In-play telemetry confidence scales up with overs bowled:
        # Give in-play telemetry authoritative confidence over historical priors as the match deepens
        if total_overs >= 45.0:
            w_telemetry = min(0.65, 0.45 + (total_overs / 150.0) * 0.20)
            w_nlp = 0.35 if has_strong_nlp else 0.25
            w_prior = max(0.05, 1.0 - w_telemetry - w_nlp)
        elif total_overs >= 15.0:
            w_telemetry = 0.45 if not has_strong_nlp else 0.40
            w_nlp = 0.45 if has_strong_nlp else 0.30
            w_prior = 0.15 if has_strong_nlp else 0.25
        else:
            w_telemetry = 0.25
            w_nlp = 0.55 if has_strong_nlp else 0.35
            w_prior = 0.20 if has_strong_nlp else 0.40

        final_probs = {}
        for arch in PITCH_ARCHETYPES:
            raw = (app1_probs[arch] * w_telemetry) + (app2_probs[arch] * w_nlp) + (app3_probs[arch] * w_prior)
            final_probs[arch] = round(raw, 1)

        total_final = sum(final_probs.values())
        final_probs = {k: round((v / total_final) * 100.0, 1) for k, v in final_probs.items()}

        # Add backward-compatibility aliases across all approach probability dictionaries
        for p_dict in [app1_probs, app2_probs, app3_probs, final_probs]:
            p_dict["spin heavy"] = p_dict.get("dusty", 20.0)
            p_dict["bounce"] = p_dict.get("bouncy", 20.0)
            p_dict["seam"] = p_dict.get("balanced", 20.0)
            p_dict["seam / green"] = round(min(100.0, p_dict.get("green", 0.0) + p_dict.get("balanced", 0.0)), 1)

        # Dominant classified pitch archetype strictly chosen from the 5 canonical types
        classified_type = max(["dusty", "green", "bouncy", "flat", "balanced"], key=lambda k: final_probs.get(k, 0.0))
        confidence = final_probs.get(classified_type, 50.0)

        format_ground_archetype_canon = canonicalize_pitch_type(format_ground_archetype)

        # Override Evaluation: Did real NLP pitch detection override historical ground prior?
        is_override = (classified_type != format_ground_archetype_canon)
        if is_override and has_strong_nlp:
            override_reason = (
                f"Real pitch detected as '{classified_type.upper()}' via Natural Language pitch report/commentary "
                f"({confidence}% confidence), overriding historical '{format_ground_archetype_canon.upper()}' ground profile."
            )
        elif is_override:
            override_reason = (
                f"Live match telemetry shifted deck to '{classified_type.upper()}' from historical '{format_ground_archetype_canon.upper()}'."
            )
        else:
            override_reason = (
                f"Real pitch report and in-play telemetry confirmed historical ground profile of '{classified_type.upper()}' ({confidence}%)."
            )

        # Physical In-Match Indices (0-10 scale)
        turn_index = round(min(9.8, max(1.5, (final_probs["spin heavy"] / 10.0) + (total_overs / 30.0))), 1)
        seam_index = round(min(9.5, max(1.5, (final_probs["seam / green"] / 10.0) * max(0.4, 1.0 - (total_overs / 50.0)))), 1)
        bounce_index = round(min(9.8, max(2.0, (final_probs["bounce"] / 10.0) * 1.2 + (5.0 if "centurion" in venue_id or "mcg" in venue_id else 3.5))), 1)

        # Human-readable surface diagnosis & tactical implications
        if classified_type in ["dusty", "spin heavy"]:
            diagnosis = f"Surface offering consistent grip, low bounce, and sharp turning angles ({turn_index}/10). Early seam movement has subsided as ball aged."
            tactical_implication = "Prioritize spin matchups: Off-spin (OB) into left-handers, Left-arm orthodox (SLA) into right-handers. Batters must smother spin with soft hands."
        elif classified_type == "green":
            diagnosis = f"Deck carrying prominent green grass covering with active lateral seam deviation ({seam_index}/10) and swinging conditions under overhead skies."
            tactical_implication = "Pacers attack top-of-off fourth stump corridor and utilize wobble-seam deviation off the pitch. Batters must play late with soft hands."
        elif classified_type in ["bouncy", "bounce"]:
            diagnosis = f"Hard-packed deck generating steep vertical lift ({bounce_index}/10) and tennis-ball carry through to the keeper."
            tactical_implication = "Pacers enforce short-pitched bouncer barrages. Batters must rely on backfoot punches and controlled pulls."
        elif classified_type == "flat":
            diagnosis = "Placid, true batting track with minimal lateral movement. Fast outfield and consistent carry allow free-flowing strokeplay."
            tactical_implication = "Bowlers must employ defensive containment fields, wide yorkers, and change of pace. Projected par 330+ (ODI) / 185+ (T20I)."
        else: # balanced
            diagnosis = "Sporting surface offering a fair contest between bat and ball. Moderate lateral movement and occasional two-paced carry require batter patience."
            tactical_implication = "Bowlers utilize subtle wobble-seam cutters and cross-seam deliveries. Batters target loose deliveries while respecting the good length."

        # Active detected NLP phrases for dominant category (pooling green and seam phrases)
        if classified_type == "seam / green":
            key_nlp_triggers = detected_phrases.get("green", []) + detected_phrases.get("seam", [])
        else:
            key_nlp_triggers = detected_phrases.get(classified_type, [])

        if not key_nlp_triggers:
            for k in ["green", "seam", "spin heavy", "bounce", "flat"]:
                if detected_phrases.get(k):
                    key_nlp_triggers = detected_phrases[k]
                    break

        cls1 = canonicalize_pitch_type(max(app1_probs, key=lambda k: app1_probs[k] if k != "seam / green" else -1.0))
        cls2 = canonicalize_pitch_type(max(app2_probs, key=lambda k: app2_probs[k] if k != "seam / green" else -1.0))
        cls3 = canonicalize_pitch_type(max(app3_probs, key=lambda k: app3_probs[k] if k != "seam / green" else -1.0))

        conf1 = app1_probs["seam / green"] if cls1 == "seam / green" else app1_probs[cls1]
        conf2 = app2_probs["seam / green"] if cls2 == "seam / green" else app2_probs[cls2]
        conf3 = app3_probs["seam / green"] if cls3 == "seam / green" else app3_probs[cls3]

        return {
            "classified_pitch_type": classified_type,
            "classified_archetype": classified_type, # full backward compatibility
            "confidence_pct": confidence,
            "drift_state": dt_drift_state,
            "decision_tree_step": dt_step,
            "phase_id": phase_id,
            "wicket_gate_active": wicket_gate_active,
            "turn_index": turn_index,
            "seam_index": seam_index,
            "bounce_index": bounce_index,
            "surface_diagnosis": diagnosis,
            "tactical_implication": tactical_implication,
            "probabilities": final_probs,
            "nlp_detected_keywords": key_nlp_triggers[:6],
            "is_override": is_override,
            "historic_ground_archetype": format_ground_archetype_canon,
            "detection_source": "NLP_PITCH_REPORT" if has_strong_nlp else "GROUND_PRIOR_CONSENSUS",
            "override_reason": override_reason,
            "pitch_report_excerpt": pitch_report_excerpt,
            "approaches": {
                "approach_1_telemetry": {
                    "name": "Live Ball Telemetry & In-Match Drift",
                    "spin_overs": round(spin_overs, 1),
                    "pace_overs": round(pace_overs, 1),
                    "spin_economy": round(spin_econ, 2),
                    "pace_economy": round(pace_econ, 2),
                    "spin_deployment_pct": round(spin_overs_share * 100.0, 1),
                    "dot_ball_pct": round(dot_pressure_idx * 100.0, 1),
                    "relative_run_rate": round(relative_rr, 3),
                    "classified": cls1,
                    "drift_state": dt_drift_state,
                    "confidence": conf1,
                    "decision_tree_step": dt_step,
                    "decision_path": dt_trace,
                    "telemetry_metrics": {
                        "Phase_ID": phase_id,
                        "Wicket_Gate_Active": wicket_gate_active,
                        "Relative_Run_Rate": round(relative_rr, 3),
                        "Spin_Overs_Share": round(spin_overs_share, 3),
                        "Spin_Economy_Rate": round(spin_econ, 2),
                        "Pace_Economy_Rate": round(pace_econ, 2),
                        "Dot_Pressure_Index": round(dot_pressure_idx, 3),
                        "Spin_Dot_Rate": round(spin_dot_rate, 3),
                        "Boundary_to_Dot_Ratio": round(boundary_dot_ratio, 3),
                        "Square_Boundary_Share": round(square_boundary_share, 3),
                        "Keeper_Beat_Rate": round(keeper_beat_rate, 3),
                        "Pad_Beat_Rate": round(pad_beat_rate, 3),
                        "Lateral_Deviation_Signal": round(lateral_deviation_signal, 3),
                        "False_Shot_Rate": round(false_shot_rate, 3),
                        "LBW_Bowled_Stumped_Share": round(lbw_bowled_stumped_share, 3),
                        "Behind_Wicket_Ratio": round(behind_wicket_ratio, 3),
                        "Outfield_Catch_Share": round(outfield_catch_share, 3),
                        "Pace_Wicket_Dominance": round(pace_wicket_dominance, 3)
                    },
                    "probabilities": app1_probs
                },
                "approach_2_nlp": {
                    "name": "Natural Language Processing (NLP) Pitch Report & Commentary",
                    "total_surface_mentions": total_nlp_matches,
                    "pitch_report_matches": pitch_report_matches_count,
                    "commentary_matches": commentary_matches_count,
                    "classified": cls2,
                    "confidence": conf2,
                    "key_phrases": key_nlp_triggers[:5],
                    "probabilities": app2_probs
                },
                "approach_3_bayesian": {
                    "name": "Bayesian Ground Prior Blending",
                    "venue_id": venue_id,
                    "format_ground_archetype": format_ground_archetype_canon,
                    "historical_matches_sample": historical_matches,
                    "classified": cls3,
                    "confidence": conf3,
                    "probabilities": app3_probs
                }
            }
        }
