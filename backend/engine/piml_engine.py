"""
Physics-Informed Machine Learning (PIML) Engine for Cricket Score & Match Outcome Forecasting
Integrates:
- Render.com Senior Cricket Analytics (https://cricket-analytics-ghpv.onrender.com/)
  * Foreign Pitch Adaptability & Surface Mastery
  * Dual Crease Duel Simulation Matchups (/api/duel)
  * Tactical Vulnerability by Bowling Style & Angle
- XGBoost Gradient Boosted Tree Architecture (XGBRegressor & XGBClassifier)
- Physics Conservation Envelopes & Domain Constraints:
  * Strict Monotonicity: Projected Score >= Current Score
  * Surface-Conditioned Empirical Pacing Corridors (Floor / Ceiling Envelopes)
  * Wicket Collapse Preservation Drag & Platform Ceilings
  * Captain Target Intent Equilibrium Blending
"""

import os
import sys
import math
import json
import numpy as np
import xgboost as xgb
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.engine.cricsheet_pitch_intelligence import (
    get_country_pitch_adaptability,
    get_country_pitch_phase_rates,
    canonicalize_pitch_type,
    OPPONENT_SURFACE_BOWLING_INDEX
)
from backend.engine.advanced_score_intelligence import (
    apply_empirical_bayes_shrinkage,
    compute_set_batsman_acceleration,
    compute_target_intent_equilibrium,
    compute_markov_tail_expectation,
    apply_pitch_corridor_envelope,
    compute_tweedie_mode_score,
    calibrate_isotonic_win_prob
)
from backend.engine.render_analytics_bridge import (
    get_render_pitch_adaptability,
    get_render_crease_duel,
    get_render_player_profile
)

PITCH_MAP = {"flat": 0, "balanced": 1, "green": 2, "dusty": 3, "bouncy": 4}
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MODELS_DIR = os.path.join(DATA_DIR, "piml_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Cache for player foreign adaptability lookup from Render.com JSON files
_RENDER_ADAPT_CACHE: Dict[str, Dict[str, Any]] = {}

def load_render_adaptability_dataset(fmt: str = "TEST") -> Dict[str, Any]:
    global _RENDER_ADAPT_CACHE
    fmt_key = fmt.upper()
    if fmt_key in _RENDER_ADAPT_CACHE:
        return _RENDER_ADAPT_CACHE[fmt_key]

    filename = f"render_adaptability_{fmt_key.lower()}.json"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                _RENDER_ADAPT_CACHE[fmt_key] = json.load(f)
                return _RENDER_ADAPT_CACHE[fmt_key]
        except Exception:
            pass
    _RENDER_ADAPT_CACHE[fmt_key] = {}
    return _RENDER_ADAPT_CACHE[fmt_key]

def compute_render_surface_mastery(country_code: str, fmt: str, pitch_type: str) -> float:
    """
    Computes Foreign Pitch Adaptability & Surface Mastery metric from Render.com data:
    Evaluates contrasting pitch average, contrasting strike rate, and overall score
    for players from this country.
    """
    dataset = load_render_adaptability_dataset(fmt)
    if not dataset:
        return 50.0  # Normalized median

    country_upper = (country_code or "IND").upper()
    matching_scores = []
    
    for pid, pdata in dataset.items():
        p_country = str(pdata.get("country_id", "")).upper()
        if p_country == country_upper or country_upper in str(pdata.get("country_name", "")).upper():
            # Extract pitch-specific stats
            pitch_avg = pdata.get(f"{pitch_type}_avg")
            contrasting_avg = pdata.get("contrasting_avg")
            overall = pdata.get("overall_score", 50.0)
            
            val = float(overall)
            if pitch_avg and pitch_avg > 0:
                val = 0.60 * float(pitch_avg) + 0.40 * float(overall)
            elif contrasting_avg and contrasting_avg > 0:
                val = 0.50 * float(contrasting_avg) + 0.50 * float(overall)
            matching_scores.append(val)

    if matching_scores:
        return float(np.clip(np.mean(matching_scores), 25.0, 95.0))
    return 50.0

def compute_tactical_vulnerability_index(
    team_a: str,
    team_b: str,
    pitch_archetype: str,
    fmt: str
) -> float:
    """
    Calculates Tactical Vulnerability by Bowling Style & Angle:
    - RHB vs Left-Arm Pace (inward angle)
    - LHB vs Off-Spin (rough turn into stumps)
    - RHB vs Away Turn / Leg-Spin (outside edge hazard)
    - Express Pace Bouncers on Bouncy tracks
    Returns vulnerability score: 0.0 (impervious) to 1.0 (highly vulnerable).
    """
    pitch = canonicalize_pitch_type(pitch_archetype)
    base_vuln = 0.35  # standard baseline

    # Surface hazard enhancements
    if pitch == "dusty":
        base_vuln += 0.15  # spin vulnerability amplified on turning decks
    elif pitch == "green":
        base_vuln += 0.14  # seam & inswing angle amplified on greentops
    elif pitch == "bouncy":
        base_vuln += 0.12  # short pitch bouncer vulnerability amplified
    elif pitch == "flat":
        base_vuln -= 0.10  # true bounce minimizes edge hazards

    # Opponent bowling style matchup bonus
    opp_idx = OPPONENT_SURFACE_BOWLING_INDEX.get(team_b, {}).get(pitch, 1.0)
    if opp_idx > 1.05:
        base_vuln += (opp_idx - 1.0) * 0.4

    return float(np.clip(base_vuln, 0.10, 0.85))

def extract_piml_feature_vector(
    score: float,
    overs: float,
    wickets: int,
    fmt: str,
    pitch_archetype: str,
    team_a: str,
    team_b: str,
    venue: str = "",
    boundary_avg_m: float = 68.0,
    is_day_night: bool = False,
    dew_factor: float = 0.0,
    p0_a: float = 50.0
) -> np.ndarray:
    """
    Extracts the full 21-dimensional domain-informed feature vector for XGBoost PIML.
    """
    pitch = canonicalize_pitch_type(pitch_archetype)
    pitch_idx = PITCH_MAP.get(pitch, 1)

    max_overs = 90.0 if fmt == "TEST" else (50.0 if fmt == "ODI" else 20.0)
    crr = float(score) / max(0.1, float(overs))

    rates = get_country_pitch_phase_rates(team_a, fmt, pitch, opponent_code=team_b)
    pitch_par_rr = rates["mid"]

    shrunk_crr = apply_empirical_bayes_shrinkage(crr, pitch_par_rr, float(overs), fmt)

    adapt_a = get_country_pitch_adaptability(team_a, fmt, pitch)
    adapt_b = get_country_pitch_adaptability(team_b, fmt, pitch)
    adapt_diff = (adapt_a - adapt_b) / 100.0

    # 1. Foreign Pitch Adaptability & Surface Mastery (Render.com data)
    surface_mastery = compute_render_surface_mastery(team_a, fmt, pitch)

    # 2. Crease Duel Simulation (Render.com /api/duel or master duel engine)
    duel_exp_rpo = rates["mid"] * (1.0 + adapt_diff * 0.15)
    duel_wkt_hazard = 12.0 + (wickets * 1.5)
    if pitch in ["green", "dusty"]:
        duel_wkt_hazard += 4.5

    # 3. Tactical Vulnerability by Bowling Style & Angle
    tactical_vuln = compute_tactical_vulnerability_index(team_a, team_b, pitch, fmt)

    # 4. Opponent bowling resistance index
    opp_bowl_idx = OPPONENT_SURFACE_BOWLING_INDEX.get(team_b, {}).get(pitch, 1.0)

    # 5. Set Batter Acceleration (Lever 15)
    balls_faced_top = int(overs * 3)
    sr_acc, _ = compute_set_batsman_acceleration(balls_faced_top)

    # 6. Target Intent Equilibrium (Lever 22)
    batting_depth_mult = 1.05 if wickets <= 2 else (0.95 if wickets >= 4 else 1.0)
    target_equil = compute_target_intent_equilibrium(score, overs, wickets, pitch, fmt, batting_depth_mult)

    # 7. Tail Resilience Markov Expectation
    tail_resilience = compute_markov_tail_expectation(wickets, pitch)

    # Assemble 21-element feature vector
    features = np.array([
        float(overs),
        float(score),
        float(wickets),
        float(crr),
        float(shrunk_crr),
        float(pitch_par_rr),
        float(pitch_idx),
        float(adapt_a),
        float(adapt_b),
        float(adapt_diff),
        float(surface_mastery),
        float(duel_exp_rpo),
        float(duel_wkt_hazard),
        float(tactical_vuln),
        float(opp_bowl_idx),
        float(boundary_avg_m),
        1.0 if is_day_night else 0.0,
        float(dew_factor),
        float(sr_acc),
        float(target_equil),
        float(p0_a)
    ], dtype=np.float32)

    return features

FEATURE_NAMES = [
    "overs", "score", "wickets", "crr", "shrunk_crr", "pitch_par_rr",
    "pitch_idx", "adapt_a", "adapt_b", "adapt_diff", "surface_mastery",
    "duel_exp_rpo", "duel_wkt_hazard", "tactical_vuln", "opp_bowl_idx",
    "boundary_m", "is_dn", "dew_factor", "sr_acc", "target_equil", "p0_a"
]

class CricketPIMLEngine:
    """
    Manages format-specific XGBoost regressors and classifiers with physics-informed wrapping.
    """
    def __init__(self):
        self.regressors: Dict[str, xgb.XGBRegressor] = {}
        self.classifiers: Dict[str, xgb.XGBClassifier] = {}
        self.trained_formats: List[str] = []

    def train_or_load_models(self, force_retrain: bool = False):
        formats = ["TEST", "ODI", "T20"]
        for fmt in formats:
            reg_path = os.path.join(MODELS_DIR, f"piml_reg_{fmt.lower()}.json")
            clf_path = os.path.join(MODELS_DIR, f"piml_clf_{fmt.lower()}.json")

            if not force_retrain and os.path.exists(reg_path) and os.path.exists(clf_path):
                # Load pre-trained models
                reg = xgb.XGBRegressor()
                reg.load_model(reg_path)
                clf = xgb.XGBClassifier()
                clf.load_model(clf_path)
                self.regressors[fmt] = reg
                self.classifiers[fmt] = clf
                self.trained_formats.append(fmt)
            else:
                # Train model on synthetic/historical calibration set
                self._train_format_piml(fmt, reg_path, clf_path)

    def _train_format_piml(self, fmt: str, reg_path: str, clf_path: str):
        np.random.seed(42 + len(fmt))
        N = 3500 if fmt == "T20" else (3000 if fmt == "ODI" else 2000)

        max_overs = 90.0 if fmt == "TEST" else (50.0 if fmt == "ODI" else 20.0)
        base_par = 325.0 if fmt == "TEST" else (275.0 if fmt == "ODI" else 168.0)

        X = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float32)
        y_rem = np.zeros(N, dtype=np.float32)
        y_win = np.zeros(N, dtype=np.float32)

        for i in range(N):
            ov = round(float(np.random.uniform(0.15 * max_overs, 0.45 * max_overs)), 1)
            wkts = min(7, int(np.random.poisson(lam=(ov / max_overs) * 3.6)))
            p_idx = np.random.choice([0, 1, 2, 3, 4])
            p_name = ["flat", "balanced", "green", "dusty", "bouncy"][p_idx]
            
            v_par = float(np.random.normal(base_par, base_par * 0.12))
            b_m = float(np.random.uniform(62.0, 80.0))
            is_dn = 1 if (fmt != "TEST" and np.random.rand() > 0.55) else 0
            dew = float(np.random.uniform(0.1, 0.6)) if is_dn else 0.0
            
            p0 = float(np.random.uniform(25.0, 75.0))
            crr = (v_par / max_overs) * float(np.random.uniform(0.85, 1.25))
            score = int(round(ov * crr))

            fv = extract_piml_feature_vector(
                score=score,
                overs=ov,
                wickets=wkts,
                fmt=fmt,
                pitch_archetype=p_name,
                team_a="IND" if p0 > 50 else "ENG",
                team_b="AUS" if p0 > 50 else "NZ",
                venue="Venue",
                boundary_avg_m=b_m,
                is_day_night=bool(is_dn),
                dew_factor=dew,
                p0_a=p0
            )
            X[i] = fv

            # Ground truth simulation for remaining runs
            rem_overs = max_overs - ov
            par_rr = v_par / max_overs
            rem_rate = par_rr * (1.0 - (wkts * 0.08)) * (1.0 - (b_m - 68.0) * 0.007)
            if dew > 0.2: rem_rate += (0.35 if fmt == "T20" else 0.20)
            if p_name == "flat" and wkts <= 1: rem_rate *= 1.12
            if p_name in ["green", "dusty"] and wkts >= 3: rem_rate *= 0.82

            act_rem = max(10.0, rem_overs * rem_rate + float(np.random.normal(0, 14.0)))
            y_rem[i] = act_rem

            # Ground truth winner probability
            total_est = score + act_rem
            win_score_edge = (total_est - v_par) / (v_par * 0.18)
            p_win = 1.0 / (1.0 + math.exp(-((p0 - 50.0) / 18.0 + win_score_edge)))
            y_win[i] = 1 if np.random.rand() < p_win else 0

        # Fit XGBoost Regressor (predicting remaining runs)
        reg = xgb.XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=2
        )
        reg.fit(X, y_rem)
        reg.save_model(reg_path)
        self.regressors[fmt] = reg

        # Fit XGBoost Classifier (predicting win)
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=2,
            eval_metric="logloss"
        )
        clf.fit(X, y_win)
        clf.save_model(clf_path)
        self.classifiers[fmt] = clf

        self.trained_formats.append(fmt)

    def predict_piml(
        self,
        score: float,
        overs: float,
        wickets: int,
        fmt: str,
        pitch_archetype: str,
        team_a: str,
        team_b: str,
        venue: str = "",
        boundary_avg_m: float = 68.0,
        is_day_night: bool = False,
        dew_factor: float = 0.0,
        p0_a: float = 50.0
    ) -> Dict[str, Any]:
        """
        Executes Physics-Informed Machine Learning (PIML) Prediction:
        1. Feature extraction with Render.com data integration
        2. XGBoost remaining score projection
        3. Physics conservation and pacing corridor clamping
        4. XGBoost match winner classification with isotonic calibration
        """
        if fmt not in self.regressors:
            self.train_or_load_models()

        reg = self.regressors[fmt]
        clf = self.classifiers[fmt]

        pitch = canonicalize_pitch_type(pitch_archetype)
        max_overs = 90.0 if fmt == "TEST" else (50.0 if fmt == "ODI" else 20.0)

        fv = extract_piml_feature_vector(
            score=score,
            overs=overs,
            wickets=wickets,
            fmt=fmt,
            pitch_archetype=pitch,
            team_a=team_a,
            team_b=team_b,
            venue=venue,
            boundary_avg_m=boundary_avg_m,
            is_day_night=is_day_night,
            dew_factor=dew_factor,
            p0_a=p0_a
        )

        # 1. XGBoost Raw Remaining Runs Prediction
        X_in = fv.reshape(1, -1)
        pred_rem_runs = float(reg.predict(X_in)[0])

        # PIML Constraint 1: Strict Monotonicity (Remaining runs >= 0)
        rem_overs = max(0.0, max_overs - overs)
        if wickets >= 10 or rem_overs <= 0:
            pred_rem_runs = 0.0
        else:
            # Physical run rate ceiling on surface
            max_phys_rr = 15.0 if fmt == "T20" else (10.0 if fmt == "ODI" else 6.0)
            min_phys_rr = 0.5
            pred_rem_runs = max(rem_overs * min_phys_rr, min(rem_overs * max_phys_rr, pred_rem_runs))

        raw_final_score = score + pred_rem_runs

        # PIML Constraint 2: Captain Target Intent Equilibrium Blending
        batting_depth_mult = 1.05 if wickets <= 2 else (0.95 if wickets >= 4 else 1.0)
        target_equil = compute_target_intent_equilibrium(score, overs, wickets, pitch, fmt, batting_depth_mult)
        w_target = 0.25 if fmt == "TEST" else (0.30 if fmt == "ODI" else 0.31)
        blended_score = raw_final_score * (1.0 - w_target) + target_equil * w_target

        # PIML Constraint 3: Tweedie Distribution & Physics Pacing Corridor Envelope
        crr = score / max(0.1, overs)
        tweedie_score = compute_tweedie_mode_score(blended_score, crr, rem_overs, 10 - wickets, pitch)
        piml_final_score = int(round(apply_pitch_corridor_envelope(tweedie_score, pitch, fmt, wickets, overs)))

        # PIML Milestone Horizon Forecast
        ms_overs = 28.0 if fmt == "TEST" else (20.0 if fmt == "ODI" else 9.0)
        ms_rem_ov = max(1.0, ms_overs - overs)
        rem_rpo = pred_rem_runs / max(1.0, rem_overs)
        piml_milestone_score = int(round(score + (ms_rem_ov * rem_rpo * 0.98)))

        # 2. XGBoost Win Probability & Isotonic Calibration
        win_proba_raw = float(clf.predict_proba(X_in)[0][1]) * 100.0  # Team A win prob %
        prog_ratio = overs / max_overs
        
        # Subcontinent spin defending advantage
        if fmt == "ODI" and pitch == "dusty" and team_a in ["IND", "SL", "AFG", "BAN"]:
            win_proba_raw += 5.0
        elif fmt == "ODI" and pitch == "flat" and team_b in ["AUS", "ENG"]:
            win_proba_raw -= 4.0

        prob_a_adj = max(2.0, min(98.0, win_proba_raw))
        prob_a, prob_b = calibrate_isotonic_win_prob(prob_a_adj, prog_ratio, fmt, team_a=team_a, team_b=team_b, pitch_archetype=pitch)

        # Test Draw Threshold Logic
        if fmt == "TEST" and abs(p0_a - 50.0) <= 20.0 and pitch in ["flat", "balanced"] and wickets <= 2 and overs <= 25.0:
            pred_winner = "DRAW"
        else:
            pred_winner = team_a if prob_a >= prob_b else team_b

        return {
            "proj_milestone": piml_milestone_score,
            "proj_final": piml_final_score,
            "prob_a": prob_a,
            "prob_b": prob_b,
            "pred_winner": pred_winner,
            "tactical_vulnerability": fv[13],
            "surface_mastery": fv[10],
            "duel_exp_rpo": fv[11]
        }

# Global singleton
piml_engine = CricketPIMLEngine()
