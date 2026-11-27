"""
Stage 4 Live Calibrator (Innings 1 & Limited-Overs Innings 2 Chases)

Calibrated, leakage-free Stage 4 models combining:
- Render 22 Master Attributes & remaining dugout/bowling resource quality
- Render 5-Surface Pitch Adaptability & offline /api/duel simulation cache
- Live ball telemetry (recent 3-over RPO, dot%, boundary%, extras spike, spin RPO delta)
- Dynamic Dew & Weather Intelligence (Day/Night dew factor, LiveDewDetector, Open-Meteo)
- Stage 3 calibrated venue, pitch, MOV-Elo, and pre-match P0 priors
- Stern DLS resource curves, Live Chase DLS Par differential, & conformal 90% bounds
"""

import logging
import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit, logit as sp_logit

from backend.engine.cricsheet_pitch_intelligence import (
    canonicalize_pitch_type,
    get_country_pitch_adaptability,
    get_country_pitch_phase_rates,
    normalize_team_code,
    resolve_venue_dossier_context,
)
from backend.engine.cricpulse_master_engine import (
    compute_22_master_attributes,
    compute_duel_simulation,
)
from backend.engine.live_dew_detector import LiveDewDetector
from backend.engine.render_analytics_bridge import (
    _load_adapt_cache,
    _load_duels_cache,
    _load_players_cache,
    resolve_render_player_id,
)
from backend.engine.stage3_ground_pitch_model import (
    compute_stage3_dual_first_innings_projections,
    load_stage3_calibrated_priors,
    normalize_venue_key,
)
from backend.engine.weather_intelligence import (
    calculate_live_dls_par_score,
    interpolate_dls_resource,
)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CALIBRATOR_PATH = os.path.join(DATA_DIR, "piml_models", "stage4_calibrator_v1.pkl")

SCORE_FEAT_NAMES: List[str] = [
    "s3_par_bat",
    "s3_hist_par",
    "era_par",
    "dls_rem_pct",
    "sc_above_dls_used",
    "prog",
    "score",
    "wickets",
    "wickets_left",
    "wkts_saved",
    "crr",
    "crr_x_wkts_left",
    "score_above_par",
    "recent_3ov_rpo",
    "rec3_minus_crr",
    "dot_pct",
    "bdry_pct",
    "has_ball_telem",
    "p_flat",
    "p_green",
    "p_dusty",
    "p_bouncy",
    "adapt_diff",
    "crease_bat_score",
    "crease_pitch_avg",
    "rem_bat_score",
    "rem_bat_dep",
    "rem_bat_pitch_avg",
    "rem_bat_adapt_sc",
    "opp_bowl_score",
    "opp_bowl_spear",
    "opp_bowl_death",
    "opp_bowl_pitch_econ",
    "wkts_x_rem_bat",
    "pitch_bat_edge",
    "duel_net_rpo",
    "duel_wkt",
    "boundary_avg_m",
    "dew_factor",
    "is_day_night",
]

SCORE_COMPACT_IDX: List[int] = [SCORE_FEAT_NAMES.index(n) for n in [
    "era_par",
    "s3_hist_par",
    "dls_rem_pct",
    "sc_above_dls_used",
    "prog",
    "score",
    "wickets",
    "wkts_saved",
    "crr",
    "crr_x_wkts_left",
    "recent_3ov_rpo",
    "rec3_minus_crr",
    "dot_pct",
    "bdry_pct",
    "adapt_diff",
    "rem_bat_score",
    "rem_bat_pitch_avg",
    "opp_bowl_score",
    "opp_bowl_pitch_econ",
    "wkts_x_rem_bat",
    "pitch_bat_edge",
    "duel_net_rpo",
    "dew_factor",
]]

CHASE_FEAT_NAMES: List[str] = [
    "p0_logit",
    "sc_above_dls_par",
    "rrr_minus_crr",
    "need_vs_dls_rem_par",
    "target_vs_par",
    "wickets_left",
    "dls_rem",
    "prog",
    "crr",
    "rrr",
    "p_flat",
    "p_dusty",
    "p_green",
    "adapt_diff",
    "rem_bat_score",
    "rem_bat_dep",
    "opp_bowl_score",
    "opp_bowl_death",
    "rem_bat_pitch_avg",
    "opp_bowl_pitch_econ",
    "duel_net_rpo",
    "recent_3ov_rpo",
    "rec3_minus_rrr",
    "dot_pct",
    "bdry_pct",
    "dew_cal",
    "is_dn_x_dew",
    "dew_tele",
    "extras_spike",
    "spin_rpo_delta",
    "ven_dew_prior",
]

_CALIBRATOR_BUNDLE: Optional[Dict[str, Any]] = None
_S3_DUAL_MEMO: Dict[Tuple[str, ...], Tuple[float, float, float]] = {}
_ATTRS_MEMO: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def load_stage4_calibrator(force_reload: bool = False) -> Optional[Dict[str, Any]]:
    """Loads calibrated Stage 4 artifact once cleanly into memory."""
    global _CALIBRATOR_BUNDLE
    if _CALIBRATOR_BUNDLE is not None and not force_reload:
        return _CALIBRATOR_BUNDLE
    if not os.path.exists(CALIBRATOR_PATH):
        return None
    try:
        with open(CALIBRATOR_PATH, "rb") as f:
            bundle = pickle.load(f)
        for fmt_k, fmt_v in bundle.items():
            if not isinstance(fmt_v, dict):
                continue
            for sub_k in ("early", "mid_late", "chase"):
                exp_d = fmt_v.get(sub_k)
                if isinstance(exp_d, dict):
                    for m_k in ("et_reg", "xgb_reg", "xgb_win"):
                        mdl = exp_d.get(m_k)
                        if mdl is not None and hasattr(mdl, "n_jobs"):
                            mdl.n_jobs = 1
        _CALIBRATOR_BUNDLE = bundle
        return _CALIBRATOR_BUNDLE
    except Exception as exc:
        logger.warning("Failed to load stage4_calibrator_v1.pkl: %s", exc)
        return None


def invalidate_stage4_calibrator_cache() -> None:
    """Clears cached model bundle and memo tables (for tests/retraining)."""
    global _CALIBRATOR_BUNDLE
    _CALIBRATOR_BUNDLE = None
    _S3_DUAL_MEMO.clear()
    _ATTRS_MEMO.clear()


def normalize_stage4_fmt(fmt: str) -> str:
    up = str(fmt or "T20I").upper().strip()
    if up in ("T20", "T20I", "IT20"):
        return "T20I"
    if "ODI" in up:
        return "ODI"
    return "TEST"


def compute_dls_rem_pct(fmt: str, overs_f: float, wkts: float) -> float:
    """Standard Stern DLS resource remaining curve (with Test survival curve)."""
    f = normalize_stage4_fmt(fmt)
    w = int(max(0, min(9, round(wkts))))
    if f == "TEST":
        rem_ov_frac = max(0.18, 1.0 - min(0.85, overs_f / 108.0))
        w_factor = max(0.08, ((10.0 - w) / 10.0) ** 0.75)
        return float(max(0.05, min(0.95, rem_ov_frac * w_factor)))

    max_ov = 20.0 if f == "T20I" else 50.0
    u = max(0.1, max_ov - overs_f)
    z_table = [100.0, 93.4, 85.1, 74.9, 62.7, 49.0, 34.9, 22.0, 11.9, 4.7]
    b_table = [0.035, 0.037, 0.041, 0.047, 0.056, 0.068, 0.086, 0.115, 0.165, 0.270]
    b = b_table[w] / 0.52 if f == "T20I" else b_table[w]
    num = 1.0 - math.exp(-b * u)
    den = 1.0 - math.exp(-b * max_ov)
    return float(max(0.02, min(0.98, (z_table[w] / 100.0) * (num / max(1e-6, den)))))


def extract_live_ball_telemetry(
    deliveries: Optional[List[Dict[str, Any]]],
    innings_num: int,
    crr: float,
    fmt: str,
) -> Dict[str, float]:
    """Extracts rolling 3-over RPO, dot%, and boundary% from deliveries with neutral fallback."""
    f = normalize_stage4_fmt(fmt)
    def_dot = 0.38 if f == "T20I" else (0.50 if f == "ODI" else 0.72)
    def_bdry = 0.16 if f == "T20I" else (0.10 if f == "ODI" else 0.05)

    if not deliveries:
        return {
            "has_ball_telem": 0.0,
            "recent_3ov_rpo": float(crr),
            "dot_pct": def_dot,
            "bdry_pct": def_bdry,
        }

    inn_dels = [
        d for d in deliveries
        if isinstance(d, dict) and int(d.get("innings") or d.get("inning") or innings_num) == innings_num
    ]
    if len(inn_dels) < 6:
        return {
            "has_ball_telem": 0.0,
            "recent_3ov_rpo": float(crr),
            "dot_pct": def_dot,
            "bdry_pct": def_bdry,
        }

    dots = 0
    bdries = 0
    legal_n = 0
    for d in inn_dels:
        is_w = bool(d.get("is_wide") or "wide" in str(d.get("extra_type", "")).lower())
        is_nb = bool(d.get("is_noball") or "noball" in str(d.get("extra_type", "")).lower())
        rt = int(d.get("runs") if (d.get("runs") is not None and not isinstance(d.get("runs"), dict)) else (d.get("runs_total") or 0))
        rb_raw = d.get("runs_batter")
        if rb_raw is None:
            rb_raw = d.get("batter_runs")
        if rb_raw is None:
            rb_raw = d.get("runs_off_bat")
        rb = int(rb_raw or 0)
        if rt == 0 and rb > 0:
            rt = rb + int(d.get("extras_runs") or (d.get("extras") if not isinstance(d.get("extras"), dict) else 0) or 0)
        if not is_w and not is_nb:
            legal_n += 1
            if rt == 0:
                dots += 1
        if rb in (4, 6):
            bdries += 1

    rec_slice = inn_dels[-18:]
    rec_runs = 0
    for d in rec_slice:
        rt = d.get("runs")
        if rt is not None and not isinstance(rt, dict):
            rec_runs += int(rt)
        else:
            rb_raw = d.get("runs_batter")
            if rb_raw is None:
                rb_raw = d.get("batter_runs")
            if rb_raw is None:
                rb_raw = d.get("runs_off_bat")
            rec_runs += int(rb_raw or 0) + int(d.get("extras_runs") or (d.get("extras") if not isinstance(d.get("extras"), dict) else 0) or 0)
    recent_3ov_rpo = float(rec_runs) / max(0.5, len(rec_slice) / 6.0)
    return {
        "has_ball_telem": 1.0,
        "recent_3ov_rpo": recent_3ov_rpo,
        "dot_pct": float(dots) / max(1, legal_n),
        "bdry_pct": float(bdries) / max(1, len(inn_dels)),
    }


def _get_cached_attrs(player: Dict[str, Any], render_fmt: str) -> Dict[str, Any]:
    pid = str(player.get("cricsheet_id") or player.get("render_id") or player.get("id") or player.get("name") or "")
    role = str(player.get("role") or "")
    k = (pid, role, render_fmt)
    res = _ATTRS_MEMO.get(k)
    if res is None:
        res = compute_22_master_attributes(player, render_fmt)
        if len(_ATTRS_MEMO) < 12000:
            _ATTRS_MEMO[k] = res
    return res


def _resolve_pid_fast(player: Dict[str, Any], p_cache: Dict[str, Any]) -> str:
    for k in ("cricsheet_id", "render_id", "id"):
        val = str(player.get(k) or "").strip()
        if val and val in p_cache:
            return val
    name = str(player.get("name") or "").strip()
    if name:
        pid = resolve_render_player_id(name)
        if pid:
            return pid
    return ""


def _resolve_stage3_pars(
    fmt: str,
    bat_team: str,
    bowl_team: str,
    venue_name: str,
    live_pitch_type: str,
    bat_xi: List[Dict[str, Any]],
    bowl_xi: List[Dict[str, Any]],
) -> Tuple[float, float, float]:
    """Computes cached Stage 3 batting par, bowling opp par, and historical venue par."""
    bat_c = normalize_team_code(bat_team)
    bowl_c = normalize_team_code(bowl_team)
    v_str = str(venue_name or "")
    sig = (fmt, bat_c, bowl_c, v_str.lower(), live_pitch_type)
    cached = _S3_DUAL_MEMO.get(sig)
    if cached is not None:
        return cached

    priors = load_stage3_calibrated_priors()
    vkey = normalize_venue_key(v_str)
    v_meta = priors.get(fmt, {}).get("venues", {}).get(vkey, {})
    dual = compute_stage3_dual_first_innings_projections(
        fmt=fmt,
        team_a_id=bat_c,
        team_b_id=bowl_c,
        xi_a=bat_xi,
        xi_b=bowl_xi,
        venue_name=v_str,
        priors=priors,
    )
    s3_bat = float(dual["if_team_a_bats_first"]["projected_score"])
    s3_opp = float(dual["if_team_b_bats_first"]["projected_score"])
    s3_hist = float(v_meta.get("mean") or priors.get(fmt, {}).get("overall_mean") or s3_bat)
    res = (s3_bat, s3_opp, s3_hist)
    if len(_S3_DUAL_MEMO) < 2000:
        _S3_DUAL_MEMO[sig] = res
    return res


def _extract_render_squad_edges(
    *,
    f: str,
    canon_pitch: str,
    wickets_i: int,
    wickets_left: float,
    st_list: List[Dict[str, Any]],
    dug_list: List[Dict[str, Any]],
    bw_list: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Extracts cached Render 22-attributes, 5-surface adaptability, and offline duel metrics."""
    render_fmt = "TEST" if f == "TEST" else ("ODI" if f == "ODI" else "T20")
    p_cache = _load_players_cache(render_fmt)
    a_cache = _load_adapt_cache(render_fmt)
    d_cache = _load_duels_cache()
    base_rr = 7.8 if f == "T20I" else (5.3 if f == "ODI" else 3.25)
    def_avg = 34.0 if f == "TEST" else (33.0 if f == "ODI" else 26.0)
    pitch_col = f"{canon_pitch}_avg"
    pitch_ec_col = f"{canon_pitch}_econ"

    def _bat_prof(p: Dict[str, Any], idx: int) -> Tuple[float, float, float, float]:
        pid = _resolve_pid_fast(p, p_cache)
        r_prof = p_cache.get(pid, {})
        r_adp = a_cache.get(pid, {})
        attrs = _get_cached_attrs(p if p.get("batting_order") else {**p, "batting_order": idx + 1}, render_fmt)
        b_sc = float(r_prof.get("bat_score") or attrs.get("b_power", 50.0))
        dep_b = float(r_prof.get("dependability_bat") or r_prof.get("resilience_bat") or 50.0)
        p_avg = float(r_adp.get(pitch_col) or r_adp.get(f"{canon_pitch}_home_avg") or 0.0) if r_adp else 0.0
        if p_avg <= 0:
            p_avg = def_avg
        ov_sc = float(r_adp.get("overall_score") or 50.0) if r_adp else 50.0
        return b_sc, dep_b, p_avg, ov_sc

    def _bowl_prof(p: Dict[str, Any]) -> Tuple[float, float, float, float]:
        pid = _resolve_pid_fast(p, p_cache)
        r_prof = p_cache.get(pid, {})
        r_adp = a_cache.get(pid, {})
        attrs = _get_cached_attrs({**p, "role": p.get("role") or "bowler"}, render_fmt)
        bw_sc = float(r_prof.get("bowl_score") or attrs.get("bw_power", 50.0))
        sp_sc = float(r_prof.get("spearhead_bowl") or r_prof.get("impact_bowl") or 50.0)
        dth_sc = float(r_prof.get("clutch_death_bowl") or r_prof.get("phase_death") or 50.0)
        p_ec = float(r_adp.get(pitch_ec_col) or 0.0) if r_adp else 0.0
        if p_ec <= 0:
            p_ec = base_rr
        return bw_sc, sp_sc, dth_sc, p_ec

    crease_profs = [_bat_prof(p, i) for i, p in enumerate(st_list)]
    rem_profs = [_bat_prof(p, wickets_i + 2 + i) for i, p in enumerate(dug_list[: max(1, 7 - wickets_i)])]
    if not rem_profs:
        rem_profs = crease_profs or [(48.0, 50.0, def_avg, 50.0)]

    seen_bw = set()
    bw_uniq = []
    for p in bw_list:
        k = str(p.get("id") or p.get("name") or "")
        if k not in seen_bw:
            seen_bw.add(k)
            bw_uniq.append(p)
        if len(bw_uniq) >= 5:
            break
    bw_profs = [_bowl_prof(p) for p in bw_uniq] or [(50.0, 50.0, 50.0, base_rr)]

    duel_rpos, duel_wkts = [], []
    for bp in (st_list + dug_list)[:3]:
        b_pid = _resolve_pid_fast(bp, p_cache)
        b_attrs = _get_cached_attrs(bp, render_fmt)
        for bwp in bw_uniq[:3]:
            bw_pid = _resolve_pid_fast(bwp, p_cache)
            rd = d_cache.get(f"{b_pid}_{bw_pid}_{render_fmt}") if (b_pid and bw_pid) else None
            if rd and rd.get("expected_runs_per_over"):
                duel_rpos.append(float(rd["expected_runs_per_over"]))
                duel_wkts.append(float(rd.get("wicket_risk_pct", 4.0)) / 100.0)
            else:
                sim_d = compute_duel_simulation(b_attrs, _get_cached_attrs(bwp, render_fmt), match_format=f, pitch_archetype=canon_pitch)
                duel_rpos.append(float(sim_d.get("expected_runs_per_ball", 1.0)) * 6.0)
                duel_wkts.append(float(sim_d.get("probabilities", {}).get("wicket", 0.04)))

    rem_bat_score = float(np.mean([x[0] for x in rem_profs]))
    rem_bat_dep = float(np.mean([x[1] for x in rem_profs]))
    rem_bat_pitch_avg = float(np.mean([x[2] for x in rem_profs]))
    rem_bat_adapt_sc = float(np.mean([x[3] for x in rem_profs]))
    opp_bowl_score = float(np.mean([x[0] for x in bw_profs]))
    opp_bowl_spear = float(np.mean([x[1] for x in bw_profs]))
    opp_bowl_death = float(np.mean([x[2] for x in bw_profs]))
    opp_bowl_pitch_econ = float(np.mean([x[3] for x in bw_profs]))

    crease_bat_score = float(np.mean([x[0] for x in crease_profs])) if crease_profs else rem_bat_score
    crease_pitch_avg = float(np.mean([x[2] for x in crease_profs])) if crease_profs else rem_bat_pitch_avg
    duel_rpo = float(np.mean(duel_rpos)) if duel_rpos else base_rr
    duel_wkt = float(np.mean(duel_wkts)) if duel_wkts else 0.04
    return {
        "crease_bat_score": crease_bat_score,
        "crease_pitch_avg": crease_pitch_avg,
        "rem_bat_score": rem_bat_score,
        "rem_bat_dep": rem_bat_dep,
        "rem_bat_pitch_avg": rem_bat_pitch_avg,
        "rem_bat_adapt_sc": rem_bat_adapt_sc,
        "opp_bowl_score": opp_bowl_score,
        "opp_bowl_spear": opp_bowl_spear,
        "opp_bowl_death": opp_bowl_death,
        "opp_bowl_pitch_econ": opp_bowl_pitch_econ,
        "duel_net_rpo": duel_rpo - base_rr,
        "duel_wkt": duel_wkt,
        "pitch_bat_edge": (rem_bat_pitch_avg / 30.0) - (opp_bowl_pitch_econ / base_rr),
        "wkts_x_rem_bat": (wickets_left / 10.0) * (rem_bat_score / 50.0),
    }


def _horizon_norm(prog: float, wickets_left: float) -> float:
    return float(max(0.22, math.sqrt(max(0.03, 1.0 - min(0.95, prog))) * ((max(1.0, wickets_left) / 10.0) ** 0.35)))


def extract_innings1_stage4_features(
    *,
    fmt: str,
    score: float,
    wickets: int,
    overs_f: float,
    p0_bat_dec: float,
    bat_team: str,
    bowl_team: str,
    venue_name: str = "",
    hist_venue_par: Optional[float] = None,
    live_pitch_type: str = "balanced",
    pitch_probs: Optional[Dict[str, float]] = None,
    bat_adapt: Optional[float] = None,
    bowl_adapt: Optional[float] = None,
    striker: Optional[Dict[str, Any]] = None,
    non_striker: Optional[Dict[str, Any]] = None,
    bowler: Optional[Dict[str, Any]] = None,
    dugout_players: Optional[List[Dict[str, Any]]] = None,
    bowlers_list: Optional[List[Dict[str, Any]]] = None,
    dew_factor: float = 0.0,
    is_day_night: bool = False,
    deliveries: Optional[List[Dict[str, Any]]] = None,
    boundary_avg_m: Optional[float] = None,
    s3_par_bat: Optional[float] = None,
    s3_par_opp: Optional[float] = None,
    s3_hist_par: Optional[float] = None,
    era_offset: float = 0.0,
) -> Dict[str, Any]:
    """Extracts unified live Stage 4 Innings-1 feature dict with Render + live ball telemetry."""
    f = normalize_stage4_fmt(fmt)
    canon_pitch = canonicalize_pitch_type(live_pitch_type)
    std_max_ov = 20.0 if f == "T20I" else (50.0 if f == "ODI" else 100.0)
    prog = float(max(0.01, min(0.95, overs_f / std_max_ov)))
    wickets_i = int(max(0, min(9, wickets)))
    wickets_left = float(10 - wickets_i)

    bat_c = normalize_team_code(bat_team)
    bowl_c = normalize_team_code(bowl_team)

    st_list = [p for p in (striker, non_striker) if isinstance(p, dict) and (p.get("name") or p.get("id"))]
    dug_list = [p for p in (dugout_players or []) if isinstance(p, dict) and p.get("is_out", 0) in (0, "0", False, None)]
    bw_list = [p for p in ([bowler] + list(bowlers_list or [])) if isinstance(p, dict) and (p.get("name") or p.get("id"))]

    bat_xi = (st_list + dug_list)[:11]
    bowl_xi = bw_list[:11]

    if s3_par_bat is None or s3_par_opp is None or s3_hist_par is None:
        r_bat, r_opp, r_hist = _resolve_stage3_pars(f, bat_c, bowl_c, venue_name, canon_pitch, bat_xi, bowl_xi)
        s3_par_bat = r_bat if s3_par_bat is None else float(s3_par_bat)
        s3_par_opp = r_opp if s3_par_opp is None else float(s3_par_opp)
        s3_hist_par = float(hist_venue_par) if (hist_venue_par and hist_venue_par > 0) else (r_hist if s3_hist_par is None else float(s3_hist_par))
    else:
        s3_par_bat = float(s3_par_bat)
        s3_par_opp = float(s3_par_opp)
        s3_hist_par = float(hist_venue_par) if (hist_venue_par and hist_venue_par > 0) else float(s3_hist_par)

    era_par = float(s3_par_bat + era_offset)
    rem_pct = compute_dls_rem_pct(f, overs_f, wickets_i)
    used_pct = max(0.06, 1.0 - rem_pct)
    sc_above_dls_used = float(score) - era_par * used_pct

    exp_wkts_cp = min(8.5, prog * (6.2 if f == "T20I" else (7.0 if f == "ODI" else 7.6)))
    wkts_saved = exp_wkts_cp - float(wickets_i)
    par_at_cp = s3_par_bat * (0.24 if f == "T20I" else (0.19 if f == "ODI" else 0.21)) * (prog / 0.20)
    score_above_par = float(score) - par_at_cp

    crr = float(score) / max(0.5, float(overs_f))
    crr_x_wkts_left = crr * (wickets_left / 10.0)
    telem = extract_live_ball_telemetry(deliveries, 1, crr, f)

    if bat_adapt is None:
        bat_adapt = get_country_pitch_adaptability(bat_c, f, canon_pitch)
    if bowl_adapt is None:
        bowl_adapt = get_country_pitch_adaptability(bowl_c, f, canon_pitch)
    adapt_diff = float(bat_adapt) - float(bowl_adapt)

    if pitch_probs:
        p_flat = float(pitch_probs.get("flat", 20.0))
        p_green = float(pitch_probs.get("green", 20.0))
        p_dusty = float(pitch_probs.get("dusty", 20.0))
        p_bouncy = float(pitch_probs.get("bouncy", 20.0))
    else:
        p_flat = 55.0 if canon_pitch == "flat" else 11.25
        p_green = 55.0 if canon_pitch == "green" else 11.25
        p_dusty = 55.0 if canon_pitch == "dusty" else 11.25
        p_bouncy = 55.0 if canon_pitch == "bouncy" else 11.25

    v_ctx = resolve_venue_dossier_context(venue_name or normalize_venue_key(venue_name), f) if venue_name else {"matched": False}
    b_avg_m = float(boundary_avg_m) if boundary_avg_m is not None else float(v_ctx.get("avg_m") or 68.0)
    eff_dew = float(dew_factor) if dew_factor > 0 else float(v_ctx.get("dew_factor") or 0.25)

    r_edges = _extract_render_squad_edges(
        f=f, canon_pitch=canon_pitch, wickets_i=wickets_i, wickets_left=wickets_left,
        st_list=st_list, dug_list=dug_list, bw_list=bw_list,
    )

    opp_rates = get_country_pitch_phase_rates(bowl_c, f, canon_pitch, opponent_code=bat_c)
    team_rates = get_country_pitch_phase_rates(bat_c, f, canon_pitch, opponent_code=bowl_c)
    rem_tank_edge = ((r_edges["rem_bat_score"] - r_edges["opp_bowl_score"]) + (opp_rates["par"] - team_rates["par"]) * (-0.15)) / 15.0

    p0_clamped = float(max(0.02, min(0.98, p0_bat_dec)))
    return {
        "fmt": f,
        "score": float(score),
        "wickets": float(wickets_i),
        "wickets_left": wickets_left,
        "prog": prog,
        "era_par": era_par,
        "s3_par_bat": s3_par_bat,
        "s3_par_opp": s3_par_opp,
        "s3_hist_par": s3_hist_par,
        "dls_rem_pct": rem_pct,
        "sc_above_dls_used": sc_above_dls_used,
        "wkts_saved": wkts_saved,
        "crr": crr,
        "crr_x_wkts_left": crr_x_wkts_left,
        "score_above_par": score_above_par,
        "recent_3ov_rpo": float(telem["recent_3ov_rpo"]),
        "rec3_minus_crr": float(telem["recent_3ov_rpo"] - crr),
        "dot_pct": float(telem["dot_pct"]),
        "bdry_pct": float(telem["bdry_pct"]),
        "has_ball_telem": float(telem["has_ball_telem"]),
        "p_flat": p_flat,
        "p_green": p_green,
        "p_dusty": p_dusty,
        "p_bouncy": p_bouncy,
        "adapt_diff": adapt_diff,
        **r_edges,
        "boundary_avg_m": b_avg_m,
        "dew_factor": eff_dew,
        "is_day_night": 1.0 if is_day_night else 0.0,
        "rem_tank_edge": rem_tank_edge,
        "p0_dec": p0_clamped,
        "p0_logit": float(sp_logit(p0_clamped)),
    }


def extract_innings2_chase_features(
    *,
    fmt: str,
    score: float,
    wickets: int,
    overs_f: float,
    target: float,
    inn1_score: float,
    p0_chase_dec: float,
    chase_team: str,
    defend_team: str,
    venue_name: str = "",
    live_pitch_type: str = "balanced",
    pitch_probs: Optional[Dict[str, float]] = None,
    chase_adapt: Optional[float] = None,
    defend_adapt: Optional[float] = None,
    striker: Optional[Dict[str, Any]] = None,
    non_striker: Optional[Dict[str, Any]] = None,
    bowler: Optional[Dict[str, Any]] = None,
    dugout_players: Optional[List[Dict[str, Any]]] = None,
    bowlers_list: Optional[List[Dict[str, Any]]] = None,
    is_day_night: bool = False,
    dew_factor: float = 0.25,
    dew_intelligence: Optional[Dict[str, Any]] = None,
    deliveries: Optional[List[Dict[str, Any]]] = None,
    scheduled_overs: Optional[float] = None,
    s3_par_chase: Optional[float] = None,
) -> Dict[str, Any]:
    """Extracts unified live Stage 4 Innings-2 Chase feature dict including Dew & DLS Chase Par."""
    f = normalize_stage4_fmt(fmt)
    canon_pitch = canonicalize_pitch_type(live_pitch_type)
    std_max_ov = float(scheduled_overs) if (scheduled_overs and float(scheduled_overs) > 0) else (20.0 if f == "T20I" else 50.0)
    prog = float(max(0.01, min(0.98, overs_f / max(1.0, std_max_ov))))
    wickets_i = int(max(0, min(9, wickets)))
    wickets_left = float(10 - wickets_i)
    chase_c = normalize_team_code(chase_team)
    defend_c = normalize_team_code(defend_team)

    st_list = [p for p in (striker, non_striker) if isinstance(p, dict) and (p.get("name") or p.get("id"))]
    dug_list = [p for p in (dugout_players or []) if isinstance(p, dict) and p.get("is_out", 0) in (0, "0", False, None)]
    bw_list = [p for p in ([bowler] + list(bowlers_list or [])) if isinstance(p, dict) and (p.get("name") or p.get("id"))]

    if s3_par_chase is None:
        r_chase, _, _ = _resolve_stage3_pars(f, chase_c, defend_c, venue_name, canon_pitch, (st_list + dug_list)[:11], bw_list[:11])
        s3_par_chase = r_chase
    else:
        s3_par_chase = float(s3_par_chase)

    eff_target = max(float(score) + 1.0, float(target)) if target > 0 else max(float(score) + 1.0, float(inn1_score) + 1.0)
    eff_inn1 = float(inn1_score) if inn1_score > 0 else max(1.0, eff_target - 1.0)
    runs_needed = max(0.0, eff_target - float(score))
    ov_rem = max(0.1, std_max_ov - overs_f)
    crr = float(score) / max(0.5, float(overs_f))
    rrr = runs_needed / ov_rem

    # Compute Stern DLS live chase par (interpolated across overs)
    if f == "T20I":
        scale = 50.0 / max(5.0, std_max_ov)
        dls_par = float(calculate_live_dls_par_score(
            first_innings_score=int(round(eff_inn1)),
            current_second_innings_overs=overs_f * scale,
            current_wickets_lost=wickets_i,
            total_scheduled_overs_team_2=50.0,
            first_innings_scheduled_overs=50.0,
            g50_par=165,
        ))
        dls_rem = float(interpolate_dls_resource(max(0.0, ov_rem * scale), wickets_i)) / 100.0
    else:
        dls_par = float(calculate_live_dls_par_score(
            first_innings_score=int(round(eff_inn1)),
            current_second_innings_overs=overs_f,
            current_wickets_lost=wickets_i,
            total_scheduled_overs_team_2=std_max_ov,
            first_innings_scheduled_overs=std_max_ov,
            g50_par=245,
        ))
        dls_rem = float(interpolate_dls_resource(max(0.0, ov_rem), wickets_i)) / 100.0

    telem = extract_live_ball_telemetry(deliveries, 2, crr, f)

    if chase_adapt is None:
        chase_adapt = get_country_pitch_adaptability(chase_c, f, canon_pitch)
    if defend_adapt is None:
        defend_adapt = get_country_pitch_adaptability(defend_c, f, canon_pitch)

    if pitch_probs:
        p_flat = float(pitch_probs.get("flat", 20.0))
        p_dusty = float(pitch_probs.get("dusty", 20.0))
        p_green = float(pitch_probs.get("green", 20.0))
    else:
        p_flat = 55.0 if canon_pitch == "flat" else 11.25
        p_dusty = 55.0 if canon_pitch == "dusty" else 11.25
        p_green = 55.0 if canon_pitch == "green" else 11.25

    v_key = normalize_venue_key(venue_name) if venue_name else ""
    ven_dew = LiveDewDetector.get_venue_dew_prior(venue_name or v_key)
    ven_dew_prior = float(ven_dew.get("calibrated_prior") or 0.25)

    tele_dict = (dew_intelligence or {}).get("telemetry", {}) if isinstance(dew_intelligence, dict) else {}
    if not tele_dict and deliveries:
        tele_dict = LiveDewDetector.evaluate_in_play_telemetry(deliveries=deliveries, match_format=f)
    dew_cal = float((dew_intelligence or {}).get("dew_factor_calibrated") or dew_factor or ven_dew_prior)

    r_edges = _extract_render_squad_edges(
        f=f, canon_pitch=canon_pitch, wickets_i=wickets_i, wickets_left=wickets_left,
        st_list=st_list, dug_list=dug_list, bw_list=bw_list,
    )

    p0_clamped = float(max(0.02, min(0.98, p0_chase_dec)))
    return {
        "fmt": f,
        "score": float(score),
        "target": eff_target,
        "runs_needed": runs_needed,
        "wickets": float(wickets_i),
        "wickets_left": wickets_left,
        "prog": prog,
        "dls_rem": dls_rem,
        "p0_dec": p0_clamped,
        "p0_logit": float(sp_logit(p0_clamped)),
        "sc_above_dls_par": float(score) - dls_par,
        "rrr_minus_crr": rrr - crr,
        "need_vs_dls_rem_par": (runs_needed / max(1.0, eff_inn1 * max(0.03, dls_rem))) - 1.0,
        "target_vs_par": (eff_target - s3_par_chase) / (20.0 if f == "T20I" else 35.0),
        "crr": crr,
        "rrr": rrr,
        "p_flat": p_flat,
        "p_dusty": p_dusty,
        "p_green": p_green,
        "adapt_diff": float(chase_adapt) - float(defend_adapt),
        **r_edges,
        "recent_3ov_rpo": float(telem["recent_3ov_rpo"]),
        "rec3_minus_rrr": float(telem["recent_3ov_rpo"] - rrr),
        "dot_pct": float(telem["dot_pct"]),
        "bdry_pct": float(telem["bdry_pct"]),
        "dew_cal": dew_cal,
        "is_dn_x_dew": (1.0 if is_day_night else 0.0) * dew_cal,
        "dew_tele": float(tele_dict.get("telemetry_dew_index") or 0.0),
        "extras_spike": float(tele_dict.get("extras_spike_ratio") or 1.0),
        "spin_rpo_delta": float(tele_dict.get("spin_rpo_delta") or 0.0),
        "ven_dew_prior": ven_dew_prior,
    }


def _eval_expert_pair(
    expert: Dict[str, Any],
    feats: Dict[str, Any],
    fmt: str,
) -> Tuple[float, float]:
    """Runs score & decisive win probability models for one Innings-1 phase expert."""
    x_full = np.array([[float(feats[n]) for n in SCORE_FEAT_NAMES]], dtype=float)
    x_f_imp = expert["imp_full"].transform(x_full)
    x_comp = x_f_imp[:, SCORE_COMPACT_IDX]
    x_c_sc = expert["sc_compact"].transform(x_comp)

    w_idx = int(max(0, min(9, round(feats["wickets"]))))
    if "rem_tbl" in expert:
        fracs = sorted(expert["rem_tbl"].keys())
        near_f = min(fracs, key=lambda fr: abs(fr - feats["prog"]))
        rem_frac = float(expert["rem_tbl"][near_f][w_idx])
    else:
        rem_frac = float(expert["rem_w"][w_idx])

    anchor = float(feats["score"]) + float(feats["era_par"]) * rem_frac
    p_huber = anchor + float(expert["huber"].predict(x_c_sc)[0])
    p_ridge = anchor + float(expert["ridge"].predict(x_c_sc)[0])
    p_xgb = anchor + float(expert["xgb_reg"].predict(x_f_imp)[0])
    p_et = anchor + float(expert["et_reg"].predict(x_f_imp)[0])

    w_s = expert["score_weights"]
    proj_sc = max(
        float(feats["score"]),
        w_s["huber"] * p_huber + w_s["ridge"] * p_ridge + w_s["xgb"] * p_xgb + w_s["et"] * p_et,
    )

    scale = 20.0 if fmt == "T20I" else (30.0 if fmt == "ODI" else 70.0)
    win_row = np.array([[
        float(feats["p0_logit"]),
        (proj_sc - float(feats["s3_par_opp"])) / scale,
        float(feats["sc_above_dls_used"]) / scale,
        float(feats["wkts_saved"]),
        float(feats["rem_tank_edge"]),
        float(feats["adapt_diff"]) / 15.0,
        float(feats["pitch_bat_edge"]),
        float(feats["duel_net_rpo"]) / 2.0,
        (float(feats["rem_bat_dep"]) - float(feats["opp_bowl_death"])) / 20.0,
        float(feats["prog"]),
        float(feats["rec3_minus_crr"]) / 2.5,
        float(feats["dot_pct"]),
        float(feats["bdry_pct"]),
        float(feats["dew_factor"]) * float(feats["is_day_night"]),
    ]], dtype=float)

    x_w_lr = np.column_stack([win_row[:, :1], expert["sc_win"].transform(win_row[:, 1:])])
    p_lr = float(np.clip(expert["lr_win"].predict_proba(x_w_lr)[0, 1], 0.02, 0.98))
    p_xgb_w = float(np.clip(expert["xgb_win"].predict_proba(win_row)[0, 1], 0.02, 0.98))

    w_win = expert["win_weights"]
    logit_p = (
        w_win["lr"] * float(sp_logit(p_lr))
        + w_win["xgb"] * float(sp_logit(p_xgb_w))
        + w_win["p0"] * float(feats["p0_logit"])
    )
    p_win = float(np.clip(expit(logit_p), 0.02, 0.98))
    return proj_sc, p_win


def predict_innings1_calibrated(
    *,
    fmt: str,
    score: float,
    wickets: int,
    overs_f: float,
    p0_bat_dec: float,
    bat_team: str,
    bowl_team: str,
    venue_name: str = "",
    hist_venue_par: Optional[float] = None,
    live_pitch_type: str = "balanced",
    bat_adapt: Optional[float] = None,
    bowl_adapt: Optional[float] = None,
    striker: Optional[Dict[str, Any]] = None,
    non_striker: Optional[Dict[str, Any]] = None,
    bowler: Optional[Dict[str, Any]] = None,
    dugout_players: Optional[List[Dict[str, Any]]] = None,
    bowlers_list: Optional[List[Dict[str, Any]]] = None,
    dew_factor: float = 0.0,
    is_day_night: bool = False,
    deliveries: Optional[List[Dict[str, Any]]] = None,
    boundary_avg_m: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Produces calibrated Innings-1 score projection, 90% conformal interval, and
    decisive batting-team win probability smoothly across all Stage 4 horizons.
    """
    bundle = load_stage4_calibrator()
    f = normalize_stage4_fmt(fmt)
    if not bundle or f not in bundle:
        return None
    fmt_bundle = bundle[f]
    if overs_f < 1.0 or wickets >= 10:
        return None

    feats = extract_innings1_stage4_features(
        fmt=f,
        score=score,
        wickets=wickets,
        overs_f=overs_f,
        p0_bat_dec=p0_bat_dec,
        bat_team=bat_team,
        bowl_team=bowl_team,
        venue_name=venue_name,
        hist_venue_par=hist_venue_par,
        live_pitch_type=live_pitch_type,
        bat_adapt=bat_adapt,
        bowl_adapt=bowl_adapt,
        striker=striker,
        non_striker=non_striker,
        bowler=bowler,
        dugout_players=dugout_players,
        bowlers_list=bowlers_list,
        dew_factor=dew_factor,
        is_day_night=is_day_night,
        deliveries=deliveries,
        boundary_avg_m=boundary_avg_m,
        era_offset=float(fmt_bundle.get("era_offset", 0.0)),
    )

    prog = float(feats["prog"])
    if prog <= 0.24:
        w_early = 1.0
    elif prog >= 0.36:
        w_early = 0.0
    else:
        w_early = (0.36 - prog) / 0.12

    sc_e, p_e = _eval_expert_pair(fmt_bundle["early"], feats, f)
    sc_m, p_m = _eval_expert_pair(fmt_bundle["mid_late"], feats, f)

    proj_score = max(float(score), w_early * sc_e + (1.0 - w_early) * sc_m)
    p_bat_win = float(np.clip(expit(w_early * float(sp_logit(p_e)) + (1.0 - w_early) * float(sp_logit(p_m))), 0.02, 0.98))

    h_norm = _horizon_norm(prog, float(feats["wickets_left"]))
    q_lo_e, q_hi_e = fmt_bundle["early"]["conf_q90_norm"]
    q_lo_m, q_hi_m = fmt_bundle["mid_late"]["conf_q90_norm"]
    q_lo = w_early * q_lo_e + (1.0 - w_early) * q_lo_m
    q_hi = w_early * q_hi_e + (1.0 - w_early) * q_hi_m
    proj_low = max(int(score), int(round(proj_score + q_lo * h_norm)))
    proj_high = max(proj_low + 8, int(round(proj_score + q_hi * h_norm)))

    fade_prog = max(0.0, min(1.0, 1.0 - max(0.0, prog - 0.50) / 0.32))
    return {
        "proj_score": int(round(proj_score)),
        "proj_low": proj_low,
        "proj_high": proj_high,
        "p_bat_win": p_bat_win,
        "fade_weight": fade_prog,
    }


def predict_innings2_chase_calibrated(
    *,
    fmt: str,
    score: float,
    wickets: int,
    overs_f: float,
    target: float,
    inn1_score: float,
    p0_chase_dec: float,
    chase_team: str,
    defend_team: str,
    venue_name: str = "",
    live_pitch_type: str = "balanced",
    chase_adapt: Optional[float] = None,
    defend_adapt: Optional[float] = None,
    striker: Optional[Dict[str, Any]] = None,
    non_striker: Optional[Dict[str, Any]] = None,
    bowler: Optional[Dict[str, Any]] = None,
    dugout_players: Optional[List[Dict[str, Any]]] = None,
    bowlers_list: Optional[List[Dict[str, Any]]] = None,
    is_day_night: bool = False,
    dew_factor: float = 0.25,
    dew_intelligence: Optional[Dict[str, Any]] = None,
    deliveries: Optional[List[Dict[str, Any]]] = None,
    scheduled_overs: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Produces calibrated Limited-Overs Innings-2 Chase probability using DLS live chase par,
    required vs current run-rate & 3-over pace, dew telemetry, and Render crew depth.
    """
    bundle = load_stage4_calibrator()
    f = normalize_stage4_fmt(fmt)
    if f not in ("T20I", "ODI") or not bundle or f not in bundle:
        return None
    chase_exp = bundle[f].get("chase")
    if not chase_exp:
        return None
    if overs_f < 1.0 or wickets >= 10 or score >= target:
        return None

    feats = extract_innings2_chase_features(
        fmt=f,
        score=score,
        wickets=wickets,
        overs_f=overs_f,
        target=target,
        inn1_score=inn1_score,
        p0_chase_dec=p0_chase_dec,
        chase_team=chase_team,
        defend_team=defend_team,
        venue_name=venue_name,
        live_pitch_type=live_pitch_type,
        chase_adapt=chase_adapt,
        defend_adapt=defend_adapt,
        striker=striker,
        non_striker=non_striker,
        bowler=bowler,
        dugout_players=dugout_players,
        bowlers_list=bowlers_list,
        is_day_night=is_day_night,
        dew_factor=dew_factor,
        dew_intelligence=dew_intelligence,
        deliveries=deliveries,
        scheduled_overs=scheduled_overs,
    )

    x_raw = np.array([[float(feats[n]) for n in CHASE_FEAT_NAMES]], dtype=float)
    x_imp = chase_exp["imp"].transform(x_raw)
    x_sc = np.column_stack([x_imp[:, :1], chase_exp["sc"].transform(x_imp[:, 1:])])

    p_lr = float(np.clip(chase_exp["lr_win"].predict_proba(x_sc)[0, 1], 0.01, 0.99))
    p_xgb = float(np.clip(chase_exp["xgb_win"].predict_proba(x_imp)[0, 1], 0.01, 0.99))
    w_c = chase_exp["win_weights"]
    logit_p = w_c["lr"] * float(sp_logit(p_lr)) + w_c["xgb"] * float(sp_logit(p_xgb)) + w_c["p0"] * float(feats["p0_logit"])
    p_chase_win = float(np.clip(expit(logit_p), 0.01, 0.99))

    std_max_ov = float(scheduled_overs) if (scheduled_overs and float(scheduled_overs) > 0) else (20.0 if f == "T20I" else 50.0)
    balls_rem = max(0.0, (std_max_ov - overs_f) * 6.0)
    if balls_rem >= 12.0:
        fade_weight = 0.88
    elif balls_rem <= 2.0:
        fade_weight = 0.0
    else:
        fade_weight = 0.88 * ((balls_rem - 2.0) / 10.0)

    return {
        "p_chase_win": p_chase_win,
        "fade_weight": fade_weight,
        "sc_above_dls_par": float(feats["sc_above_dls_par"]),
    }
