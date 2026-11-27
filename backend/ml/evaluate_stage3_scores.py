"""
Leakage-Free Calibration & Out-of-Sample Accuracy Evaluation for Stage 3 (ODI & T20I)
===================================================================================
- Calibrates ground, pitch-type, country, player-adapt fallback, and L2 Ridge weights
  strictly on 2018-01-01 to 2024-12-31 Cricsheet Senior Men's Internationals
  (using 5-fold out-of-fold CV inside the training window so Ridge learns true generalization).
- Evaluates Stage 3 pre-match projected scores against actual final 1st-innings scores
  on strictly out-of-sample 2025-2026 matches (plus post-2026-10-25 holdout slice).
"""

from __future__ import annotations

import json
import math
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from backend.engine.cricsheet_pitch_intelligence import (
    FORMAT_GROUND_PITCH_DOSSIERS,
    canonicalize_pitch_type,
    get_ground_pitch_archetype,
    normalize_team_code,
)
from backend.engine.pitch_history_model import get_historical_venue_profile
from backend.engine.render_analytics_bridge import _load_players_cache
from backend.engine.stage3_ground_pitch_model import (
    CALIBRATED_PRIORS_PATH,
    CANONICAL_PITCHES,
    TEAM_NAME_TO_CODE,
    compute_stage3_xi_score_features,
    infer_host_country_code,
    invalidate_stage3_priors_cache,
    normalize_city_key,
    normalize_venue_key,
    predict_stage3_calibrated_score,
)
from backend.engine.stage3_xi_locked import calculate_stage3_xi_locked

DEFAULT_CRICSHEET_ZIP_DIR = Path(
    r"C:\Users\bhash\.gemini\antigravity\brain\7500a1f7-932d-4128-87dd-5d7a38d31d98\scratch\cricsheet_eval"
)
SENIOR_TEAMS = {
    "India", "Australia", "England", "South Africa", "New Zealand",
    "Pakistan", "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan",
    "Zimbabwe", "Ireland", "Netherlands", "Scotland",
}


_CALIBRATION_CFG: Dict[str, Dict[str, Any]] = {
    "ODI": {
        "half_life": 900.0,
        "overall_default": 268.0,
        "lam_t": 28.0,
        "lam_v": 8.0,
        "lam_home": 60.0,
        "l2_val": 600.0,
        "huber_delta": 55.0,
        "tol_a": 30.0,
        "tol_b": 45.0,
        "default_band": 45,
    },
    "T20I": {
        "half_life": 420.0,
        "overall_default": 168.0,
        "lam_t": 8.0,
        "lam_v": 4.0,
        "lam_home": 60.0,
        "l2_val": 260.0,
        "huber_delta": None,
        "tol_a": 20.0,
        "tol_b": 30.0,
        "default_band": 28,
    },
    "TEST": {
        "half_life": 1100.0,
        "overall_default": 318.0,
        "lam_t": 20.0,
        "lam_v": 8.0,
        "lam_home": 50.0,
        "l2_val": 450.0,
        "huber_delta": 85.0,
        "tol_a": 50.0,
        "tol_b": 80.0,
        "default_band": 95,
    },
}


def classify_match_pitch_telemetry(match_json: Dict[str, Any], fmt: str, players_cache: Dict[str, Any]) -> str:
    """Classifies historical match surface into one of the 5 canonical archetypes from match telemetry."""
    innings = match_json.get("innings", [])
    if not innings:
        return "balanced"
    pace_balls = spin_balls = pace_runs = spin_runs = pace_wkts = spin_wkts = total_wkts = 0
    total_runs = total_balls = 0
    reg = match_json.get("info", {}).get("registry", {}).get("people", {})
    for inn in innings[:2]:
        for ov in inn.get("overs", []):
            for d in ov.get("deliveries", []):
                r = d.get("runs", {}).get("total", 0)
                total_runs += r
                et = d.get("extras", {})
                if "wides" not in et and "noballs" not in et:
                    total_balls += 1
                bw_id = reg.get(d.get("bowler", ""), "")
                b_style = str(players_cache.get(bw_id, {}).get("bowling_style") or "").lower()
                is_spin = any(k in b_style for k in ("spin", "offbreak", "legbreak", "orthodox", "chinaman", "wrist", "slow"))
                if is_spin:
                    spin_balls += 1
                    spin_runs += r
                else:
                    pace_balls += 1
                    pace_runs += r
                for _ in d.get("wickets", []):
                    total_wkts += 1
                    if is_spin:
                        spin_wkts += 1
                    else:
                        pace_wkts += 1

    match_rpo = total_runs / max(1.0, total_balls / 6.0)
    spin_ball_share = spin_balls / max(1, spin_balls + pace_balls)
    spin_wkt_share = spin_wkts / max(1, total_wkts)
    spin_econ = spin_runs / max(1.0, spin_balls / 6.0) if spin_balls >= 12 else 99.0
    pace_econ = pace_runs / max(1.0, pace_balls / 6.0) if pace_balls >= 12 else 99.0

    if fmt == "TEST":
        if match_rpo >= 3.67 or (total_wkts <= 18 and match_rpo >= 3.45):
            return "flat"
        if spin_wkt_share >= 0.36 or (spin_ball_share >= 0.34 and spin_econ < pace_econ):
            return "dusty"
        if match_rpo <= 2.95 and pace_wkts >= max(10, int(total_wkts * 0.65)):
            return "green"
        if (pace_wkts / max(1, total_wkts)) >= 0.72 and match_rpo > 2.95:
            return "bouncy"
        return "balanced"

    flat_rpo = 6.15 if fmt == "ODI" else 8.95
    low_rpo = 4.65 if fmt == "ODI" else 7.15
    if match_rpo >= flat_rpo or (total_wkts <= 11 and match_rpo >= (flat_rpo - 0.35)):
        return "flat"
    if spin_wkt_share >= 0.48 or (spin_ball_share >= 0.44 and spin_econ < pace_econ):
        return "dusty"
    if match_rpo <= low_rpo and pace_wkts >= max(8, int(total_wkts * 0.65)):
        return "green"
    if (pace_wkts / max(1, total_wkts)) >= 0.68 and match_rpo >= (low_rpo + 0.15):
        return "bouncy"
    return "balanced"


def extract_clean_matches_from_zip(zip_path: Path, fmt_label: str) -> List[Dict[str, Any]]:
    """Parses senior men's matches with full clean 1st innings (no rain washout/curtailment)."""
    render_fmt = "TEST" if fmt_label == "TEST" else ("ODI" if fmt_label == "ODI" else "T20")
    players_cache = _load_players_cache(render_fmt)
    rows: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for fn in sorted(z.namelist()):
            if not fn.endswith(".json"):
                continue
            d = json.loads(z.read(fn))
            info = d.get("info", {})
            teams = info.get("teams", [])
            dt = (info.get("dates") or [""])[0]
            if info.get("gender") != "male" or len(teams) != 2 or not all(t in SENIOR_TEAMS for t in teams):
                continue
            if dt < "2018-01-01":
                continue
            outcome = info.get("outcome", {})
            if outcome.get("result") in ("no result", "abandoned"):
                continue
            innings = d.get("innings", [])
            if not innings:
                continue

            inn0 = innings[0]
            overs_list = inn0.get("overs", [])
            if not overs_list:
                continue

            runs1 = 0
            wkts1 = 0
            bat_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])  # runs, balls, outs
            for ov in overs_list:
                for dl in ov.get("deliveries", []):
                    runs1 += dl.get("runs", {}).get("total", 0)
                    bat_n = dl.get("batter")
                    et = dl.get("extras", {})
                    if bat_n:
                        bat_stats[bat_n][0] += dl.get("runs", {}).get("batter", 0)
                        if "wides" not in et:
                            bat_stats[bat_n][1] += 1
                    for w in dl.get("wickets", []):
                        wkts1 += 1
                        po = w.get("player_out")
                        if po:
                            bat_stats[po][2] += 1

            max_ov1 = overs_list[-1].get("over", 0) + 1
            if fmt_label == "TEST":
                if len(innings) < 2 and wkts1 < 10 and not inn0.get("declared"):
                    continue
            else:
                min_full = 45 if fmt_label == "ODI" else 18
                if max_ov1 < min_full and wkts1 < 10:
                    continue

            bat_team_name = inn0.get("team") or teams[0]
            bowl_team_name = teams[1] if bat_team_name == teams[0] else teams[0]
            bat_code = TEAM_NAME_TO_CODE.get(bat_team_name, normalize_team_code(bat_team_name))
            bowl_code = TEAM_NAME_TO_CODE.get(bowl_team_name, normalize_team_code(bowl_team_name))

            toss = info.get("toss", {})
            toss_winner_name = toss.get("winner") or bat_team_name
            toss_dec = toss.get("decision") or "bat"
            toss_winner_code = TEAM_NAME_TO_CODE.get(toss_winner_name, normalize_team_code(toss_winner_name))

            reg = info.get("registry", {}).get("people", {})
            players_by_team = info.get("players", {})

            def _build_xi(t_name: str, t_code: str) -> List[Dict[str, Any]]:
                out_list = []
                for idx, pname in enumerate(players_by_team.get(t_name, [])[:11]):
                    cid = reg.get(pname, "")
                    r_rec = players_cache.get(cid, {})
                    out_list.append({
                        "id": cid or pname.lower().replace(" ", "_"),
                        "cricsheet_id": cid,
                        "render_id": cid,
                        "name": pname,
                        "country": t_name,
                        "team_id": t_code,
                        "batting_order": idx + 1,
                        "is_confirmed_xi": 1,
                        "role": str(r_rec.get("player_role") or ("top_order_batter" if idx < 5 else "all_rounder")),
                        "bowling_style": r_rec.get("bowling_style") or "RF",
                        "batting_hand": "left" if "left" in str(r_rec.get("batting_style") or "").lower() else "right",
                    })
                return out_list

            # Also summarize per-player batting/bowling contributions across BOTH innings for train-only fallback stats
            player_contrib_bat: Dict[str, Tuple[int, int, int]] = {}
            player_contrib_bowl: Dict[str, Tuple[int, int, int]] = defaultdict(lambda: [0, 0, 0])  # runs, balls, wkts
            for inn in innings[:2]:
                for ov in inn.get("overs", []):
                    for dl in ov.get("deliveries", []):
                        bn = dl.get("batter")
                        bwn = dl.get("bowler")
                        et = dl.get("extras", {})
                        if bn:
                            pid_b = reg.get(bn, "")
                            if pid_b:
                                cur = player_contrib_bat.get(pid_b, [0, 0, 0])
                                cur[0] += dl.get("runs", {}).get("batter", 0)
                                if "wides" not in et:
                                    cur[1] += 1
                                player_contrib_bat[pid_b] = cur
                        if bwn:
                            pid_bw = reg.get(bwn, "")
                            if pid_bw:
                                cur_bw = player_contrib_bowl[pid_bw]
                                cur_bw[0] += dl.get("runs", {}).get("total", 0)
                                if "wides" not in et and "noballs" not in et:
                                    cur_bw[1] += 1
                        for w in dl.get("wickets", []):
                            po = w.get("player_out")
                            if po and reg.get(po):
                                cur_o = player_contrib_bat.get(reg[po], [0, 0, 0])
                                cur_o[2] += 1
                                player_contrib_bat[reg[po]] = cur_o
                            if bwn and reg.get(bwn) and w.get("kind") not in ("run out", "retired hurt", "retired out", "obstructing the field"):
                                player_contrib_bowl[reg[bwn]][2] += 1

            p_type = classify_match_pitch_telemetry(d, fmt_label, players_cache)
            win_raw = outcome.get("winner")
            win_code = TEAM_NAME_TO_CODE.get(win_raw, normalize_team_code(win_raw)) if win_raw else None
            is_draw = outcome.get("result") == "draw"
            rows.append({
                "match_id": fn.replace(".json", ""),
                "format": fmt_label,
                "date": dt,
                "venue": info.get("venue", ""),
                "city": info.get("city", ""),
                "venue_key": normalize_venue_key(info.get("venue", "")),
                "city_key": normalize_city_key(info.get("city", "")),
                "bat_code": bat_code,
                "bowl_code": bowl_code,
                "toss_winner_code": toss_winner_code,
                "toss_decision": toss_dec,
                "toss_winner_bats": (toss_winner_code == bat_code),
                "winner_code": win_code,
                "is_draw": is_draw,
                "bat_xi": _build_xi(bat_team_name, bat_code),
                "bowl_xi": _build_xi(bowl_team_name, bowl_code),
                "actual_score": int(runs1),
                "actual_wkts": int(wkts1),
                "pitch_type": p_type,
                "bat_contrib": player_contrib_bat,
                "bowl_contrib": {k: list(v) for k, v in player_contrib_bowl.items()},
            })

    rows.sort(key=lambda r: r["date"])
    return rows


def _recency_weight(dt_str: str, ref_dt_str: str, half_life_days: float = 900.0) -> float:
    try:
        d0 = datetime.strptime(dt_str[:10], "%Y-%m-%d")
        d_ref = datetime.strptime(ref_dt_str[:10], "%Y-%m-%d")
        days = max(0.0, (d_ref - d0).days)
        return math.exp(-(math.log(2.0) / half_life_days) * days)
    except Exception:
        return 1.0


def build_empirical_priors_from_train_rows(fmt: str, train_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Builds recency-weighted ground/pitch, mixed-effects team/venue/home, and player priors strictly from `train_rows`."""
    cfg = _CALIBRATION_CFG[fmt]
    half_life = cfg["half_life"]
    ref_date = max((r["date"] for r in train_rows), default="2024-12-31")
    w_all = [_recency_weight(r["date"], ref_date, half_life_days=half_life) for r in train_rows]
    w_arr = np.array(w_all, dtype=float)
    ys = np.array([r["actual_score"] for r in train_rows], dtype=float)
    overall_mean = float(np.average(ys, weights=w_arr)) if len(ys) > 0 else cfg["overall_default"]

    by_pitch: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    by_venue: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    by_venue_pitch: Dict[str, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    by_city: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    by_city_pitch: Dict[str, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    by_host: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    by_month: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    bat_country_pitch: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    bowl_country_pitch: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    p_bat_stats: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    p_bat_pitch: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    p_bowl_stats: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    p_bowl_pitch: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))

    for r, wt in zip(train_rows, w_all):
        pt = r["pitch_type"]
        sc = r["actual_score"]
        by_pitch[pt].append((sc, wt))
        if r["venue_key"]:
            by_venue[r["venue_key"]].append((sc, wt))
            by_venue_pitch[r["venue_key"]][pt].append((sc, wt))
        if r["city_key"]:
            by_city[r["city_key"]].append((sc, wt))
            by_city_pitch[r["city_key"]][pt].append((sc, wt))
        hc = infer_host_country_code(r["venue_key"], r["city_key"], r["venue"])
        if hc:
            by_host[hc].append((float(sc), wt))
        mo = r["date"][5:7]
        if len(mo) == 2:
            by_month[mo].append((float(sc), wt))

        for pid, vals in r["bat_contrib"].items():
            for tgt in (p_bat_stats[pid], p_bat_pitch[pid][pt]):
                tgt[0] += vals[0] * wt
                tgt[1] += vals[1] * wt
                tgt[2] += vals[2] * wt
        for pid, vals in r["bowl_contrib"].items():
            for tgt in (p_bowl_stats[pid], p_bowl_pitch[pid][pt]):
                tgt[0] += vals[0] * wt
                tgt[1] += vals[1] * wt
                tgt[2] += vals[2] * wt

    def _wmean(pairs: List[Tuple[float, float]], default_val: float) -> float:
        if not pairs:
            return default_val
        ws = sum(w for _, w in pairs)
        if ws <= 0:
            return default_val
        return sum(v * w for v, w in pairs) / ws

    tot_w = max(1e-6, sum(w_all))
    global_pitch_pars = {pt: round(_wmean(by_pitch[pt], overall_mean), 2) for pt in CANONICAL_PITCHES}
    global_pitch_probs = {pt: round(sum(w for _, w in by_pitch[pt]) / tot_w, 4) for pt in CANONICAL_PITCHES}

    # Joint mixed-effects solver separating team batting strength, bowling suppression, venue, & home edge
    bat_teams = sorted({r["bat_code"] for r in train_rows})
    bowl_teams = sorted({r["bowl_code"] for r in train_rows})
    venues_list = sorted({r["venue_key"] for r in train_rows if r["venue_key"]})
    bat_idx = {c: i for i, c in enumerate(bat_teams)}
    bowl_idx = {c: i for i, c in enumerate(bowl_teams)}
    ven_idx = {v: i for i, v in enumerate(venues_list)}
    nb, nbw, nv = len(bat_teams), len(bowl_teams), len(venues_list)
    X_me = np.zeros((len(train_rows), nb + nbw + nv + 2), dtype=float)
    for row_i, r in enumerate(train_rows):
        X_me[row_i, bat_idx[r["bat_code"]]] = 1.0
        X_me[row_i, nb + bowl_idx[r["bowl_code"]]] = 1.0
        if r["venue_key"] in ven_idx:
            X_me[row_i, nb + nbw + ven_idx[r["venue_key"]]] = 1.0
        hc = infer_host_country_code(r["venue_key"], r["city_key"], r["venue"])
        if hc:
            X_me[row_i, -2] = 1.0 if r["bat_code"] == hc else 0.0
            X_me[row_i, -1] = 1.0 if r["bowl_code"] == hc else 0.0

    lam_t = cfg["lam_t"]
    lam_v = cfg["lam_v"]
    lam_vec = np.array([lam_t] * nb + [lam_t] * nbw + [lam_v] * nv + [cfg["lam_home"], cfg["lam_home"]], dtype=float)
    Xw = X_me * np.sqrt(w_arr)[:, None]
    beta = np.linalg.solve(Xw.T @ Xw + np.diag(lam_vec), Xw.T @ ((ys - overall_mean) * np.sqrt(w_arr)))
    me_bat = {c: round(float(beta[i]), 3) for i, c in enumerate(bat_teams)}
    me_bowl = {c: round(float(beta[nb + i]), 3) for i, c in enumerate(bowl_teams)}
    me_venue = {v: round(float(beta[nb + nbw + i]), 3) for i, v in enumerate(venues_list)}
    me_home = {"bat_home": round(float(beta[-2]), 3), "bowl_home": round(float(beta[-1]), 3)}

    for r, wt in zip(train_rows, w_all):
        pt = r["pitch_type"]
        resid = float(r["actual_score"]) - global_pitch_pars[pt]
        bat_country_pitch[r["bat_code"]][pt].append((resid, wt))
        bowl_country_pitch[r["bowl_code"]][pt].append((resid, wt))

    def _pack_loc(
        loc_scores: Dict[str, List[Tuple[int, float]]],
        loc_pitch: Dict[str, Dict[str, List[Tuple[int, float]]]],
    ) -> Dict[str, Any]:
        packed = {}
        for k, pairs in loc_scores.items():
            sw = sum(w for _, w in pairs)
            p_counts = {pt: len(loc_pitch[k][pt]) for pt in CANONICAL_PITCHES if loc_pitch[k][pt]}
            p_w = {pt: round(sum(w for _, w in loc_pitch[k][pt]), 2) for pt in CANONICAL_PITCHES if loc_pitch[k][pt]}
            p_means = {
                pt: round(_wmean(loc_pitch[k][pt], overall_mean), 2)
                for pt in CANONICAL_PITCHES
                if loc_pitch[k][pt]
            }
            packed[k] = {
                "n": len(pairs),
                "w_eff": round(sw, 2),
                "mean": round(_wmean(pairs, overall_mean), 2),
                "pitch_counts": p_counts,
                "pitch_w": p_w,
                "pitch_means": p_means,
            }
        return packed

    shrunk_bat_pitch = {
        c: {pt: round((sum(w for _, w in prs) / (sum(w for _, w in prs) + 8.0)) * _wmean(prs, 0.0), 2) for pt, prs in pm.items()}
        for c, pm in bat_country_pitch.items()
    }
    shrunk_bowl_pitch = {
        c: {pt: round((sum(w for _, w in prs) / (sum(w for _, w in prs) + 8.0)) * _wmean(prs, 0.0), 2) for pt, prs in pm.items()}
        for c, pm in bowl_country_pitch.items()
    }
    raw_host = {
        hc: ((sum(w for _, w in prs) / (sum(w for _, w in prs) + 6.0)) * (_wmean(prs, overall_mean) - overall_mean) + overall_mean)
        for hc, prs in by_host.items()
    }
    host_means = {hc: round(0.45 * hv + 0.55 * overall_mean, 2) for hc, hv in raw_host.items()}
    month_means = {
        m: round((sum(w for _, w in prs) / (sum(w for _, w in prs) + 14.0)) * (_wmean(prs, overall_mean) - overall_mean), 2)
        for m, prs in by_month.items()
    }

    player_bat_adapt = {}
    for pid, vals in p_bat_stats.items():
        if vals[1] >= 12.0:
            player_bat_adapt[pid] = {
                "balls": round(vals[1], 1),
                "avg": round(vals[0] / max(1.0, vals[2]), 2),
                "sr": round((vals[0] * 100.0) / max(1.0, vals[1]), 2),
                "by_pitch": {
                    pt: {
                        "balls": round(pv[1], 1),
                        "avg": round(pv[0] / max(1.0, pv[2]), 2),
                        "sr": round((pv[0] * 100.0) / max(1.0, pv[1]), 2),
                    }
                    for pt, pv in p_bat_pitch[pid].items()
                    if pv[1] >= 10.0
                },
            }

    player_bowl_adapt = {}
    for pid, vals in p_bowl_stats.items():
        if vals[1] >= 18.0:
            player_bowl_adapt[pid] = {
                "balls": round(vals[1], 1),
                "econ": round((vals[0] * 6.0) / max(1.0, vals[1]), 2),
                "wkt_rate": round((vals[2] * 24.0) / max(1.0, vals[1]), 2),
                "by_pitch": {
                    pt: {
                        "balls": round(pv[1], 1),
                        "econ": round((pv[0] * 6.0) / max(1.0, pv[1]), 2),
                    }
                    for pt, pv in p_bowl_pitch[pid].items()
                    if pv[1] >= 12.0
                },
            }

    # Chronological MOV-Elo, Form EMA, Venue Chase Bias & H2H strictly over `train_rows`
    home_elo_boost = 45.0 if fmt == "T20I" else (55.0 if fmt == "ODI" else 70.0)
    k_elo_base = 28.0 if fmt == "T20I" else 24.0
    default_bat1_win = 0.48 if fmt == "T20I" else (0.49 if fmt == "ODI" else 0.52)
    elo_map: Dict[str, float] = defaultdict(lambda: 1500.0)
    form_pairs: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    ven_bat1_pairs: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    h2h_pairs: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r, wt in sorted(zip(train_rows, w_all), key=lambda x: (x[0]["date"], x[0]["match_id"])):
        b_c, bw_c = r["bat_code"], r["bowl_code"]
        hc = infer_host_country_code(r["venue_key"], r["city_key"], r["venue"])
        eb = elo_map[b_c] + (home_elo_boost if hc and b_c == hc else 0.0)
        ebw = elo_map[bw_c] + (home_elo_boost if hc and bw_c == hc else 0.0)
        p_exp = 1.0 / (1.0 + math.pow(10.0, -(eb - ebw) / 400.0))
        wc = r.get("winner_code")
        if wc in (b_c, bw_c):
            res_b = 1.0 if wc == b_c else 0.0
            res_bw = 1.0 - res_b
            if r["venue_key"]:
                ven_bat1_pairs[r["venue_key"]].append((res_b, wt))
        else:
            res_b, res_bw = 0.5, 0.5
        k_eff = k_elo_base * (2.2 / (abs(eb - ebw) * 0.001 + 2.2))
        d_elo = k_eff * (res_b - p_exp)
        elo_map[b_c] += d_elo
        elo_map[bw_c] -= d_elo
        form_pairs[b_c].append((res_b, wt))
        form_pairs[bw_c].append((res_bw, wt))
        pk = "_".join(sorted((b_c, bw_c)))
        h2h_pairs[pk].append((1.0 if wc == pk.split("_")[0] else (0.0 if wc == pk.split("_")[1] else 0.5), wt))

    team_elo = {c: round(float(v), 1) for c, v in elo_map.items()}
    team_form = {
        c: round((sum(w for _, w in prs) * _wmean(prs, 0.5) + 4.0 * 0.5) / (sum(w for _, w in prs) + 4.0), 4)
        for c, prs in form_pairs.items()
    }
    venue_bat1_win_pct = {
        v: round((sum(w for _, w in prs) * _wmean(prs, default_bat1_win) + 8.0 * default_bat1_win) / (sum(w for _, w in prs) + 8.0), 4)
        for v, prs in ven_bat1_pairs.items()
    }
    h2h_win_pct = {
        k: round((sum(w for _, w in prs) * _wmean(prs, 0.5) + 3.0 * 0.5) / (sum(w for _, w in prs) + 3.0), 4)
        for k, prs in h2h_pairs.items()
    }

    return {
        "overall_mean": round(overall_mean, 2),
        "global_pitch_pars": global_pitch_pars,
        "global_pitch_probs": global_pitch_probs,
        "venues": _pack_loc(by_venue, by_venue_pitch),
        "cities": _pack_loc(by_city, by_city_pitch),
        "host_means": host_means,
        "month_means": month_means,
        "me_bat": me_bat,
        "me_bowl": me_bowl,
        "me_venue": me_venue,
        "me_home": me_home,
        "country_bat_pitch_residuals": shrunk_bat_pitch,
        "country_bowl_pitch_residuals": shrunk_bowl_pitch,
        "player_bat_adapt": player_bat_adapt,
        "player_bowl_adapt": player_bowl_adapt,
        "team_elo": team_elo,
        "team_form": team_form,
        "venue_bat1_win_pct": venue_bat1_win_pct,
        "h2h_win_pct": h2h_win_pct,
    }


def _extract_feature_vector(feat: Dict[str, Any]) -> np.ndarray:
    return np.array([
        feat["same_ground_pitch_par"],
        feat["global_pitch_par"],
        feat["ground_overall_par"],
        feat.get("me_venue_par", feat["ground_overall_par"]),
        feat.get("me_bat_rating", 0.0),
        feat.get("me_bowl_rating", 0.0),
        feat.get("home_delta", 0.0),
        feat["bat_country_adapt"] - feat["bowl_country_adapt"],
        feat["emp_bat_country_residual"] - feat["emp_bowl_country_residual"],
        feat["xi_bat_adapt"],
        feat["xi_bowl_adapt"],
        feat["xi_bat_strength"] - feat["xi_bowl_strength"],
        feat.get("xi_bat_index", 0.0),
        feat.get("xi_bowl_index", 0.0),
        feat["duel_run_delta"],
        feat.get("month_eff", 0.0),
        feat["toss_winner_bats"],
    ], dtype=float)


def fit_ridge_stage3_model(
    fmt: str,
    train_rows: List[Dict[str, Any]],
    l2_alpha: float = 250.0,
) -> Dict[str, Any]:
    """
    Fits softmax-convex, sign-constrained, recency-weighted Ridge weights using 5-fold out-of-fold priors
    strictly inside `train_rows`.
    """
    cfg = _CALIBRATION_CFG[fmt]
    half_life = cfg["half_life"]
    l2_val = cfg["l2_val"]
    huber_delta = cfg["huber_delta"]

    n = len(train_rows)
    n_folds = 5
    fold_idx = np.arange(n) % n_folds
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    d_list: List[str] = []
    w_list: List[float] = []
    ref_date = max((r["date"] for r in train_rows), default="2024-12-31")

    for f_id in range(n_folds):
        sub_train = [train_rows[i] for i in range(n) if fold_idx[i] != f_id]
        sub_val = [train_rows[i] for i in range(n) if fold_idx[i] == f_id]
        fold_priors = {fmt: build_empirical_priors_from_train_rows(fmt, sub_train)}
        for r in sub_val:
            feat = compute_stage3_xi_score_features(
                fmt=fmt,
                bat_team_id=r["bat_code"],
                bowl_team_id=r["bowl_code"],
                bat_xi=r["bat_xi"],
                bowl_xi=r["bowl_xi"],
                venue_name=r["venue"],
                city=r["city"],
                toss_bat_first_known=True,
                toss_winner_bats=r["toss_winner_bats"],
                pitch_override=None,
                priors=fold_priors,
                match_date=r["date"],
            )
            X_list.append(_extract_feature_vector(feat))
            y_list.append(float(r["actual_score"]))
            d_list.append(r["date"])
            w_list.append(_recency_weight(r["date"], ref_date, half_life_days=half_life))

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    sample_w = np.array(w_list, dtype=float)
    sample_w = sample_w / np.mean(sample_w)

    X_base = X[:, :4]
    adj_cols = X[:, 4:]
    adj_means = np.average(adj_cols, axis=0, weights=sample_w)
    adj_stds = np.maximum(0.5, np.std(adj_cols, axis=0))
    X_adj_centered = adj_cols - adj_means

    prior_base = np.array([0.35, 0.25, 0.15, 0.25])
    x0 = np.concatenate([[0.0], np.zeros(4), np.zeros(adj_cols.shape[1])])
    bounds = [
        (-25.0, 25.0),
        (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0),
        (0.0, 1.3),    # me_bat_rating (>= 0)
        (0.0, 1.3),    # me_bowl_rating (>= 0)
        (0.0, 1.2),    # home_delta (>= 0)
        (0.0, 0.8),    # country_adapt_diff (>= 0)
        (0.0, 0.7),    # emp_country_residual (>= 0)
        (0.0, 1.0),    # xi_bat_adapt_diff (>= 0)
        (-1.0, 0.0),   # xi_bowl_adapt_diff (<= 0)
        (0.0, 1.2),    # xi_strength_diff (>= 0)
        (0.0, 2.0),    # xi_bat_index (>= 0)
        (0.0, 2.0),    # xi_bowl_index (>= 0, higher means leaky bowling)
        (0.0, 0.9),    # duel_run_delta (>= 0)
        (0.0, 1.2),    # month_eff (>= 0)
        (-6.0, 6.0),   # toss_winner_bats
    ]

    def loss_fn(theta: np.ndarray) -> float:
        inter = theta[0]
        w_base = softmax(theta[1:5])
        w_adj = theta[5:]
        pred = inter + X_base @ w_base + X_adj_centered @ w_adj
        res = y - pred
        if huber_delta is None:
            data_l = float(np.mean(sample_w * (res ** 2)))
        else:
            ar = np.abs(res)
            hub = np.where(ar <= huber_delta, 0.5 * (res ** 2), huber_delta * (ar - 0.5 * huber_delta))
            data_l = float(np.mean(sample_w * (2.0 * hub)))
        reg_base = 180.0 * float(np.sum((w_base - prior_base) ** 2))
        reg_adj = (l2_val / max(1, n)) * float(np.sum((w_adj * adj_stds) ** 2)) + 0.002 * (inter ** 2)
        return data_l + reg_base + reg_adj

    opt = minimize(loss_fn, x0, method="L-BFGS-B", bounds=bounds)
    theta = opt.x
    inter = float(theta[0])
    w_base = softmax(theta[1:5])
    w_adj = theta[5:]

    preds = inter + X_base @ w_base + X_adj_centered @ w_adj
    resid_std = float(np.sqrt(np.average((y - preds) ** 2, weights=sample_w)))

    # Era-drift debiaser from the final 300-day OOF training tail (fixes recent scoring inflation cleanly)
    tail_pairs = [
        (float(y[i] - preds[i]), _recency_weight(d_list[i], ref_date, half_life_days=240.0))
        for i in range(len(y))
    ]
    t_ws = sum(w for _, w in tail_pairs)
    raw_era_shift = sum(r_v * w for r_v, w in tail_pairs) / max(1e-6, t_ws + 28.0)
    max_shift = 7.0 if fmt == "T20I" else (6.0 if fmt == "ODI" else 14.0)
    # Only correct secular upward scoring drift; avoid fitting short-term dips negatively
    era_shift = round(max(0.0, min(max_shift, raw_era_shift)), 2)

    adj_names = [
        "me_bat_rating",
        "me_bowl_rating",
        "home_delta",
        "country_adapt_diff",
        "emp_country_residual",
        "xi_bat_adapt_diff",
        "xi_bowl_adapt_diff",
        "xi_strength_diff",
        "xi_bat_index",
        "xi_bowl_index",
        "duel_run_delta",
        "month_eff",
        "toss_winner_bats",
    ]
    mean_keys = [
        "me_bat_rating",
        "me_bowl_rating",
        "home_delta",
        "country_adapt_diff",
        "emp_country_residual",
        "xi_bat_adapt",
        "xi_bowl_adapt",
        "xi_strength_diff",
        "xi_bat_index",
        "xi_bowl_index",
        "duel_run_delta",
        "month_eff",
        "toss_winner_bats",
    ]
    weights = {
        "same_ground_pitch_par": round(float(w_base[0]), 4),
        "global_pitch_par": round(float(w_base[1]), 4),
        "ground_overall_par": round(float(w_base[2]), 4),
        "me_venue_par": round(float(w_base[3]), 4),
    }
    feature_means = {}
    for j, fname in enumerate(adj_names):
        weights[fname] = round(float(w_adj[j]), 5)
        feature_means[mean_keys[j]] = round(float(adj_means[j]), 4)

    return {
        "intercept": round(inter, 4),
        "weights": weights,
        "feature_means": feature_means,
        "era_shift": era_shift,
        "residual_std": round(resid_std, 2),
        "oof_train_mae": round(float(np.mean(np.abs(y - preds))), 2),
        "oof_train_rmse": round(float(np.sqrt(np.mean((y - preds) ** 2))), 2),
        "n_train": n,
    }


def _compute_metrics(preds: List[int], lows: List[int], highs: List[int], actuals: List[int], fmt: str) -> Dict[str, Any]:
    y = np.array(actuals, dtype=float)
    p = np.array(preds, dtype=float)
    lo = np.array(lows, dtype=float)
    hi = np.array(highs, dtype=float)
    err = np.abs(y - p)
    bias = float(np.mean(p - y))
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    tol_a = _CALIBRATION_CFG[fmt]["tol_a"]
    tol_b = _CALIBRATION_CFG[fmt]["tol_b"]
    rel_err = err / np.maximum(1.0, y)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / max(1e-9, ss_tot))
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else 0.0
    ci_cov = float(np.mean((y >= lo) & (y <= hi)) * 100.0)
    return {
        "n": len(actuals),
        "mae": round(mae, 2),
        "median_ae": round(float(np.median(err)), 2),
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        "r": round(corr, 3),
        "r2": round(r2, 3),
        f"within_{int(tol_a)}_runs_pct": round(float(np.mean(err <= tol_a) * 100.0), 1),
        f"within_{int(tol_b)}_runs_pct": round(float(np.mean(err <= tol_b) * 100.0), 1),
        "within_10pct_score_pct": round(float(np.mean(rel_err <= 0.10) * 100.0), 1),
        "within_15pct_score_pct": round(float(np.mean(rel_err <= 0.15) * 100.0), 1),
        "ci_80_coverage_pct": round(ci_cov, 1),
    }


def _match_baseline_legacy_stage3(r: Dict[str, Any]) -> Tuple[int, int, int]:
    """Evaluates existing legacy `calculate_stage3_xi_locked` baseline pre-upgrade."""
    fmt = r["format"]
    dossier = None
    v_low = r["venue_key"]
    for dk, dv in FORMAT_GROUND_PITCH_DOSSIERS.items():
        if normalize_venue_key(dv.get("name")) == v_low or dk == v_low.replace(" ", "_"):
            dossier = dict(dv)
            break
    if dossier:
        venue_dict = {
            "id": dossier["id"],
            "name": dossier["name"],
            "country": dossier.get("country", ""),
            "dimensions_avg_boundary_m": dossier.get("boundary_dimensions", {}).get("avg_m", 70.0),
            "dew_factor": dossier.get("dew_factor", 0.25),
            "avg_first_innings_score_odi": dossier.get("format_pars", {}).get("ODI", 275),
            "avg_first_innings_score_t20": dossier.get("format_pars", {}).get("T20I", 166),
            "avg_first_innings_score_test": dossier.get("format_pars", {}).get("TEST", 330),
        }
    else:
        vp = get_historical_venue_profile(r["city_key"] or v_low, fmt)
        venue_dict = {
            "id": vp.get("id", v_low),
            "name": r["venue"],
            "country": vp.get("country", "Neutral"),
            "dimensions_avg_boundary_m": vp.get("boundary_avg_m", 70.0),
            "dew_factor": vp.get("dew_factor", 0.30),
            "avg_first_innings_score_odi": vp.get("avg_first_innings_score_odi", 275),
            "avg_first_innings_score_t20": vp.get("avg_first_innings_score_t20", 166),
            "avg_first_innings_score_test": vp.get("avg_first_innings_score_test", 330),
        }

    # Note: temporarily skip upgraded model inside baseline helper if patched
    toss_win = r["toss_winner_code"]
    res = calculate_stage3_xi_locked(
        confirmed_xi_a=r["bat_xi"],
        confirmed_xi_b=r["bowl_xi"],
        toss_winner_id=toss_win,
        toss_decision=r["toss_decision"],
        venue=venue_dict,
        team_a_id=r["bat_code"],
        team_b_id=r["bowl_code"],
        p2_squad_prob_a=50.0,
        match_format=fmt,
        match_date=r["date"],
        use_ground_pitch_model=False,
    )
    return int(res["par_score_projected"]), int(res["par_score_min"]), int(res["par_score_max"])


def _fit_stage3_win_calibrator(fmt: str, train_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fits an L2-regularized Stage 3 Win Probability logistic meta-calibrator strictly chronologically
    on `train_rows` combining chronological MOV-Elo, rolling form, bilateral H2H, and venue chase bias.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    home_boost = 45.0 if fmt == "T20I" else (55.0 if fmt == "ODI" else 70.0)
    k_elo = 28.0 if fmt == "T20I" else 24.0
    def_bat1 = 0.48 if fmt == "T20I" else (0.49 if fmt == "ODI" else 0.52)

    elo_c: Dict[str, float] = defaultdict(lambda: 1500.0)
    form_c: Dict[str, List[float]] = defaultdict(list)
    ven_c: Dict[str, List[float]] = defaultdict(list)
    h2h_c: Dict[str, List[float]] = defaultdict(list)

    X_w, y_w = [], []
    for r in sorted(train_rows, key=lambda x: (x["date"], x["match_id"])):
        b_c, bw_c = r["bat_code"], r["bowl_code"]
        hc = infer_host_country_code(r["venue_key"], r["city_key"], r["venue"])
        eb = elo_c[b_c] + (home_boost if hc and b_c == hc else 0.0)
        ebw = elo_c[bw_c] + (home_boost if hc and bw_c == hc else 0.0)
        elo_diff = eb - ebw
        p_exp = 1.0 / (1.0 + math.pow(10.0, -elo_diff / 400.0))

        fa = (sum(form_c[b_c][-12:]) + 4.0 * 0.5) / (len(form_c[b_c][-12:]) + 4.0)
        fb = (sum(form_c[bw_c][-12:]) + 4.0 * 0.5) / (len(form_c[bw_c][-12:]) + 4.0)
        vb_list = ven_c.get(r["venue_key"], [])
        v_b1 = (sum(vb_list[-20:]) + 8.0 * def_bat1) / (len(vb_list[-20:]) + 8.0)

        pk_a, pk_b = sorted((b_c, bw_c))
        h_list = h2h_c.get(f"{pk_a}_{pk_b}", [])
        raw_h = (sum(h_list[-12:]) + 3.0 * 0.5) / (len(h_list[-12:]) + 3.0)
        h_b = raw_h if b_c == pk_a else (1.0 - raw_h)

        wc = r.get("winner_code")
        if wc in (b_c, bw_c):
            res_b = 1.0 if wc == b_c else 0.0
            X_w.append([
                elo_diff / 100.0,
                fa - fb,
                h_b - 0.5,
                v_b1 - 0.5,
                (1.0 if r["toss_winner_bats"] else -1.0) * (v_b1 - 0.5),
            ])
            y_w.append(int(res_b))
            if r["venue_key"]:
                ven_c[r["venue_key"]].append(res_b)
        else:
            res_b = 0.5

        k_eff = k_elo * (2.2 / (abs(elo_diff) * 0.001 + 2.2))
        d_elo = k_eff * (res_b - p_exp)
        elo_c[b_c] += d_elo
        elo_c[bw_c] -= d_elo
        form_c[b_c].append(res_b)
        form_c[bw_c].append(1.0 - res_b)
        h2h_c[f"{pk_a}_{pk_b}"].append(res_b if b_c == pk_a else (1.0 - res_b))

    if len(X_w) < 30:
        return {"intercept": 0.0, "weights": [0.45, 0.25, 0.15, 0.20, 0.15], "means": [0.0] * 5, "stds": [1.0] * 5}

    X_arr = np.array(X_w, dtype=float)
    y_arr = np.array(y_w, dtype=int)
    sc = StandardScaler()
    Xs = sc.fit_transform(X_arr)
    clf = LogisticRegression(C=0.05, max_iter=1000).fit(Xs, y_arr)
    return {
        "intercept": round(float(clf.intercept_[0]), 5),
        "weights": [round(float(w), 5) for w in clf.coef_[0]],
        "means": [round(float(m), 5) for m in sc.mean_],
        "stds": [round(float(s), 5) for s in np.maximum(1e-4, sc.scale_)],
    }


def _fit_stage3_full_chronological_win_priors(zpath: Path, fmt: str, split_date: str) -> Dict[str, Any]:
    """
    Replays ALL historical male matches (< split_date) in strict chronological order to converge
    MOV-Elo, SENA/Subcontinent cluster Elo, rolling L12 win rate, RPO margin, H2H, venue chase bias,
    and an L2-regularized LogisticRegression win calibrator without any post-split leakage.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sena_set = {"AUS", "ENG", "NZ", "SA"}
    sub_set = {"IND", "PAK", "SL", "BAN", "AFG"}
    home_boost = 45.0 if fmt == "T20I" else (55.0 if fmt == "ODI" else 70.0)
    k_elo = 28.0 if fmt == "T20I" else 24.0
    def_bat1 = 0.48 if fmt == "T20I" else (0.49 if fmt == "ODI" else 0.52)

    raw_list = []
    with zipfile.ZipFile(zpath, "r") as z:
        for fn in sorted(z.namelist()):
            if not fn.endswith(".json"):
                continue
            d = json.loads(z.read(fn))
            info = d.get("info", {})
            if info.get("gender") != "male":
                continue
            teams = info.get("teams", [])
            dt = (info.get("dates") or [""])[0]
            if len(teams) != 2 or not dt or dt >= split_date:
                continue
            outcome = info.get("outcome", {})
            if outcome.get("result") in ("no result", "abandoned"):
                continue
            inns = d.get("innings", [])
            if not inns:
                continue
            raw_list.append((dt, fn, d))
    raw_list.sort(key=lambda x: (x[0], x[1]))

    elo_c: Dict[str, float] = defaultdict(lambda: 1500.0)
    elo_cl: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 1500.0))
    form_c: Dict[str, List[float]] = defaultdict(list)
    rpo_c: Dict[str, List[float]] = defaultdict(list)
    ven_c: Dict[str, List[float]] = defaultdict(list)
    h2h_c: Dict[str, List[float]] = defaultdict(list)

    X_w, y_w, w_samps = [], [], []
    for dt, _, d in raw_list:
        info = d.get("info", {})
        teams = info.get("teams", [])
        inns = d.get("innings", [])
        b_name = inns[0].get("team") or teams[0]
        bw_name = teams[1] if b_name == teams[0] else teams[0]
        b_c = TEAM_NAME_TO_CODE.get(b_name, normalize_team_code(b_name))
        bw_c = TEAM_NAME_TO_CODE.get(bw_name, normalize_team_code(bw_name))
        venue = info.get("venue", "") or ""
        city = info.get("city", "") or ""
        vk = normalize_venue_key(venue)
        ck = normalize_city_key(city)
        hc = infer_host_country_code(vk, ck, venue)
        cl = "SENA" if hc in sena_set else ("SUBCONT" if hc in sub_set else "OTHER")

        toss = info.get("toss", {})
        tw_name = toss.get("winner") or b_name
        tw_code = TEAM_NAME_TO_CODE.get(tw_name, normalize_team_code(tw_name))
        toss_bat = 1.0 if tw_code == b_c else -1.0

        eb = elo_c[b_c] + (home_boost if hc and b_c == hc else 0.0)
        ebw = elo_c[bw_c] + (home_boost if hc and bw_c == hc else 0.0)
        elo_diff = eb - ebw
        cl_diff = elo_cl[cl][b_c] - elo_cl[cl][bw_c]
        p_exp = 1.0 / (1.0 + math.pow(10.0, -elo_diff / 400.0))

        fa = (sum(form_c[b_c][-12:]) + 4.0 * 0.5) / (len(form_c[b_c][-12:]) + 4.0)
        fb = (sum(form_c[bw_c][-12:]) + 4.0 * 0.5) / (len(form_c[bw_c][-12:]) + 4.0)
        ra = (sum(rpo_c[b_c][-10:]) / max(1, len(rpo_c[b_c][-10:]))) if rpo_c[b_c] else 0.0
        rb = (sum(rpo_c[bw_c][-10:]) / max(1, len(rpo_c[bw_c][-10:]))) if rpo_c[bw_c] else 0.0

        vb_list = ven_c.get(vk, [])
        v_b1 = (sum(vb_list[-25:]) + 8.0 * def_bat1) / (len(vb_list[-25:]) + 8.0)

        pk_a, pk_b = sorted((b_c, bw_c))
        h_list = h2h_c.get(f"{pk_a}_{pk_b}", [])
        raw_h = (sum(h_list[-12:]) + 3.0 * 0.5) / (len(h_list[-12:]) + 3.0)
        h_b = raw_h if b_c == pk_a else (1.0 - raw_h)

        # Extract RPO margin for MOV-Elo
        inn_rpos = []
        for inn in inns[:2]:
            ir, ib = 0, 0
            for ov in inn.get("overs", []):
                for dl in ov.get("deliveries", []):
                    ir += dl.get("runs", {}).get("total", 0)
                    et = dl.get("extras", {})
                    if "wides" not in et and "noballs" not in et:
                        ib += 1
            inn_rpos.append((inn.get("team"), ir / max(1.0, ib / 6.0)))
        if len(inn_rpos) >= 2:
            rpo_m = inn_rpos[0][1] - inn_rpos[1][1] if inn_rpos[0][0] == b_name else inn_rpos[1][1] - inn_rpos[0][1]
        else:
            rpo_m = 0.0

        win_raw = info.get("outcome", {}).get("winner")
        wc = TEAM_NAME_TO_CODE.get(win_raw, normalize_team_code(win_raw)) if win_raw else None
        is_sen = (b_name in SENIOR_TEAMS and bw_name in SENIOR_TEAMS)
        if dt >= "2016-01-01" and wc in (b_c, bw_c):
            res_b = 1.0 if wc == b_c else 0.0
            X_w.append([
                elo_diff / 100.0,
                cl_diff / 100.0,
                fa - fb,
                ra - rb,
                h_b - 0.5,
                v_b1 - 0.5,
                toss_bat * (v_b1 - 0.5),
            ])
            y_w.append(int(res_b))
            w_samps.append(1.5 if is_sen else 0.75)
            if vk:
                ven_c[vk].append(res_b)
        elif wc in (b_c, bw_c):
            res_b = 1.0 if wc == b_c else 0.0
            if vk:
                ven_c[vk].append(res_b)
        else:
            res_b = 0.5

        mov = max(0.75, min(2.0, math.log1p(abs(rpo_m)) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))))
        d_elo = (k_elo * mov) * (res_b - p_exp)
        elo_c[b_c] += d_elo
        elo_c[bw_c] -= d_elo
        p_cl = 1.0 / (1.0 + math.pow(10.0, -cl_diff / 400.0))
        elo_cl[cl][b_c] += 20.0 * (res_b - p_cl)
        elo_cl[cl][bw_c] -= 20.0 * (res_b - p_cl)
        form_c[b_c].append(res_b)
        form_c[bw_c].append(1.0 - res_b)
        rpo_c[b_c].append(rpo_m)
        rpo_c[bw_c].append(-rpo_m)
        h2h_c[f"{pk_a}_{pk_b}"].append(res_b if b_c == pk_a else (1.0 - res_b))

    team_elo = {c: round(float(v), 1) for c, v in elo_c.items()}
    team_elo_cluster = {cl: {c: round(float(v), 1) for c, v in mp.items()} for cl, mp in elo_cl.items()}
    team_form = {c: round((sum(v[-12:]) + 4.0 * 0.5) / (len(v[-12:]) + 4.0), 4) for c, v in form_c.items()}
    team_rpo_margin = {c: round(float(sum(v[-10:]) / max(1, len(v[-10:]))), 4) for c, v in rpo_c.items()}
    venue_bat1 = {v: round((sum(vals[-25:]) + 8.0 * def_bat1) / (len(vals[-25:]) + 8.0), 4) for v, vals in ven_c.items()}
    h2h_map = {k: round((sum(vals[-12:]) + 3.0 * 0.5) / (len(vals[-12:]) + 3.0), 4) for k, vals in h2h_c.items()}

    win_model = {"intercept": 0.0, "weights": [0.55, 0.15, 0.15, 0.10, 0.08, 0.12, 0.08], "means": [0.0] * 7, "stds": [1.0] * 7}
    if len(X_w) >= 40:
        Xa = np.array(X_w, dtype=float)
        ya = np.array(y_w, dtype=int)
        wa = np.array(w_samps, dtype=float)
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa)
        clf = LogisticRegression(C=0.05, max_iter=1000).fit(Xs, ya, sample_weight=wa)
        win_model = {
            "intercept": round(float(clf.intercept_[0]), 5),
            "weights": [round(float(w), 5) for w in clf.coef_[0]],
            "means": [round(float(m), 5) for m in sc.mean_],
            "stds": [round(float(s), 5) for s in np.maximum(1e-4, sc.scale_)],
        }

    return {
        "team_elo": team_elo,
        "team_elo_cluster": team_elo_cluster,
        "team_form": team_form,
        "team_rpo_margin": team_rpo_margin,
        "venue_bat1_win_pct": venue_bat1,
        "h2h_win_pct": h2h_map,
        "win_model": win_model,
    }


def run_calibration_and_evaluation(
    split_date: str = "2025-01-01",
    save_priors: bool = True,
) -> Dict[str, Any]:
    priors_all: Dict[str, Any] = {
        "metadata": {
            "train_cutoff_exclusive": split_date,
            "sources": [
                "Cricsheet Official Men's ODI & T20I Archives (2018-01-01 to 2024-12-31 training split)",
                "Render Analytics (https://cricket-analytics-ghpv.onrender.com/) cached player 22-attr, 5-surface adaptability & duel snapshots",
            ],
        }
    }
    eval_report: Dict[str, Any] = {"split_date": split_date, "formats": {}}

    for fmt, zname in [("ODI", "odis_male_json.zip"), ("T20I", "t20s_male_json.zip"), ("TEST", "tests_male_json.zip")]:
        zpath = DEFAULT_CRICSHEET_ZIP_DIR / zname
        if not zpath.exists():
            continue
        rows = extract_clean_matches_from_zip(zpath, fmt)
        train_rows = [r for r in rows if r["date"] < split_date]
        test_rows = [r for r in rows if r["date"] >= split_date]
        post_render_rows = [r for r in test_rows if r["date"] > "2026-10-25"]

        fmt_priors = build_empirical_priors_from_train_rows(fmt, train_rows)
        score_model = fit_ridge_stage3_model(fmt, train_rows, l2_alpha=250.0)
        fmt_priors["score_model"] = score_model
        chron_win = _fit_stage3_full_chronological_win_priors(zpath, fmt, split_date=split_date)
        fmt_priors.update(chron_win)
        priors_all[fmt] = fmt_priors

        single_prior = {fmt: fmt_priors}
        actuals = [r["actual_score"] for r in test_rows]
        fmt_mean = int(round(fmt_priors["overall_mean"]))

        # Baseline 1: Format Global Mean
        def_band = _CALIBRATION_CFG[fmt]["default_band"]
        b1_preds = [fmt_mean] * len(test_rows)
        b1_lo = [fmt_mean - def_band] * len(test_rows)
        b1_hi = [fmt_mean + def_band] * len(test_rows)

        # Baseline 2: Legacy Stage 3 Heuristic
        b2_preds, b2_lo, b2_hi = [], [], []
        for r in test_rows:
            p, l, h = _match_baseline_legacy_stage3(r)
            b2_preds.append(p)
            b2_lo.append(l)
            b2_hi.append(h)

        # Upgraded Stage 3 Mode 1: Pure Zero-Leakage Pre-Match
        m1_preds, m1_lo, m1_hi = [], [], []
        post_preds, post_actuals, post_lo, post_hi = [], [], [], []
        for r in test_rows:
            feat = compute_stage3_xi_score_features(
                fmt=fmt,
                bat_team_id=r["bat_code"],
                bowl_team_id=r["bowl_code"],
                bat_xi=r["bat_xi"],
                bowl_xi=r["bowl_xi"],
                venue_name=r["venue"],
                city=r["city"],
                toss_bat_first_known=True,
                toss_winner_bats=r["toss_winner_bats"],
                pitch_override=None,
                priors=single_prior,
                match_date=r["date"],
            )
            p, l, h = predict_stage3_calibrated_score(feat, priors=single_prior)
            m1_preds.append(p)
            m1_lo.append(l)
            m1_hi.append(h)
            if r["date"] > "2026-10-25":
                post_preds.append(p)
                post_lo.append(l)
                post_hi.append(h)
                post_actuals.append(r["actual_score"])

        # Upgraded Stage 3 Mode 2: Pitch Type Declared Pre-Match (curator pitch report scenario)
        m2_preds, m2_lo, m2_hi = [], [], []
        for r in test_rows:
            feat2 = compute_stage3_xi_score_features(
                fmt=fmt,
                bat_team_id=r["bat_code"],
                bowl_team_id=r["bowl_code"],
                bat_xi=r["bat_xi"],
                bowl_xi=r["bowl_xi"],
                venue_name=r["venue"],
                city=r["city"],
                toss_bat_first_known=True,
                toss_winner_bats=r["toss_winner_bats"],
                pitch_override=r["pitch_type"],
                priors=single_prior,
                match_date=r["date"],
            )
            p2, l2, h2 = predict_stage3_calibrated_score(feat2, priors=single_prior)
            m2_preds.append(p2)
            m2_lo.append(l2)
            m2_hi.append(h2)

        eval_report["formats"][fmt] = {
            "n_train_2018_2024": len(train_rows),
            "n_test_2025_2026": len(test_rows),
            "oof_train_mae": score_model["oof_train_mae"],
            "oof_train_rmse": score_model["oof_train_rmse"],
            "global_pitch_pars_train": fmt_priors["global_pitch_pars"],
            "score_model_weights": score_model["weights"],
            "baseline_format_mean": _compute_metrics(b1_preds, b1_lo, b1_hi, actuals, fmt),
            "baseline_legacy_stage3": _compute_metrics(b2_preds, b2_lo, b2_hi, actuals, fmt),
            "upgraded_stage3_zero_leakage_prematch": _compute_metrics(m1_preds, m1_lo, m1_hi, actuals, fmt),
            "upgraded_stage3_with_pitch_declared": _compute_metrics(m2_preds, m2_lo, m2_hi, actuals, fmt),
            "post_oct25_2026_strict_slice": _compute_metrics(post_preds, post_lo, post_hi, post_actuals, fmt) if post_actuals else None,
        }

    if save_priors:
        CALIBRATED_PRIORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATED_PRIORS_PATH, "w", encoding="utf-8") as fh:
            json.dump(priors_all, fh, indent=2)
        invalidate_stage3_priors_cache()

    return eval_report


if __name__ == "__main__":
    rep = run_calibration_and_evaluation(split_date="2025-01-01", save_priors=True)
    print(json.dumps(rep, indent=2))
