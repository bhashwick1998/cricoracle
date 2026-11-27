"""
Trains Stage 4 Live Innings-1 Calibrator (stage4_calibrator_v1.pkl) strictly on
pre-cutoff historical Senior Men's Cricsheet matches with full Render 22-attribute,
5-surface adaptability, offline duel simulation, and Stage 3 prior parity.

Usage:
    python -m backend.ml.train_stage4_calibrator --cutoff-date 2025-01-01
"""

import argparse
import json
import math
import os
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from scipy.special import expit, logit as sp_logit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.engine.cricsheet_pitch_intelligence import canonicalize_pitch_type, normalize_team_code
from backend.engine.live_pitch_classifier import LivePitchClassifier
from backend.engine.render_analytics_bridge import _load_players_cache
from backend.engine.stage3_ground_pitch_model import (
    TEAM_NAME_TO_CODE,
    compute_stage3_dual_first_innings_projections,
    load_stage3_calibrated_priors,
    normalize_venue_key,
)
from backend.engine.stage4_calibrator import (
    CALIBRATOR_PATH,
    SCORE_COMPACT_IDX,
    SCORE_FEAT_NAMES,
    _horizon_norm,
    extract_innings1_stage4_features,
    invalidate_stage4_calibrator_cache,
)
from backend.ml.evaluate_stage3_scores import DEFAULT_CRICSHEET_ZIP_DIR, SENIOR_TEAMS

CHECKPOINTS = [0.10, 0.20, 0.35, 0.55, 0.75]


def _extract_training_rows(cutoff_date: str) -> Dict[str, List[Dict[str, Any]]]:
    priors = load_stage3_calibrated_priors()
    lpc = LivePitchClassifier()
    ds: Dict[str, List[Dict[str, Any]]] = {}

    specs = [
        ("T20I", "t20s_male_json.zip", 20.0),
        ("ODI", "odis_male_json.zip", 50.0),
        ("TEST", "tests_male_json.zip", 100.0),
    ]
    for fmt, zname, std_max_ov in specs:
        render_fmt = "TEST" if fmt == "TEST" else ("ODI" if fmt == "ODI" else "T20")
        p_cache = _load_players_cache(render_fmt)
        rows: List[Dict[str, Any]] = []
        cp_targets = [(fr, int(round(fr * std_max_ov * 6))) for fr in CHECKPOINTS]

        with zipfile.ZipFile(DEFAULT_CRICSHEET_ZIP_DIR / zname, "r") as z:
            for fn in sorted(z.namelist()):
                if not fn.endswith(".json"):
                    continue
                d = json.loads(z.read(fn))
                info = d.get("info", {})
                teams = info.get("teams", [])
                dt = (info.get("dates") or [""])[0]
                if (
                    info.get("gender") != "male"
                    or len(teams) != 2
                    or not all(t in SENIOR_TEAMS for t in teams)
                    or dt < "2018-01-01"
                    or dt >= cutoff_date
                ):
                    continue
                innings = d.get("innings", [])
                if not innings:
                    continue
                inn0 = innings[0]
                overs_list = inn0.get("overs", [])
                if not overs_list:
                    continue
                outcome = info.get("outcome", {})
                if outcome.get("result") in ("no result", "abandoned"):
                    continue

                tot_runs1, tot_wkts1 = 0, 0
                for ov in overs_list:
                    for dl in ov.get("deliveries", []):
                        tot_runs1 += int(dl.get("runs", {}).get("total", 0))
                        tot_wkts1 += len(dl.get("wickets", []))
                max_ov1 = overs_list[-1].get("over", 0) + 1
                if fmt == "TEST":
                    if len(innings) < 2 and tot_wkts1 < 10 and not inn0.get("declared"):
                        continue
                else:
                    min_full = 45 if fmt == "ODI" else 18
                    if max_ov1 < min_full and tot_wkts1 < 10:
                        continue

                reg = info.get("registry", {}).get("people", {})
                team_a_name = inn0.get("team") or teams[0]
                team_b_name = teams[1] if team_a_name == teams[0] else teams[0]
                code_a = TEAM_NAME_TO_CODE.get(team_a_name, normalize_team_code(team_a_name))
                code_b = TEAM_NAME_TO_CODE.get(team_b_name, normalize_team_code(team_b_name))
                venue_name = info.get("venue", "") or ""
                city_name = info.get("city", "") or ""
                vkey = normalize_venue_key(venue_name)
                toss = info.get("toss", {})
                toss_win_code = TEAM_NAME_TO_CODE.get(toss.get("winner") or team_a_name, normalize_team_code(toss.get("winner") or team_a_name))
                toss_dec = toss.get("decision") or "bat"
                winner_name = outcome.get("winner")
                bat1_won = None if (outcome.get("result") in ("draw", "tie") or not winner_name) else (1 if winner_name == team_a_name else 0)

                def _build_xi(t_name: str, t_code: str) -> List[Dict[str, Any]]:
                    out = []
                    for idx, pn in enumerate(info.get("players", {}).get(t_name, [])[:11]):
                        cid = reg.get(pn, "")
                        r_rec = p_cache.get(cid, {})
                        out.append({
                            "id": cid or pn.lower().replace(" ", "_"),
                            "cricsheet_id": cid,
                            "render_id": cid,
                            "name": pn,
                            "country": t_name,
                            "team_id": t_code,
                            "batting_order": idx + 1,
                            "is_confirmed_xi": 1,
                            "role": str(r_rec.get("player_role") or ("top_order_batter" if idx < 5 else "all_rounder")),
                            "bowling_style": r_rec.get("bowling_style") or "RF",
                            "batting_hand": "left" if "left" in str(r_rec.get("batting_style") or "").lower() else "right",
                        })
                    return out

                xi_a = _build_xi(team_a_name, code_a)
                xi_b = _build_xi(team_b_name, code_b)

                dual = compute_stage3_dual_first_innings_projections(
                    fmt=fmt,
                    team_a_id=code_a,
                    team_b_id=code_b,
                    xi_a=xi_a,
                    xi_b=xi_b,
                    venue_name=venue_name,
                    city=city_name,
                    toss_winner_id=toss_win_code,
                    toss_decision=toss_dec,
                    match_date=dt,
                    priors=priors,
                )
                s3_par_a = float(dual["if_team_a_bats_first"]["projected_score"])
                s3_par_b = float(dual["if_team_b_bats_first"]["projected_score"])
                p0_a = float(dual.get("p0_team_a", 50.0))
                p0_b = float(dual.get("p0_team_b", 50.0))
                p0_dec = max(0.02, min(0.98, p0_a / max(1.0, p0_a + p0_b)))
                v_meta = priors.get(fmt, {}).get("venues", {}).get(vkey, {})
                hist_par = float(v_meta.get("mean") or priors.get(fmt, {}).get("overall_mean") or s3_par_a)

                score, wkts, legal_balls = 0, 0, 0
                out_players = set()
                delivs_so_far: List[Dict[str, Any]] = []
                last_striker, last_ns, last_bowler = "", "", ""
                cp_idx = 0

                for ov in overs_list:
                    ov_i = int(ov.get("over", 0))
                    lin_ov = 0
                    for dl in ov.get("deliveries", []):
                        et = dl.get("extras", {})
                        r_tot = int(dl.get("runs", {}).get("total", 0))
                        r_bat = int(dl.get("runs", {}).get("batter", 0))
                        is_legal = ("wides" not in et and "noballs" not in et)
                        score += r_tot
                        if is_legal:
                            legal_balls += 1
                            lin_ov += 1
                        for w_obj in dl.get("wickets", []):
                            wkts += 1
                            out_players.add(w_obj.get("player_out") or dl.get("batter", ""))
                        last_striker = dl.get("batter", "")
                        last_ns = dl.get("non_striker", "")
                        last_bowler = dl.get("bowler", "")
                        bw_rec = p_cache.get(reg.get(last_bowler, ""), {})
                        delivs_so_far.append({
                            "innings": 1,
                            "over": ov_i,
                            "ball": max(1, lin_ov),
                            "runs": r_tot,
                            "batter_runs": r_bat,
                            "is_wicket": len(dl.get("wickets", [])) > 0,
                            "bowler_style": str(bw_rec.get("bowling_style") or ""),
                            "commentary": "",
                        })
                        while cp_idx < len(cp_targets) and legal_balls >= cp_targets[cp_idx][1]:
                            frac, _ = cp_targets[cp_idx]
                            cp_idx += 1
                            if wkts >= 10:
                                continue
                            overs_f = legal_balls / 6.0
                            lpe = lpc.classify_live_pitch(
                                venue_id=vkey,
                                match_format=fmt,
                                current_overs=int(legal_balls // 6),
                                current_balls=int(legal_balls % 6),
                                current_score=score,
                                current_wickets=wkts,
                                deliveries=delivs_so_far[-36:],
                                match_day=1 + int(overs_f // 90.0),
                            )
                            live_pitch = canonicalize_pitch_type(lpe.get("classified_pitch_type", "balanced"))
                            p_probs = lpe.get("probabilities", {})

                            st_obj = next((p for p in xi_a if p["name"] == last_striker), {"name": last_striker})
                            ns_obj = next((p for p in xi_a if p["name"] == last_ns), {"name": last_ns})
                            bw_obj = next((p for p in xi_b if p["name"] == last_bowler), {"name": last_bowler})
                            dugout = [p for p in xi_a if p["name"] not in out_players and p["name"] not in (last_striker, last_ns)]
                            bowlers = [p for p in xi_b if "bowl" in str(p.get("role", "")).lower() or "all" in str(p.get("role", "")).lower()][:5] or xi_b[6:]

                            feat_raw = extract_innings1_stage4_features(
                                fmt=fmt,
                                score=score,
                                wickets=wkts,
                                overs_f=overs_f,
                                p0_bat_dec=p0_dec,
                                bat_team=code_a,
                                bowl_team=code_b,
                                venue_name=venue_name,
                                hist_venue_par=hist_par,
                                live_pitch_type=live_pitch,
                                pitch_probs=p_probs,
                                striker=st_obj,
                                non_striker=ns_obj,
                                bowler=bw_obj,
                                dugout_players=dugout,
                                bowlers_list=bowlers,
                                s3_par_bat=s3_par_a,
                                s3_par_opp=s3_par_b,
                                s3_hist_par=hist_par,
                                era_offset=0.0,
                            )
                            rows.append({
                                "date": dt,
                                "date_yr": float(dt[:4]) + (float(dt[5:7]) - 1.0) / 12.0 if len(dt) >= 7 else 2022.0,
                                "cp_frac": frac,
                                "actual_1st_score": float(tot_runs1),
                                "actual_rem_runs": float(tot_runs1 - score),
                                "actual_bat1_won": bat1_won,
                                "feats": feat_raw,
                            })
        rows.sort(key=lambda r: (r["date"], r["cp_frac"]))
        ds[fmt] = rows
        print(f"Prepared pre-{cutoff_date} {fmt}: {len(rows)} checkpoints", flush=True)
    return ds


def _recency_weights(rows: List[Dict[str, Any]], tau_years: float) -> np.ndarray:
    if tau_years >= 99.0 or not rows:
        return np.ones(len(rows), dtype=float)
    max_yr = max(r["date_yr"] for r in rows)
    return np.array([math.exp(-(max_yr - r["date_yr"]) / tau_years) for r in rows], dtype=float)


def _fit_expert(
    fmt: str,
    rows: List[Dict[str, Any]],
    era_offset: float,
    multi_horizon: bool,
) -> Dict[str, Any]:
    # Adjust era_par & sc_above_dls_used deterministically with pre-cutoff era_offset
    rows_adj = []
    for r in rows:
        f_d = dict(r["feats"])
        if era_offset != 0.0:
            f_d["era_par"] = f_d["s3_par_bat"] + era_offset
            used_pct = max(0.06, 1.0 - f_d["dls_rem_pct"])
            f_d["sc_above_dls_used"] = f_d["score"] - f_d["era_par"] * used_pct
        rows_adj.append({**r, "feats": f_d})

    cut = int(len(rows_adj) * 0.78)
    tr_v, va_v = rows_adj[:cut], rows_adj[cut:]

    def _fit_rem_stats(sub_rows):
        if multi_horizon:
            tbl = {}
            for fr in CHECKPOINTS:
                sub = [r for r in sub_rows if abs(r["cp_frac"] - fr) < 0.03]
                by_w = {w: [] for w in range(10)}
                for r in sub:
                    w = int(max(0, min(9, round(r["feats"]["wickets"]))))
                    by_w[w].append(r["actual_rem_runs"] / max(50.0, r["feats"]["era_par"]))
                m_all = float(np.mean([r["actual_rem_runs"] / max(50.0, r["feats"]["era_par"]) for r in sub])) if sub else 0.5
                tbl[fr] = {w: (sum(by_w[w]) + 10.0 * max(0.06, m_all * (1.0 - 0.08 * w))) / (len(by_w[w]) + 10.0) for w in range(10)}
            return tbl
        by_w = {w: [] for w in range(10)}
        for r in sub_rows:
            w = int(max(0, min(9, round(r["feats"]["wickets"]))))
            by_w[w].append(r["actual_rem_runs"] / max(50.0, r["feats"]["era_par"]))
        m_all = float(np.mean([r["actual_rem_runs"] / max(50.0, r["feats"]["era_par"]) for r in sub_rows]))
        return {w: (sum(by_w[w]) + 10.0 * max(0.08, m_all * (1.0 - 0.08 * w))) / (len(by_w[w]) + 10.0) for w in range(10)}

    def _apply_anc(rlist, tbl):
        out = []
        for r in rlist:
            w = int(max(0, min(9, round(r["feats"]["wickets"]))))
            frac = tbl[min(CHECKPOINTS, key=lambda fr: abs(fr - r["cp_frac"]))][w] if multi_horizon else tbl[w]
            out.append(r["feats"]["score"] + r["feats"]["era_par"] * frac)
        return np.array(out, dtype=float)

    X_all = np.array([[float(r["feats"][n]) for n in SCORE_FEAT_NAMES] for r in rows_adj], dtype=float)
    y_tot = np.array([r["actual_1st_score"] for r in rows_adj], dtype=float)
    sc_all = np.array([r["feats"]["score"] for r in rows_adj], dtype=float)

    # Tune (tau, ridge_alpha) strictly on pre-cutoff temporal validation
    tbl_val = _fit_rem_stats(tr_v)
    anc_tr_v = _apply_anc(tr_v, tbl_val)
    anc_va_v = _apply_anc(va_v, tbl_val)

    best_cfg, best_mae = (2.2, 200.0), 9999.0
    for tau in (1.2, 2.2, 4.0, 100.0):
        sw_v = _recency_weights(tr_v, tau)
        imp_v = SimpleImputer(strategy="median")
        Xf_tv = imp_v.fit_transform(X_all[:cut])
        Xf_vv = imp_v.transform(X_all[cut:])
        sc_v = StandardScaler()
        Xs_tv = sc_v.fit_transform(Xf_tv[:, SCORE_COMPACT_IDX])
        Xs_vv = sc_v.transform(Xf_vv[:, SCORE_COMPACT_IDX])
        for alpha in (50.0, 140.0, 300.0, 600.0):
            r_v = Ridge(alpha=alpha).fit(Xs_tv, y_tot[:cut] - anc_tr_v, sample_weight=sw_v)
            pv = np.maximum(sc_all[cut:], anc_va_v + r_v.predict(Xs_vv))
            mae_v = float(np.mean(np.abs(pv - y_tot[cut:])))
            if mae_v < best_mae:
                best_mae, best_cfg = mae_v, (tau, alpha)
    b_tau, b_alpha = best_cfg

    # Fit validation ensemble on tr_v -> measure val MAE weights & normalized conformal 90% quantiles
    sw_v = _recency_weights(tr_v, b_tau)
    imp_v = SimpleImputer(strategy="median")
    Xf_tv = imp_v.fit_transform(X_all[:cut])
    Xf_vv = imp_v.transform(X_all[cut:])
    sc_v = StandardScaler()
    Xs_tv = sc_v.fit_transform(Xf_tv[:, SCORE_COMPACT_IDX])
    Xs_vv = sc_v.transform(Xf_vv[:, SCORE_COMPACT_IDX])
    y_res_v = y_tot[:cut] - anc_tr_v

    h_v = HuberRegressor(alpha=b_alpha * 0.12, epsilon=1.45, max_iter=600).fit(Xs_tv, y_res_v, sample_weight=sw_v)
    r_v = Ridge(alpha=b_alpha).fit(Xs_tv, y_res_v, sample_weight=sw_v)
    x_v = xgb.XGBRegressor(
        n_estimators=120, learning_rate=0.025, max_depth=2, min_child_weight=8,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=4.0, reg_lambda=16.0, random_state=42, n_jobs=2
    ).fit(Xf_tv, y_res_v, sample_weight=sw_v)
    et_v = ExtraTreesRegressor(
        n_estimators=150, max_depth=5, min_samples_leaf=max(10, len(tr_v) // 35),
        max_features=0.65, random_state=42, n_jobs=2
    ).fit(Xf_tv, y_res_v, sample_weight=sw_v)

    pv_h = np.maximum(sc_all[cut:], anc_va_v + h_v.predict(Xs_vv))
    pv_r = np.maximum(sc_all[cut:], anc_va_v + r_v.predict(Xs_vv))
    pv_x = np.maximum(sc_all[cut:], anc_va_v + x_v.predict(Xf_vv))
    pv_e = np.maximum(sc_all[cut:], anc_va_v + et_v.predict(Xf_vv))

    # Candidate score blends evaluated strictly on pre-cutoff val slice
    candidates = [
        {"huber": 0.35, "ridge": 0.25, "xgb": 0.20, "et": 0.20},
        {"huber": 0.45, "ridge": 0.25, "xgb": 0.15, "et": 0.15},
        {"huber": 0.25, "ridge": 0.20, "xgb": 0.25, "et": 0.30},
        {"huber": 0.60, "ridge": 0.40, "xgb": 0.00, "et": 0.00},
    ]
    best_sw, best_bm = candidates[0], 9999.0
    for w_cand in candidates:
        p_cand = w_cand["huber"] * pv_h + w_cand["ridge"] * pv_r + w_cand["xgb"] * pv_x + w_cand["et"] * pv_e
        m_c = float(np.mean(np.abs(p_cand - y_tot[cut:])))
        if m_c < best_bm:
            best_bm, best_sw = m_c, w_cand

    p_val_best = best_sw["huber"] * pv_h + best_sw["ridge"] * pv_r + best_sw["xgb"] * pv_x + best_sw["et"] * pv_e
    norms_val = np.array([_horizon_norm(r["feats"]["prog"], r["feats"]["wickets_left"]) for r in va_v], dtype=float)
    scaled_err_v = (y_tot[cut:] - p_val_best) / norms_val
    q_tail = 0.025 if fmt == "TEST" else 0.035
    conf_q90_norm = (float(np.quantile(scaled_err_v, q_tail)), float(np.quantile(scaled_err_v, 1.0 - q_tail)))

    # Final Score models refitted on full pre-cutoff train
    tbl_full = _fit_rem_stats(rows_adj)
    anc_full = _apply_anc(rows_adj, tbl_full)
    y_res_full = y_tot - anc_full
    sw_full = _recency_weights(rows_adj, b_tau)

    imp_full = SimpleImputer(strategy="median")
    Xf_full = imp_full.fit_transform(X_all)
    sc_compact = StandardScaler()
    Xc_full = sc_compact.fit_transform(Xf_full[:, SCORE_COMPACT_IDX])

    huber_f = HuberRegressor(alpha=b_alpha * 0.12, epsilon=1.45, max_iter=600).fit(Xc_full, y_res_full, sample_weight=sw_full)
    ridge_f = Ridge(alpha=b_alpha).fit(Xc_full, y_res_full, sample_weight=sw_full)
    xgb_reg = xgb.XGBRegressor(
        n_estimators=130, learning_rate=0.025, max_depth=2, min_child_weight=8,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=4.0, reg_lambda=16.0, random_state=42, n_jobs=1
    ).fit(Xf_full, y_res_full, sample_weight=sw_full)
    et_reg = ExtraTreesRegressor(
        n_estimators=160, max_depth=5, min_samples_leaf=max(10, len(rows_adj) // 35),
        max_features=0.65, random_state=42, n_jobs=1
    ).fit(Xf_full, y_res_full, sample_weight=sw_full)

    # ---------- Win Probability (Bayesian Log-Odds Update on p0_logit) ----------
    dec_rows = [r for r in rows_adj if r["actual_bat1_won"] is not None]
    yw = np.array([r["actual_bat1_won"] for r in dec_rows], dtype=int)
    Xd_imp = imp_full.transform(np.array([[float(r["feats"][n]) for n in SCORE_FEAT_NAMES] for r in dec_rows], dtype=float))
    Xd_sc = sc_compact.transform(Xd_imp[:, SCORE_COMPACT_IDX])
    anc_dec = _apply_anc(dec_rows, tbl_full)
    proj_dec = np.maximum(
        [r["feats"]["score"] for r in dec_rows],
        anc_dec
        + best_sw["huber"] * huber_f.predict(Xd_sc)
        + best_sw["ridge"] * ridge_f.predict(Xd_sc)
        + best_sw["xgb"] * xgb_reg.predict(Xd_imp)
        + best_sw["et"] * et_reg.predict(Xd_imp),
    )

    scale = 20.0 if fmt == "T20I" else (30.0 if fmt == "ODI" else 70.0)
    Xw = np.array([
        [
            float(r["feats"]["p0_logit"]),
            (float(proj_dec[idx]) - float(r["feats"]["s3_par_opp"])) / scale,
            float(r["feats"]["sc_above_dls_used"]) / scale,
            float(r["feats"]["wkts_saved"]),
            float(r["feats"]["rem_tank_edge"]),
            float(r["feats"]["adapt_diff"]) / 15.0,
            float(r["feats"]["pitch_bat_edge"]),
            float(r["feats"]["duel_net_rpo"]) / 2.0,
            (float(r["feats"]["rem_bat_dep"]) - float(r["feats"]["opp_bowl_death"])) / 20.0,
            float(r["feats"]["prog"]),
        ]
        for idx, r in enumerate(dec_rows)
    ], dtype=float)

    cut_w = int(len(dec_rows) * 0.78)
    sw_w = _recency_weights(dec_rows, 3.5)
    sc_wv = StandardScaler()
    Xwv_tr = np.column_stack([Xw[:cut_w, :1], sc_wv.fit_transform(Xw[:cut_w, 1:])])
    Xwv_va = np.column_stack([Xw[cut_w:, :1], sc_wv.transform(Xw[cut_w:, 1:])])
    best_wc, best_wll = 0.06, 999.0
    for c_val in (0.01, 0.03, 0.08, 0.20, 0.50):
        cl = LogisticRegression(C=c_val, max_iter=1000).fit(Xwv_tr, yw[:cut_w], sample_weight=sw_w[:cut_w])
        pv = np.clip(cl.predict_proba(Xwv_va)[:, 1], 0.02, 0.98)
        ll = log_loss(yw[cut_w:], pv)
        if ll < best_wll:
            best_wll, best_wc = ll, c_val

    sc_win = StandardScaler()
    Xw_lr_full = np.column_stack([Xw[:, :1], sc_win.fit_transform(Xw[:, 1:])])
    lr_win = LogisticRegression(C=best_wc, max_iter=1000).fit(Xw_lr_full, yw, sample_weight=sw_w)
    xgb_win = xgb.XGBClassifier(
        n_estimators=85, learning_rate=0.025, max_depth=2, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=3.5, reg_lambda=15.0,
        eval_metric="logloss", random_state=42, n_jobs=2
    ).fit(Xw, yw, sample_weight=sw_w)

    # Check LR vs Fuse weights on val split cleanly
    cl_v = LogisticRegression(C=best_wc, max_iter=1000).fit(Xwv_tr, yw[:cut_w], sample_weight=sw_w[:cut_w])
    xgb_v = xgb.XGBClassifier(
        n_estimators=85, learning_rate=0.025, max_depth=2, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=3.5, reg_lambda=15.0,
        eval_metric="logloss", random_state=42, n_jobs=2
    ).fit(Xw[:cut_w], yw[:cut_w], sample_weight=sw_w[:cut_w])
    p_lr_va = np.clip(cl_v.predict_proba(Xwv_va)[:, 1], 0.02, 0.98)
    p_xgb_va = np.clip(xgb_v.predict_proba(Xw[cut_w:])[:, 1], 0.02, 0.98)
    p0_lo_va = Xw[cut_w:, 0]
    win_cands = [
        {"lr": 0.65, "xgb": 0.20, "p0": 0.15},
        {"lr": 0.55, "xgb": 0.25, "p0": 0.20},
        {"lr": 0.80, "xgb": 0.10, "p0": 0.10},
        {"lr": 0.45, "xgb": 0.25, "p0": 0.30},
    ]
    best_ww, best_bll = win_cands[0], 999.0
    for wc_d in win_cands:
        p_blend = np.clip(expit(wc_d["lr"] * sp_logit(p_lr_va) + wc_d["xgb"] * sp_logit(p_xgb_va) + wc_d["p0"] * p0_lo_va), 0.02, 0.98)
        ll_b = log_loss(yw[cut_w:], p_blend)
        if ll_b < best_bll:
            best_bll, best_ww = ll_b, wc_d

    out: Dict[str, Any] = {
        "imp_full": imp_full,
        "sc_compact": sc_compact,
        "huber": huber_f,
        "ridge": ridge_f,
        "xgb_reg": xgb_reg,
        "et_reg": et_reg,
        "score_weights": best_sw,
        "conf_q90_norm": conf_q90_norm,
        "sc_win": sc_win,
        "lr_win": lr_win,
        "xgb_win": xgb_win,
        "win_weights": best_ww,
        "meta": {"tau": b_tau, "alpha": b_alpha, "win_c": best_wc, "val_mae": round(best_bm, 2), "val_ll": round(best_bll, 4)},
    }
    if multi_horizon:
        out["rem_tbl"] = tbl_full
    else:
        out["rem_w"] = tbl_full
    return out


def train_and_save(cutoff_date: str = "2025-01-01") -> None:
    ds = _extract_training_rows(cutoff_date)
    bundle: Dict[str, Any] = {"cutoff_date": cutoff_date}

    for fmt, tail_n in [("T20I", 40), ("ODI", 40), ("TEST", 25)]:
        all_rows = ds[fmt]
        early_rows = [r for r in all_rows if abs(r["cp_frac"] - 0.20) < 0.03]
        # Pre-cutoff recent scoring drift offset (T20I 2023-24 modern boom; ODI stays 0 to avoid spin-series slump)
        tail_res = [r["actual_1st_score"] - r["feats"]["s3_par_bat"] for r in early_rows[-tail_n:]]
        era_offset = (float(np.mean(tail_res)) * 0.55 if len(tail_res) >= 10 else 0.0) if fmt == "T20I" else 0.0

        early_exp = _fit_expert(fmt, early_rows, era_offset=era_offset, multi_horizon=False)
        mid_exp = _fit_expert(fmt, all_rows, era_offset=era_offset, multi_horizon=True)

        bundle[fmt] = {
            "era_offset": round(era_offset, 2),
            "early": early_exp,
            "mid_late": mid_exp,
        }
        print(
            f"Trained {fmt}: era_offset={era_offset:+.2f} | "
            f"early(n={len(early_rows)}) val_mae={early_exp['meta']['val_mae']}, val_ll={early_exp['meta']['val_ll']} | "
            f"mid_late(n={len(all_rows)}) val_mae={mid_exp['meta']['val_mae']}, val_ll={mid_exp['meta']['val_ll']}",
            flush=True,
        )

    os.makedirs(os.path.dirname(CALIBRATOR_PATH), exist_ok=True)
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump(bundle, f)
    invalidate_stage4_calibrator_cache()
    print(f"Saved calibrated Stage 4 artifact to {CALIBRATOR_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 4 Innings-1 Live Calibrator")
    parser.add_argument("--cutoff-date", type=str, default="2025-01-01")
    args = parser.parse_args()
    train_and_save(cutoff_date=args.cutoff_date)


if __name__ == "__main__":
    main()
