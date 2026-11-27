"""
Stage 3 Pre-Match ODI & T20I Score Projection Engine (Ground × Pitch × Render Analytics × Duel Simulator)
========================================================================================================
Combines all 6 empirical & tactical signals prior to first ball (XI_AND_TOSS_LOCKED / XI_LOCKED):
1. Ground Historical Pitch Type Distribution P(pitch | venue, city, format, XI_spin_ratio)
2. Same-Ground Pitch-Type Empirical Bayes Score Means E[score | venue, format, pitch]
3. Format-Wide Pitch-Type Average Scores E[score | format, pitch]
4. Separated Country Batting/Bowling Ratings, Home Familiarity & Pitch Adaptability
5. Player 5-Surface Pitch Adaptability & Empirical XI Form (Cricket Analytics snapshots + Cricsheet priors)
6. Confirmed XI (7×5) Duel Simulator Matrix (aligned 22-attribute Bayesian simulation + Cricket Analytics cache)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cricsheet_pitch_intelligence import (
    COUNTRY_PITCH_ADAPTABILITY,
    FORMAT_GROUND_PITCH_DOSSIERS,
    canonicalize_pitch_type,
    normalize_team_code,
)
from .cricpulse_master_engine import compute_22_master_attributes, compute_duel_simulation
from .render_analytics_bridge import (
    _load_adapt_cache,
    _load_duels_cache,
    _load_players_cache,
    resolve_render_player_id,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CALIBRATED_PRIORS_PATH = DATA_DIR / "stage3_calibrated_priors.json"
CANONICAL_PITCHES = ("flat", "bouncy", "balanced", "green", "dusty")

TEAM_NAME_TO_CODE = {
    "India": "IND",
    "Australia": "AUS",
    "England": "ENG",
    "South Africa": "SA",
    "Pakistan": "PAK",
    "New Zealand": "NZ",
    "West Indies": "WI",
    "Sri Lanka": "SL",
    "Bangladesh": "BAN",
    "Afghanistan": "AFG",
    "Zimbabwe": "ZIM",
    "Ireland": "IRE",
    "Netherlands": "NED",
    "Scotland": "SCO",
    "Namibia": "NAM",
    "Nepal": "NEP",
    "United States of America": "USA",
    "USA": "USA",
    "United Arab Emirates": "UAE",
    "UAE": "UAE",
    "Oman": "OMA",
    "Canada": "CAN",
}

COUNTRY_ALIASES = {
    "india": "IND",
    "australia": "AUS",
    "england": "ENG",
    "south africa": "SA",
    "new zealand": "NZ",
    "pakistan": "PAK",
    "sri lanka": "SL",
    "west indies": "WI",
    "bangladesh": "BAN",
    "afghanistan": "AFG",
    "zimbabwe": "ZIM",
    "ireland": "IRE",
    "netherlands": "NED",
    "scotland": "SCO",
    "united arab emirates": "UAE",
    "uae": "UAE",
}

CITY_TO_COUNTRY: Dict[str, str] = {
    "mackay": "AUS", "cairns": "AUS", "carrara": "AUS", "canberra": "AUS",
    "townsville": "AUS", "darwin": "AUS", "melbourne": "AUS", "sydney": "AUS",
    "perth": "AUS", "brisbane": "AUS", "adelaide": "AUS", "hobart": "AUS",
    "faisalabad": "PAK", "multan": "PAK", "rawalpindi": "PAK", "lahore": "PAK",
    "karachi": "PAK", "peshawar": "PAK",
    "mount maunganui": "NZ", "tauranga": "NZ", "nelson": "NZ", "napier": "NZ",
    "queenstown": "NZ", "hamilton": "NZ", "auckland": "NZ", "wellington": "NZ",
    "christchurch": "NZ", "dunedin": "NZ",
    "southampton": "ENG", "manchester": "ENG", "leeds": "ENG", "birmingham": "ENG",
    "nottingham": "ENG", "london": "ENG", "cardiff": "ENG", "bristol": "ENG",
    "chester le street": "ENG", "taunton": "ENG",
    "dublin": "IRE", "belfast": "IRE", "magheramason": "IRE",
    "amstelveen": "NED", "rotterdam": "NED", "utrecht": "NED", "the hague": "NED", "voorburg": "NED",
    "edinburgh": "SCO", "aberdeen": "SCO", "glasgow": "SCO",
    "harare": "ZIM", "bulawayo": "ZIM", "kwekwe": "ZIM",
    "dhaka": "BAN", "mirpur": "BAN", "chattogram": "BAN", "chittagong": "BAN", "sylhet": "BAN",
    "colombo": "SL", "kandy": "SL", "pallekele": "SL", "galle": "SL", "dambulla": "SL", "hambantota": "SL",
    "sharjah": "UAE", "dubai": "UAE", "abu dhabi": "UAE", "al amerat": "OMA",
    "mumbai": "IND", "kolkata": "IND", "chennai": "IND", "bengaluru": "IND", "bangalore": "IND",
    "ahmedabad": "IND", "delhi": "IND", "hyderabad": "IND", "pune": "IND", "mohali": "IND",
    "chandigarh": "IND", "dharamshala": "IND", "jaipur": "IND", "rajkot": "IND", "indore": "IND",
    "lucknow": "IND", "guwahati": "IND", "nagpur": "IND", "ranchi": "IND",
    "visakhapatnam": "IND", "thiruvananthapuram": "IND", "kanpur": "IND", "cuttack": "IND",
    "johannesburg": "SA", "centurion": "SA", "cape town": "SA", "durban": "SA",
    "gqeberha": "SA", "port elizabeth": "SA", "paarl": "SA", "bloemfontein": "SA",
    "bridgetown": "WI", "port of spain": "WI", "tarouba": "WI", "gros islet": "WI",
    "kingstown": "WI", "st georges": "WI", "providence": "WI", "north sound": "WI",
}

VENUE_COUNTRY_MAP: Dict[str, str] = {}
for _dk, _dv in FORMAT_GROUND_PITCH_DOSSIERS.items():
    _cname = COUNTRY_ALIASES.get(str(_dv.get("country") or "").strip().lower(), "")
    if _cname:
        if _dv.get("name"):
            _vn = re.sub(r"[^a-z0-9\s]", " ", str(_dv["name"]).lower().split(",", 1)[0]).strip()
            _vn = re.sub(r"\s+", " ", _vn)
            if _vn:
                VENUE_COUNTRY_MAP[_vn] = _cname
        if _dv.get("city"):
            _cn = re.sub(r"[^a-z0-9\s]", " ", str(_dv["city"]).lower()).strip()
            _cn = re.sub(r"\s+", " ", _cn)
            if _cn:
                CITY_TO_COUNTRY.setdefault(_cn, _cname)

_PRIORS_MEM_CACHE: Optional[Dict[str, Any]] = None
_ALIGNED_PLAYERS_CACHE: Dict[str, Dict[str, Any]] = {}
_NEUTRAL_BAT_SIM = {"player_id": "neutral_bat", "b_power": 55.0, "bw_power": 35.0, "batting_style": "right"}
_NEUTRAL_BWL_SIM = {"player_id": "neutral_bwl", "b_power": 35.0, "bw_power": 55.0, "bowling_style": "RF"}


_FMT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "TEST": {
        "render_fmt": "TEST",
        "overall_mean": 318.0,
        "global_pars": {"flat": 430.0, "bouncy": 310.0, "balanced": 325.0, "green": 240.0, "dusty": 285.0},
        "k_ven": 2.8,
        "k_vp": 2.2,
        "k_smooth": 2.8,
        "base_avg": 32.0,
        "base_sr": 54.0,
        "base_econ": 3.25,
        "base_wkt": 0.55,
        "base_rpo": 3.30,
        "innings_overs": 95.0,
        "me_bowl_econ_scale": 0.008,
        "bowl_idx_econ_scale": 18.0,
        "bowl_idx_wkt_scale": 12.0,
        "default_wkt_prob": 0.032,
        "default_resid_std": 95.0,
        "clip": (110.0, 620.0),
        "b_shift": 140.0,
    },
    "ODI": {
        "render_fmt": "ODI",
        "overall_mean": 268.0,
        "global_pars": {"flat": 335.0, "bouncy": 262.0, "balanced": 240.0, "green": 195.0, "dusty": 220.0},
        "k_ven": 3.5,
        "k_vp": 3.0,
        "k_smooth": 3.5,
        "base_avg": 34.0,
        "base_sr": 88.0,
        "base_econ": 5.4,
        "base_wkt": 0.72,
        "base_rpo": 5.36,
        "innings_overs": 50.0,
        "me_bowl_econ_scale": 0.015,
        "bowl_idx_econ_scale": 10.0,
        "bowl_idx_wkt_scale": 8.0,
        "default_wkt_prob": 0.022,
        "default_resid_std": 45.0,
        "clip": (150.0, 400.0),
        "b_shift": 110.0,
    },
    "T20I": {
        "render_fmt": "T20",
        "overall_mean": 168.0,
        "global_pars": {"flat": 198.0, "bouncy": 165.0, "balanced": 154.0, "green": 127.0, "dusty": 141.0},
        "k_ven": 2.2,
        "k_vp": 2.2,
        "k_smooth": 2.8,
        "base_avg": 24.0,
        "base_sr": 132.0,
        "base_econ": 8.1,
        "base_wkt": 1.25,
        "base_rpo": 8.20,
        "innings_overs": 20.0,
        "me_bowl_econ_scale": 0.03,
        "bowl_idx_econ_scale": 6.0,
        "bowl_idx_wkt_scale": 8.0,
        "default_wkt_prob": 0.055,
        "default_resid_std": 28.0,
        "clip": (105.0, 250.0),
        "b_shift": 70.0,
    },
}


def _normalize_fmt_key(fmt: Optional[str]) -> str:
    up = str(fmt or "").upper()
    if "TEST" in up:
        return "TEST"
    if "ODI" in up:
        return "ODI"
    return "T20I"


def invalidate_stage3_priors_cache() -> None:
    """Clears cached Stage 3 empirical priors so updated calibrations reload immediately."""
    global _PRIORS_MEM_CACHE
    _PRIORS_MEM_CACHE = None
    _ALIGNED_PLAYERS_CACHE.clear()


def load_stage3_calibrated_priors() -> Dict[str, Any]:
    """Loads pre-computed training-split venue/pitch/country priors & calibrated regression weights."""
    global _PRIORS_MEM_CACHE
    if _PRIORS_MEM_CACHE is not None:
        return _PRIORS_MEM_CACHE
    if CALIBRATED_PRIORS_PATH.exists():
        try:
            with open(CALIBRATED_PRIORS_PATH, "r", encoding="utf-8") as fh:
                _PRIORS_MEM_CACHE = json.load(fh)
                return _PRIORS_MEM_CACHE
        except Exception:
            pass
    _PRIORS_MEM_CACHE = {}
    return _PRIORS_MEM_CACHE


def normalize_venue_key(venue: Optional[str]) -> str:
    """Canonicalizes ground names across Cricsheet JSONs and DB records."""
    s = str(venue or "").lower().strip()
    if "," in s:
        s = s.split(",", 1)[0].strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_city_key(city: Optional[str]) -> str:
    s = str(city or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def infer_host_country_code(venue_key: str, city_key: str, venue_raw: Optional[str] = None) -> str:
    """Infers host country code (IND, AUS, ENG, etc.) for home/away & regional priors."""
    if venue_key in VENUE_COUNTRY_MAP:
        return VENUE_COUNTRY_MAP[venue_key]
    if city_key in CITY_TO_COUNTRY:
        return CITY_TO_COUNTRY[city_key]
    raw_low = re.sub(r"[^a-z0-9\s]", " ", str(venue_raw or "").lower())
    for ck, cc in CITY_TO_COUNTRY.items():
        if ck and (ck in venue_key or ck in city_key or ck in raw_low):
            return cc
    return ""


def _get_aligned_players_cache(render_fmt: str) -> Dict[str, Any]:
    """
    Centers Render batting/bowling sub-attribute vectors onto a common 55.0 cohort baseline
    so missingness of newly-debuted players never skews XI strength or Duel Simulator downward.
    """
    if render_fmt in _ALIGNED_PLAYERS_CACHE:
        return _ALIGNED_PLAYERS_CACHE[render_fmt]
    raw = _load_players_cache(render_fmt)
    if not raw:
        return {}
    keys = (
        "bat_score", "accel_bat", "acceleration_bat", "dependability_bat", "clutch_bat",
        "bowl_score", "spearhead_bowl", "xw_bowl", "clutch_death_bowl",
    )
    means: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in raw.values() if r.get(k) is not None]
        means[k] = float(np.mean(vals)) if vals else 52.0
    aligned: Dict[str, Any] = {}
    for pid, r in raw.items():
        rc = dict(r)
        for k in keys:
            if rc.get(k) is not None:
                rc[k] = round(max(20.0, min(95.0, float(rc[k]) + (55.0 - means[k]))), 2)
        aligned[pid] = rc
    _ALIGNED_PLAYERS_CACHE[render_fmt] = aligned
    return aligned


def _is_spin_style(style_or_role: str) -> bool:
    s = str(style_or_role or "").lower()
    return any(
        tok in s
        for tok in ("spin", "offbreak", "legbreak", "orthodox", "chinaman", "wrist", "slow", "sla", "ob", "lbg", "laws")
    )


def _is_pace_style(style_or_role: str) -> bool:
    s = str(style_or_role or "").lower()
    return any(tok in s for tok in ("fast", "medium", "pacer", "seam", "rf", "rmf", "lf", "lmf"))


def _resolve_render_player_record(
    player: Dict[str, Any],
    fmt: str,
    players_cache: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    pid = str(player.get("render_id") or player.get("cricsheet_id") or player.get("id") or "").strip()
    if pid and pid in players_cache:
        return players_cache[pid]
    name = str(player.get("name") or "").strip()
    if name:
        resolved = resolve_render_player_id(name, str(player.get("country") or player.get("team_id") or ""))
        if resolved and resolved in players_cache:
            return players_cache[resolved]
        name_low = name.lower()
        for rec in players_cache.values():
            if str(rec.get("name") or "").strip().lower() == name_low:
                return rec
    return None


def _resolve_render_adapt_record(
    player: Dict[str, Any],
    fmt: str,
    adapt_cache: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    pid = str(player.get("render_id") or player.get("cricsheet_id") or player.get("id") or "").strip()
    if pid and pid in adapt_cache:
        return adapt_cache[pid]
    name = str(player.get("name") or "").strip()
    if name:
        if name.lower() in adapt_cache:
            return adapt_cache[name.lower()]
        resolved = resolve_render_player_id(name, str(player.get("country") or player.get("team_id") or ""))
        if resolved and resolved in adapt_cache:
            return adapt_cache[resolved]
    return None


def compute_xi_bowling_profile(
    xi_players: List[Dict[str, Any]],
    fmt: str,
    players_cache: Dict[str, Any],
) -> Dict[str, float]:
    """Extracts pure pre-match spin vs pace bowling count & share from confirmed XIs."""
    n_spin = 0
    n_pace = 0
    for p in xi_players:
        rec = _resolve_render_player_record(p, fmt, players_cache) or {}
        role = str(rec.get("player_role") or p.get("role") or "").upper()
        b_style = str(rec.get("bowling_style") or p.get("bowling_style") or "")
        bowl_sc = float(rec.get("bowl_score") or 50.0)
        is_viable_bowler = ("BOWL" in role) or ("ALL" in role) or ("spinner" in role.lower()) or ("pacer" in role.lower()) or bowl_sc >= 52.0
        if not is_viable_bowler and not b_style:
            continue
        if _is_spin_style(b_style) or "spinner" in role.lower():
            n_spin += 1
        elif _is_pace_style(b_style) or "pacer" in role.lower():
            n_pace += 1
    tot = max(1, n_spin + n_pace)
    return {
        "n_spin": float(n_spin),
        "n_pace": float(n_pace),
        "spin_ratio": float(n_spin) / float(tot),
    }


def resolve_stage3_pitch_distribution_and_pars(
    fmt: str,
    venue_name: Optional[str],
    city: Optional[str] = None,
    pitch_override: Optional[str] = None,
    combined_spin_ratio: Optional[float] = None,
    priors: Optional[Dict[str, Any]] = None,
    venue_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes:
    - P(pitch | venue, city, format, XI_spin_ratio)
    - Empirical Bayes score mean for each pitch type on this same ground
    - Global format average score per pitch type
    - Weighted ground-pitch par & global pitch-type par
    """
    fmt_key = _normalize_fmt_key(fmt)
    cfg = _FMT_DEFAULTS[fmt_key]
    pr = priors if priors is not None else load_stage3_calibrated_priors()
    fmt_priors = pr.get(fmt_key, {})

    default_global_pars = cfg["global_pars"]
    global_pitch_pars = {
        p: float(fmt_priors.get("global_pitch_pars", {}).get(p, default_global_pars[p]))
        for p in CANONICAL_PITCHES
    }
    global_pitch_probs = {
        p: float(fmt_priors.get("global_pitch_probs", {}).get(p, 0.20))
        for p in CANONICAL_PITCHES
    }
    overall_fmt_mean = float(fmt_priors.get("overall_mean", cfg["overall_mean"]))

    v_key = normalize_venue_key(venue_name or venue_id)
    c_key = normalize_city_key(city)
    if not c_key and venue_name and "," in str(venue_name):
        c_key = normalize_city_key(str(venue_name).split(",", 1)[1])
    venues_map = fmt_priors.get("venues", {})
    cities_map = fmt_priors.get("cities", {})

    v_entry = venues_map.get(v_key)
    lookup_source = "venue_exact"
    if not v_entry and venue_id:
        dossier = FORMAT_GROUND_PITCH_DOSSIERS.get(str(venue_id).lower().replace("-", "_"), {})
        if dossier:
            v_entry = venues_map.get(normalize_venue_key(dossier.get("name")))
            if not c_key:
                c_key = normalize_city_key(dossier.get("city"))
    if not v_entry and c_key and c_key in cities_map:
        v_entry = cities_map[c_key]
        lookup_source = "city_fallback"
    elif not v_entry:
        lookup_source = "global_fallback"

    host_code = infer_host_country_code(v_key, c_key, venue_name)
    host_means = fmt_priors.get("host_means", {})
    host_par = float(host_means.get(host_code, overall_fmt_mean)) if host_code else overall_fmt_mean

    # Base ground par shrunk gently toward host/format prior preserving strong venue identity
    k_ven = cfg["k_ven"]
    if v_entry and float(v_entry.get("w_eff") or v_entry.get("n") or 0) > 0:
        n_v = float(v_entry.get("n") or 0)
        w_eff = float(v_entry.get("w_eff") or n_v)
        raw_v_mean = float(v_entry["mean"])
        w_v = w_eff / (w_eff + k_ven)
        ground_par = w_v * raw_v_mean + (1.0 - w_v) * host_par
        w_src = v_entry.get("pitch_w") or v_entry.get("pitch_counts") or {}
        raw_counts = {p: float(w_src.get(p) or 0.0) for p in CANONICAL_PITCHES}
        raw_means = {p: v_entry.get("pitch_means", {}).get(p) for p in CANONICAL_PITCHES}
        tot_vw = sum(raw_counts.values())
    else:
        n_v = 0.0
        w_eff = 0.0
        tot_vw = 0.0
        ground_par = host_par
        raw_counts = {p: 0.0 for p in CANONICAL_PITCHES}
        raw_means = {p: None for p in CANONICAL_PITCHES}

    # Same-ground pitch-type Empirical Bayes score means (60% ground identity + 40% pitch archetype anchor)
    k_vp = cfg["k_vp"]
    ground_pitch_eb_pars: Dict[str, float] = {}
    for p in CANONICAL_PITCHES:
        anchor_p = 0.60 * ground_par + 0.40 * global_pitch_pars[p]
        n_vp = raw_counts[p]
        m_vp = raw_means[p]
        if m_vp is not None and n_vp > 0:
            w_vp = n_vp / (n_vp + k_vp)
            ground_pitch_eb_pars[p] = round(w_vp * float(m_vp) + (1.0 - w_vp) * anchor_p, 2)
        else:
            ground_pitch_eb_pars[p] = round(anchor_p, 2)

    # Prior pitch distribution at this venue (Dirichlet-smoothed toward global distribution)
    if pitch_override:
        p_canon = canonicalize_pitch_type(pitch_override)
        pitch_probs = {p: (1.0 if p == p_canon else 0.0) for p in CANONICAL_PITCHES}
        dominant_pitch = p_canon
    else:
        k_smooth = cfg["k_smooth"]
        pitch_probs = {
            p: (raw_counts[p] + k_smooth * global_pitch_probs[p]) / (tot_vw + k_smooth)
            for p in CANONICAL_PITCHES
        }

        if combined_spin_ratio is not None:
            spin_excess = max(-0.25, min(0.25, float(combined_spin_ratio) - 0.48))
            pitch_probs["dusty"] *= max(0.65, 1.0 + spin_excess * 1.4)
            pitch_probs["green"] *= max(0.65, 1.0 - spin_excess * 1.0)
            pitch_probs["bouncy"] *= max(0.75, 1.0 - spin_excess * 0.5)

        tot_prob = sum(pitch_probs.values()) or 1.0
        pitch_probs = {p: round(pitch_probs[p] / tot_prob, 4) for p in CANONICAL_PITCHES}
        dominant_pitch = max(pitch_probs, key=lambda k: pitch_probs[k])

    same_ground_pitch_par = sum(pitch_probs[p] * ground_pitch_eb_pars[p] for p in CANONICAL_PITCHES)
    global_pitch_par = sum(pitch_probs[p] * global_pitch_pars[p] for p in CANONICAL_PITCHES)
    me_venue_offset = float(fmt_priors.get("me_venue", {}).get(v_key, host_par - overall_fmt_mean))

    return {
        "venue_key": v_key,
        "city_key": c_key,
        "host_country_code": host_code,
        "lookup_source": lookup_source,
        "venue_sample_size": int(n_v),
        "ground_overall_par": round(ground_par, 2),
        "me_venue_par": round(overall_fmt_mean + me_venue_offset, 2),
        "dominant_pitch": dominant_pitch,
        "pitch_probs": pitch_probs,
        "ground_pitch_eb_pars": ground_pitch_eb_pars,
        "global_pitch_pars": global_pitch_pars,
        "same_ground_pitch_weighted_par": round(same_ground_pitch_par, 2),
        "global_pitch_weighted_par": round(global_pitch_par, 2),
    }


def _player_surface_adapt_score(
    player: Dict[str, Any],
    pitch_type: str,
    fmt: str,
    adapt_rec: Optional[Dict[str, Any]],
    fallback_player_adapt: Optional[Dict[str, Any]],
    is_bowler: bool,
    bench_avg: float,
    bench_sr: float,
    bench_econ: float,
    bench_wkt: float,
) -> float:
    """Computes normalized 0-100 surface adaptability score on `pitch_type` using Render data + Cricsheet shrinkage."""
    score = None
    if adapt_rec:
        overall = float(adapt_rec.get("overall_score") or 65.0)
        avg_v = adapt_rec.get(f"{pitch_type}_avg")
        sr_v = adapt_rec.get(f"{pitch_type}_sr")
        balls_v = float(adapt_rec.get(f"{pitch_type}_balls") or 0.0)
        if not is_bowler and avg_v is not None and float(avg_v) > 0:
            sr_val = float(sr_v) if (sr_v is not None and float(sr_v) > 0) else bench_sr
            raw_sc = 50.0 + ((float(avg_v) - bench_avg) / 15.0) * 20.0 + ((sr_val - bench_sr) / 25.0) * 16.0
            w_b = balls_v / (balls_v + 70.0)
            score = w_b * max(25.0, min(98.0, raw_sc)) + (1.0 - w_b) * overall
        else:
            score = overall

    if score is None and fallback_player_adapt:
        by_pitch = fallback_player_adapt.get("by_pitch") if "by_pitch" in fallback_player_adapt else fallback_player_adapt
        p_map = (by_pitch or {}).get(pitch_type, {})
        if not is_bowler and fallback_player_adapt.get("balls"):
            wb = float(fallback_player_adapt["balls"]) / (float(fallback_player_adapt["balls"]) + 90.0)
            ov_avg = wb * min(75.0, float(fallback_player_adapt.get("avg", bench_avg))) + (1.0 - wb) * bench_avg
            ov_sr = wb * float(fallback_player_adapt.get("sr", bench_sr)) + (1.0 - wb) * bench_sr
            base_ad = 50.0 + ((ov_avg - bench_avg) / 15.0) * 18.0 + ((ov_sr - bench_sr) / 25.0) * 15.0
            if p_map and float(p_map.get("balls") or 0.0) > 0:
                raw_sc = 50.0 + ((float(p_map["avg"]) - bench_avg) / 15.0) * 18.0 + ((float(p_map.get("sr", bench_sr)) - bench_sr) / 25.0) * 15.0
                w_p = float(p_map["balls"]) / (float(p_map["balls"]) + 75.0)
                score = w_p * raw_sc + (1.0 - w_p) * base_ad
            else:
                score = base_ad
        elif is_bowler and fallback_player_adapt.get("balls"):
            wb = float(fallback_player_adapt["balls"]) / (float(fallback_player_adapt["balls"]) + 90.0)
            ov_ec = wb * float(fallback_player_adapt.get("econ", bench_econ)) + (1.0 - wb) * bench_econ
            ov_wk = wb * float(fallback_player_adapt.get("wkt_rate", bench_wkt)) + (1.0 - wb) * bench_wkt
            base_ad = 52.0 + ((bench_econ - ov_ec) / 1.5) * 18.0 + (ov_wk - bench_wkt) * 8.0
            if p_map and float(p_map.get("balls") or 0.0) > 0:
                raw_sc = 52.0 + ((bench_econ - float(p_map["econ"])) / 1.5) * 18.0
                w_p = float(p_map["balls"]) / (float(p_map["balls"]) + 60.0)
                score = w_p * raw_sc + (1.0 - w_p) * base_ad
            else:
                score = base_ad

    if score is None:
        p_adapt = (player.get("pitch_adaptability") or {}).get(pitch_type)
        score = float(p_adapt) if p_adapt is not None else (55.0 if is_bowler else 60.0)
    return float(max(20.0, min(99.0, score)))


def _build_render_backed_attrs(
    player: Dict[str, Any],
    fmt: str,
    opp_code: str,
    players_cache: Dict[str, Any],
    career_rec: Optional[Dict[str, Any]] = None,
    team_offset: float = 0.0,
    bench_avg: float = 30.0,
    bench_sr: float = 105.0,
    bench_econ: float = 6.5,
    is_bowler_slot: bool = False,
) -> Dict[str, Any]:
    """Resolves 22 attributes & power vectors from aligned Render snapshots, falling back to Cricsheet EB stats or team priors."""
    rec = _resolve_render_player_record(player, fmt, players_cache)
    if rec:
        acc = float(rec.get("accel_bat") or rec.get("acceleration_bat") or 55.0)
        dep_b = float(rec.get("dependability_bat") or 55.0)
        clu_b = float(rec.get("clutch_bat") or 55.0)
        spear = float(rec.get("spearhead_bowl") or 55.0)
        xw = float(rec.get("xw_bowl") or 55.0)
        clu_bw = float(rec.get("clutch_death_bowl") or 55.0)
        bat_sc = float(rec.get("bat_score") or 55.0)
        bowl_sc = float(rec.get("bowl_score") or 55.0)
        return {
            "player_id": rec.get("id") or player.get("cricsheet_id") or player.get("id"),
            "name": rec.get("name") or player.get("name"),
            "bat_score": bat_sc,
            "bowl_score": bowl_sc,
            "b_power": round(0.35 * acc + 0.35 * dep_b + 0.30 * clu_b, 2),
            "bw_power": round(0.35 * spear + 0.35 * xw + 0.30 * clu_bw, 2),
            "bowling_style": rec.get("bowling_style") or player.get("bowling_style") or "RF",
            "batting_style": rec.get("batting_style") or player.get("batting_hand") or "right",
            "role": rec.get("player_role") or player.get("role") or "BATTER",
        }

    if career_rec and float(career_rec.get("balls") or 0.0) > 0:
        wb = float(career_rec["balls"]) / (float(career_rec["balls"]) + 90.0)
        if not is_bowler_slot and career_rec.get("avg") is not None:
            p_av = wb * min(75.0, float(career_rec["avg"])) + (1.0 - wb) * bench_avg
            p_sr = wb * float(career_rec.get("sr", bench_sr)) + (1.0 - wb) * bench_sr
            sc = max(25.0, min(90.0, 55.0 + (p_av - bench_avg) * 0.9 + (p_sr - bench_sr) * 0.35))
            return {
                "player_id": player.get("cricsheet_id") or player.get("id"),
                "name": player.get("name"),
                "bat_score": round(sc, 2),
                "bowl_score": 38.0,
                "b_power": round(sc, 2),
                "bw_power": 35.0,
                "bowling_style": player.get("bowling_style") or "RF",
                "batting_style": player.get("batting_hand") or "right",
                "role": player.get("role") or "BATTER",
            }
        if is_bowler_slot and career_rec.get("econ") is not None:
            p_ec = wb * float(career_rec["econ"]) + (1.0 - wb) * bench_econ
            sc = max(30.0, min(90.0, 55.0 + (bench_econ - p_ec) * 8.0))
            return {
                "player_id": player.get("cricsheet_id") or player.get("id"),
                "name": player.get("name"),
                "bat_score": 35.0,
                "bowl_score": round(sc, 2),
                "b_power": 35.0,
                "bw_power": round(sc, 2),
                "bowling_style": player.get("bowling_style") or "RF",
                "batting_style": player.get("batting_hand") or "right",
                "role": player.get("role") or "BOWLER",
            }

    # Conditioned neutral cohort baseline using team's separated prior rating
    if not is_bowler_slot:
        sc = max(38.0, min(72.0, 55.0 + team_offset * 0.25))
    else:
        sc = max(38.0, min(72.0, 55.0 - team_offset * 0.25))
    return {
        "player_id": player.get("cricsheet_id") or player.get("id"),
        "name": player.get("name"),
        "bat_score": round(sc if not is_bowler_slot else 35.0, 2),
        "bowl_score": round(sc if is_bowler_slot else 38.0, 2),
        "b_power": round(sc if not is_bowler_slot else 35.0, 2),
        "bw_power": round(sc if is_bowler_slot else 35.0, 2),
        "bowling_style": player.get("bowling_style") or "RF",
        "batting_style": player.get("batting_hand") or "right",
        "role": player.get("role") or ("BOWLER" if is_bowler_slot else "BATTER"),
    }


def compute_stage3_xi_score_features(
    fmt: str,
    bat_team_id: str,
    bowl_team_id: str,
    bat_xi: List[Dict[str, Any]],
    bowl_xi: List[Dict[str, Any]],
    venue_name: Optional[str],
    city: Optional[str] = None,
    venue_id: Optional[str] = None,
    toss_bat_first_known: bool = True,
    toss_winner_bats: bool = True,
    pitch_override: Optional[str] = None,
    priors: Optional[Dict[str, Any]] = None,
    match_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes the complete Stage 3 feature set for projecting the batting-first team's innings score
    using all 6 signals (ground pitch distribution, same-ground pitch scores, global pitch-type means,
    separated country batting/bowling ratings + pitch adaptability, Render/career player adaptability & form,
    and Render-backed 7×5 Duel Simulator matrix).
    """
    fmt_key = _normalize_fmt_key(fmt)
    cfg = _FMT_DEFAULTS[fmt_key]
    render_fmt = cfg["render_fmt"]
    pr = priors if priors is not None else load_stage3_calibrated_priors()
    fmt_priors = pr.get(fmt_key, {})

    players_cache = _get_aligned_players_cache(render_fmt)
    adapt_cache = _load_adapt_cache(render_fmt)
    duels_cache = _load_duels_cache()

    bat_code = normalize_team_code(bat_team_id)
    bowl_code = normalize_team_code(bowl_team_id)

    bowl_prof_a = compute_xi_bowling_profile(bat_xi, render_fmt, players_cache)
    bowl_prof_b = compute_xi_bowling_profile(bowl_xi, render_fmt, players_cache)
    combined_spin_ratio = (bowl_prof_a["spin_ratio"] + bowl_prof_b["spin_ratio"]) / 2.0

    # 1, 2, 3: Ground Pitch Distribution + Same-Ground Pitch Scores + Global Pitch-Type Means
    ground_info = resolve_stage3_pitch_distribution_and_pars(
        fmt=fmt_key,
        venue_name=venue_name,
        city=city,
        venue_id=venue_id,
        pitch_override=pitch_override,
        combined_spin_ratio=combined_spin_ratio,
        priors=pr,
    )
    p_probs = ground_info["pitch_probs"]
    dom_pitch = ground_info["dominant_pitch"]
    host_code = ground_info.get("host_country_code", "")

    # 4. Separated Country Mixed-Effects Ratings + Home Edge + Pitch Adaptability
    me_bat = float(fmt_priors.get("me_bat", {}).get(bat_code, 0.0))
    me_bowl = float(fmt_priors.get("me_bowl", {}).get(bowl_code, 0.0))
    me_home = fmt_priors.get("me_home", {})
    bat_is_home = 1.0 if (host_code and bat_code == host_code) else 0.0
    bowl_is_home = 1.0 if (host_code and bowl_code == host_code) else 0.0
    home_delta = float(me_home.get("bat_home", 0.0)) * bat_is_home + float(me_home.get("bowl_home", 0.0)) * bowl_is_home

    c_bat_map = COUNTRY_PITCH_ADAPTABILITY.get(bat_code, {}).get(fmt_key, {})
    c_bowl_map = COUNTRY_PITCH_ADAPTABILITY.get(bowl_code, {}).get(fmt_key, {})
    bat_country_adapt = sum(p_probs[p] * float(c_bat_map.get(p, 75.0)) for p in CANONICAL_PITCHES)
    bowl_country_adapt = sum(p_probs[p] * float(c_bowl_map.get(p, 75.0)) for p in CANONICAL_PITCHES)

    sep_bat_res = fmt_priors.get("country_bat_pitch_residuals", {})
    sep_bowl_res = fmt_priors.get("country_bowl_pitch_residuals", {})
    emp_country = fmt_priors.get("country_pitch_residuals", {})
    if sep_bat_res:
        emp_bat_res = sum(p_probs[p] * float(sep_bat_res.get(bat_code, {}).get(p, 0.0)) for p in CANONICAL_PITCHES)
        emp_bowl_res = sum(p_probs[p] * float(sep_bowl_res.get(bowl_code, {}).get(p, 0.0)) for p in CANONICAL_PITCHES)
    else:
        emp_bat_res = sum(p_probs[p] * float(emp_country.get(bat_code, {}).get(p, 0.0)) for p in CANONICAL_PITCHES)
        emp_bowl_res = sum(p_probs[p] * float(emp_country.get(bowl_code, {}).get(p, 0.0)) for p in CANONICAL_PITCHES)

    # 5. Player Pitch Adaptability + Empirical Form across Top-7 Batters & Tail + Top-5 Bowlers
    train_bat_adapt = fmt_priors.get("player_bat_adapt", {})
    train_bowl_adapt = fmt_priors.get("player_bowl_adapt", {})

    base_avg_fmt = cfg["base_avg"]
    base_sr_fmt = cfg["base_sr"]
    base_econ_fmt = cfg["base_econ"]
    base_wkt_fmt = cfg["base_wkt"]

    bench_avg = base_avg_fmt + me_bat * 0.12
    bench_sr = base_sr_fmt + me_bat * (0.10 if fmt_key == "TEST" else 0.25)
    bench_econ = base_econ_fmt + me_bowl * cfg["me_bowl_econ_scale"]
    bench_wkt = base_wkt_fmt

    bat_weights = [1.25, 1.25, 1.15, 1.10, 1.0, 0.90, 0.75]
    bat_adapt_weighted = 0.0
    bat_sr_weighted = 0.0
    bat_avg_weighted = 0.0
    bat_w_sum = 0.0
    tail_avgs: List[float] = []
    bat_attrs_list: List[Dict[str, Any]] = []
    bat_power_vals: List[float] = []

    for idx, p in enumerate(bat_xi[:9]):
        pid_c = str(p.get("render_id") or p.get("cricsheet_id") or p.get("id") or "")
        fb_bat = train_bat_adapt.get(pid_c)
        if fb_bat and fb_bat.get("balls"):
            wb = float(fb_bat["balls"]) / (float(fb_bat["balls"]) + 90.0)
            p_sr = wb * float(fb_bat.get("sr", bench_sr)) + (1.0 - wb) * bench_sr
            p_av = wb * min(75.0, float(fb_bat.get("avg", bench_avg))) + (1.0 - wb) * bench_avg
        else:
            p_sr, p_av = bench_sr, bench_avg

        if idx >= 6:
            tail_avgs.append(p_av)
        if idx >= 7:
            continue

        w = bat_weights[idx] if idx < len(bat_weights) else 0.75
        rec_a = _resolve_render_adapt_record(p, render_fmt, adapt_cache)
        sc_p = sum(
            p_probs[pitch] * _player_surface_adapt_score(
                p, pitch, fmt_key, rec_a, fb_bat, is_bowler=False,
                bench_avg=bench_avg, bench_sr=bench_sr, bench_econ=bench_econ, bench_wkt=bench_wkt,
            )
            for pitch in CANONICAL_PITCHES
        )
        bat_adapt_weighted += sc_p * w
        bat_sr_weighted += p_sr * w
        bat_avg_weighted += p_av * w
        bat_w_sum += w

        attrs = _build_render_backed_attrs(
            p, render_fmt, bowl_code, players_cache,
            career_rec=fb_bat, team_offset=me_bat,
            bench_avg=bench_avg, bench_sr=bench_sr, bench_econ=bench_econ, is_bowler_slot=False,
        )
        attrs["_pitch_adapt_score"] = sc_p
        bat_attrs_list.append(attrs)
        bat_power_vals.append(float(attrs.get("bat_score") or 55.0))

    xi_bat_adapt = bat_adapt_weighted / max(0.1, bat_w_sum)
    xi_bat_sr = bat_sr_weighted / max(0.1, bat_w_sum)
    xi_bat_avg = bat_avg_weighted / max(0.1, bat_w_sum)
    xi_tail_avg = float(np.mean(tail_avgs)) if tail_avgs else (bench_avg * 0.5)
    xi_bat_strength = float(np.mean(bat_power_vals)) if bat_power_vals else 55.0
    xi_bat_index = 0.45 * (xi_bat_sr - base_sr_fmt) + 0.60 * (xi_bat_avg - base_avg_fmt) + 0.30 * (xi_tail_avg - base_avg_fmt * 0.5)

    scored_bowlers: List[Tuple[float, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]] = []
    for idx, p in enumerate(bowl_xi):
        pid_c = str(p.get("render_id") or p.get("cricsheet_id") or p.get("id") or "")
        fb_bw = train_bowl_adapt.get(pid_c)
        role_hint = str(p.get("role") or "").upper()
        slot_bowl = ("BOWL" in role_hint) or ("ALL" in role_hint) or idx >= max(0, len(bowl_xi) - 5)
        attrs = _build_render_backed_attrs(
            p, render_fmt, bat_code, players_cache,
            career_rec=fb_bw, team_offset=me_bowl,
            bench_avg=bench_avg, bench_sr=bench_sr, bench_econ=bench_econ, is_bowler_slot=slot_bowl,
        )
        role_up = str(attrs.get("role") or "").upper()
        bw_sc = float(attrs.get("bowl_score") or 55.0)
        is_bw = ("BOWL" in role_up) or ("ALL" in role_up) or bw_sc >= 54.0 or idx >= max(0, len(bowl_xi) - 5)
        rank_bonus = 20.0 if ("BOWL" in role_up or "ALL" in role_up) else 0.0
        scored_bowlers.append((bw_sc + rank_bonus if is_bw else bw_sc - 15.0, p, attrs, fb_bw))
    scored_bowlers.sort(key=lambda x: x[0], reverse=True)
    top5_bowlers = scored_bowlers[:5]

    bowl_adapt_vals: List[float] = []
    bowl_power_vals: List[float] = []
    bowl_econ_vals: List[float] = []
    bowl_wkt_vals: List[float] = []
    bowl_attrs_list: List[Dict[str, Any]] = []
    for _, p, attrs, fb_bw in top5_bowlers:
        sc_bw = sum(
            p_probs[pitch] * _player_surface_adapt_score(
                p, pitch, fmt_key, None, fb_bw, is_bowler=True,
                bench_avg=bench_avg, bench_sr=bench_sr, bench_econ=bench_econ, bench_wkt=bench_wkt,
            )
            for pitch in CANONICAL_PITCHES
        )
        if fb_bw and fb_bw.get("balls"):
            wb = float(fb_bw["balls"]) / (float(fb_bw["balls"]) + 90.0)
            p_ec = wb * float(fb_bw.get("econ", bench_econ)) + (1.0 - wb) * bench_econ
            p_wk = wb * float(fb_bw.get("wkt_rate", bench_wkt)) + (1.0 - wb) * bench_wkt
        else:
            p_ec, p_wk = bench_econ, bench_wkt
        attrs["_pitch_adapt_score"] = sc_bw
        bowl_adapt_vals.append(sc_bw)
        bowl_power_vals.append(float(attrs.get("bowl_score") or 55.0))
        bowl_econ_vals.append(p_ec)
        bowl_wkt_vals.append(p_wk)
        bowl_attrs_list.append(attrs)

    xi_bowl_adapt = float(np.mean(bowl_adapt_vals)) if bowl_adapt_vals else 55.0
    xi_bowl_strength = float(np.mean(bowl_power_vals)) if bowl_power_vals else 55.0
    xi_bowl_econ = float(np.mean(bowl_econ_vals)) if bowl_econ_vals else bench_econ
    xi_bowl_wkt = float(np.mean(bowl_wkt_vals)) if bowl_wkt_vals else bench_wkt
    xi_bowl_index = (xi_bowl_econ - base_econ_fmt) * cfg["bowl_idx_econ_scale"] - (xi_bowl_wkt - base_wkt_fmt) * cfg["bowl_idx_wkt_scale"]

    # 6. Confirmed XI (7×5) Duel Simulator Matrix anchored to a neutral simulator baseline
    base_fmt_rpo = cfg["base_rpo"]
    neutral_sim = compute_duel_simulation(
        batter_attrs=_NEUTRAL_BAT_SIM,
        bowler_attrs=_NEUTRAL_BWL_SIM,
        match_format=fmt_key,
        pitch_archetype=dom_pitch,
        batter_hand="right",
        bowler_style="RF",
        pitch_adaptability_delta=0.0,
    )
    neutral_sim_rpo = float(neutral_sim["expected_runs_per_ball"]) * 6.0

    duel_deltas: List[float] = []
    duel_wkt_samples: List[float] = []
    cached_duel_hits = 0

    for b_idx, b_attr in enumerate(bat_attrs_list):
        b_w = bat_weights[b_idx] if b_idx < len(bat_weights) else 0.75
        b_pid = str(b_attr.get("player_id") or "")
        for bw_attr in bowl_attrs_list:
            bw_pid = str(bw_attr.get("player_id") or "")
            adapt_delta = (float(b_attr.get("_pitch_adapt_score", 60.0)) - float(bw_attr.get("_pitch_adapt_score", 55.0))) / 25.0
            d_sim = compute_duel_simulation(
                batter_attrs=b_attr,
                bowler_attrs=bw_attr,
                match_format=fmt_key,
                pitch_archetype=dom_pitch,
                batter_hand=str(b_attr.get("batting_style") or "right"),
                bowler_style=str(bw_attr.get("bowling_style") or "RF"),
                pitch_adaptability_delta=adapt_delta,
            )
            sim_rpo = float(d_sim["expected_runs_per_ball"]) * 6.0
            delta_rpo = sim_rpo - neutral_sim_rpo
            p_wkt = float(d_sim["probabilities"]["wicket"])

            cache_k = f"{b_pid}_{bw_pid}_{render_fmt}"
            d_cached = duels_cache.get(cache_k) if (b_pid and bw_pid) else None
            if d_cached and d_cached.get("expected_runs_per_over"):
                cached_duel_hits += 1
                cached_anom = float(d_cached["expected_runs_per_over"]) - base_fmt_rpo
                delta_rpo = 0.65 * delta_rpo + 0.35 * cached_anom
                p_wkt = 0.65 * p_wkt + 0.35 * (float(d_cached.get("wicket_risk_pct") or 18.0) / 600.0)

            duel_deltas.append(delta_rpo * b_w)
            duel_wkt_samples.append(p_wkt)

    if duel_deltas:
        norm_w = sum(bat_weights[i] if i < len(bat_weights) else 0.75 for i in range(len(bat_attrs_list))) * max(1, len(bowl_attrs_list))
        mean_delta_rpo = sum(duel_deltas) / max(0.1, norm_w)
        duel_mean_rpo = base_fmt_rpo + mean_delta_rpo
        duel_mean_wkt_prob = float(np.mean(duel_wkt_samples))
    else:
        mean_delta_rpo = 0.0
        duel_mean_rpo = base_fmt_rpo
        duel_mean_wkt_prob = cfg["default_wkt_prob"]

    innings_overs = cfg["innings_overs"]
    duel_run_delta = mean_delta_rpo * (innings_overs * 0.45)

    mo = str(match_date or "")[5:7]
    month_eff = float(fmt_priors.get("month_means", {}).get(mo, 0.0)) if len(mo) == 2 else 0.0

    return {
        "format": fmt_key,
        "ground_info": ground_info,
        "same_ground_pitch_par": ground_info["same_ground_pitch_weighted_par"],
        "global_pitch_par": ground_info["global_pitch_weighted_par"],
        "ground_overall_par": ground_info["ground_overall_par"],
        "me_venue_par": ground_info["me_venue_par"],
        "dominant_pitch": dom_pitch,
        "combined_spin_ratio": round(combined_spin_ratio, 3),
        "me_bat_rating": round(me_bat, 2),
        "me_bowl_rating": round(me_bowl, 2),
        "home_delta": round(home_delta, 2),
        "bat_country_adapt": round(bat_country_adapt, 2),
        "bowl_country_adapt": round(bowl_country_adapt, 2),
        "emp_bat_country_residual": round(emp_bat_res, 2),
        "emp_bowl_country_residual": round(emp_bowl_res, 2),
        "xi_bat_adapt": round(xi_bat_adapt, 2),
        "xi_bowl_adapt": round(xi_bowl_adapt, 2),
        "xi_bat_strength": round(xi_bat_strength, 2),
        "xi_bowl_strength": round(xi_bowl_strength, 2),
        "xi_bat_index": round(xi_bat_index, 2),
        "xi_bowl_index": round(xi_bowl_index, 2),
        "duel_mean_rpo": round(duel_mean_rpo, 3),
        "duel_mean_wkt_prob": round(duel_mean_wkt_prob, 4),
        "duel_run_delta": round(duel_run_delta, 2),
        "cached_duel_hits": cached_duel_hits,
        "month_eff": round(month_eff, 2),
        "toss_winner_bats": 1.0 if (toss_bat_first_known and toss_winner_bats) else 0.0,
    }


def predict_stage3_calibrated_score(
    features: Dict[str, Any],
    priors: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, int]:
    """
    Applies L2-regularized, sign-constrained weights (fitted strictly on 2018-2024 historical matches)
    to project 1st-innings score + calibrated 10th/90th percentile band.
    """
    fmt_key = _normalize_fmt_key(features["format"])
    cfg = _FMT_DEFAULTS[fmt_key]
    pr = priors if priors is not None else load_stage3_calibrated_priors()
    model = pr.get(fmt_key, {}).get("score_model", {})

    sg_par = float(features["same_ground_pitch_par"])
    gp_par = float(features["global_pitch_par"])
    go_par = float(features["ground_overall_par"])
    mv_par = float(features.get("me_venue_par", go_par))

    if model and "weights" in model:
        w = model["weights"]
        fm = model.get("feature_means", {})
        c_vals = {
            "me_bat_rating": float(features.get("me_bat_rating", 0.0)) - float(fm.get("me_bat_rating", 0.0)),
            "me_bowl_rating": float(features.get("me_bowl_rating", 0.0)) - float(fm.get("me_bowl_rating", 0.0)),
            "home_delta": float(features.get("home_delta", 0.0)) - float(fm.get("home_delta", 0.0)),
            "country_adapt_diff": (float(features["bat_country_adapt"]) - float(features["bowl_country_adapt"])) - float(fm.get("country_adapt_diff", 0.0)),
            "emp_country_residual": (float(features["emp_bat_country_residual"]) - float(features["emp_bowl_country_residual"])) - float(fm.get("emp_country_residual", 0.0)),
            "xi_bat_adapt_diff": float(features["xi_bat_adapt"]) - float(fm.get("xi_bat_adapt", 55.0)),
            "xi_bowl_adapt_diff": float(features["xi_bowl_adapt"]) - float(fm.get("xi_bowl_adapt", 55.0)),
            "xi_strength_diff": (float(features["xi_bat_strength"]) - float(features["xi_bowl_strength"])) - float(fm.get("xi_strength_diff", 0.0)),
            "xi_bat_index": float(features.get("xi_bat_index", 0.0)) - float(fm.get("xi_bat_index", 0.0)),
            "xi_bowl_index": float(features.get("xi_bowl_index", 0.0)) - float(fm.get("xi_bowl_index", 0.0)),
            "duel_run_delta": float(features["duel_run_delta"]) - float(fm.get("duel_run_delta", 0.0)),
            "month_eff": float(features.get("month_eff", 0.0)) - float(fm.get("month_eff", 0.0)),
            "toss_winner_bats": float(features["toss_winner_bats"]) - float(fm.get("toss_winner_bats", 0.4)),
        }
        raw_pred = (
            float(model.get("intercept", 0.0))
            + float(w.get("same_ground_pitch_par", 0.35)) * sg_par
            + float(w.get("global_pitch_par", 0.25)) * gp_par
            + float(w.get("ground_overall_par", 0.15)) * go_par
            + float(w.get("me_venue_par", 0.25)) * mv_par
        )
        for k, v in c_vals.items():
            raw_pred += float(w.get(k, 0.0)) * v
        resid_std = float(model.get("residual_std", cfg["default_resid_std"]))
    else:
        raw_pred = (
            0.35 * sg_par
            + 0.25 * gp_par
            + 0.15 * go_par
            + 0.25 * mv_par
            + 0.35 * float(features.get("me_bat_rating", 0.0))
            + 0.35 * float(features.get("me_bowl_rating", 0.0))
            + 0.25 * (features["xi_bat_adapt"] - 55.0)
            - 0.20 * (features["xi_bowl_adapt"] - 55.0)
            + 0.35 * (features["xi_bat_strength"] - features["xi_bowl_strength"])
            + 0.25 * (features["bat_country_adapt"] - features["bowl_country_adapt"])
            + 0.25 * features["duel_run_delta"]
        )
        resid_std = cfg["default_resid_std"]

    clip_lo, clip_hi = cfg["clip"]
    proj = int(round(max(clip_lo, min(clip_hi, raw_pred))))
    band = int(round(1.2816 * resid_std))
    proj_min = int(round(max(clip_lo - 20.0, proj - band)))
    proj_max = int(round(min(clip_hi + 20.0, proj + band)))
    return proj, proj_min, proj_max


def compute_stage3_dual_first_innings_projections(
    fmt: str,
    team_a_id: str,
    team_b_id: str,
    xi_a: List[Dict[str, Any]],
    xi_b: List[Dict[str, Any]],
    venue_name: Optional[str],
    city: Optional[str] = None,
    venue_id: Optional[str] = None,
    toss_winner_id: Optional[str] = None,
    toss_decision: Optional[str] = None,
    pitch_override: Optional[str] = None,
    boundary_multiplier: float = 1.0,
    match_date: Optional[str] = None,
    priors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes Stage 3 calibrated 1st-innings projections for BOTH scenarios:
    - If Country A plays 1st inning
    - If Country B plays 1st inning
    ready immediately once CricAPI provides schedule + Playing 11 + venue (with or without toss).
    """
    fmt_key = _normalize_fmt_key(fmt)
    code_a = normalize_team_code(team_a_id)
    code_b = normalize_team_code(team_b_id)
    toss_known = bool(toss_winner_id and toss_decision)
    toss_code = normalize_team_code(toss_winner_id) if toss_known else ""
    b_shift = int(round((float(boundary_multiplier) - 1.0) * _FMT_DEFAULTS[fmt_key]["b_shift"]))

    def _run_side(bat_id: str, bat_code: str, bowl_id: str, b_xi: List[Dict[str, Any]], bw_xi: List[Dict[str, Any]]) -> Dict[str, Any]:
        feats = compute_stage3_xi_score_features(
            fmt=fmt_key,
            bat_team_id=bat_id,
            bowl_team_id=bowl_id,
            bat_xi=b_xi,
            bowl_xi=bw_xi,
            venue_name=venue_name,
            city=city,
            venue_id=venue_id,
            toss_bat_first_known=toss_known,
            toss_winner_bats=(toss_code == bat_code) if toss_known else False,
            pitch_override=pitch_override,
            priors=priors,
            match_date=match_date,
        )
        proj, p_min, p_max = predict_stage3_calibrated_score(feats, priors=priors)
        proj_adj = proj + b_shift
        min_adj = p_min + b_shift
        max_adj = p_max + b_shift
        return {
            "batting_first_id": bat_id,
            "bowling_first_id": bowl_id,
            "projected_score": proj_adj,
            "min_score": min_adj,
            "max_score": max_adj,
            "score_range": f"{min_adj} - {max_adj}",
            "features": feats,
        }

    side_a = _run_side(team_a_id, code_a, team_b_id, xi_a, xi_b)
    side_b = _run_side(team_b_id, code_b, team_a_id, xi_b, xi_a)

    return {
        "format": fmt_key,
        "toss_confirmed": toss_known,
        "if_team_a_bats_first": side_a,
        "if_team_b_bats_first": side_b,
    }
