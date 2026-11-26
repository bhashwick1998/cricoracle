"""
Render Analytics Bridge: Integration with https://cricket-analytics-ghpv.onrender.com/
Connects CricOracle with Render.com's Senior International Cricket Intelligence Platform:
- 22 Master Attributes (0-100 normalized scale, median 50.0)
- 5-Surface Pitch Adaptability (GREEN, DUSTY, BOUNCY, FLAT, BALANCED)
- Dual Simulated Matchups (/api/duel) with empirical Bayesian crease duel outcomes
- Automatic disk & in-memory caching with graceful offline fallbacks
"""

import os
import re
import json
import tempfile
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

RENDER_BASE_URL = "https://cricket-analytics-ghpv.onrender.com"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LOOKUP_FILE = os.path.join(DATA_DIR, "render_player_lookup.json")
PLAYERS_TEST_FILE = os.path.join(DATA_DIR, "render_players_test.json")
ADAPT_TEST_FILE = os.path.join(DATA_DIR, "render_adaptability_test.json")
DUELS_CACHE_FILE = os.path.join(DATA_DIR, "render_duels_cache.json")

# In-memory caches for < 1ms response latency
_memory_lookup: Optional[Dict[str, Any]] = None
_memory_players: Dict[str, Dict[str, Any]] = {}
_memory_adapt: Dict[str, Dict[str, Any]] = {}
_memory_duels: Dict[str, Any] = {}


def invalidate_render_caches() -> None:
    """Clears in-memory Render caches so refreshed JSON snapshots load cleanly."""
    global _memory_lookup, _memory_players, _memory_adapt, _memory_duels
    _memory_lookup = None
    _memory_players.clear()
    _memory_adapt.clear()
    _memory_duels.clear()

# High-priority curated alias map for instant resolution
KNOWN_ALIASES = {
    # Sri Lanka
    "kamindu mendis": "08548b13",
    "k_mendis_kam_sl": "08548b13",
    "phkd mendis": "08548b13",
    "dhananjaya de silva": "7d608e12",
    "d_desilva_sl": "7d608e12",
    "dm de silva": "7d608e12",
    "dinesh chandimal": "8dd02a98",
    "d_chandimal_sl": "8dd02a98",
    "ld chandimal": "8dd02a98",
    "kusal mendis": "5d1e7582",
    "k_mendis_sl": "5d1e7582",
    "bkg mendis": "5d1e7582",
    "pasindu sooriyabandara": "cc17536f",
    "p_sooriyabandara_sl": "cc17536f",
    "dps sooriyabandara": "cc17536f",
    "lahiru udara": "07b6c3c1",
    "l_udara_sl": "07b6c3c1",
    "lu igalagamage": "07b6c3c1",
    "sonal dinusha": "4bbfddd5",
    "s_dinusha_sl": "4bbfddd5",
    "gs dinusha": "4bbfddd5",
    "milan rathnayake": "afe830a2",
    "m_rathnayake_sl": "afe830a2",
    "rmmp rathnayake": "afe830a2",
    "lahiru kumara": "4751caa3",
    "l_kumara_sl": "4751caa3",
    "cbrls kumara": "4751caa3",
    "asitha fernando": "de3d549a",
    "a_fernando_sl": "de3d549a",
    "am fernando": "de3d549a",
    "dilshan madushanka": "de7d833e",
    "d_madushanka_sl": "de7d833e",
    "wanindu hasaranga": "a97c8ec2",
    "w_hasaranga_sl": "a97c8ec2",
    "pwh de silva": "a97c8ec2",
    "prabath jayasuriya": "127bc5ca",
    "p_jayasuriya_sl": "127bc5ca",
    "pathum nissanka": "8ee36b18",
    "p_nissanka_sl": "8ee36b18",
    "charith asalanka": "732c038e",
    "c_asalanka_sl": "732c038e",
    "dimuth karunaratne": "400e2d4d",
    "d_karunaratne_sl": "400e2d4d",
    "angelo mathews": "89d031bf",
    "a_mathews_sl": "89d031bf",

    # Pakistan
    "shaheen afridi": "45a7e761",
    "shaheen shah afridi": "45a7e761",
    "s_afridi_pak": "45a7e761",
    "mohammad abbas": "a089f93f",
    "m_abbas_pak": "a089f93f",
    "naseem shah": "9c9af282",
    "n_shah_pak": "9c9af282",
    "babar azam": "8a75e999",
    "b_azam_pak": "8a75e999",
    "saud shakeel": "d07c1b2f",
    "s_shakeel_pak": "d07c1b2f",
    "salman agha": "89c16049",
    "agha salman": "89c16049",
    "s_ali_agha_pak": "89c16049",
    "mohammad rizwan": "2f26ac1a",
    "m_rizwan_pak": "2f26ac1a",
    "aamer jamal": "a8e54ef4",
    "a_jamal_pak": "a8e54ef4",
    "saim ayub": "33609a8c",
    "s_ayub_pak": "33609a8c",
    "abdullah shafique": "fc2fffb5",
    "a_shafique_pak": "fc2fffb5",
    "shan masood": "6843a783",
    "s_masood_pak": "6843a783",
    "haris rauf": "24bb1c2f",
    "h_rauf_pak": "24bb1c2f",
    "hasan ali": "2911de16",
    "h_ali_pak": "2911de16",
    "sajid khan": "bbc192a4",
    "s_khan_pak": "bbc192a4",
    "nauman ali": "54c58a5e",
    "n_ali_pak": "54c58a5e",
    "abrar ahmed": "abb7c76c",
    "a_ahmed_pak": "abb7c76c",
    "mir hamza": "cb1a782a",
    "m_hamza_pak": "cb1a782a",
    "khurram shahzad": "27c5715d",
    "k_shahzad_pak": "27c5715d",

    # India & Global
    "virat kohli": "ba607b88",
    "v_kohli": "ba607b88",
    "rohit sharma": "d9818804",
    "r_sharma": "d9818804",
    "jasprit bumrah": "d6db0d30",
    "j_bumrah": "d6db0d30",
    "mitchell starc": "3fb19989",
    "m_starc": "3fb19989",
    "steve smith": "c867201c",
    "s_smith": "c867201c",
    "pat cummins": "13a8f4c4",
    "p_cummins": "13a8f4c4"
}

def _load_lookup() -> Dict[str, Any]:
    global _memory_lookup
    if _memory_lookup is not None:
        return _memory_lookup
    if os.path.exists(LOOKUP_FILE):
        try:
            with open(LOOKUP_FILE, 'r', encoding='utf-8') as f:
                _memory_lookup = json.load(f)
                return _memory_lookup
        except Exception:
            pass
    _memory_lookup = {}
    return _memory_lookup

def _load_players_cache(fmt: str = "TEST") -> Dict[str, Any]:
    global _memory_players
    fmt_key = fmt.upper()
    if fmt_key in _memory_players:
        return _memory_players[fmt_key]
    file_path = PLAYERS_TEST_FILE if fmt_key == "TEST" else os.path.join(DATA_DIR, f"render_players_{fmt_key.lower()}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                _memory_players[fmt_key] = json.load(f)
                return _memory_players[fmt_key]
        except Exception:
            pass
    _memory_players[fmt_key] = {}
    return _memory_players[fmt_key]

def _load_adapt_cache(fmt: str = "TEST") -> Dict[str, Any]:
    global _memory_adapt
    fmt_key = fmt.upper()
    if fmt_key in _memory_adapt:
        return _memory_adapt[fmt_key]
    file_path = ADAPT_TEST_FILE if fmt_key == "TEST" else os.path.join(DATA_DIR, f"render_adaptability_{fmt_key.lower()}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                _memory_adapt[fmt_key] = json.load(f)
                return _memory_adapt[fmt_key]
        except Exception:
            pass
    _memory_adapt[fmt_key] = {}
    return _memory_adapt[fmt_key]

def _load_duels_cache() -> Dict[str, Any]:
    global _memory_duels
    if _memory_duels:
        return _memory_duels
    if os.path.exists(DUELS_CACHE_FILE):
        try:
            with open(DUELS_CACHE_FILE, 'r', encoding='utf-8') as f:
                _memory_duels = json.load(f)
                return _memory_duels
        except Exception:
            pass
    _memory_duels = {}
    return _memory_duels

def _save_duels_cache(cache: Dict[str, Any]):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".render_duels.", suffix=".tmp", dir=DATA_DIR)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
            os.replace(tmp_name, DUELS_CACHE_FILE)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
    except Exception:
        pass

def resolve_render_player_id(player_name_or_id: str, team_country: Optional[str] = None) -> Optional[str]:
    """
    Resolves local player names/IDs to Render.com unique player IDs.
    Checks curated aliases first, then fuzzy candidate lookup.
    """
    if not player_name_or_id:
        return None
    k = player_name_or_id.strip().lower()
    if k in KNOWN_ALIASES:
        return KNOWN_ALIASES[k]

    # Clean underscores and suffixes
    clean_k = re.sub(r'_[a-z]{2,3}$', '', k).replace('_', ' ')
    if clean_k in KNOWN_ALIASES:
        return KNOWN_ALIASES[clean_k]

    lookup = _load_lookup()
    if k in lookup:
        return lookup[k]['id']
    if clean_k in lookup:
        return lookup[clean_k]['id']

    # Substring search
    tokens = [t for t in re.split(r'[\s_]+', clean_k) if len(t) > 2 and t not in ['mendis', 'khan', 'sharma', 'singh', 'ahmed', 'ali']]
    if tokens:
        last_t = tokens[-1]
        for name_key, v in lookup.items():
            if last_t in name_key:
                if team_country and v.get('country') and team_country.lower() not in v['country'].lower():
                    continue
                return v['id']

    return None

def get_render_player_profile(player_name_or_id: str, fmt: str = "TEST") -> Optional[Dict[str, Any]]:
    """
    Returns 22 Master Attributes & composite rankings from Render.com for a player.
    """
    pid = resolve_render_player_id(player_name_or_id)
    if not pid:
        return None
    players = _load_players_cache(fmt)
    if pid in players:
        return players[pid]
    return None

def get_render_pitch_adaptability(player_name_or_id: str, fmt: str = "TEST") -> Optional[Dict[str, Any]]:
    """
    Returns 5-Surface Pitch Adaptability (GREEN, DUSTY, BOUNCY, FLAT, BALANCED) from Render.com.
    Automatically substitutes null foreign surface averages with home surface experience.
    """
    pid = resolve_render_player_id(player_name_or_id)
    if not pid:
        return None
    adapt_map = _load_adapt_cache(fmt)
    if pid in adapt_map:
        record = dict(adapt_map[pid])
        surfaces = ["dusty", "green", "bouncy", "flat", "balanced"]
        for s in surfaces:
            if record.get(f"{s}_avg") is None and record.get(f"{s}_home_avg") is not None:
                record[f"{s}_avg"] = record.get(f"{s}_home_avg")
                if record.get(f"{s}_sr") in [None, 0, 0.0] and record.get(f"{s}_home_sr"):
                    record[f"{s}_sr"] = record.get(f"{s}_home_sr")
                record[f"{s}_source"] = "HOME_SURFACE_EXPERIENCE"
            elif record.get(f"{s}_avg") is not None and not record.get(f"{s}_source"):
                record[f"{s}_source"] = "FOREIGN_DATA"
        return record
    return None

def get_render_crease_duel(
    batter_name_or_id: str,
    bowler_name_or_id: str,
    fmt: str = "TEST",
    team_bat: Optional[str] = None,
    team_bowl: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetches dual simulated matchup between striker and bowler from Render.com /api/duel.
    Returns:
    - expected_runs_per_over: Expected runs conceded/scored
    - wicket_risk_pct: Dismissal hazard over 6 balls
    - probabilities: dot_pct, rotation_pct, boundary_pct, wicket_pct
    - power ratings & tactical edge
    """
    b_id = resolve_render_player_id(batter_name_or_id, team_bat)
    bw_id = resolve_render_player_id(bowler_name_or_id, team_bowl)

    if not b_id or not bw_id:
        return None

    render_fmt = "T20" if fmt.upper() in ["T20", "T20I"] else fmt.upper()
    cache_key = f"{b_id}_{bw_id}_{render_fmt}"
    duels_cache = _load_duels_cache()
    if cache_key in duels_cache:
        return duels_cache[cache_key]

    # Live query with 3.5s timeout
    url = f"{RENDER_BASE_URL}/api/duel?batter_id={b_id}&bowler_id={bw_id}&format={render_fmt}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 CricOracle/2.0'})
    try:
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if data.get('success'):
                sim_res = {
                    'batter_id': b_id,
                    'bowler_id': bw_id,
                    'batter_name': data.get('batter', {}).get('name'),
                    'bowler_name': data.get('bowler', {}).get('name'),
                    'expected_runs_per_over': data.get('simulated_over', {}).get('expected_runs', 5.5),
                    'wicket_risk_pct': data.get('simulated_over', {}).get('wicket_risk_pct', 20.0),
                    'dot_pct': data.get('probabilities', {}).get('dot_pct', 45.0),
                    'boundary_pct': data.get('probabilities', {}).get('boundary_pct', 12.0),
                    'rotation_pct': data.get('probabilities', {}).get('rotation_pct', 35.0),
                    'batter_power_rating': data.get('batter', {}).get('power_rating', 50.0),
                    'bowler_power_rating': data.get('bowler', {}).get('power_rating', 50.0),
                    'tactical_edge': data.get('simulated_over', {}).get('tactical_edge', 'NEUTRAL'),
                    '_updated_at': datetime.now(timezone.utc).isoformat()
                }
                duels_cache[cache_key] = sim_res
                _save_duels_cache(duels_cache)
                return sim_res
    except Exception:
        pass

    return None
