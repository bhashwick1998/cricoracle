"""
Cricpulse Analytics Intelligence Provider
Connects CricOracle with Cricpulse AI (https://cricket-analytics-ghpv.onrender.com)
to extract advanced player rankings, 18-attribute ratings, Head-to-Head Duel simulations,
country foreign adaptability indices, venue pitch archetypes, and two-player matchup comparisons.
Includes bi-directional pitch linking (including BALANCED) and smart player ID resolution.
"""

import httpx
import logging
import re
import urllib.parse
from typing import Dict, Any, List, Optional

logger = logging.getLogger("cricpulse_provider")

# 5 Canonical Pitch Archetypes linked across Render.com and CricOracle
PITCH_MAP_RENDER_TO_LOCAL = {
    "BALANCED": "balanced",
    "FLAT": "flat",
    "DUSTY": "dusty",
    "GREEN": "green",
    "BOUNCY": "bouncy"
}

PITCH_MAP_LOCAL_TO_RENDER = {
    "balanced": "BALANCED",
    "flat": "FLAT",
    "dusty": "DUSTY",
    "green": "GREEN",
    "bouncy": "BOUNCY"
}

def canonicalize_render_pitch(pitch: str) -> str:
    """Normalizes any pitch string to Render.com uppercase archetype."""
    if not pitch:
        return "BALANCED"
    p = str(pitch).strip().upper()
    if p in ["DUSTY", "SPIN_HEAVY", "SPIN", "TURNER"]:
        return "DUSTY"
    if p in ["GREEN", "LUSH_GREEN", "SEAM"]:
        return "GREEN"
    if p in ["BOUNCY", "BOUNCE", "PACE"]:
        return "BOUNCY"
    if p in ["FLAT", "ROAD"]:
        return "FLAT"
    return "BALANCED"

class CricpulseAnalyticsProvider:
    BASE_URL = "https://cricket-analytics-ghpv.onrender.com"

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CricOracle/2.0",
            "Accept": "application/json"
        }
        self._cache: Dict[str, Any] = {}
        self._player_id_cache: Dict[str, str] = {
            "p_babar_azam": "8a75e999",
            "p_shaheen_afridi": "45a7e761",
            "p_kamindu_mendis": "08548b13",
            "rsharma_ind": "a1cbdeeb",
            "vkohli_ind": "ba607b88",
            "babar_azam": "8a75e999",
            "shaheen_afridi": "45a7e761",
            "kamindu_mendis": "08548b13"
        }

    async def resolve_render_player_id(self, player_identifier: str) -> Optional[str]:
        """
        Resolves local player IDs, slugs, or full names to Render's 8-character Cricsheet hex ID.
        Example: 'p_babar_azam' -> '8a75e999', 'Kamindu Mendis' -> '08548b13'
        """
        if not player_identifier:
            return None
        pid = str(player_identifier).strip()
        
        # 1. Already an 8-character hex ID?
        if len(pid) == 8 and re.match(r"^[0-9a-fA-F]{8}$", pid):
            return pid.lower()

        # 2. Check in-memory cache
        low_id = pid.lower()
        if low_id in self._player_id_cache:
            return self._player_id_cache[low_id]

        # 3. Clean search term from slug or name
        search_term = pid
        if search_term.startswith("p_"):
            search_term = search_term[2:].replace("_", " ").title()
        elif "_" in search_term:
            search_term = search_term.replace("_", " ").title()

        try:
            encoded = urllib.parse.quote(search_term)
            url = f"{self.BASE_URL}/api/players?search={encoded}"
            res = await self._get(url)
            if isinstance(res, dict) and res.get("players"):
                top_hit = res["players"][0]
                resolved = top_hit.get("id") or top_hit.get("player_id")
                if resolved:
                    self._player_id_cache[low_id] = str(resolved).lower()
                    return str(resolved).lower()
        except Exception as e:
            logger.warning(f"Could not resolve player ID for '{player_identifier}': {e}")

        return None

    async def get_stats(self) -> Dict[str, Any]:
        """Global analytics database coverage (balls, matches, players, venues)."""
        url = f"{self.BASE_URL}/api/stats"
        return await self._get(url)

    async def get_countries(self) -> List[Dict[str, Any]]:
        """List of all tracked cricket nations and classification tiers."""
        url = f"{self.BASE_URL}/api/countries"
        res = await self._get(url)
        return res.get("countries", []) if isinstance(res, dict) else []

    async def get_venues(self, match_format: str = "ODI") -> List[Dict[str, Any]]:
        """158 venue dossiers with PQI, format-specific pitch archetypes, toss bias, and tactical suits."""
        fmt = "T20" if match_format.upper() in ["T20", "T20I"] else match_format.upper()
        url = f"{self.BASE_URL}/api/venues?format={fmt}"
        res = await self._get(url)
        venues = res.get("venues", []) if isinstance(res, dict) else []
        for v in venues:
            # Map Render pitch archetypes to canonical lowercase for local consumption
            render_pitch = v.get("pitch_type") or v.get("archetype") or "BALANCED"
            v["pitch_type_canonical"] = PITCH_MAP_RENDER_TO_LOCAL.get(render_pitch.upper(), "balanced")
            v["pitch_type_render"] = render_pitch.upper()
        return venues

    async def get_country_adaptability(self, match_format: str = "ODI", gender: str = "MALE", surface_mode: str = "touring") -> List[Dict[str, Any]]:
        """
        Country Foreign Adaptability Index across SENA and foreign surfaces including BALANCED.
        Supports surface_mode='touring' vs surface_mode='home' (5 Home Surface Baselines),
        covering all Top 20 ICC nations (11 Full Members + 9 Associates) and Dual Top Touring Performers.
        """
        fmt = "T20" if match_format.upper() in ["T20", "T20I"] else match_format.upper()
        url = f"{self.BASE_URL}/api/adaptability/countries?format={fmt}&gender={gender.upper()}"
        res = await self._get(url)
        countries = res.get("countries", []) if isinstance(res, dict) else []

        if not countries:
            # Resilient fallback using Top 20 ICC Country Adaptability Engine
            from backend.engine.cricsheet_pitch_intelligence import (
                COUNTRY_PITCH_ADAPTABILITY,
                COUNTRY_HOME_SURFACES,
                COUNTRY_TOP_TOURING_PERFORMERS,
                ICC_FULL_MEMBERS
            )
            fmt_key = "T20I" if fmt in ["T20", "T20I"] else fmt
            country_names = {
                "IND": "India", "AUS": "Australia", "ENG": "England", "SA": "South Africa",
                "NZ": "New Zealand", "PAK": "Pakistan", "WI": "West Indies", "SL": "Sri Lanka",
                "BAN": "Bangladesh", "AFG": "Afghanistan", "ZIM": "Zimbabwe", "IRE": "Ireland",
                "NED": "Netherlands", "SCO": "Scotland", "NAM": "Namibia", "USA": "United States",
                "NEP": "Nepal", "OMA": "Oman", "UAE": "United Arab Emirates", "CAN": "Canada", "PNG": "Papua New Guinea"
            }
            for code, data_dict in COUNTRY_PITCH_ADAPTABILITY.items():
                if fmt_key not in data_dict:
                    continue
                c_scores = data_dict[fmt_key]
                home_scores = COUNTRY_HOME_SURFACES.get(code, {}).get(fmt_key, {})
                top_p = COUNTRY_TOP_TOURING_PERFORMERS.get(code, {})
                favored = max(c_scores.items(), key=lambda x: x[1])[0]
                overall = round(sum(c_scores.values()) / len(c_scores), 1)

                countries.append({
                    "country_id": code,
                    "country_name": country_names.get(code, code),
                    "is_full_member": code in ICC_FULL_MEMBERS,
                    "tier": "FULL_MEMBER" if code in ICC_FULL_MEMBERS else "ASSOCIATE",
                    "overall_score": overall,
                    "favored_pitch": favored.upper(),
                    "favored_pitch_canonical": favored,
                    "scores": c_scores,
                    "home_surfaces": home_scores,
                    "top_foreign_batter": top_p.get("batter"),
                    "top_foreign_bowler": top_p.get("bowler")
                })
        else:
            for c in countries:
                fav = c.get("favored_pitch", "BALANCED")
                c["favored_pitch_canonical"] = PITCH_MAP_RENDER_TO_LOCAL.get(fav.upper(), "balanced")

        return countries

    async def get_player_adaptability(
        self,
        match_format: str = "ODI",
        gender: str = "MALE",
        role: str = "BATTER",
        min_balls: int = 150,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Player Foreign Pitch Adaptability & Surface Mastery scores across DUSTY, GREEN, BOUNCY, FLAT, BALANCED.
        Supports Role Switcher (🏏 Batters | 🎯 Bowlers) and Bowler Adaptability Engine.
        """
        fmt = "T20" if match_format.upper() in ["T20", "T20I"] else match_format.upper()
        role_param = "BOWLER" if "bowl" in str(role).lower() else "BATTER"
        url = f"{self.BASE_URL}/api/adaptability/players?format={fmt}&gender={gender.upper()}&role={role_param}&min_balls={min_balls}&limit={limit}"
        res = await self._get(url)
        players = res.get("players", []) if isinstance(res, dict) else []

        if not players:
            # Fallback to local cache files
            from backend.engine.render_analytics_bridge import _load_adapt_cache
            from backend.engine.cricsheet_pitch_intelligence import compute_bowler_foreign_adaptability_all_formats
            adapt_cache = _load_adapt_cache(fmt)
            raw_players = list(adapt_cache.values()) if adapt_cache else []
            filtered = []
            for p in raw_players:
                p_role = str(p.get("player_role", "BATTER")).upper()
                if role_param == "BOWLER" and "BOWL" in p_role:
                    filtered.append(p)
                elif role_param == "BATTER" and "BOWL" not in p_role:
                    filtered.append(p)

            players = filtered[:limit] if filtered else raw_players[:limit]

        for p in players:
            fav = p.get("favored_pitch", "BALANCED")
            p["favored_pitch_canonical"] = PITCH_MAP_RENDER_TO_LOCAL.get(str(fav).upper(), "balanced")
            # Fallback to home surface experience when foreign average is null
            for s in ["dusty", "green", "bouncy", "flat", "balanced"]:
                f_avg = p.get(f"{s}_avg")
                h_avg = p.get(f"{s}_home_avg")
                h_balls = p.get(f"{s}_home_balls") or 0
                if f_avg is None and h_avg is not None and h_balls > 0:
                    p[f"{s}_avg"] = h_avg
                    if p.get(f"{s}_sr") in [None, 0, 0.0] and p.get(f"{s}_home_sr"):
                        p[f"{s}_sr"] = p.get(f"{s}_home_sr")
                    p[f"{s}_source"] = "HOME_SURFACE_EXPERIENCE"
                elif f_avg is not None and not p.get(f"{s}_source"):
                    p[f"{s}_source"] = "FOREIGN_DATA"

                # Bowler economy fallback
                f_econ = p.get(f"{s}_econ")
                h_econ = p.get(f"{s}_home_econ")
                if (f_econ is None or f_econ == 0.0) and h_econ is not None and h_econ > 0.0:
                    p[f"{s}_econ"] = h_econ
                    p[f"{s}_bowler_source"] = "HOME_SURFACE_EXPERIENCE"

        return players

    async def search_players(self, query: str) -> List[Dict[str, Any]]:
        """Searches players by name in Cricpulse database."""
        encoded = urllib.parse.quote(query)
        url = f"{self.BASE_URL}/api/players?search={encoded}"
        res = await self._get(url)
        return res.get("players", []) if isinstance(res, dict) else []

    async def get_player_rankings(
        self,
        match_format: str = "ODI",
        role: str = "BATTER",
        lens: str = "balanced",
        phase: str = "all",
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Master player rankings with 18 multi-factor attributes."""
        fmt = "T20" if match_format.upper() in ["T20", "T20I"] else match_format.upper()
        url = f"{self.BASE_URL}/api/players?format={fmt}&role={role.upper()}&lens={lens}&phase={phase}&limit={limit}"
        res = await self._get(url)
        return res.get("players", []) if isinstance(res, dict) else []

    async def get_player_dossier(self, player_id: str) -> Dict[str, Any]:
        """Comprehensive player dossier with radar attributes, percentile badges, and career stats."""
        resolved_id = await self.resolve_render_player_id(player_id) or player_id
        url = f"{self.BASE_URL}/api/players/{resolved_id}"
        return await self._get(url)

    async def simulate_duel(self, batter_id: str, bowler_id: str, match_format: str = "ODI") -> Dict[str, Any]:
        """
        Ball-by-Ball Head-to-Head Duel Simulator.
        Automatically resolves batter and bowler IDs, hits Render.com, and falls back to local engine if needed.
        """
        fmt = "T20" if match_format.upper() in ["T20", "T20I"] else match_format.upper()
        resolved_b = await self.resolve_render_player_id(batter_id) or batter_id
        resolved_bw = await self.resolve_render_player_id(bowler_id) or bowler_id

        url = f"{self.BASE_URL}/api/duel?batter_id={resolved_b}&bowler_id={resolved_bw}&format={fmt}"
        res = await self._get(url)
        if isinstance(res, dict) and res.get("success") is True and "probabilities" in res:
            return res

        # High-fidelity fallback using Cricpulse Master Engine Duel Simulator
        try:
            from backend.engine.cricpulse_master_engine import compute_22_master_attributes, compute_duel_simulation
            from backend.engine.player_strengths import get_player_dossier
            bat_data = get_player_dossier(batter_id, "batter")
            bowl_data = get_player_dossier(bowler_id, "pacer")
            bat_attrs = compute_22_master_attributes(bat_data, match_format)
            bowl_attrs = compute_22_master_attributes(bowl_data, match_format)
            duel_sim = compute_duel_simulation(bat_attrs, bowl_attrs, match_format=match_format)
            return {
                "success": True,
                "format": match_format,
                "batter": bat_attrs,
                "bowler": bowl_attrs,
                "duel": duel_sim,
                "engine": "Cricpulse Bayesian Duel Simulator (Local Resiliency)"
            }
        except Exception as e:
            logger.error(f"Error in local duel fallback: {e}")
            return res if isinstance(res, dict) else {"success": False, "error": str(e)}

    async def compare_players(self, p1_id: str, p2_id: str, match_format: str = "ODI", lens: str = "balanced") -> Dict[str, Any]:
        """Two-player comparison across 18 multi-factor attributes, phase mastery, and head-to-head tally."""
        fmt = "T20" if match_format.upper() in ["T20", "T20I"] else match_format.upper()
        resolved_p1 = await self.resolve_render_player_id(p1_id) or p1_id
        resolved_p2 = await self.resolve_render_player_id(p2_id) or p2_id

        url = f"{self.BASE_URL}/api/players/compare?p1={resolved_p1}&p2={resolved_p2}&format={fmt}&lens={lens}"
        res = await self._get(url)
        if isinstance(res, dict) and res.get("success") is not False and "error" not in res and "player_1" in res:
            return res

        # High-fidelity fallback using Cricpulse Master Engine (22 attributes + 4 lenses)
        try:
            from backend.engine.cricpulse_master_engine import compute_22_master_attributes, compare_players_across_lenses
            from backend.engine.player_strengths import get_player_dossier
            p1_data = get_player_dossier(p1_id, "batter")
            p2_data = get_player_dossier(p2_id, "pacer")
            p1_attrs = compute_22_master_attributes(p1_data, match_format)
            p2_attrs = compute_22_master_attributes(p2_data, match_format)
            lens_title = lens.capitalize() if lens else "Balanced"
            comparison = compare_players_across_lenses(p1_attrs, p2_attrs, lens_title)
            return {
                "success": True,
                "format": match_format,
                "lens": lens,
                "player_1": p1_attrs,
                "player_2": p2_attrs,
                "comparison": comparison,
                "engine": "Cricpulse AI Master Engine (18-Attribute Radar - Local Resiliency)"
            }
        except Exception as e:
            logger.error(f"Error in local player compare fallback: {e}")
            return res if isinstance(res, dict) else {"success": False, "error": str(e)}

    async def _get(self, url: str) -> Any:
        if url in self._cache:
            return self._cache[url]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=self.headers)
                if resp.status_code == 200:
                    data = resp.json()
                    self._cache[url] = data
                    return data
                else:
                    logger.warning(f"Cricpulse request failed ({resp.status_code}): {url}")
                    return {"success": False, "status_code": resp.status_code}
        except Exception as e:
            logger.error(f"Error fetching from Cricpulse {url}: {e}")
            return {"success": False, "error": str(e)}

cricpulse_provider = CricpulseAnalyticsProvider()
