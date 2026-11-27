"""
Live Weather & Dew Telemetry Service
Fetches real-time wind speed, wind direction, relative humidity, and dew point
from the global ECMWF/GFS-backed Open-Meteo meteorological feed (no API key required).
Calculates physical dew condensation spread and dynamic dew factors.
"""

import time
import math
import json
import urllib.request
from typing import Dict, Any, Optional, Tuple

# Cache weather lookups for 10 minutes to prevent redundant requests during ball-by-ball stepping
_WEATHER_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 600.0

def calculate_dew_point_magnus(temp_c: float, relative_humidity: float) -> float:
    """
    Computes exact dew point temperature (in °C) using the Magnus-Tetens formula.
    """
    a = 17.27
    b = 237.7
    rh = max(1.0, min(100.0, float(relative_humidity)))
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return round(dew_point, 2)


def get_live_stadium_weather(
    latitude: float,
    longitude: float,
    venue_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Queries real-time meteorological conditions for stadium coordinates.
    Source: Open-Meteo (European Centre for Medium-Range Weather Forecasts / NOAA GFS).
    Returns:
      - wind_kmh: Wind speed at 10m height (km/h)
      - wind_direction_deg: 0-360 degrees (0 = North, 90 = East, 180 = South, 270 = West)
      - temperature_c: Ambient 2m air temperature (°C)
      - humidity: Relative humidity (%)
      - dew_point_c: Dew point temperature (°C)
      - dew_spread_c: Ambient - Dew Point delta (°C)
      - dew_risk: 'HIGH', 'MODERATE', or 'LOW'
      - dew_factor: Dynamic 0.0 to 1.0 factor for the match engine
    """
    cache_key = f"{round(latitude, 3)},{round(longitude, 3)}"
    now = time.time()

    if cache_key in _WEATHER_CACHE:
        cached_time, cached_data = _WEATHER_CACHE[cache_key]
        if (now - cached_time) < CACHE_TTL_SECONDS:
            return cached_data

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m,relative_humidity_2m,dew_point_2m,wind_speed_10m,wind_direction_10m"
    )

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CricOracle-CricketAnalytics/2.0"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
            current = data.get("current", {})

            temp_c = float(current.get("temperature_2m") or 25.0)
            rh = float(current.get("relative_humidity_2m") or 60.0)
            dew_point_c = float(current.get("dew_point_2m") or calculate_dew_point_magnus(temp_c, rh))
            wind_kmh = float(current.get("wind_speed_10m") or 10.0)
            wind_dir = float(current.get("wind_direction_10m") or 0.0)

            dew_spread = max(0.0, temp_c - dew_point_c)

            # Dew physical condensation risk model:
            # Condensation begins when spread <= 2.5°C and RH >= 65%
            if dew_spread <= 1.8 and rh >= 70.0:
                dew_risk = "HIGH"
                dew_factor = min(1.0, 0.50 + ((100.0 - rh) * -0.015) + ((2.0 - dew_spread) * 0.20))
            elif dew_spread <= 3.5 and rh >= 58.0:
                dew_risk = "MODERATE"
                dew_factor = 0.25 + ((3.5 - dew_spread) * 0.07)
            else:
                dew_risk = "LOW"
                dew_factor = max(0.05, 0.20 - (dew_spread * 0.02))

            result = {
                "source": "Open-Meteo (ECMWF/NOAA GFS Global Telemetry)",
                "venue_name": venue_name or f"Coords({latitude}, {longitude})",
                "wind_kmh": round(wind_kmh, 1),
                "wind_direction_deg": round(wind_dir, 0),
                "temperature_c": round(temp_c, 1),
                "humidity": round(rh, 1),
                "dew_point_c": round(dew_point_c, 1),
                "dew_spread_c": round(dew_spread, 1),
                "dew_risk": dew_risk,
                "dew_factor": round(float(dew_factor), 3)
            }

            _WEATHER_CACHE[cache_key] = (now, result)
            return result

    except Exception as e:
        # Graceful fallback if network drops or offline
        fallback = {
            "source": f"Fallback Default (offline: {str(e)})",
            "venue_name": venue_name or "Unknown Venue",
            "wind_kmh": 12.0,
            "wind_direction_deg": 180.0,
            "temperature_c": 26.0,
            "humidity": 65.0,
            "dew_point_c": 18.5,
            "dew_spread_c": 7.5,
            "dew_risk": "LOW",
            "dew_factor": 0.20
        }
        _WEATHER_CACHE[cache_key] = (now - CACHE_TTL_SECONDS + 60.0, fallback)
        return fallback


def get_venue_live_weather_bundle(venue_hint: Optional[str], fmt: str = "ODI") -> Optional[Dict[str, Any]]:
    """
    Resolves venue coordinates and returns a weather_data dict formatted for LiveDewDetector
    and weather_intelligence (Open-Meteo cached for 10 min; non-blocking offline fallback).
    """
    if not venue_hint:
        return None
    try:
        from .cricsheet_pitch_intelligence import FORMAT_GROUND_PITCH_DOSSIERS, resolve_venue_dossier_context
        v_ctx = resolve_venue_dossier_context(str(venue_hint), fmt)
        vk = v_ctx.get("venue_key")
        coords = FORMAT_GROUND_PITCH_DOSSIERS.get(vk, {}).get("coordinates") if vk else None
        if not coords or "lat" not in coords or "lon" not in coords:
            return None
        w = get_live_stadium_weather(coords["lat"], coords["lon"], str(venue_hint))
        return {
            "source": w.get("source"),
            "temp_c": w.get("temperature_c"),
            "relative_humidity_pct": w.get("humidity"),
            "wind_speed_kmh": w.get("wind_kmh"),
            "dew_point_c": w.get("dew_point_c"),
            "dew_spread_c": w.get("dew_spread_c"),
            "dew_factor": w.get("dew_factor"),
            "dew_risk": w.get("dew_risk"),
            "humidity_pct": w.get("humidity"),
            "temperature_c": w.get("temperature_c"),
        }
    except Exception:
        return None
