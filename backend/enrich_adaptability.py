import urllib.request
import json
import time
import os

BASE_URL = "https://cricket-analytics-ghpv.onrender.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
surfaces = ["dusty", "green", "bouncy", "flat", "balanced"]

def fetch_with_retry(url, max_retries=5, delay=3.0):
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 CricOracle'})
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            print(f"  Attempt {attempt}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries:
                time.sleep(delay)
    return None

def fetch_and_enrich(fmt: str):
    print(f"\n==================== PROCESSING {fmt} ====================")
    url = f"{BASE_URL}/api/adaptability/players?format={fmt}&limit=400"
    data = fetch_with_retry(url)
    if not data or not data.get('players'):
        print(f"Failed to fetch {fmt} players after retries. Preserving existing cache.")
        return

    players = data.get('players', [])
    print(f"Successfully fetched {len(players)} players from Render for {fmt}")

    adapt_map = {}
    null_replaced_count = 0
    sl_replaced = []

    for p in players:
        p_name = p.get('player_name', '').strip()
        p_id = p.get('player_id')
        country = p.get('country_name', '')
        p_role = str(p.get('player_role', 'BATTER')).upper()
        is_bowler = 'BOWL' in p_role or p_role in ['ALL_ROUNDER', 'BOWLER']

        for s in surfaces:
            f_avg = p.get(f"{s}_avg")
            h_avg = p.get(f"{s}_home_avg")
            h_balls = p.get(f"{s}_home_balls") or 0
            
            # 1. Batting Average Substitution:
            # If foreign average is not found (null / None), but player has played on this surface at home
            if f_avg is None and h_avg is not None and h_balls > 0:
                p[f"{s}_avg"] = h_avg
                if p.get(f"{s}_sr") in [None, 0, 0.0] and p.get(f"{s}_home_sr"):
                    p[f"{s}_sr"] = p.get(f"{s}_home_sr")
                p[f"{s}_source"] = "HOME_SURFACE_EXPERIENCE"
                null_replaced_count += 1
                if 'sri' in country.lower() or p.get('country_id') in ['SRI', 'SL']:
                    sl_replaced.append((p_name, f"{s}_avg", h_avg, f"{h_balls} home balls"))
            elif f_avg is not None:
                p[f"{s}_source"] = "FOREIGN_DATA"
            else:
                p[f"{s}_source"] = "NO_EXPERIENCE"
            
            # 2. Bowling Economy / Out Substitution for Bowlers/All-Rounders:
            f_econ = p.get(f"{s}_econ")
            h_econ = p.get(f"{s}_home_econ")
            if is_bowler:
                if (f_econ is None or f_econ == 0.0) and h_econ is not None and h_econ > 0.0:
                    p[f"{s}_econ"] = h_econ
                    p[f"{s}_bowler_source"] = "HOME_SURFACE_EXPERIENCE"
                    null_replaced_count += 1
                    if 'sri' in country.lower() or p.get('country_id') in ['SRI', 'SL']:
                        sl_replaced.append((p_name, f"{s}_econ", h_econ, "home bowler econ"))

        if p_id:
            adapt_map[p_id] = p
        if p_name:
            adapt_map[p_name.lower()] = p

    cache_file = os.path.join(DATA_DIR, f"render_adaptability_{fmt.lower()}.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(adapt_map, f, indent=2)

    print(f"Total nulls replaced with home stats in {fmt}: {null_replaced_count}")
    print(f"Saved {len(adapt_map)} mappings to {cache_file}")
    if sl_replaced:
        print("Sri Lankan player replacements:")
        for name, metric, val, note in sl_replaced[:12]:
            print(f"  • {name} - {metric}: {val} ({note})")

for f in ["TEST", "ODI", "T20"]:
    fetch_and_enrich(f)
