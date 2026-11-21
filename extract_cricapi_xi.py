#!/usr/bin/env python3
"""
extract_cricapi_xi.py - CricAPI Playing XI Extractor with Strict Credit-Saving Caching

Designed for strict quota constraints (100 API hits per day max).
- Match listing costs 1 hit (cached locally for 15-30 minutes to save credits).
- Squad extraction costs 10 hits (cached permanently per match to never re-fetch).
- Prevents calling match_squad if hasSquad == False (saving 10 hits each time!).
- Automatic Quota Guard blocks requests before exceeding daily limit.
- Extracts ONLY the first 11 players (Playing XI / Squad) per team.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

# API Key provided by user
DEFAULT_API_KEY = "d0e9f357-ae08-4700-9634-1a26bc27501b"

# Cache Directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cricapi_cache")
CURRENT_MATCHES_CACHE = os.path.join(CACHE_DIR, "current_matches.json")
ALL_MATCHES_CACHE = os.path.join(CACHE_DIR, "all_matches.json")
QUOTA_TRACKER_FILE = os.path.join(CACHE_DIR, "quota_tracker.json")

# Quota settings
DAILY_CALL_LIMIT = 100
SAFETY_RESERVE = 10  # Stop making live calls when hits >= 90
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes


def ensure_cache_dir():
    """Ensure cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_quota_tracker():
    """Load persistent quota usage tracker."""
    ensure_cache_dir()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    if os.path.exists(QUOTA_TRACKER_FILE):
        try:
            with open(QUOTA_TRACKER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("date") == today_str:
                    return data
        except Exception:
            pass

    # Default baseline
    return {
        "date": today_str,
        "hitsToday": 23,  # Current verified usage today
        "hitsLimit": DAILY_CALL_LIMIT,
        "lastUpdated": datetime.now(timezone.utc).isoformat()
    }


def save_quota_tracker(hits_today, hits_limit=DAILY_CALL_LIMIT):
    """Save persistent quota usage tracker."""
    ensure_cache_dir()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = {
        "date": today_str,
        "hitsToday": hits_today,
        "hitsLimit": hits_limit,
        "lastUpdated": datetime.now(timezone.utc).isoformat()
    }
    with open(QUOTA_TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def update_quota_from_response(resp_json):
    """Update quota tracker if info field is present in response."""
    info = resp_json.get("info", {})
    if "hitsToday" in info:
        hits_today = info.get("hitsToday", 0)
        hits_limit = info.get("hitsLimit", DAILY_CALL_LIMIT)
        save_quota_tracker(hits_today, hits_limit)


def can_make_api_call(cost=1):
    """Check if API call is allowed under daily quota limit."""
    quota = load_quota_tracker()
    hits_today = quota.get("hitsToday", 0)
    hits_limit = quota.get("hitsLimit", DAILY_CALL_LIMIT)
    
    if hits_today + cost > (hits_limit - SAFETY_RESERVE):
        print(f"\n[QUOTA GUARD BLOCKED] Daily API budget protection!")
        print(f"Hits used today: {hits_today}/{hits_limit} (Cost of this request: {cost} hits).")
        print(f"Safety reserve is {SAFETY_RESERVE} hits. Request blocked to prevent hitting limit.")
        return False
    return True


def print_quota_status():
    """Print current quota usage."""
    quota = load_quota_tracker()
    hits_today = quota.get("hitsToday", 0)
    hits_limit = quota.get("hitsLimit", DAILY_CALL_LIMIT)
    rem = hits_limit - hits_today
    print(f"\n================ CRICAPI QUOTA TRACKER ================")
    print(f"Date:         {quota.get('date')}")
    print(f"Hits Used:    {hits_today} / {hits_limit}")
    print(f"Hits Left:    {rem}")
    print(f"Squad Budget: ~{rem // 10} squad calls remaining today (10 hits each)")
    print(f"Last Updated: {quota.get('lastUpdated')}")
    print(f"=======================================================\n")


def get_cached_or_fresh_matches(api_key, endpoint_type="current", force_refresh=False):
    """
    Fetch match lists with local caching to conserve credits.
    - endpoint_type='current': https://api.cricapi.com/v1/currentMatches (live + today)
    - endpoint_type='all': https://api.cricapi.com/v1/matches (upcoming fixtures)
    Costs 1 hit only if cache is stale or missing; 0 hits if reading from cache.
    """
    ensure_cache_dir()
    cache_file = CURRENT_MATCHES_CACHE if endpoint_type == "current" else ALL_MATCHES_CACHE

    if not force_refresh and os.path.exists(cache_file):
        try:
            mtime = os.path.getmtime(cache_file)
            age = time.time() - mtime
            if age < CACHE_TTL_SECONDS:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
        except Exception:
            pass

    # If cache exists but expired and quota reached, fallback to existing cache
    if not can_make_api_call(cost=1):
        if os.path.exists(cache_file):
            print(f"[QUOTA NOTICE] Preserving credits: reading cached {endpoint_type} matches from disk.")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "failure", "reason": "Quota limit reached"}

    import requests
    endpoint = "currentMatches" if endpoint_type == "current" else "matches"
    url = f"https://api.cricapi.com/v1/{endpoint}?apikey={api_key}&offset=0"
    print(f"[API CALL] Fetching {endpoint} from CricAPI (Costs 1 hit)...")
    try:
        resp = requests.get(url, timeout=15).json()
        if resp.get("status") == "success":
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(resp, f, indent=2)
            update_quota_from_response(resp)
            quota = load_quota_tracker()
            print(f"[API SUCCESS] Cached {len(resp.get('data', []))} matches. Hits today: {quota.get('hitsToday')}/{quota.get('hitsLimit')}.")
        return resp
    except Exception as e:
        print(f"[API ERROR] Failed to fetch {endpoint}: {e}")
        if os.path.exists(cache_file):
            print("[FALLBACK] Using existing cached matches.")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "failure", "reason": str(e)}


def get_all_active_matches(api_key, force_refresh=False):
    """Combine currentMatches and upcoming matches while avoiding duplicate API calls."""
    cm_data = get_cached_or_fresh_matches(api_key, "current", force_refresh=force_refresh)
    am_data = get_cached_or_fresh_matches(api_key, "all", force_refresh=False)

    merged = []
    seen = set()

    for m in cm_data.get("data", []):
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            merged.append(m)

    for m in am_data.get("data", []):
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            merged.append(m)

    return merged


def filter_matches(all_matches):
    """
    Categorize matches into:
    1. ABOUT TO START (Not started yet, starting within next 24 hours)
    2. LIVE / IN-PROGRESS (Started, not ended)
    3. UPCOMING (Future fixtures > 24 hours)
    """
    now = datetime.now(timezone.utc)
    about_to_start = []
    live_matches = []
    upcoming_fixtures = []

    for m in all_matches:
        if m.get("matchEnded", False):
            continue

        mid = m.get("id")
        name = m.get("name", "Unknown Match")
        match_type = m.get("matchType", "")
        status = m.get("status", "")
        has_squad = m.get("hasSquad", False)
        match_started = m.get("matchStarted", False)
        dt_str = m.get("dateTimeGMT")

        diff_hours = None
        if dt_str:
            try:
                dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
                diff_hours = (dt - now).total_seconds() / 3600
            except Exception:
                pass

        item = {
            "id": mid,
            "name": name,
            "matchType": match_type,
            "status": status,
            "hasSquad": has_squad,
            "matchStarted": match_started,
            "dateTimeGMT": dt_str,
            "diff_hours": diff_hours
        }

        if match_started:
            live_matches.append(item)
        elif diff_hours is not None and diff_hours <= 24:
            about_to_start.append(item)
        else:
            upcoming_fixtures.append(item)

    # Sort about_to_start by closest start time
    about_to_start.sort(key=lambda x: (x["diff_hours"] if x["diff_hours"] is not None else 9999))
    # Sort live by most recent
    live_matches.sort(key=lambda x: abs(x["diff_hours"] if x["diff_hours"] is not None else 9999))

    return about_to_start, live_matches, upcoming_fixtures


def get_match_squad(match_id, match_name="", has_squad=True, api_key=DEFAULT_API_KEY, force_refresh=False):
    """
    Fetch match squad with permanent file caching and credit guards.
    - If already in cache: 0 hits spent.
    - If has_squad is False: skips API call, 0 hits spent (saves 10 hits!).
    - Otherwise: calls CricAPI match_squad (costs 10 hits) and caches response.
    """
    ensure_cache_dir()
    squad_cache_file = os.path.join(CACHE_DIR, f"squad_{match_id}.json")

    # 1. Check local file cache first (ZERO API HITS)
    if not force_refresh and os.path.exists(squad_cache_file):
        try:
            with open(squad_cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                print(f"[CACHE HIT] Loaded squad for '{match_name or match_id}' from local disk. (0 API hits spent!)")
                return cached_data
        except Exception:
            pass

    # 2. Check has_squad flag to prevent wasting 10 credits on empty squad
    if not has_squad:
        print(f"\n[CREDIT SAVER GUARD] Squad not available on CricAPI yet (hasSquad: False).")
        print(f"Skipping match_squad API call to save 10 credits! Match: {match_name or match_id}")
        return {"status": "skipped", "reason": "hasSquad is False. Squad not yet announced or uploaded by CricAPI."}

    # 3. Check quota guard
    if not can_make_api_call(cost=10):
        return {"status": "failure", "reason": "Quota protection triggered. Remaining hits too low for 10-hit squad call."}

    # 4. Make live API call
    import requests
    url = f"https://api.cricapi.com/v1/match_squad?apikey={api_key}&id={match_id}"
    print(f"\n[API CALL] Requesting squad from CricAPI for ID: {match_id} (Costs 10 hits)...")
    try:
        response = requests.get(url, timeout=15)
        resp_json = response.json()
        
        update_quota_from_response(resp_json)
        quota = load_quota_tracker()

        if resp_json.get("status") == "success":
            with open(squad_cache_file, "w", encoding="utf-8") as f:
                json.dump(resp_json, f, indent=2)
            print(f"[API SUCCESS] Squad saved to cache: squad_{match_id}.json. Hits today: {quota.get('hitsToday')}/{quota.get('hitsLimit')}.")
            return resp_json
        else:
            print(f"[API NOTICE] CricAPI returned status '{resp_json.get('status')}': {resp_json.get('reason')}")
            return resp_json
    except Exception as e:
        print(f"[API ERROR] Request failed: {e}")
        return {"status": "failure", "reason": str(e)}


def display_playing_xi(squad_json, match_name=""):
    """
    Extract ONLY 11 players for each team and print formatted output.
    Follows user's exact requested specification:
      ================ {team_name} (Playing XI / Squad) ================
      {idx}. {name:<22} | Role: {role:<18} | Bat: {batting} | Bowl: {bowling}
    """
    if squad_json.get("status") == "skipped":
        print(f"Notice: {squad_json.get('reason')}")
        return

    if squad_json.get("status") != "success":
        print("Error:", squad_json.get("reason", "Could not fetch squad"))
        return

    teams = squad_json.get("data", [])
    if not teams:
        print("Notice: No team squads returned in response data.")
        return

    if match_name:
        print(f"\n================================================================================")
        print(f" MATCH: {match_name}")
        print(f"================================================================================")

    for team in teams:
        team_name = team.get("teamName", "Unknown Team")
        players = team.get("players", [])

        print(f"\n================ {team_name} (Playing XI / Squad) ================")
        # Slicing strictly to first 11 players as requested
        for idx, player in enumerate(players[:11], start=1):
            name = player.get("name", "Unknown")
            role = player.get("role", "Player")
            batting = player.get("battingStyle", "")
            bowling = player.get("bowlingStyle", "")

            # Matching user's exact specification with optional bowling style info
            bowl_str = f" | Bowl: {bowling}" if bowling else ""
            print(f"{idx:2d}. {name:<24} | Role: {role:<18} | Bat: {batting:<15}{bowl_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract only 11 players (Playing XI / Squad) for cricket matches about to start with strict credit conservation."
    )
    parser.add_argument("--key", type=str, default=DEFAULT_API_KEY, help="CricAPI API key")
    parser.add_argument("--id", type=str, help="Specific CricAPI Match ID to extract Playing XI for")
    parser.add_argument("--pick", type=int, help="Select match by index [1, 2, 3...] from the listed matches")
    parser.add_argument("--list", action="store_true", help="List all matches about to start and live matches with their Match IDs without fetching squads")
    parser.add_argument("--quota", action="store_true", help="Display current CricAPI hit quota usage")
    parser.add_argument("--refresh", action="store_true", help="Force refresh matches cache from CricAPI (costs 1 hit)")

    args = parser.parse_args()
    api_key = args.key or os.environ.get("CRICAPI_KEY", DEFAULT_API_KEY)

    # 1. Quota check
    if args.quota:
        print_quota_status()
        return

    # 2. Match discovery
    all_matches = get_all_active_matches(api_key, force_refresh=args.refresh)
    about_to_start, live_matches, upcoming = filter_matches(all_matches)

    # Direct Match ID extraction
    if args.id:
        match_id = args.id.strip()
        print(f"Target Match ID: {match_id}")
        match_name = ""
        has_squad = True
        for m in all_matches:
            if m.get("id") == match_id:
                match_name = m.get("name", "")
                has_squad = m.get("hasSquad", True)
                break

        squad_json = get_match_squad(match_id, match_name=match_name, has_squad=has_squad, api_key=api_key)
        display_playing_xi(squad_json, match_name=match_name)
        print_quota_status()
        return

    selectable_matches = []

    print("\n" + "=" * 80)
    print(" CRICAPI MATCH FINDER: MATCHES ABOUT TO START & LIVE")
    print("=" * 80)

    print(f"\n[1] MATCHES ABOUT TO START (Next 24h): {len(about_to_start)} found")
    print("-" * 80)
    if not about_to_start:
        print("  No fixtures scheduled to start in the next 24 hours.")
    else:
        for m in about_to_start:
            selectable_matches.append(m)
            idx = len(selectable_matches)
            squad_indicator = "[SQUAD READY]" if m["hasSquad"] else "[SQUAD PENDING]"
            time_str = f"in {m['diff_hours']:.1f} hours" if m['diff_hours'] is not None and m['diff_hours'] > 0 else (m.get('dateTimeGMT') or 'Starting Soon')
            print(f"  [{idx:2d}] Match ID: {m['id']}")
            print(f"       Fixture:  {m['name']}")
            print(f"       Starts:   {time_str} | Squad: {squad_indicator} | Type: {m['matchType']}")
            print()

    print(f"\n[2] LIVE / RECENTLY STARTED MATCHES: {len(live_matches)} found")
    print("-" * 80)
    for m in live_matches[:10]:
        selectable_matches.append(m)
        idx = len(selectable_matches)
        squad_indicator = "[SQUAD READY]" if m["hasSquad"] else "[SQUAD PENDING]"
        print(f"  [{idx:2d}] Match ID: {m['id']}")
        print(f"       Fixture:  {m['name']}")
        print(f"       Status:   {m['status']} | Squad: {squad_indicator}")
        print()

    if args.list:
        print_quota_status()
        print("To extract 11 players, run:")
        print("  python extract_cricapi_xi.py --id <MATCH_ID>")
        print("  or")
        print("  python extract_cricapi_xi.py --pick <INDEX_NUMBER>")
        return

    # If --pick <INDEX> was specified:
    if args.pick is not None:
        idx = args.pick - 1
        if 0 <= idx < len(selectable_matches):
            target = selectable_matches[idx]
            print("=" * 80)
            print(f"PICKED MATCH [{args.pick}]: {target['name']}")
            print(f"Match ID: {target['id']} (hasSquad: {target['hasSquad']})")
            print("=" * 80)
            squad_json = get_match_squad(target["id"], match_name=target["name"], has_squad=target["hasSquad"], api_key=api_key)
            display_playing_xi(squad_json, match_name=target["name"])
            print_quota_status()
            return
        else:
            print(f"Invalid pick index {args.pick}. Must be between 1 and {len(selectable_matches)}.")
            return

    # Automatic selection:
    # First: look for match about to start with squad ready
    target = None
    for m in about_to_start:
        if m["hasSquad"]:
            target = m
            break

    # Second: look for live match with squad ready
    if not target:
        for m in live_matches:
            if m["hasSquad"]:
                target = m
                break

    if target:
        print("=" * 80)
        print(f"AUTO-SELECTED MATCH: {target['name']}")
        print(f"Match ID: {target['id']} (hasSquad: {target['hasSquad']})")
        print("=" * 80)
        squad_json = get_match_squad(target["id"], match_name=target["name"], has_squad=target["hasSquad"], api_key=api_key)
        display_playing_xi(squad_json, match_name=target["name"])
    else:
        print("\nNotice: None of the upcoming matches currently have squads announced (hasSquad: False).")
        print("CricAPI will show [SQUAD READY] once toss occurs and official Playing XI is submitted.")
        print("To save credits, no calls were made to empty squad endpoints.")

    print_quota_status()


if __name__ == "__main__":
    main()
