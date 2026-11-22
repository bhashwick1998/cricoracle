"""
CricOracle SQLite Database Seeder
Seeds rich initial test data covering:
- 12 International venues with format-specific archetypes & boundary dimensions
- 11 National teams with elo ratings & condition performance indices
- International player rosters across all formats
- 16 Real fixtures across all 4 stages:
    0. LIVE 2nd Test: BAN vs WI (Sylhet - Day 1 Stumps, 235/4 in 70 ov)
    1. LIVE 1st Test: AFG vs ZIM (Sharjah - Day 1 Session 3 LIVE IN-PLAY, 276/4 in 76.3 ov, NOT STUMPED)
    2. SQUAD ANNOUNCED 2nd Test: AFG vs ZIM (Sharjah, Nov 12-16, Propagated 15-Man Test Squad)
    3. SQUAD ANNOUNCED 2nd ODI: NZ vs IND (Wellington, Nov 7)
    4. SQUAD ANNOUNCED 3rd ODI: NZ vs IND (Seddon Park, Nov 10, Propagated)
    5. SQUAD ANNOUNCED 4th ODI: NZ vs IND (Bay Oval, Nov 13, Propagated)
    6. SQUAD ANNOUNCED 5th ODI: NZ vs IND (Bay Oval, Nov 15, Propagated)
    7. SQUAD ANNOUNCED 1st T20I: AUS vs ENG (MCG, Nov 13)
    8. SQUAD ANNOUNCED 2nd T20I: AUS vs ENG (SCG, Nov 15, Propagated)
    9. SQUAD ANNOUNCED 1st T20I: IND vs SA (Wankhede, Nov 27)
    10. SQUAD ANNOUNCED 2nd T20I: IND vs SA (Eden Gardens, Nov 29, Propagated)
    11. UPCOMING T20I: NZ vs IND (Eden Park, Nov 10)
    12. UPCOMING ODI: AUS vs ENG (Adelaide, Nov 18)
    13. UPCOMING ODI: SA vs BAN (Centurion, Nov 14)
    14. UPCOMING T20I: PAK vs SL (Karachi, Nov 19)
    15. UPCOMING ODI: NZ vs IND (Napier, Nov 21)
"""

import asyncio
from backend.database.db import get_db, init_db, _db_write_lock
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "cricoracle.db"

async def seed_database():
    await init_db()
    conn = await get_db()
    try:
        # Clear existing tables cleanly (respecting foreign key hierarchy)
        await conn.execute("DELETE FROM model_learning_logs")
        await conn.execute("DELETE FROM match_deliveries")
        await conn.execute("DELETE FROM concussion_substitutions")
        await conn.execute("DELETE FROM match_squads")
        await conn.execute("DELETE FROM matches")
        await conn.execute("DELETE FROM venue_pitch_formats")
        await conn.execute("DELETE FROM player_aliases")
        await conn.execute("DELETE FROM players")
        await conn.execute("DELETE FROM teams")
        await conn.execute("DELETE FROM venues")

        # 1. Venues (with historical pitch profiles)
        venues = [
            ("wellington", "Sky Stadium (Westpac)", "Wellington", "New Zealand", "sena", "green", 0.28, 166, 285, 0.526, 0.20, 72.0),
            ("seddon_park", "Seddon Park", "Hamilton", "New Zealand", "sena", "balanced", 0.35, 172, 292, 0.531, 0.25, 68.0),
            ("bay_oval", "Bay Oval", "Mount Maunganui", "New Zealand", "sena", "balanced", 0.32, 172, 268, 0.485, 0.22, 68.0),
            ("eden_park", "Eden Park", "Auckland", "New Zealand", "sena", "flat", 0.25, 185, 310, 0.548, 0.30, 64.0),
            ("sharjah", "Sharjah Cricket Stadium", "Sharjah", "United Arab Emirates", "subcontinent", "flat", 0.46, 178, 275, 0.582, 0.55, 62.0),
            ("dubai_dsc", "Dubai International Stadium", "Dubai", "United Arab Emirates", "subcontinent", "balanced", 0.52, 158, 255, 0.628, 0.60, 68.0),
            ("mcg", "Melbourne Cricket Ground", "Melbourne", "Australia", "sena", "bouncy", 0.32, 164, 285, 0.545, 0.30, 82.0),
            ("scg", "Sydney Cricket Ground", "Sydney", "Australia", "sena", "dusty", 0.48, 166, 280, 0.500, 0.35, 76.0),
            ("adelaide", "Adelaide Oval", "Adelaide", "Australia", "sena", "flat", 0.38, 175, 290, 0.515, 0.25, 74.0),
            ("perth", "Perth Stadium (Optus Stadium)", "Perth", "Australia", "sena", "bouncy", 0.26, 165, 268, 0.480, 0.12, 72.0),
            ("centurion", "SuperSport Park", "Centurion", "South Africa", "sena", "bouncy", 0.24, 180, 295, 0.543, 0.20, 75.0),
            ("karachi", "National Stadium", "Karachi", "Pakistan", "subcontinent", "flat", 0.44, 182, 290, 0.552, 0.50, 70.0),
            ("mirpur", "Sher-e-Bangla National Stadium", "Mirpur", "Bangladesh", "subcontinent", "dusty", 0.65, 148, 240, 0.521, 0.40, 68.0),
            ("sylhet", "Sylhet International Cricket Stadium", "Sylhet", "Bangladesh", "subcontinent", "dusty", 0.62, 154, 248, 0.514, 0.30, 70.0),
            ("wankhede", "Wankhede Stadium", "Mumbai", "India", "subcontinent", "flat", 0.42, 192, 310, 0.640, 0.65, 68.0),
            ("eden_gardens", "Eden Gardens", "Kolkata", "India", "subcontinent", "dusty", 0.52, 176, 280, 0.587, 0.55, 71.0),
            ("premadasa", "R. Premadasa Stadium", "Colombo", "Sri Lanka", "subcontinent", "dusty", 0.68, 152, 245, 0.53, 0.40, 72.0),
            ("lords", "Lord's Cricket Ground", "London", "England", "sena", "balanced", 0.28, 168, 275, 0.51, 0.20, 74.0),
            ("the_oval", "The Oval", "London", "England", "sena", "flat", 0.34, 174, 285, 0.52, 0.25, 71.0),
            ("rawalpindi", "Rawalpindi Cricket Stadium", "Rawalpindi", "Pakistan", "subcontinent", "balanced", 0.38, 185, 295, 0.540, 0.40, 71.5),
            ("lahore", "Gaddafi Stadium", "Lahore", "Pakistan", "subcontinent", "flat", 0.44, 180, 290, 0.535, 0.50, 71.0)
        ]

        # Dynamically ensure all venues in FORMAT_GROUND_PITCH_DOSSIERS exist in venues table
        from ..engine.cricsheet_pitch_intelligence import FORMAT_GROUND_PITCH_DOSSIERS
        existing_v_ids = set(v[0] for v in venues)
        for v_id, dossier in FORMAT_GROUND_PITCH_DOSSIERS.items():
            if v_id not in existing_v_ids:
                b_dim = dossier.get("boundary_dimensions", {})
                avg_m = b_dim.get("avg_m", 68.0)
                dew = dossier.get("dew_factor", 0.25)
                fmt_pars = dossier.get("format_pars", {})
                t20_par = fmt_pars.get("T20I", 170)
                odi_par = fmt_pars.get("ODI", 270)
                venues.append((
                    v_id,
                    dossier.get("name", v_id),
                    dossier.get("city", ""),
                    dossier.get("country", ""),
                    dossier.get("region_cluster", "other"),
                    dossier.get("format_archetypes", {}).get("T20I", "flat"),
                    0.45,
                    t20_par,
                    odi_par,
                    0.52,
                    dew,
                    avg_m
                ))
                existing_v_ids.add(v_id)

        await conn.executemany("""
            INSERT INTO venues (id, name, city, country, region_cluster, pitch_type, spin_wicket_ratio, avg_first_innings_score_t20, avg_first_innings_score_odi, chase_win_pct, dew_factor, dimensions_avg_boundary_m)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, venues)

        # 2. Teams (International Full Members)
        teams = [
            ("IND", "India", "IND", "#0284c7", 1725.0, 1695.0, 1620.0, 88.0, 84.0, 64.0, 66.0, 66.0),
            ("AUS", "Australia", "AUS", "#EAB308", 1715.0, 1595.0, 1685.0, 74.0, 92.0, 66.0, 60.0, 64.0),
            ("ENG", "England", "ENG", "#DC2626", 1665.0, 1575.0, 1635.0, 72.0, 86.0, 64.0, 60.0, 62.0),
            ("SA", "South Africa", "SA", "#16A34A", 1645.0, 1535.0, 1605.0, 76.0, 89.0, 59.0, 62.0, 63.0),
            ("PAK", "Pakistan", "PAK", "#059669", 1615.0, 1635.0, 1545.0, 82.0, 87.0, 58.0, 58.0, 61.0),
            ("NZ", "New Zealand", "NZ", "#27272A", 1635.0, 1545.0, 1625.0, 75.0, 85.0, 60.0, 60.0, 60.0),
            ("WI", "West Indies", "WI", "#831843", 1585.0, 1535.0, 1515.0, 70.0, 80.0, 62.0, 56.0, 65.0),
            ("SL", "Sri Lanka", "SL", "#1D4ED8", 1565.0, 1605.0, 1485.0, 86.0, 72.0, 55.0, 57.0, 56.0),
            ("BAN", "Bangladesh", "BAN", "#047857", 1545.0, 1575.0, 1455.0, 81.0, 70.0, 53.0, 55.0, 54.0),
            ("AFG", "Afghanistan", "AFG", "#0056B3", 1570.0, 1610.0, 1460.0, 90.0, 68.0, 56.0, 62.0, 58.0),
            ("ZIM", "Zimbabwe", "ZIM", "#DC2626", 1460.0, 1450.0, 1420.0, 65.0, 68.0, 48.0, 50.0, 48.0)
        ]
        await conn.executemany("""
            INSERT INTO teams (id, name, short_name, color, elo_home, elo_away_subcontinent, elo_away_sena, spin_index, pace_index, powerplay_rating, middle_overs_rating, death_overs_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, teams)

        # 3. Comprehensive Player Roster across all formats
        players = [
            # India Real ODI & T20 Squad
            ("rsharma_ind", "Rohit Sharma", "IND", "top_order_batter", "right", "NONE", 2100, 0, 140.5, 32.5, 0, 0, 66.0, 58.0, 52.0, 10.0, 10.0, 10.0, 1.05, 0.94, 0.95, 0, "tier_1_ipl"),
            ("vkohli_ind", "Virat Kohli", "IND", "top_order_batter", "right", "RMF", 2450, 48, 138.2, 51.8, 8.2, 24.0, 58.0, 65.0, 68.0, 15.0, 15.0, 10.0, 1.08, 0.98, 0.98, 0, "tier_1_ipl"),
            ("sgill_ind", "Shubman Gill", "IND", "top_order_batter", "right", "NONE", 1850, 0, 102.5, 58.2, 0, 0, 68.0, 64.0, 62.0, 10.0, 10.0, 10.0, 1.10, 1.08, 0.98, 0, "tier_1_ipl"),
            ("siyer_ind", "Shreyas Iyer", "IND", "middle_order_batter", "right", "NONE", 1620, 0, 101.0, 47.5, 0, 0, 52.0, 68.0, 66.0, 10.0, 10.0, 10.0, 1.18, 0.94, 0.94, 0, "tier_1_ipl"),
            ("klrahul_ind", "KL Rahul", "IND", "wicket_keeper", "right", "NONE", 1950, 0, 136.0, 49.5, 0, 0, 56.0, 62.0, 64.0, 10.0, 10.0, 10.0, 1.04, 1.02, 0.96, 0, "tier_1_ipl"),
            ("rpant_ind", "Rishabh Pant", "IND", "wicket_keeper", "left", "NONE", 1100, 0, 126.0, 34.5, 0, 0, 64.0, 60.0, 68.0, 10.0, 10.0, 10.0, 1.02, 1.04, 0.92, 0, "tier_1_ipl"),
            ("hardik_ind", "Hardik Pandya", "IND", "all_rounder", "right", "RF", 1250, 920, 142.0, 33.8, 8.4, 21.5, 55.0, 64.0, 72.0, 58.0, 56.0, 62.0, 1.02, 1.04, 0.96, 0, "tier_1_ipl"),
            ("rjadeja_ind", "Ravindra Jadeja", "IND", "all_rounder", "left", "SLA", 1450, 1680, 128.0, 32.5, 4.88, 32.0, 45.0, 58.0, 62.0, 55.0, 68.0, 58.0, 1.15, 0.94, 0.95, 0, "tier_1_ipl"),
            ("akshar_ind", "Axar Patel", "IND", "all_rounder", "left", "SLA", 480, 880, 144.0, 23.5, 7.3, 19.5, 42.0, 54.0, 60.0, 52.0, 66.0, 56.0, 1.12, 0.89, 0.88, 0, "tier_1_ipl"),
            ("wsundar_ind", "Washington Sundar", "IND", "all_rounder", "left", "OB", 680, 890, 86.0, 29.5, 4.85, 28.0, 44.0, 54.0, 58.0, 50.0, 64.0, 55.0, 1.12, 1.02, 0.92, 0, "tier_1_ipl"),
            ("k_yadav_ind", "Kuldeep Yadav", "IND", "frontline_spinner", "left", "LAWS", 60, 920, 85.0, 8.0, 6.7, 13.8, 10.0, 15.0, 15.0, 42.0, 74.0, 48.0, 1.18, 0.92, 0.90, 0, "tier_1_ipl"),
            ("jbumrah_ind", "Jasprit Bumrah", "IND", "frontline_pacer", "right", "RF", 85, 1420, 95.0, 7.5, 6.2, 14.2, 15.0, 15.0, 20.0, 72.0, 68.0, 78.0, 1.0, 1.02, 0.99, 0, "tier_1_ipl"),
            ("asir_ind", "Mohammed Siraj", "IND", "frontline_pacer", "right", "RF", 40, 680, 90.0, 6.0, 8.1, 16.5, 15.0, 15.0, 15.0, 68.0, 52.0, 55.0, 0.98, 0.97, 0.85, 0, "tier_1_ipl"),
            ("arsheep_ind", "Arshdeep Singh", "IND", "frontline_pacer", "left", "RF", 35, 1120, 90.0, 6.5, 8.3, 13.5, 15.0, 15.0, 15.0, 65.0, 54.0, 69.0, 1.0, 0.98, 0.92, 0, "tier_1_ipl"),
            ("harshit_rana_ind", "Harshit Rana", "IND", "frontline_pacer", "right", "RF", 30, 480, 105.0, 8.5, 8.1, 16.5, 15.0, 15.0, 15.0, 66.0, 56.0, 68.0, 1.02, 0.96, 0.88, 0, "tier_1_ipl"),
            # India T20 Specialists
            ("skyadav_ind", "Suryakumar Yadav", "IND", "middle_order_batter", "right", "NONE", 1250, 0, 168.0, 42.0, 0, 0, 55.0, 72.0, 74.0, 10.0, 10.0, 10.0, 1.02, 0.96, 0.95, 0, "tier_1_ipl"),
            ("jaiswal_ind", "Yashasvi Jaiswal", "IND", "top_order_batter", "left", "NONE", 650, 0, 161.0, 36.5, 0, 0, 70.0, 60.0, 55.0, 10.0, 10.0, 10.0, 1.05, 0.95, 0.92, 0, "tier_1_ipl"),
            ("rinku_ind", "Rinku Singh", "IND", "finisher", "left", "NONE", 380, 0, 172.0, 58.0, 0, 0, 40.0, 62.0, 76.0, 10.0, 10.0, 10.0, 1.04, 0.96, 0.88, 0, "tier_1_ipl"),
            ("sanju_ind", "Sanju Samson", "IND", "wicket_keeper", "right", "NONE", 610, 0, 148.0, 26.5, 0, 0, 62.0, 56.0, 58.0, 10.0, 10.0, 10.0, 1.02, 0.94, 0.82, 0, "tier_1_ipl"),
            ("t_varma_ind", "Tilak Varma", "IND", "middle_order_batter", "left", "OB", 520, 40, 142.0, 38.0, 7.5, 24.0, 50.0, 62.0, 64.0, 15.0, 25.0, 15.0, 1.04, 0.95, 0.90, 0, "tier_1_ipl"),
            ("j_sharma_ind", "Jitesh Sharma", "IND", "wicket_keeper", "right", "NONE", 410, 0, 150.0, 28.0, 0, 0, 55.0, 58.0, 66.0, 10.0, 10.0, 10.0, 1.02, 0.94, 0.85, 0, "tier_1_ipl"),
            ("r_bishnoi_ind", "Ravi Bishnoi", "IND", "frontline_spinner", "right", "LBG", 50, 980, 85.0, 8.0, 7.1, 16.5, 10.0, 15.0, 15.0, 40.0, 72.0, 54.0, 1.12, 0.94, 0.92, 0, "tier_1_ipl"),
            ("a_khan_ind", "Avesh Khan", "IND", "frontline_pacer", "right", "RF", 40, 820, 95.0, 7.5, 8.2, 16.0, 15.0, 15.0, 20.0, 66.0, 54.0, 68.0, 0.98, 0.98, 0.88, 0, "tier_1_ipl"),

            # New Zealand Real ODI & T20 Squad
            ("d_conway_nz", "Devon Conway", "NZ", "top_order_batter", "left", "NONE", 1120, 0, 133.0, 39.5, 0, 0, 58.0, 56.0, 54.0, 10.0, 10.0, 10.0, 0.98, 1.03, 0.95, 0, "tier_1_ipl"),
            ("w_young_nz", "Will Young", "NZ", "top_order_batter", "right", "NONE", 1120, 0, 89.0, 42.0, 0, 0, 58.0, 60.0, 56.0, 10.0, 10.0, 10.0, 0.95, 1.08, 0.92, 0, "tier_1_ipl"),
            ("k_williamson_nz", "Kane Williamson", "NZ", "top_order_batter", "right", "OB", 1850, 80, 126.0, 33.5, 7.8, 25.0, 52.0, 60.0, 58.0, 15.0, 20.0, 10.0, 1.02, 1.02, 0.98, 0, "tier_1_ipl"),
            ("d_mitchell_nz", "Daryl Mitchell", "NZ", "middle_order_batter", "right", "RMF", 980, 240, 141.0, 32.0, 8.8, 22.0, 54.0, 64.0, 66.0, 35.0, 40.0, 35.0, 1.04, 1.01, 0.92, 0, "tier_1_ipl"),
            ("t_latham_nz", "Tom Latham", "NZ", "wicket_keeper", "left", "NONE", 2150, 0, 86.0, 35.8, 0, 0, 55.0, 60.0, 58.0, 10.0, 10.0, 10.0, 1.10, 1.04, 0.96, 0, "tier_1_ipl"),
            ("g_phillips_nz", "Glenn Phillips", "NZ", "middle_order_batter", "right", "OB", 1240, 380, 146.0, 32.5, 7.6, 21.0, 48.0, 66.0, 70.0, 25.0, 55.0, 40.0, 1.0, 1.02, 0.92, 0, "tier_1_ipl"),
            ("m_santner_nz", "Mitchell Santner", "NZ", "all_rounder", "left", "SLA", 580, 1680, 124.0, 18.0, 6.9, 19.5, 35.0, 45.0, 50.0, 48.0, 72.0, 58.0, 1.12, 0.96, 0.96, 0, "tier_1_ipl"),
            ("r_ravindra_nz", "Rachin Ravindra", "NZ", "all_rounder", "left", "SLA", 540, 420, 138.0, 28.5, 7.8, 22.0, 55.0, 58.0, 56.0, 35.0, 54.0, 42.0, 1.06, 0.98, 0.88, 0, "tier_1_ipl"),
            ("m_henry_nz", "Matt Henry", "NZ", "frontline_pacer", "right", "RF", 80, 1680, 88.0, 12.0, 5.15, 24.5, 15.0, 15.0, 20.0, 75.0, 60.0, 70.0, 0.92, 1.18, 0.96, 0, "tier_1_ipl"),
            ("l_ferguson_nz", "Lockie Ferguson", "NZ", "frontline_pacer", "right", "RF", 30, 890, 80.0, 5.0, 7.5, 14.2, 10.0, 10.0, 15.0, 65.0, 66.0, 72.0, 0.96, 1.03, 0.90, 0, "tier_1_ipl"),
            ("t_boult_nz", "Trent Boult", "NZ", "frontline_pacer", "left", "RF", 45, 1480, 85.0, 6.0, 7.8, 15.0, 15.0, 15.0, 15.0, 72.0, 55.0, 70.0, 0.98, 1.04, 0.98, 0, "tier_1_ipl"),
            ("j_duffy_nz", "Jacob Duffy", "NZ", "frontline_pacer", "right", "RMF", 30, 420, 75.0, 8.0, 5.60, 26.0, 10.0, 10.0, 15.0, 62.0, 54.0, 62.0, 0.90, 1.06, 0.85, 0, "tier_2_bbl_psl"),
            ("i_sodhi_nz", "Ish Sodhi", "NZ", "frontline_spinner", "right", "LBG", 60, 1540, 82.0, 8.5, 5.50, 27.0, 10.0, 15.0, 15.0, 35.0, 70.0, 52.0, 1.15, 0.95, 0.90, 0, "tier_1_ipl"),
            ("m_bracewell_nz", "Michael Bracewell", "NZ", "all_rounder", "left", "OB", 480, 540, 118.0, 38.0, 5.30, 28.0, 40.0, 55.0, 65.0, 35.0, 58.0, 45.0, 1.06, 1.04, 0.90, 0, "tier_1_ipl"),
            ("f_allen_nz", "Finn Allen", "NZ", "top_order_batter", "right", "NONE", 820, 0, 162.0, 24.5, 0, 0, 74.0, 52.0, 50.0, 10.0, 10.0, 10.0, 0.95, 1.04, 0.88, 0, "tier_1_ipl"),
            ("m_chapman_nz", "Mark Chapman", "NZ", "middle_order_batter", "left", "SLA", 680, 80, 134.0, 27.5, 8.4, 24.0, 44.0, 56.0, 62.0, 15.0, 25.0, 20.0, 1.02, 1.0, 0.85, 0, "tier_2_bbl_psl"),
            ("t_southee_nz", "Tim Southee", "NZ", "frontline_pacer", "right", "RMF", 140, 1860, 120.0, 12.0, 8.1, 16.0, 20.0, 20.0, 25.0, 66.0, 58.0, 68.0, 0.97, 1.03, 0.98, 0, "tier_1_ipl"),

            # Australia Full T20 Squad
            ("t_head_aus", "Travis Head", "AUS", "top_order_batter", "left", "OB", 1100, 210, 158.0, 34.0, 8.5, 22.0, 74.0, 58.0, 52.0, 25.0, 40.0, 20.0, 0.94, 1.06, 0.95, 0, "tier_1_ipl"),
            ("m_marsh_aus", "Mitchell Marsh", "AUS", "all_rounder", "right", "RMF", 1280, 540, 136.0, 30.5, 8.6, 21.0, 58.0, 60.0, 60.0, 45.0, 48.0, 48.0, 0.95, 1.05, 0.95, 0, "tier_1_ipl"),
            ("g_maxwell_aus", "Glenn Maxwell", "AUS", "all_rounder", "right", "OB", 1650, 820, 155.0, 30.0, 7.6, 24.0, 48.0, 68.0, 72.0, 35.0, 60.0, 45.0, 0.98, 1.02, 0.92, 0, "tier_1_ipl"),
            ("p_cummins_aus", "Pat Cummins", "AUS", "frontline_pacer", "right", "RF", 210, 1260, 135.0, 15.0, 7.3, 18.0, 25.0, 25.0, 35.0, 66.0, 62.0, 68.0, 0.97, 1.03, 0.95, 0, "tier_1_ipl"),
            ("m_starc_aus", "Mitchell Starc", "AUS", "frontline_pacer", "left", "RF", 95, 1380, 110.0, 10.0, 7.7, 15.2, 20.0, 20.0, 25.0, 74.0, 56.0, 72.0, 0.96, 1.04, 0.96, 0, "tier_1_ipl"),
            ("a_zampa_aus", "Adam Zampa", "AUS", "frontline_spinner", "right", "LBG", 60, 1640, 85.0, 7.0, 7.2, 17.0, 10.0, 15.0, 15.0, 35.0, 74.0, 52.0, 1.05, 0.97, 0.98, 0, "tier_1_ipl"),
            ("j_inglis_aus", "Josh Inglis", "AUS", "wicket_keeper", "right", "NONE", 650, 0, 152.0, 31.0, 0, 0, 64.0, 60.0, 60.0, 10.0, 10.0, 10.0, 0.96, 1.04, 0.92, 0, "tier_1_ipl"),
            ("m_stoinis_aus", "Marcus Stoinis", "AUS", "all_rounder", "right", "RMF", 1150, 580, 142.0, 29.5, 8.7, 19.5, 52.0, 62.0, 68.0, 40.0, 52.0, 50.0, 0.96, 1.04, 0.94, 0, "tier_1_ipl"),
            ("t_david_aus", "Tim David", "AUS", "finisher", "right", "OB", 680, 80, 162.0, 34.0, 9.2, 28.0, 42.0, 64.0, 76.0, 15.0, 20.0, 15.0, 0.98, 1.02, 0.90, 0, "tier_1_ipl"),
            ("n_ellis_aus", "Nathan Ellis", "AUS", "frontline_pacer", "right", "RF", 35, 780, 90.0, 8.0, 8.1, 14.8, 15.0, 15.0, 20.0, 60.0, 58.0, 74.0, 0.98, 1.02, 0.90, 0, "tier_1_ipl"),
            ("j_hazlewood_aus", "Josh Hazlewood", "AUS", "frontline_pacer", "right", "RF", 40, 1150, 85.0, 6.0, 7.2, 17.5, 15.0, 15.0, 15.0, 74.0, 60.0, 66.0, 0.98, 1.05, 0.96, 0, "tier_1_ipl"),
            ("s_abbott_aus", "Sean Abbott", "AUS", "all_rounder", "right", "RMF", 320, 640, 128.0, 18.0, 8.6, 18.0, 30.0, 45.0, 50.0, 55.0, 52.0, 60.0, 0.95, 1.02, 0.88, 0, "tier_2_bbl_psl"),
            ("m_wade_aus", "Matthew Wade", "AUS", "wicket_keeper", "left", "NONE", 1120, 0, 134.0, 26.0, 0, 0, 50.0, 56.0, 65.0, 10.0, 10.0, 10.0, 0.96, 1.02, 0.86, 0, "tier_1_ipl"),
            ("x_bartlett_aus", "Xavier Bartlett", "AUS", "frontline_pacer", "right", "RF", 20, 420, 90.0, 8.0, 7.6, 15.2, 15.0, 15.0, 15.0, 68.0, 54.0, 64.0, 0.96, 1.02, 0.88, 0, "tier_2_bbl_psl"),
            ("s_johnson_aus", "Spencer Johnson", "AUS", "frontline_pacer", "left", "RF", 15, 380, 85.0, 6.0, 7.5, 16.0, 15.0, 15.0, 15.0, 70.0, 54.0, 65.0, 0.96, 1.03, 0.86, 0, "tier_2_bbl_psl"),
            ("m_renshaw_aus", "Matt Renshaw", "AUS", "top_order_batter", "left", "OB", 850, 60, 128.0, 34.0, 7.8, 24.0, 64.0, 62.0, 50.0, 15.0, 20.0, 15.0, 0.92, 1.05, 0.90, 0, "tier_1_ipl"),
            ("c_connolly_aus", "Cooper Connolly", "AUS", "all_rounder", "left", "SLA", 420, 180, 142.0, 31.0, 7.4, 21.0, 55.0, 65.0, 68.0, 35.0, 58.0, 42.0, 0.96, 1.06, 0.95, 0, "tier_2_bbl_psl"),
            ("c_green_aus", "Cameron Green", "AUS", "all_rounder", "right", "RF", 720, 480, 146.0, 35.0, 7.8, 22.0, 62.0, 68.0, 66.0, 55.0, 58.0, 60.0, 0.96, 1.05, 0.94, 0, "tier_1_ipl"),
            ("l_scott_aus", "Liam Scott", "AUS", "all_rounder", "right", "RMF", 380, 260, 126.0, 29.5, 7.6, 22.0, 50.0, 58.0, 56.0, 45.0, 52.0, 48.0, 0.94, 1.02, 0.88, 1, "tier_2_bbl_psl"),
            ("j_edwards_aus", "Jack Edwards", "AUS", "all_rounder", "right", "RMF", 540, 340, 136.0, 32.0, 7.5, 20.0, 58.0, 64.0, 68.0, 52.0, 56.0, 62.0, 0.95, 1.04, 0.92, 0, "tier_2_bbl_psl"),
            ("o_peake_aus", "Oliver Peake", "AUS", "all_rounder", "left", "OB", 320, 120, 124.0, 35.0, 7.2, 25.0, 52.0, 66.0, 58.0, 25.0, 45.0, 30.0, 0.95, 1.02, 0.90, 1, "tier_2_bbl_psl"),
            ("a_carey_aus", "Alex Carey", "AUS", "wicket_keeper", "left", "NONE", 1100, 0, 132.0, 34.0, 0, 0, 56.0, 64.0, 62.0, 10.0, 10.0, 10.0, 0.96, 1.04, 0.94, 0, "tier_1_ipl"),
            ("m_beardman_aus", "Mahli Beardman", "AUS", "frontline_pacer", "right", "RF", 30, 480, 85.0, 8.0, 7.3, 14.5, 15.0, 15.0, 15.0, 75.0, 58.0, 70.0, 0.96, 1.08, 0.95, 1, "tier_2_bbl_psl"),
            ("l_morris_aus", "Lance Morris", "AUS", "frontline_pacer", "right", "RF", 40, 620, 80.0, 7.0, 7.4, 14.0, 15.0, 15.0, 15.0, 78.0, 56.0, 72.0, 0.95, 1.08, 0.94, 0, "tier_2_bbl_psl"),

            # England Full T20 Squad
            ("j_buttler_eng", "Jos Buttler", "ENG", "wicket_keeper", "right", "NONE", 2150, 0, 145.0, 35.0, 0, 0, 68.0, 64.0, 72.0, 10.0, 10.0, 10.0, 0.98, 1.02, 0.98, 0, "tier_1_ipl"),
            ("p_salt_eng", "Phil Salt", "ENG", "top_order_batter", "right", "NONE", 850, 0, 166.0, 34.5, 0, 0, 72.0, 58.0, 54.0, 10.0, 10.0, 10.0, 0.98, 1.02, 0.92, 0, "tier_1_ipl"),
            ("j_archer_eng", "Jofra Archer", "ENG", "frontline_pacer", "right", "RF", 50, 620, 115.0, 12.0, 7.4, 15.0, 15.0, 15.0, 20.0, 72.0, 60.0, 72.0, 0.97, 1.03, 0.90, 0, "tier_1_ipl"),
            ("a_rashid_eng", "Adil Rashid", "ENG", "frontline_spinner", "right", "LBG", 80, 1820, 85.0, 8.0, 7.3, 16.8, 10.0, 15.0, 15.0, 40.0, 72.0, 52.0, 1.06, 0.96, 0.96, 0, "tier_1_ipl"),
            ("w_jacks_eng", "Will Jacks", "ENG", "all_rounder", "right", "OB", 620, 240, 156.0, 30.5, 8.2, 22.0, 68.0, 60.0, 58.0, 30.0, 48.0, 30.0, 0.98, 1.02, 0.92, 0, "tier_1_ipl"),
            ("h_brook_eng", "Harry Brook", "ENG", "middle_order_batter", "right", "RMF", 840, 60, 148.0, 34.0, 8.5, 30.0, 56.0, 66.0, 68.0, 15.0, 20.0, 15.0, 1.00, 1.04, 0.94, 0, "tier_1_ipl"),
            ("l_livingstone_eng", "Liam Livingstone", "ENG", "all_rounder", "right", "OB", 980, 440, 149.0, 28.0, 8.1, 21.0, 52.0, 64.0, 72.0, 35.0, 55.0, 42.0, 0.98, 1.02, 0.92, 0, "tier_1_ipl"),
            ("s_curran_eng", "Sam Curran", "ENG", "all_rounder", "left", "LMF", 860, 1120, 132.0, 23.5, 8.4, 18.0, 45.0, 54.0, 62.0, 62.0, 56.0, 68.0, 0.98, 1.02, 0.94, 0, "tier_1_ipl"),
            ("j_overton_eng", "Jamie Overton", "ENG", "all_rounder", "right", "RF", 310, 420, 155.0, 24.0, 8.6, 17.5, 40.0, 55.0, 66.0, 58.0, 54.0, 62.0, 0.96, 1.02, 0.86, 0, "tier_2_bbl_psl"),
            ("r_topley_eng", "Reece Topley", "ENG", "frontline_pacer", "left", "LMF", 35, 780, 85.0, 6.0, 7.8, 17.0, 15.0, 15.0, 15.0, 68.0, 56.0, 64.0, 0.98, 1.03, 0.90, 0, "tier_1_ipl"),
            ("m_wood_eng", "Mark Wood", "ENG", "frontline_pacer", "right", "RF", 50, 920, 110.0, 9.0, 8.1, 14.5, 15.0, 15.0, 15.0, 76.0, 60.0, 70.0, 0.96, 1.05, 0.92, 0, "tier_1_ipl"),
            ("j_bethell_eng", "Jacob Bethell", "ENG", "all_rounder", "left", "SLA", 280, 160, 140.0, 28.0, 7.8, 24.0, 50.0, 56.0, 58.0, 30.0, 52.0, 35.0, 0.98, 1.00, 0.84, 0, "tier_2_bbl_psl"),
            ("b_carse_eng", "Brydon Carse", "ENG", "frontline_pacer", "right", "RF", 80, 480, 120.0, 14.0, 8.2, 18.0, 20.0, 25.0, 30.0, 64.0, 54.0, 62.0, 0.96, 1.02, 0.86, 0, "tier_2_bbl_psl"),
            ("d_mousley_eng", "Dan Mousley", "ENG", "all_rounder", "left", "OB", 240, 180, 135.0, 25.0, 7.9, 25.0, 48.0, 54.0, 55.0, 30.0, 50.0, 35.0, 0.98, 1.00, 0.82, 0, "domestic_list_a"),
            ("s_mahmood_eng", "Saqib Mahmood", "ENG", "frontline_pacer", "right", "RF", 25, 420, 85.0, 7.0, 8.0, 16.0, 15.0, 15.0, 15.0, 65.0, 54.0, 66.0, 0.96, 1.02, 0.85, 0, "tier_2_bbl_psl"),
            ("b_duckett_eng", "Ben Duckett", "ENG", "top_order_batter", "left", "NONE", 1200, 0, 152.0, 38.0, 0, 0, 74.0, 66.0, 58.0, 10.0, 10.0, 10.0, 0.96, 1.04, 0.96, 0, "tier_1_ipl"),
            ("j_root_eng", "Joe Root", "ENG", "top_order_batter", "right", "OB", 2400, 360, 128.0, 48.0, 7.1, 28.0, 68.0, 78.0, 64.0, 30.0, 52.0, 30.0, 0.99, 1.02, 0.98, 0, "tier_1_ipl"),
            ("j_cox_eng", "Jordan Cox", "ENG", "top_order_batter", "right", "NONE", 520, 0, 138.0, 31.0, 0, 0, 62.0, 58.0, 56.0, 10.0, 10.0, 10.0, 0.95, 1.02, 0.86, 0, "tier_2_bbl_psl"),
            ("t_banton_eng", "Tom Banton", "ENG", "wicket_keeper", "right", "NONE", 720, 0, 146.0, 30.0, 0, 0, 70.0, 58.0, 64.0, 10.0, 10.0, 10.0, 0.97, 1.02, 0.90, 0, "tier_1_ipl"),
            ("l_dawson_eng", "Liam Dawson", "ENG", "all_rounder", "right", "SLA", 650, 880, 132.0, 27.0, 6.9, 21.0, 48.0, 58.0, 62.0, 55.0, 68.0, 58.0, 1.02, 0.98, 0.92, 0, "tier_1_ipl"),
            ("g_atkinson_eng", "Gus Atkinson", "ENG", "frontline_pacer", "right", "RF", 60, 780, 110.0, 12.0, 7.5, 15.0, 20.0, 25.0, 30.0, 72.0, 58.0, 68.0, 0.96, 1.04, 0.94, 0, "tier_1_ipl"),
            ("j_tongue_eng", "Josh Tongue", "ENG", "frontline_pacer", "right", "RF", 20, 420, 85.0, 6.0, 7.6, 15.5, 15.0, 15.0, 15.0, 70.0, 56.0, 66.0, 0.95, 1.03, 0.88, 0, "tier_2_bbl_psl"),
            ("s_baker_eng", "Sonny Baker", "ENG", "frontline_pacer", "right", "RF", 15, 340, 80.0, 5.0, 7.7, 16.0, 15.0, 15.0, 15.0, 68.0, 54.0, 64.0, 0.94, 1.02, 0.85, 1, "tier_2_bbl_psl"),

            # South Africa Full T20 Squad
            ("h_klaasen_sa", "Heinrich Klaasen", "SA", "middle_order_batter", "right", "NONE", 940, 0, 158.0, 38.5, 0, 0, 44.0, 72.0, 74.0, 10.0, 10.0, 10.0, 1.06, 0.96, 0.95, 0, "tier_1_ipl"),
            ("k_rabada_sa", "Kagiso Rabada", "SA", "frontline_pacer", "right", "RF", 85, 1280, 110.0, 11.0, 7.9, 15.5, 15.0, 15.0, 20.0, 70.0, 58.0, 72.0, 0.98, 1.03, 0.96, 0, "tier_1_ipl"),
            ("a_markram_sa", "Aiden Markram", "SA", "top_order_batter", "right", "OB", 1350, 420, 145.0, 38.0, 7.4, 25.0, 64.0, 66.0, 62.0, 35.0, 55.0, 35.0, 1.02, 1.02, 0.96, 0, "tier_1_ipl"),
            ("q_dekock_sa", "Quinton de Kock", "SA", "wicket_keeper", "left", "NONE", 2150, 0, 141.0, 34.0, 0, 0, 72.0, 60.0, 58.0, 10.0, 10.0, 10.0, 1.04, 1.02, 0.96, 0, "tier_1_ipl"),
            ("d_miller_sa", "David Miller", "SA", "middle_order_batter", "left", "NONE", 1750, 0, 144.0, 36.5, 0, 0, 45.0, 66.0, 74.0, 10.0, 10.0, 10.0, 1.02, 1.00, 0.95, 0, "tier_1_ipl"),
            ("t_stubbs_sa", "Tristan Stubbs", "SA", "middle_order_batter", "right", "OB", 680, 80, 160.0, 36.0, 7.8, 24.0, 55.0, 68.0, 72.0, 20.0, 30.0, 20.0, 1.05, 0.98, 0.92, 0, "tier_1_ipl"),
            ("m_jansen_sa", "Marco Jansen", "SA", "all_rounder", "right", "LF", 420, 840, 135.0, 22.0, 8.1, 16.8, 35.0, 50.0, 60.0, 68.0, 58.0, 66.0, 0.98, 1.04, 0.94, 0, "tier_1_ipl"),
            ("k_maharaj_sa", "Keshav Maharaj", "SA", "frontline_spinner", "right", "SLA", 110, 1260, 95.0, 12.0, 6.9, 21.0, 20.0, 25.0, 25.0, 48.0, 72.0, 54.0, 1.10, 0.98, 0.95, 0, "tier_1_ipl"),
            ("g_coetzee_sa", "Gerald Coetzee", "SA", "frontline_pacer", "right", "RF", 80, 620, 125.0, 15.0, 8.4, 14.5, 20.0, 25.0, 35.0, 72.0, 58.0, 68.0, 0.96, 1.04, 0.90, 0, "tier_1_ipl"),
            ("a_nortje_sa", "Anrich Nortje", "SA", "frontline_pacer", "right", "RF", 40, 980, 90.0, 7.0, 7.8, 14.2, 15.0, 15.0, 15.0, 75.0, 60.0, 72.0, 0.98, 1.04, 0.94, 0, "tier_1_ipl"),
            ("t_shamsi_sa", "Tabraiz Shamsi", "SA", "frontline_spinner", "right", "LAWS", 50, 1340, 80.0, 6.0, 7.2, 17.5, 10.0, 15.0, 15.0, 38.0, 74.0, 52.0, 1.12, 0.95, 0.92, 0, "tier_1_ipl"),
            ("r_hendricks_sa", "Reeza Hendricks", "SA", "top_order_batter", "right", "NONE", 1120, 0, 132.0, 32.0, 0, 0, 62.0, 58.0, 54.0, 10.0, 10.0, 10.0, 1.00, 1.00, 0.90, 0, "tier_1_ipl"),
            ("r_rickelton_sa", "Ryan Rickelton", "SA", "wicket_keeper", "left", "NONE", 480, 0, 146.0, 30.0, 0, 0, 66.0, 56.0, 54.0, 10.0, 10.0, 10.0, 1.00, 1.02, 0.88, 0, "tier_1_ipl"),
            ("l_ngidi_sa", "Lungi Ngidi", "SA", "frontline_pacer", "right", "RF", 40, 920, 95.0, 8.0, 8.2, 15.0, 15.0, 15.0, 15.0, 68.0, 56.0, 68.0, 0.98, 1.02, 0.92, 0, "tier_1_ipl"),
            ("b_fortuin_sa", "Bjorn Fortuin", "SA", "frontline_spinner", "right", "SLA", 80, 540, 90.0, 11.0, 7.1, 22.0, 20.0, 25.0, 25.0, 45.0, 68.0, 50.0, 1.08, 0.96, 0.86, 0, "tier_2_bbl_psl"),

            # Afghanistan Official Test Squad
            ("h_shahidi_afg", "Hashmatullah Shahidi", "AFG", "middle_order_batter", "left", "OB", 1400, 100, 118.0, 39.0, 7.8, 0, 55, 65, 55, 50, 50, 50, 1.08, 1.0, 0.9, 0, "tier_1_ipl"),
            ("r_shah_afg", "Rahmat Shah", "AFG", "top_order_batter", "right", "LBG", 1650, 420, 115.0, 38.5, 7.2, 28.0, 56.0, 64.0, 60.0, 25.0, 48.0, 25.0, 1.10, 0.95, 0.92, 0, "tier_1_ipl"),
            ("i_alikhil_afg", "Ikram Alikhil", "AFG", "wicket_keeper", "left", "NONE", 850, 0, 108.0, 32.0, 0, 0, 52.0, 58.0, 54.0, 10.0, 10.0, 10.0, 1.05, 0.92, 0.88, 0, "domestic_list_a"),
            ("i_zadran_afg", "Ibrahim Zadran", "AFG", "top_order_batter", "right", "RMF", 1200, 50, 125.0, 41.0, 7.5, 0, 60, 65, 60, 50, 50, 50, 1.1, 1.0, 0.9, 0, "tier_1_ipl"),
            ("r_gurbaz_afg", "Rahmanullah Gurbaz", "AFG", "wicket_keeper", "right", "NONE", 1300, 0, 142.0, 34.0, 8.0, 0, 65, 60, 55, 50, 50, 50, 1.1, 1.0, 0.95, 0, "tier_1_ipl"),
            ("a_omarzai_afg", "Azmatullah Omarzai", "AFG", "all_rounder", "right", "RMF", 850, 950, 135.0, 36.0, 7.4, 21.0, 60, 60, 60, 60, 60, 60, 1.05, 1.0, 0.9, 0, "tier_1_ipl"),
            ("r_khan_afg", "Rashid Khan", "AFG", "all_rounder", "right", "LBG", 620, 2180, 134.0, 16.0, 6.2, 13.5, 30.0, 45.0, 55.0, 52.0, 82.0, 72.0, 1.15, 0.98, 0.98, 0, "tier_1_ipl"),
            ("q_ahmad_afg", "Qais Ahmad", "AFG", "frontline_spinner", "right", "LBG", 300, 1400, 115.0, 15.0, 6.8, 19.0, 45, 45, 45, 65, 68, 65, 1.1, 1.0, 0.85, 0, "tier_1_ipl"),
            ("m_saleem_afg", "Mohammad Saleem Safi", "AFG", "frontline_pacer", "right", "RF", 100, 600, 95.0, 10.0, 6.8, 20.0, 35, 35, 35, 60, 60, 60, 1.0, 1.0, 0.8, 0, "tier_1_ipl"),
            ("z_akbar_afg", "Zia-ur-Rehman Akbar", "AFG", "frontline_spinner", "right", "SLA", 150, 900, 100.0, 12.0, 6.9, 21.0, 40, 40, 40, 62, 62, 62, 1.05, 1.0, 0.8, 0, "tier_1_ipl"),
            ("f_farooqi_afg", "Fazalhaq Farooqi", "AFG", "frontline_pacer", "left", "LF", 150, 1300, 95.0, 9.0, 6.9, 17.5, 35, 35, 35, 68, 68, 68, 1.08, 1.05, 0.9, 0, "tier_1_ipl"),
            ("z_sharifi_afg", "Ziaur Rahman Sharifi", "AFG", "frontline_pacer", "right", "RF", 100, 500, 90.0, 8.0, 7.2, 22.0, 35, 35, 35, 58, 58, 58, 1.0, 1.0, 0.75, 0, "tier_1_ipl"),
            ("s_kamal_afg", "Shahidullah Kamal", "AFG", "all_rounder", "left", "SLA", 400, 300, 120.0, 26.0, 7.5, 26.0, 50, 50, 50, 52, 52, 52, 1.0, 1.0, 0.75, 0, "tier_1_ipl"),
            ("y_ahmadzai_afg", "Yamin Ahmadzai", "AFG", "frontline_pacer", "right", "RF", 80, 850, 85.0, 8.0, 7.4, 20.0, 20.0, 25.0, 25.0, 68.0, 56.0, 62.0, 1.05, 1.00, 0.84, 0, "domestic_list_a"),
            ("a_zazai_afg", "Afsar Zazai", "AFG", "wicket_keeper", "right", "NONE", 920, 0, 112.0, 29.5, 0, 0, 50.0, 56.0, 52.0, 10.0, 10.0, 10.0, 1.04, 0.90, 0.85, 0, "domestic_list_a"),

            # Zimbabwe Official Test Squad
            ("b_bennett_zim", "Brian Bennett", "ZIM", "middle_order_batter", "right", "OB", 600, 200, 135.0, 38.0, 7.2, 24.0, 55, 60, 55, 50, 50, 50, 1.0, 1.0, 0.85, 0, "tier_1_ipl"),
            ("b_taylor_zim", "Brendan Taylor", "ZIM", "wicket_keeper", "right", "OB", 1500, 100, 130.0, 42.5, 7.5, 0, 65, 65, 60, 50, 50, 50, 1.1, 1.0, 0.95, 0, "tier_1_ipl"),
            ("b_curran_zim", "Ben Curran", "ZIM", "top_order_batter", "left", "NONE", 300, 0, 115.0, 35.0, 8.0, 0, 50, 50, 50, 50, 50, 50, 1.0, 1.0, 0.8, 0, "tier_1_ipl"),
            ("i_kaia_zim", "Innocent Kaia", "ZIM", "top_order_batter", "right", "OB", 400, 0, 120.0, 32.0, 8.0, 0, 50, 50, 50, 50, 50, 50, 1.0, 1.0, 0.8, 0, "tier_1_ipl"),
            ("c_ervine_zim", "Craig Ervine", "ZIM", "top_order_batter", "left", "OB", 1400, 0, 122.0, 36.0, 8.0, 0, 55, 60, 55, 50, 50, 50, 1.0, 1.0, 0.85, 0, "tier_1_ipl"),
            ("a_naqvi_zim", "Antum Naqvi", "ZIM", "all_rounder", "right", "OB", 350, 300, 125.0, 32.0, 7.0, 22.0, 50, 50, 50, 50, 50, 50, 1.0, 1.0, 0.8, 0, "tier_1_ipl"),
            ("w_madhevere_zim", "Wessly Madhevere", "ZIM", "all_rounder", "right", "OB", 700, 400, 125.0, 28.0, 7.2, 25.0, 50, 50, 50, 50, 50, 50, 1.0, 1.0, 0.8, 0, "tier_1_ipl"),
            ("t_tsiga_zim", "Tafadzwa Tsiga", "ZIM", "wicket_keeper", "right", "NONE", 200, 0, 110.0, 24.0, 8.0, 0, 45, 45, 45, 50, 50, 50, 1.0, 1.0, 0.7, 0, "tier_1_ipl"),
            ("g_cremer_zim", "Graeme Cremer", "ZIM", "frontline_spinner", "right", "LBG", 300, 1200, 105.0, 18.0, 6.8, 20.0, 40, 40, 40, 60, 65, 60, 1.05, 1.0, 0.8, 0, "tier_1_ipl"),
            ("n_nyamhuri_zim", "Newman Nyamhuri", "ZIM", "frontline_pacer", "left", "LMF", 100, 400, 95.0, 12.0, 7.5, 22.0, 35, 35, 35, 55, 55, 55, 1.0, 1.0, 0.7, 0, "tier_1_ipl"),
            ("b_muzarabani_zim", "Blessing Muzarabani", "ZIM", "frontline_pacer", "right", "RF", 50, 890, 85.0, 6.0, 7.5, 18.0, 15.0, 15.0, 15.0, 62.0, 56.0, 64.0, 0.96, 1.02, 0.90, 0, "tier_2_bbl_psl"),
            ("r_ngarava_zim", "Richard Ngarava", "ZIM", "frontline_pacer", "left", "LMF", 120, 1150, 95.0, 11.0, 7.3, 17.5, 30, 35, 35, 65, 58, 65, 1.05, 1.02, 0.88, 0, "tier_2_bbl_psl"),
            ("b_evans_zim", "Brad Evans", "ZIM", "all_rounder", "right", "RF", 350, 680, 122.0, 22.0, 7.9, 19.5, 40.0, 50.0, 56.0, 58.0, 56.0, 62.0, 1.02, 1.02, 0.85, 0, "domestic_list_a"),
            ("w_masakadza_zim", "Wellington Masakadza", "ZIM", "all_rounder", "left", "SLA", 480, 920, 114.0, 19.0, 6.9, 24.0, 35.0, 48.0, 50.0, 45.0, 66.0, 52.0, 1.08, 0.95, 0.86, 0, "tier_2_bbl_psl"),
            ("v_masekesa_zim", "Vincent Masekesa", "ZIM", "frontline_spinner", "right", "LBG", 60, 420, 90.0, 8.0, 7.2, 25.0, 15.0, 20.0, 20.0, 38.0, 64.0, 48.0, 1.05, 0.92, 0.80, 0, "domestic_list_a"),
            ("s_raza_zim", "Sikandar Raza", "ZIM", "all_rounder", "right", "OB", 1320, 1180, 135.0, 26.0, 7.2, 19.5, 45.0, 60.0, 62.0, 40.0, 66.0, 52.0, 1.05, 0.95, 0.92, 0, "tier_1_ipl"),

            # Pakistan Test Squad (15 Declared Players)
            ("s_masood_pak", "Shan Masood", "PAK", "top_order_batter", "left", "RMF", 1420, 60, 62.0, 32.5, 5.4, 30.0, 54.0, 58.0, 52.0, 15.0, 20.0, 15.0, 1.05, 0.98, 0.95, 0, "tier_1_ipl"),
            ("b_azam_pak", "Babar Azam", "PAK", "top_order_batter", "right", "OB", 2850, 80, 128.0, 46.8, 6.2, 36.0, 62.0, 72.0, 65.0, 15.0, 20.0, 10.0, 1.10, 0.96, 0.98, 0, "tier_1_ipl"),
            ("a_shafique_pak", "Abdullah Shafique", "PAK", "top_order_batter", "right", "OB", 1650, 0, 68.0, 39.2, 0, 0, 58.0, 60.0, 54.0, 10.0, 10.0, 10.0, 1.08, 0.94, 0.94, 0, "tier_2_bbl_psl"),
            ("s_ayub_pak", "Saim Ayub", "PAK", "top_order_batter", "left", "OB", 850, 120, 134.0, 31.0, 6.8, 25.0, 65.0, 56.0, 55.0, 15.0, 25.0, 15.0, 1.04, 0.95, 0.92, 0, "tier_2_bbl_psl"),
            ("s_shakeel_pak", "Saud Shakeel", "PAK", "middle_order_batter", "left", "SLA", 1750, 140, 65.0, 56.5, 4.2, 38.0, 52.0, 70.0, 62.0, 20.0, 35.0, 20.0, 1.15, 0.95, 0.96, 0, "tier_2_bbl_psl"),
            ("m_rizwan_pak", "Mohammad Rizwan", "PAK", "wicket_keeper", "right", "NONE", 2100, 0, 126.0, 42.5, 0, 0, 58.0, 66.0, 65.0, 10.0, 10.0, 10.0, 1.08, 0.98, 0.96, 0, "tier_1_ipl"),
            ("s_ali_agha_pak", "Salman Ali Agha", "PAK", "all_rounder", "right", "OB", 1450, 920, 85.0, 41.0, 4.8, 30.0, 50.0, 62.0, 60.0, 40.0, 64.0, 45.0, 1.12, 0.94, 0.92, 0, "tier_2_bbl_psl"),
            ("a_jamal_pak", "Aamer Jamal", "PAK", "all_rounder", "right", "RF", 520, 880, 122.0, 26.5, 5.2, 22.0, 45.0, 52.0, 62.0, 68.0, 45.0, 65.0, 0.95, 1.12, 0.90, 0, "tier_2_bbl_psl"),
            ("s_afridi_pak", "Shaheen Shah Afridi", "PAK", "frontline_pacer", "left", "LF", 180, 1850, 115.0, 14.5, 4.6, 21.0, 20.0, 20.0, 25.0, 75.0, 55.0, 72.0, 0.96, 1.15, 0.95, 0, "tier_1_ipl"),
            ("n_shah_pak", "Naseem Shah", "PAK", "frontline_pacer", "right", "RF", 120, 1420, 98.0, 12.0, 4.4, 22.5, 15.0, 15.0, 20.0, 74.0, 54.0, 70.0, 0.95, 1.12, 0.92, 0, "tier_1_ipl"),
            ("n_ali_pak", "Noman Ali", "PAK", "frontline_spinner", "left", "SLA", 140, 2450, 55.0, 15.0, 2.95, 27.0, 15.0, 15.0, 15.0, 38.0, 76.0, 48.0, 1.20, 0.88, 0.92, 0, "tier_2_bbl_psl"),
            ("s_khan_pak", "Sajid Khan", "PAK", "frontline_spinner", "right", "OB", 110, 1980, 60.0, 14.0, 3.20, 26.0, 15.0, 15.0, 15.0, 36.0, 78.0, 45.0, 1.22, 0.86, 0.90, 0, "tier_2_bbl_psl"),
            ("m_hamza_pak", "Mir Hamza", "PAK", "frontline_pacer", "left", "LF", 40, 950, 70.0, 8.0, 3.90, 24.0, 15.0, 15.0, 15.0, 70.0, 52.0, 62.0, 0.94, 1.08, 0.88, 0, "tier_2_bbl_psl"),
            ("s_ahmed_pak", "Sarfaraz Ahmed", "PAK", "wicket_keeper", "right", "NONE", 1850, 0, 78.0, 37.5, 0, 0, 52.0, 62.0, 58.0, 10.0, 10.0, 10.0, 1.08, 0.92, 0.92, 0, "tier_2_bbl_psl"),
            ("m_huraira_pak", "Mohammad Huraira", "PAK", "top_order_batter", "right", "NONE", 450, 0, 65.0, 48.0, 0, 0, 56.0, 58.0, 52.0, 10.0, 10.0, 10.0, 1.06, 0.92, 0.88, 0, "domestic_list_a"),

            # Sri Lanka Test Squad (15 Declared Players)
            ("w_hasaranga_sl", "Wanindu Hasaranga", "SL", "all_rounder", "right", "LBG", 540, 1580, 126.0, 16.5, 6.8, 14.5, 30.0, 45.0, 52.0, 48.0, 78.0, 64.0, 1.15, 0.92, 0.96, 0, "tier_1_ipl"),
            ("d_desilva_sl", "Dhananjaya de Silva", "SL", "all_rounder", "right", "OB", 2400, 1680, 76.0, 39.8, 3.65, 34.0, 52.0, 66.0, 60.0, 42.0, 65.0, 48.0, 1.15, 0.92, 0.96, 0, "tier_1_ipl"),
            ("d_karunaratne_sl", "Dimuth Karunaratne", "SL", "top_order_batter", "left", "RMF", 2900, 40, 58.0, 40.5, 4.5, 40.0, 56.0, 68.0, 54.0, 15.0, 20.0, 10.0, 1.14, 0.94, 0.96, 0, "tier_2_bbl_psl"),
            ("p_nissanka_sl", "Pathum Nissanka", "SL", "top_order_batter", "right", "NONE", 1450, 0, 88.0, 42.2, 0, 0, 62.0, 66.0, 60.0, 10.0, 10.0, 10.0, 1.12, 0.98, 0.95, 0, "tier_1_ipl"),
            ("d_chandimal_sl", "Dinesh Chandimal", "SL", "middle_order_batter", "right", "NONE", 2800, 0, 72.0, 44.2, 0, 0, 54.0, 68.0, 62.0, 10.0, 10.0, 10.0, 1.12, 0.95, 0.96, 0, "tier_1_ipl"),
            ("a_mathews_sl", "Angelo Mathews", "SL", "all_rounder", "right", "RMF", 3200, 950, 68.0, 45.4, 3.8, 35.0, 50.0, 70.0, 64.0, 40.0, 52.0, 45.0, 1.12, 0.96, 0.95, 0, "tier_1_ipl"),
            ("k_mendis_kam_sl", "Kamindu Mendis", "SL", "all_rounder", "left", "OB", 980, 520, 82.0, 62.8, 4.10, 30.0, 58.0, 74.0, 68.0, 38.0, 64.0, 48.0, 1.18, 0.96, 0.96, 0, "tier_1_ipl"),
            ("k_mendis_sl", "Kusal Mendis", "SL", "wicket_keeper", "right", "NONE", 2200, 0, 92.0, 36.5, 0, 0, 60.0, 62.0, 64.0, 10.0, 10.0, 10.0, 1.08, 0.96, 0.94, 0, "tier_1_ipl"),
            ("r_mendis_sl", "Ramesh Mendis", "SL", "all_rounder", "right", "OB", 750, 1850, 74.0, 22.0, 3.45, 29.0, 44.0, 52.0, 55.0, 42.0, 70.0, 50.0, 1.18, 0.88, 0.92, 0, "tier_2_bbl_psl"),
            ("p_jayasuriya_sl", "Prabath Jayasuriya", "SL", "frontline_spinner", "right", "SLA", 120, 2650, 50.0, 10.5, 3.05, 25.5, 15.0, 15.0, 15.0, 38.0, 84.0, 50.0, 1.25, 0.85, 0.94, 0, "tier_2_bbl_psl"),
            ("a_fernando_sl", "Asitha Fernando", "SL", "frontline_pacer", "right", "RF", 90, 1450, 82.0, 8.0, 3.85, 23.0, 15.0, 15.0, 20.0, 72.0, 55.0, 68.0, 0.96, 1.10, 0.92, 0, "tier_1_ipl"),
            ("v_fernando_sl", "Vishwa Fernando", "SL", "frontline_pacer", "left", "LF", 80, 1250, 75.0, 7.5, 3.95, 24.5, 15.0, 15.0, 15.0, 70.0, 52.0, 64.0, 0.95, 1.08, 0.90, 0, "tier_2_bbl_psl"),
            ("l_kumara_sl", "Lahiru Kumara", "SL", "frontline_pacer", "right", "RF", 60, 1100, 90.0, 6.5, 4.30, 22.0, 15.0, 15.0, 15.0, 72.0, 50.0, 66.0, 0.94, 1.12, 0.90, 0, "tier_2_bbl_psl"),
            ("s_samarawickrama_sl", "Sadeera Samarawickrama", "SL", "wicket_keeper", "right", "NONE", 850, 0, 84.0, 34.0, 0, 0, 55.0, 60.0, 58.0, 10.0, 10.0, 10.0, 1.08, 0.94, 0.92, 0, "tier_2_bbl_psl"),
            ("m_rathnayake_sl", "Milan Rathnayake", "SL", "all_rounder", "right", "RMF", 380, 550, 85.0, 32.0, 4.10, 28.0, 45.0, 55.0, 58.0, 64.0, 45.0, 60.0, 0.98, 1.05, 0.90, 0, "domestic_list_a"),
            ("j_vandersay_sl", "Jeffrey Vandersay", "SL", "frontline_spinner", "right", "LBG", 80, 780, 70.0, 11.0, 3.60, 28.0, 15.0, 15.0, 15.0, 36.0, 74.0, 48.0, 1.16, 0.90, 0.88, 0, "tier_2_bbl_psl"),

            # Additional Pakistan Shaheens / Test Squad Players
            ("a_awais_pak", "Azan Awais", "PAK", "top_order_batter", "left", "NONE", 250, 0, 68.0, 36.5, 0, 0, 52.0, 58.0, 50.0, 10.0, 10.0, 10.0, 1.05, 0.92, 0.88, 1, "domestic_list_a"),
            ("t_arif_pak", "Tariq Arif", "PAK", "top_order_batter", "right", "NONE", 200, 0, 65.0, 34.0, 0, 0, 50.0, 55.0, 48.0, 10.0, 10.0, 10.0, 1.04, 0.90, 0.88, 1, "domestic_list_a"),
            ("m_taha_pak", "Mohammad Taha", "PAK", "middle_order_batter", "left", "SLA", 300, 150, 72.0, 35.0, 5.2, 32.0, 50.0, 56.0, 52.0, 20.0, 30.0, 20.0, 1.05, 0.90, 0.88, 1, "domestic_list_a"),
            ("g_ghori_pak", "Ghulam Ghori", "PAK", "wicket_keeper", "right", "NONE", 220, 0, 70.0, 32.0, 0, 0, 50.0, 54.0, 50.0, 10.0, 10.0, 10.0, 1.04, 0.90, 0.85, 1, "domestic_list_a"),
            ("a_minhas_pak", "Arafat Minhas", "PAK", "all_rounder", "left", "SLA", 400, 350, 85.0, 32.0, 4.8, 28.0, 48.0, 55.0, 54.0, 35.0, 55.0, 45.0, 1.06, 0.92, 0.88, 1, "tier_2_bbl_psl"),
            ("razaullah_pak", "Razaullah", "PAK", "frontline_pacer", "right", "RF", 50, 450, 60.0, 10.0, 4.5, 26.0, 15.0, 15.0, 15.0, 62.0, 50.0, 58.0, 1.02, 0.95, 0.85, 1, "domestic_list_a"),
            ("m_abbas_pak", "Mohammad Abbas", "PAK", "frontline_pacer", "right", "RFM", 120, 2800, 45.0, 8.5, 2.5, 21.0, 15.0, 15.0, 15.0, 75.0, 70.0, 60.0, 1.08, 1.15, 0.95, 0, "tier_2_bbl_psl"),
            ("saqib_khan_pak", "Saqib Khan", "PAK", "frontline_pacer", "right", "RF", 40, 380, 55.0, 7.0, 4.8, 25.0, 15.0, 15.0, 15.0, 60.0, 50.0, 55.0, 1.02, 0.95, 0.85, 1, "domestic_list_a"),
            ("sameer_minhas_pak", "Sameer Minhas", "PAK", "top_order_batter", "right", "NONE", 180, 0, 65.0, 30.0, 0, 0, 48.0, 52.0, 48.0, 10.0, 10.0, 10.0, 1.02, 0.90, 0.85, 1, "domestic_list_a"),
            ("ali_shabbir_pak", "Ali Shabbir", "PAK", "frontline_pacer", "left", "LF", 30, 320, 50.0, 6.0, 5.0, 27.0, 15.0, 15.0, 15.0, 58.0, 48.0, 52.0, 1.00, 0.92, 0.85, 1, "domestic_list_a"),
            ("m_suleman_pak", "Mohammad Suleman", "PAK", "middle_order_batter", "right", "OB", 250, 80, 68.0, 34.0, 5.5, 35.0, 48.0, 54.0, 50.0, 15.0, 25.0, 15.0, 1.04, 0.90, 0.85, 1, "domestic_list_a"),
            ("s_baig_pak", "Saad Baig", "PAK", "wicket_keeper", "left", "NONE", 320, 0, 75.0, 35.0, 0, 0, 52.0, 56.0, 54.0, 10.0, 10.0, 10.0, 1.05, 0.92, 0.88, 1, "domestic_list_a"),

            # Additional Sri Lanka Test Squad Players
            ("p_sooriyabandara_sl", "Pasindu Sooriyabandara", "SL", "top_order_batter", "right", "NONE", 380, 0, 66.0, 42.0, 0, 0, 54.0, 62.0, 52.0, 10.0, 10.0, 10.0, 1.08, 0.92, 0.90, 1, "domestic_list_a"),
            ("l_udara_sl", "Lahiru Udara", "SL", "top_order_batter", "right", "NONE", 420, 0, 72.0, 38.0, 0, 0, 56.0, 60.0, 54.0, 10.0, 10.0, 10.0, 1.08, 0.92, 0.90, 1, "domestic_list_a"),
            ("s_dinusha_sl", "Sonal Dinusha", "SL", "all_rounder", "left", "SLA", 280, 320, 68.0, 33.0, 4.2, 30.0, 48.0, 55.0, 50.0, 30.0, 55.0, 40.0, 1.10, 0.90, 0.88, 1, "domestic_list_a"),
            ("d_madushanka_sl", "Dilshan Madushanka", "SL", "frontline_pacer", "left", "LF", 60, 1200, 80.0, 7.5, 4.8, 22.0, 15.0, 15.0, 15.0, 72.0, 55.0, 68.0, 1.05, 1.05, 0.92, 0, "tier_1_ipl"),
            ("k_nuwantha_sl", "Kasun Nuwantha", "SL", "frontline_pacer", "right", "RF", 40, 280, 60.0, 6.0, 5.1, 28.0, 15.0, 15.0, 15.0, 56.0, 46.0, 52.0, 1.02, 0.92, 0.85, 1, "domestic_list_a"),
            ("e_malinga_sl", "Eshan Malinga", "SL", "frontline_pacer", "right", "RF", 35, 310, 65.0, 7.0, 4.9, 26.0, 15.0, 15.0, 15.0, 58.0, 48.0, 54.0, 1.04, 0.94, 0.86, 1, "domestic_list_a"),
            ("a_bandara_sl", "Ashen Bandara", "SL", "middle_order_batter", "right", "LBG", 550, 120, 82.0, 34.0, 6.2, 32.0, 50.0, 58.0, 56.0, 20.0, 35.0, 25.0, 1.08, 0.92, 0.90, 0, "tier_2_bbl_psl"),

            # LIVE MATCH: West Indies Squad
            ("b_king_wi", "Brandon King", "WI", "top_order_batter", "right", "NONE", 980, 0, 134.0, 32.5, 0, 0, 66.0, 58.0, 54.0, 10.0, 10.0, 10.0, 0.92, 1.02, 0.95, 0, "tier_1_ipl"),
            ("j_dasilva_wi", "Joshua Da Silva", "WI", "wicket_keeper", "right", "NONE", 850, 0, 62.0, 28.5, 0, 0, 50.0, 54.0, 52.0, 10.0, 10.0, 10.0, 0.95, 1.00, 0.94, 0, "tier_2_bbl_psl"),
            ("k_wickham_wi", "Kevin Wickham", "WI", "top_order_batter", "right", "NONE", 420, 0, 75.0, 31.0, 0, 0, 52.0, 55.0, 52.0, 10.0, 10.0, 10.0, 0.96, 0.98, 0.88, 0, "domestic_list_a"),
            ("k_hodge_wi", "Kavem Hodge", "WI", "middle_order_batter", "right", "SLA", 640, 380, 72.0, 34.0, 3.80, 36.0, 52.0, 58.0, 54.0, 25.0, 40.0, 25.0, 0.90, 1.05, 0.92, 0, "tier_2_bbl_psl"),
            ("a_jangoo_wi", "Amir Jangoo", "WI", "wicket_keeper", "left", "NONE", 320, 0, 78.0, 27.0, 0, 0, 54.0, 52.0, 50.0, 10.0, 10.0, 10.0, 0.94, 0.96, 0.85, 0, "domestic_list_a"),
            ("j_greaves_wi", "Justin Greaves", "WI", "all_rounder", "right", "RMF", 560, 480, 76.0, 30.5, 4.10, 32.0, 48.0, 54.0, 52.0, 35.0, 48.0, 40.0, 0.95, 1.00, 0.90, 0, "tier_2_bbl_psl"),
            ("r_chase_wi", "Roston Chase", "WI", "all_rounder", "right", "OB", 1450, 1850, 75.0, 28.5, 3.45, 35.0, 46.0, 56.0, 54.0, 42.0, 65.0, 48.0, 1.10, 0.95, 0.96, 0, "tier_1_ipl"),
            ("s_joseph_wi", "Shamar Joseph", "WI", "frontline_pacer", "right", "RF", 80, 680, 95.0, 14.0, 3.85, 18.5, 15.0, 15.0, 20.0, 72.0, 62.0, 70.0, 0.95, 1.15, 0.94, 0, "tier_1_ipl"),
            ("j_bishop_wi", "Joshua Bishop", "WI", "frontline_spinner", "left", "SLA", 40, 520, 78.0, 10.0, 3.60, 34.0, 10.0, 15.0, 15.0, 30.0, 64.0, 45.0, 1.12, 0.90, 0.86, 0, "tier_2_bbl_psl"),
            ("j_seales_wi", "Jayden Seales", "WI", "frontline_pacer", "right", "RF", 50, 1120, 70.0, 8.0, 3.50, 21.0, 15.0, 15.0, 20.0, 70.0, 60.0, 68.0, 0.96, 1.12, 0.95, 0, "tier_1_ipl"),
            ("j_warrican_wi", "Jomel Warrican", "WI", "frontline_spinner", "left", "SLA", 80, 1580, 65.0, 9.0, 3.20, 30.0, 10.0, 15.0, 15.0, 32.0, 68.0, 48.0, 1.18, 0.88, 0.92, 0, "tier_2_bbl_psl"),
            # West Indies Reserves / Concussion Subs
            ("a_athanaze_wi", "Alick Athanaze", "WI", "top_order_batter", "left", "OB", 650, 60, 68.0, 32.0, 4.20, 35.0, 50.0, 56.0, 52.0, 15.0, 25.0, 15.0, 1.05, 0.95, 0.88, 0, "tier_2_bbl_psl"),
            ("t_imlach_wi", "Tevin Imlach", "WI", "wicket_keeper", "right", "NONE", 420, 0, 58.0, 30.0, 0, 0, 48.0, 52.0, 50.0, 10.0, 10.0, 10.0, 0.98, 0.98, 0.85, 0, "domestic_list_a"),
            ("g_motie_wi", "Gudakesh Motie", "WI", "frontline_spinner", "left", "SLA", 110, 1420, 82.0, 15.0, 3.10, 24.0, 15.0, 15.0, 20.0, 40.0, 75.0, 52.0, 1.20, 0.90, 0.92, 0, "tier_1_ipl"),
            ("a_phillip_wi", "Anderson Phillip", "WI", "frontline_pacer", "right", "RF", 40, 780, 65.0, 8.0, 3.80, 26.0, 15.0, 15.0, 15.0, 68.0, 55.0, 62.0, 0.95, 1.05, 0.86, 0, "domestic_list_a"),

            # LIVE MATCH: Bangladesh Squad
            ("s_islam_ban", "Shadman Islam", "BAN", "top_order_batter", "left", "NONE", 880, 0, 48.0, 30.5, 0, 0, 52.0, 54.0, 50.0, 10.0, 10.0, 10.0, 1.08, 0.88, 0.92, 0, "domestic_list_a"),
            ("t_hasan_ban", "Tanzid Hasan", "BAN", "top_order_batter", "left", "NONE", 650, 0, 82.0, 28.0, 0, 0, 60.0, 56.0, 54.0, 10.0, 10.0, 10.0, 1.04, 0.95, 0.94, 0, "tier_1_ipl"),
            ("n_shanto_ban", "Najmul Hossain Shanto", "BAN", "top_order_batter", "left", "OB", 1420, 80, 68.0, 35.0, 5.2, 35.0, 56.0, 62.0, 58.0, 15.0, 20.0, 15.0, 1.12, 0.92, 0.98, 0, "tier_1_ipl"),
            ("t_hridoy_ban", "Towhid Hridoy", "BAN", "middle_order_batter", "right", "NONE", 780, 0, 88.0, 38.0, 0, 0, 54.0, 64.0, 65.0, 10.0, 10.0, 10.0, 1.14, 0.95, 0.95, 0, "tier_1_ipl"),
            ("m_rahim_ban", "Mushfiqur Rahim", "BAN", "middle_order_batter", "right", "NONE", 2600, 0, 72.0, 38.5, 0, 0, 50.0, 65.0, 68.0, 10.0, 10.0, 10.0, 1.16, 0.94, 0.98, 0, "tier_1_ipl"),
            ("l_das_ban", "Litton Das", "BAN", "wicket_keeper", "right", "NONE", 1850, 0, 126.0, 36.0, 0, 0, 62.0, 60.0, 62.0, 10.0, 10.0, 10.0, 1.08, 0.96, 0.94, 0, "tier_1_ipl"),
            ("m_hasan_miraz_ban", "Mehidy Hasan Miraz", "BAN", "all_rounder", "right", "OB", 1120, 1850, 78.0, 24.5, 3.45, 32.0, 48.0, 56.0, 58.0, 45.0, 68.0, 55.0, 1.18, 0.92, 0.96, 0, "tier_1_ipl"),
            ("n_hasan_ban", "Nayeem Hasan", "BAN", "all_rounder", "right", "OB", 340, 890, 65.0, 18.0, 3.65, 34.0, 35.0, 45.0, 48.0, 40.0, 65.0, 48.0, 1.14, 0.88, 0.90, 0, "domestic_list_a"),
            ("t_ahmed_ban", "Taskin Ahmed", "BAN", "frontline_pacer", "right", "RF", 90, 1420, 85.0, 11.0, 4.20, 22.0, 15.0, 15.0, 20.0, 70.0, 58.0, 66.0, 0.98, 1.04, 0.92, 0, "tier_1_ipl"),
            ("t_islam_ban", "Taijul Islam", "BAN", "frontline_spinner", "left", "SLA", 110, 2150, 62.0, 12.0, 2.95, 28.0, 15.0, 15.0, 15.0, 42.0, 78.0, 52.0, 1.22, 0.85, 0.96, 0, "tier_2_bbl_psl"),
            ("h_mahmud_ban", "Hasan Mahmud", "BAN", "frontline_pacer", "right", "RF", 50, 980, 80.0, 8.5, 3.90, 21.0, 15.0, 15.0, 20.0, 68.0, 56.0, 65.0, 0.98, 1.05, 0.90, 0, "tier_2_bbl_psl"),
            # Bangladesh Reserves / Concussion Subs
            ("z_hasan_ban", "Zakir Hasan", "BAN", "top_order_batter", "left", "NONE", 620, 0, 52.0, 31.0, 0, 0, 50.0, 52.0, 48.0, 10.0, 10.0, 10.0, 1.06, 0.86, 0.88, 0, "domestic_list_a"),
            ("m_hasan_joy_ban", "Mahmudul Hasan Joy", "BAN", "top_order_batter", "right", "OB", 580, 20, 45.0, 28.5, 6.0, 0, 48.0, 50.0, 48.0, 10.0, 15.0, 10.0, 1.04, 0.88, 0.86, 0, "domestic_list_a"),
            ("k_ahmed_ban", "Khaled Ahmed", "BAN", "frontline_pacer", "right", "RF", 30, 680, 70.0, 6.0, 4.10, 25.0, 15.0, 15.0, 15.0, 65.0, 52.0, 60.0, 0.96, 1.02, 0.85, 0, "domestic_list_a"),
            ("t_sakib_ban", "Tanzim Hasan Sakib", "BAN", "frontline_pacer", "right", "RF", 40, 540, 85.0, 10.0, 4.30, 22.0, 15.0, 15.0, 15.0, 66.0, 54.0, 64.0, 0.96, 1.02, 0.88, 0, "tier_2_bbl_psl")
        ]
        await conn.executemany("""
            INSERT INTO players (
                id, name, team_id, role, batting_hand, bowling_style,
                intl_balls_faced, intl_balls_bowled, t20_batting_sr, t20_batting_avg,
                t20_economy, t20_bowling_sr, powerplay_batting_rating, middle_batting_rating,
                death_batting_rating, powerplay_bowling_rating, middle_bowling_rating, death_bowling_rating,
                foreign_condition_ratio_subcontinent, foreign_condition_ratio_sena,
                recent_selection_weight, is_debutant, league_experience_tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, players)

        # 4. Player Cricsheet Aliases
        aliases = [
            ("vkohli_ind", "V Kohli", "IND"),
            ("rsharma_ind", "RG Sharma", "IND"),
            ("sgill_ind", "Shubman Gill", "IND"),
            ("siyer_ind", "SS Iyer", "IND"),
            ("skyadav_ind", "SA Yadav", "IND"),
            ("jbumrah_ind", "JJ Bumrah", "IND"),
            ("p_cummins_aus", "PJ Cummins", "AUS"),
            ("j_buttler_eng", "JC Buttler", "ENG"),
            ("r_khan_afg", "Rashid Khan Arman", "AFG"),
            ("s_raza_zim", "Sikandar Raza Butt", "ZIM"),
            ("w_hasaranga_sl", "PWH de Silva", "SL"),
            ("t_islam_ban", "Taijul Islam", "BAN"),
            ("k_hodge_wi", "Kavem Hodge", "WI"),
            ("s_masood_pak", "Shan Masood", "PAK"),
            ("b_azam_pak", "Babar Azam", "PAK"),
            ("s_afridi_pak", "Shaheen Shah Afridi", "PAK"),
            ("n_shah_pak", "Naseem Shah", "PAK"),
            ("m_rizwan_pak", "Mohammad Rizwan", "PAK"),
            ("d_desilva_sl", "Dhananjaya de Silva", "SL"),
            ("k_mendis_sl", "BKG Mendis", "SL"),
            ("p_jayasuriya_sl", "Prabath Jayasuriya", "SL"),
            ("k_mendis_kam_sl", "Kamindu Mendis", "SL")
        ]
        await conn.executemany("""
            INSERT INTO player_aliases (cricsheet_id, alias, country)
            VALUES (?, ?, ?)
        """, aliases)

        # 5. REAL UPCOMING & LIVE INTERNATIONAL FIXTURES
        matches = [
            # 0. BAN vs WI - 2nd Test (COMPLETED)
            (
                "m_ban_wi_2026_live", "West Indies Tour of Bangladesh 2026 - 2nd Test", "TEST",
                "BAN", "WI", "sylhet", "2026-11-05T03:30:00Z", 0,
                "COMPLETED", "WI", "bat",
                4, 193, 6, 54, 3, 193,
                None, None, None,
                0.0, 100.0, 52.0, 53.5, 51.5,
                215, 195, 258,
                0.0, 0.0, "West Indies reached target of 193 to win 2nd Test by 4 wickets",
                0.0, "West Indies won by 4 wickets (Target 193 reached) • 2nd Test Completed"
            ),
            # 1. LIVE MATCH: AFG vs ZIM - 1st Test (LIVE IN-PLAY RIGHT NOW: Day 1 Session 3 • 276/4 in 76.3 ov • NOT STUMPED)
            (
                "m_zim_afg_2026_test_1", "Afghanistan vs Zimbabwe in UAE, 2026 - 1st Test", "TEST",
                "AFG", "ZIM", "sharjah", "2026-11-05T06:00:00Z", 0,
                "LIVE_BALL_BY_BALL", "ZIM", "bat",
                1, 276, 4, 76, 3, None,
                "b_bennett_zim", "a_naqvi_zim", "h_shahidi_afg",
                23.6, 70.9, 48.0, 50.0, 50.0,
                382, 345, 425,
                0.0, 2.5, "Brendan Taylor brilliant 131 sets formidable platform; Brian Bennett 57* anchoring Session 3",
                0.6, "Day 1: 3rd Session (Live In-Play) • Zimbabwe 276/4 (76.3 ov) - Brian Bennett 57* (105b), Antum Naqvi 2* (6b)"
            ),
            # 2. UPCOMING MATCH: AFG vs ZIM - 2nd Test (November 12-16, 2026 at Sharjah)
            (
                "m_zim_afg_2026_test_2", "Afghanistan vs Zimbabwe in UAE, 2026 - 2nd Test", "TEST",
                "AFG", "ZIM", "sharjah", "2026-11-12T06:00:00Z", 0,
                "UPCOMING_NO_SQUAD", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                52.0, 48.0, 52.0, 52.0, 52.0,
                365, 330, 410,
                0.0, 0.0, "Sharjah spin deterioration gives Rashid Khan & Qais Ahmad lethal day 4/5 turn",
                0.0, "2nd Test Scheduled at Sharjah • Official 15-Man Tour Test Squads In Effect"
            ),
            # 3. NZ vs IND - 2nd ODI (Saturday, Nov 7, 2026 - 15:00 NZDT / 02:00 UTC)
            (
                "m_nz_ind_2026_odi_2", "India Tour of New Zealand 2026 - 2nd ODI", "ODI",
                "NZ", "IND", "wellington", "2026-11-07T02:00:00Z", 1,
                "LIVE_BALL_BY_BALL", "NZ", "bat",
                2, 113, 7, 26, 0, 135,
                "klrahul_ind", "uncapped_auqib_nabi", "j_duffy_nz",
                28.5, 71.5, 45.0, 40.8, 45.7,
                134, 120, 150,
                -4.2, 4.9, "Jacob Duffy & Kyle Jamieson dismantle top-order before rain interruption",
                16.8, "Rain Stoppage: India 113/7 (26.0 ov) - Ahead by +4 runs on DLS Par (109)"
            ),
            # 4. NZ vs IND - 3rd ODI (LIVE: Rain Interruption - Covers On)
            (
                "m_nz_ind_2026_odi_3", "India Tour of New Zealand 2026 - 3rd ODI", "ODI",
                "NZ", "IND", "seddon_park", "2026-11-10T02:00:00Z", 1,
                "LIVE_BALL_BY_BALL", "IND", "bowl",
                2, 142, 3, 24, 0, 286,
                "klrahul_ind", "rpant_ind", "t_boult_nz",
                38.5, 61.5, 46.8, 43.8, 46.8,
                285, 268, 320,
                -3.0, 0.0, "Seddon Park true bounce benefits KL Rahul & Rishabh Pant anchor chase; India +4 on DLS Par",
                0.0, "Rain Interruption (Covers On): India 142/3 (24.0 ov) - Ahead by +4 runs on DLS Par (138)"
            ),
            # 5. NZ vs IND - 4th ODI (Friday, Nov 13, 2026 - 15:00 NZDT / 02:00 UTC)
            (
                "m_nz_ind_2026_odi_4", "India Tour of New Zealand 2026 - 4th ODI", "ODI",
                "NZ", "IND", "bay_oval", "2026-11-13T02:00:00Z", 1,
                "UPCOMING_NO_SQUAD", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                47.5, 52.5, 47.5, 47.5, 47.5,
                280, 250, 310,
                0.0, 0.0, "Bay Oval true seam balance aids pacers in opening 10 overs",
                0.0, "Scheduled Fixture"
            ),
            # 6. NZ vs IND - 5th ODI (Sunday, Nov 15, 2026 - 15:00 NZDT / 02:00 UTC)
            (
                "m_nz_ind_2026_odi_5", "India Tour of New Zealand 2026 - 5th ODI", "ODI",
                "NZ", "IND", "bay_oval", "2026-11-15T02:00:00Z", 1,
                "UPCOMING_NO_SQUAD", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                48.0, 52.0, 48.0, 48.0, 48.0,
                280, 250, 310,
                0.0, 0.0, "Series finale at Bay Oval with D/N dew conditions",
                0.0, "Scheduled Fixture"
            ),
            # 7. AUS vs ENG - 1st ODI (Friday, Nov 13, 2026 - 11:30 Local Perth / 03:30 UTC at Perth Stadium)
            (
                "m_aus_eng_2026_odi_1", "England Tour of Australia 2026 - 1st ODI", "ODI",
                "AUS", "ENG", "perth", "2026-11-13T03:30:00Z", 1,
                "SQUAD_ANNOUNCED", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                55.3, 44.7, 55.3, 55.3, 55.3,
                268, 240, 295,
                0.0, 0.0, "Optus Stadium hard drop-in deck creates steep vertical carry; pace attack affinity +12.4%",
                0.0, "15-Player Squads Declared • Official Playing XI announced at Toss"
            ),
            # 8. AUS vs ENG - 2nd ODI (Sunday, Nov 15, 2026 at Adelaide Oval)
            (
                "m_aus_eng_2026_odi_2", "England Tour of Australia 2026 - 2nd ODI", "ODI",
                "AUS", "ENG", "adelaide", "2026-11-15T03:30:00Z", 1,
                "SQUAD_ANNOUNCED", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                54.0, 46.0, 54.0, 54.0, 54.0,
                288, 255, 315,
                0.0, 0.0, "Adelaide Oval true batting surface with short square boundaries",
                0.0, "15-Player Squads Declared • Stage 2 Squad Fit Calibrated"
            ),
            # 8B. AUS vs ENG - 3rd ODI (Wednesday, Nov 18, 2026 at MCG)
            (
                "m_aus_eng_2026_odi_3", "England Tour of Australia 2026 - 3rd ODI", "ODI",
                "AUS", "ENG", "mcg", "2026-11-18T03:30:00Z", 1,
                "SQUAD_ANNOUNCED", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                56.2, 43.8, 56.2, 56.2, 56.2,
                285, 250, 310,
                0.0, 0.0, "MCG vast boundary dimensions favor athletic outfield running and tall seamers",
                0.0, "15-Player Squads Declared • Stage 2 Squad Fit Calibrated"
            ),
            # 9. IND vs SA - 1st T20I (Friday, Nov 27, 2026)
            (
                "m_ind_sa_2026_t20_1", "South Africa Tour of India 2026 - 1st T20I", "T20I",
                "IND", "SA", "wankhede", "2026-11-27T13:30:00Z", 1,
                "SQUAD_ANNOUNCED", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                59.4, 40.6, 59.4, 59.4, 59.4,
                192, 175, 218,
                0.0, 0.0, "Wankhede red soil pitch & evening dew boost chasing team (+14.5%)",
                0.0, "15-Man T20 Squad Announced • Stage 1 Projections Ready"
            ),
            # 10. IND vs SA - 2nd T20I (Sunday, Nov 29, 2026)
            (
                "m_ind_sa_2026_t20_2", "South Africa Tour of India 2026 - 2nd T20I", "T20I",
                "IND", "SA", "eden_gardens", "2026-11-29T13:30:00Z", 1,
                "UPCOMING_NO_SQUAD", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                58.7, 41.3, 58.7, 58.7, 58.7,
                178, 160, 200,
                0.0, 0.0, "Eden Gardens turning track enhances Indian spin trio (+8.2% economy)",
                0.0, "Macro Condition-Adjusted Prediction • Squads TBA"
            ),
            # 10A. AUS vs ENG - 1st T20I (MCG) - Official Senior Men's International (AUS 191/7 in 19.5 ov def ENG 187/9 in 20 ov)
            (
                "m_aus_eng_2026_t20_1", "England Tour of Australia 2026 - 1st T20I", "T20I",
                "AUS", "ENG", "mcg", "2026-11-20T08:00:00Z", 1,
                "COMPLETED", "AUS", "field",
                2, 191, 7, 19, 5, 188,
                None, None, None,
                100.0, 0.0, 58.0, 58.0, 58.0,
                188, 175, 195,
                0.0, 0.0, "MCG true bounce track - Australia chase 188 with 3 balls to spare",
                0.0, "Australia won by 3 wkts"
            ),
            # 10B. AUS vs ENG - 2nd T20I (Sunday, Nov 15, 2026 at SCG) - PURE STAGE 1 MACRO PRIOR (AUS 57.1% vs ENG 42.9%)
            (
                "m_aus_eng_2026_t20_2", "England Tour of Australia 2026 - 2nd T20I", "T20I",
                "AUS", "ENG", "scg", "2026-11-15T08:00:00Z", 1,
                "UPCOMING_NO_SQUAD", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                57.1, 42.9, 57.1, 57.1, 57.1,
                166, 148, 184,
                0.0, 0.0, "SCG pitch conditions with spin ratio 0.48 and home advantage give Australia 57.1% Macro Elo edge",
                0.0, "Stage 1 Macro Prediction (AUS 57.1% vs ENG 42.9%) • Sydney Cricket Ground • Squads TBA"
            ),
            # 11. NZ vs IND - 1st ODI (Wednesday, Nov 4, 2026 - COMPLETED)
            (
                "m_nz_ind_2026_odi_1", "India Tour of New Zealand 2026 - 1st ODI", "ODI",
                "NZ", "IND", "eden_park", "2026-11-04T02:00:00Z", 1,
                "COMPLETED", "NZ", "bat",
                2, 276, 6, 48, 2, 276,
                None, None, None,
                0.0, 100.0, 48.0, 52.0, 48.0,
                275, 250, 310,
                0.0, 0.0, "India chased 276 comfortably with KL Rahul 88* and Shubman Gill 74",
                0.0, "India won by 4 wickets (Target 276 reached in 48.2 ov) - 1st ODI Completed"
            ),
            # 14. PAK vs SL - 1st Test (Mon, Nov 09 - Fri, Nov 13, 2026 at Rawalpindi)
            (
                "m_pak_sl_2026_test_1", "Sri Lanka Tour of Pakistan 2026-27 - 1st Test", "TEST",
                "PAK", "SL", "rawalpindi", "2026-11-09T04:30:00Z", 0,
                "SQUAD_ANNOUNCED", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                54.8, 45.2, 54.8, 54.8, 54.8,
                365, 320, 420,
                0.0, 0.0, "Rawalpindi flat track offers run scoring early; reverse swing and spinner wear decisive from Day 4",
                0.0, "Official 1st Test at Rawalpindi • Declared 15-Member Squads"
            ),
            # 14B. PAK vs SL - 2nd Test (Tue, Nov 17 - Sat, Nov 21, 2026 at Gaddafi Stadium, Lahore)
            (
                "m_pak_sl_2026_test_2", "Sri Lanka Tour of Pakistan 2026-27 - 2nd Test", "TEST",
                "PAK", "SL", "lahore", "2026-11-17T04:30:00Z", 0,
                "UPCOMING_NO_SQUAD", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                55.2, 44.8, 55.2, 55.2, 55.2,
                380, 330, 440,
                0.0, 0.0, "Lahore classic batting paradise Days 1-3, abrasive soil turns sharply Days 4-5",
                0.0, "Official 2nd Test at Gaddafi Stadium, Lahore • Tour Squad Propagated"
            ),
            # 15. NZ vs IND - 1st Test (Nov 19-23, 2026 at Basin Reserve, Wellington)
            (
                "m_nz_ind_2026_test_1", "India Tour of New Zealand 2026 - 1st Test", "TEST",
                "NZ", "IND", "wellington", "2026-11-19T22:00:00Z", 0,
                "UPCOMING_NO_SQUAD", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                51.0, 49.0, 51.0, 51.0, 51.0,
                330, 290, 380,
                0.0, 0.0, "Basin Reserve green seamer heavily tests top-order technique against Southee and Boult",
                0.0, "Scheduled 1st Test at Basin Reserve"
            ),
            # 16. NZ vs IND - 2nd Test (Nov 27-Dec 01, 2026 at Hagley Oval, Christchurch)
            (
                "m_nz_ind_2026_test_2", "India Tour of New Zealand 2026 - 2nd Test", "TEST",
                "NZ", "IND", "eden_park", "2026-11-27T22:00:00Z", 0,
                "UPCOMING_NO_SQUAD", None, None,
                1, 0, 0, 0, 0, None,
                None, None, None,
                50.5, 49.5, 50.5, 50.5, 50.5,
                340, 300, 390,
                0.0, 0.0, "Pace and carry at Hagley Oval favors fast bowling quartets",
                0.0, "Scheduled 2nd Test at Hagley Oval"
            )
        ]

        await conn.executemany("""
            INSERT INTO matches (
                id, series_name, format, team_a_id, team_b_id, venue_id, scheduled_start, is_day_night,
                state, toss_winner_id, toss_decision, current_innings, current_score, current_wickets,
                current_overs, current_balls, target_runs, current_striker_id, current_non_striker_id,
                current_bowler_id, p_team_a, p_team_b, p1_macro, p2_squad, p0_pre_match,
                par_score_projected, par_score_min, par_score_max, delta_squad_shift, delta_toss_shift,
                biggest_matchup_adv, last_wpa_swing, last_event_description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, matches)

        # Closed BAN vs WI 2nd Test
        await conn.execute("""
            UPDATE matches SET 
                state = 'COMPLETED',
                is_stumps = 0, 
                stumps_day = 5, 
                stumps_label = NULL,
                current_innings = 4,
                current_score = 193,
                current_wickets = 6,
                current_overs = 54,
                current_balls = 3,
                target_runs = 193,
                innings_1_team_id = 'BAN',
                innings_1_score = 248,
                innings_1_wickets = 10,
                innings_1_overs = 82.3,
                innings_2_team_id = 'WI',
                innings_2_score = 275,
                innings_2_wickets = 10,
                innings_2_overs = 89.1,
                innings_3_team_id = 'BAN',
                innings_3_score = 219,
                innings_3_wickets = 10,
                innings_3_overs = 78.4,
                innings_4_team_id = 'WI',
                innings_4_score = 193,
                innings_4_wickets = 6,
                innings_4_overs = 54.3,
                p_team_a = 0.0,
                p_team_b = 100.0,
                p_draw = 0.0,
                match_status_text = 'West Indies won by 4 wickets (Target 193 reached) • 2nd Test Completed',
                last_event_description = 'West Indies won by 4 wickets • 2nd Test Completed'
            WHERE id = 'm_ban_wi_2026_live'
        """)

        # India vs New Zealand 1st ODI - Match Abandoned without a ball bowled (No Result - Rain)
        await conn.execute("""
            UPDATE matches SET
                state = 'COMPLETED',
                current_score = 0,
                current_wickets = 0,
                current_overs = 0,
                current_balls = 0,
                target_runs = NULL,
                dls_target = NULL,
                dls_par_score = NULL,
                overs_lost = 50.0,
                weather_condition = 'TORRENTIAL_RAIN_WASHED_OUT',
                rain_probability_pct = 100.0,
                innings_1_team_id = 'NZ',
                innings_1_score = NULL,
                innings_1_wickets = NULL,
                innings_1_overs = NULL,
                innings_2_team_id = 'IND',
                innings_2_score = NULL,
                innings_2_wickets = NULL,
                innings_2_overs = NULL,
                p_team_a = 50.0,
                p_team_b = 50.0,
                p_draw = 0.0,
                match_status_text = 'Match abandoned without a ball bowled (No Result - Rain)',
                last_event_description = 'Match abandoned without a ball bowled due to persistent rain and wet outfield at Seddon Park, Hamilton'
            WHERE id = 'm_nz_ind_2026_odi_1'
        """)

        # India vs New Zealand 3rd ODI - Stage 3 Confirmed Playing 11 & Toss Locked
        await conn.execute("""
            UPDATE matches SET
                state = 'XI_AND_TOSS_LOCKED',
                toss_winner_id = 'IND',
                toss_decision = 'bowl',
                pitch_report = 'Pitch is dry with expanding cracks. Spinners will thrive.',
                current_innings = 1,
                current_score = 0,
                current_wickets = 0,
                current_overs = 0,
                current_balls = 0,
                target_runs = NULL,
                dls_target = NULL,
                dls_par_score = NULL,
                p_team_a = 48.0,
                p_team_b = 52.0,
                par_score_projected = 275,
                match_status_text = 'Playing XI & Toss Locked • Stage 3 Pre-Match Intelligence Active',
                last_event_description = 'Toss: India won the toss and elected to bowl. Pitch is dry with expanding cracks.'
            WHERE id = 'm_nz_ind_2026_odi_3'
        """)

        # AUS vs ENG 1st T20I (MCG)
        await conn.execute("""
            UPDATE matches SET
                state = 'COMPLETED',
                format = 'T20I',
                current_innings = 2,
                current_score = 191,
                current_wickets = 7,
                current_overs = 19,
                current_balls = 5,
                target_runs = 188,
                innings_1_team_id = 'ENG',
                innings_1_score = 187,
                innings_1_wickets = 9,
                innings_1_overs = 20.0,
                innings_2_team_id = 'AUS',
                innings_2_score = 191,
                innings_2_wickets = 7,
                innings_2_overs = 19.5,
                p_team_a = 100.0,
                p_team_b = 0.0,
                p_draw = 0.0,
                match_status_text = 'Australia won by 3 wkts',
                last_event_description = 'Australia won by 3 wkts (Target 188 reached in 19.5 ov)'
            WHERE id = 'm_aus_eng_2026_t20_1'
        """)

        # NZ vs IND 1st Test (Basin Reserve) - Day 4 Stumps
        await conn.execute("""
            UPDATE matches SET
                state = 'LIVE_BALL_BY_BALL',
                toss_winner_id = 'IND',
                toss_decision = 'bowl',
                current_innings = 3,
                current_score = 119,
                current_wickets = 2,
                current_overs = 45,
                current_balls = 0,
                is_stumps = 1,
                stumps_day = 4,
                stumps_label = '🌙 STUMPS • DAY 4 CLOSE OF PLAY',
                innings_1_team_id = 'NZ',
                innings_1_score = 368,
                innings_1_wickets = 10,
                innings_1_overs = 121.0,
                innings_2_team_id = 'IND',
                innings_2_score = 720,
                innings_2_wickets = 9,
                innings_2_overs = 166.0,
                innings_2_declared = 1,
                innings_3_team_id = 'NZ',
                innings_3_score = 119,
                innings_3_wickets = 2,
                innings_3_overs = 45.0,
                innings_4_team_id = 'IND',
                innings_4_score = NULL,
                batting_tank_pct = 76.0,
                bowling_tank_pct = 72.0,
                p_team_a = 5.0,
                p_team_b = 83.0,
                p_draw = 12.0,
                match_status_text = 'Day 4: Stumps - New Zealand 119/2 (45.0 ov) • Trail by 233 runs',
                last_event_description = 'Stumps Day 4: New Zealand 119/2 (Conway 55*, Nicholls 48*) • Trail India (720/9d) by 233 runs'
            WHERE id = 'm_nz_ind_2026_test_1'
        """)

        # 6. Seed Official 15-Man Match Squads for all fixtures
        async with conn.execute("SELECT id, team_id, role FROM players ORDER BY id") as p_cur:
            all_players = [dict(r) for r in await p_cur.fetchall()]

        players_by_team = {}
        for p in all_players:
            players_by_team.setdefault(p["team_id"], []).append(p)

        async with conn.execute("SELECT id, team_a_id, team_b_id, state FROM matches") as m_cur:
            all_matches = [dict(r) for r in await m_cur.fetchall()]

        squad_inserts = []
        for m in all_matches:
            mid = m["id"]
            m_state = m["state"]
            for tid in [m["team_a_id"], m["team_b_id"]]:
                t_roster = players_by_team.get(tid, [])
                for idx, p in enumerate(t_roster[:15]):
                    pid = p["id"]
                    is_xi = 1 if (m_state in ("XI_AND_TOSS_LOCKED", "LIVE_BALL_BY_BALL", "COMPLETED") and idx < 11) else 0
                    is_prob = 1 if idx < 11 else 0
                    sub_st = "CONFIRMED_XI" if is_xi else ("PROBABLE_XI" if is_prob else "BENCH_RESERVE")
                    b_order = (idx + 1) if idx < 11 else 99
                    squad_inserts.append((mid, tid, pid, 1, is_prob, is_xi, b_order, sub_st))

        await conn.executemany("""
            INSERT INTO match_squads (match_id, team_id, player_id, is_in_squad, is_probable_xi, is_confirmed_xi, batting_order, sub_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, squad_inserts)

        # 7. Propagate Bilateral Series Squads
        from backend.engine.series_squad_propagator import propagate_series_squads_async
        await propagate_series_squads_async(conn)

        await conn.commit()
    finally:
        await conn.close()
