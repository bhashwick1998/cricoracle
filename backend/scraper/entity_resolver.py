"""
Entity Resolution & Debutant Auto-Detection Engine
Resolves external player names (ESPNcricinfo, Cricbuzz, abbreviations) to Cricsheet IDs
using exact alias lookups, RapidFuzz token_set_ratio with country filtering, and automatic debutant creation.
"""

from typing import Optional, Dict, Any, List
import rapidfuzz
from rapidfuzz import fuzz
from ..engine.debut_priors import compute_debutant_prior

# Predefined franchise database for players without extensive international caps
FRANCHISE_DATABASE = {
    "angkrish_raghuvanshi": {"name": "Angkrish Raghuvanshi", "team": "IND", "role": "top_order_batter", "league_tier": "tier_1_ipl", "t20_batting_sr": 155.0, "t20_batting_avg": 32.0, "bowling_style": "NONE"},
    "mayank_yadav": {"name": "Mayank Yadav", "team": "IND", "role": "frontline_pacer", "league_tier": "tier_1_ipl", "t20_economy": 7.4, "t20_bowling_sr": 13.5, "bowling_style": "RF"},
    "harshit_rana": {"name": "Harshit Rana", "team": "IND", "role": "frontline_pacer", "league_tier": "tier_1_ipl", "t20_economy": 8.1, "t20_bowling_sr": 15.0, "bowling_style": "RF"},
    "cooper_connolly": {"name": "Cooper Connolly", "team": "AUS", "role": "all_rounder", "league_tier": "tier_2_bbl_psl", "t20_batting_sr": 140.0, "t20_batting_avg": 27.0, "bowling_style": "SLA"},
    "jake_fraser_mcgurk": {"name": "Jake Fraser-McGurk", "team": "AUS", "role": "top_order_batter", "league_tier": "tier_1_ipl", "t20_batting_sr": 185.0, "t20_batting_avg": 34.0, "bowling_style": "NONE"},
    "jacob_bethell": {"name": "Jacob Bethell", "team": "ENG", "role": "all_rounder", "league_tier": "tier_2_bbl_psl", "t20_batting_sr": 142.0, "t20_batting_avg": 26.0, "bowling_style": "SLA"},
    "john_turner": {"name": "John Turner", "team": "ENG", "role": "frontline_pacer", "league_tier": "tier_2_bbl_psl", "t20_economy": 7.8, "t20_bowling_sr": 14.8, "bowling_style": "RF"},
    "kwena_maphaka": {"name": "Kwena Maphaka", "team": "SA", "role": "frontline_pacer", "league_tier": "tier_1_ipl", "t20_economy": 8.8, "t20_bowling_sr": 16.0, "bowling_style": "LAWS"},
}

GLOBAL_CRICBUZZ_CRICSHEET_ALIASES: Dict[str, str] = {
    # India
    "v kohli": "vkohli_ind", "virat kohli": "vkohli_ind",
    "rg sharma": "rsharma_ind", "rohit sharma": "rsharma_ind", "r sharma": "rsharma_ind",
    "shubman gill": "sgill_ind", "s gill": "sgill_ind",
    "ss iyer": "siyer_ind", "shreyas iyer": "siyer_ind", "s iyer": "siyer_ind",
    "sa yadav": "skyadav_ind", "suryakumar yadav": "skyadav_ind",
    "jj bumrah": "jbumrah_ind", "jasprit bumrah": "jbumrah_ind", "j bumrah": "jbumrah_ind",
    "kl rahul": "klrahul_ind", "k l rahul": "klrahul_ind",
    "hardik pandya": "hardik_ind", "hh pandya": "hardik_ind", "h pandya": "hardik_ind",
    "ravindra jadeja": "r_jadeja_ind", "ra jadeja": "r_jadeja_ind", "r jadeja": "r_jadeja_ind",
    "axar patel": "a_patel_ind", "ar patel": "a_patel_ind", "a patel": "a_patel_ind",
    "kuldeep yadav": "k_yadav_ind", "k yadav": "k_yadav_ind",
    "mohammed siraj": "m_siraj_ind", "m siraj": "m_siraj_ind", "md siraj": "m_siraj_ind",
    "washington sundar": "w_sundar_ind", "w sundar": "w_sundar_ind",
    "rishabh pant": "r_pant_ind", "rp pant": "r_pant_ind", "r pant": "r_pant_ind",
    # New Zealand
    "jacob duffy": "j_duffy_nz", "ja duffy": "j_duffy_nz", "j duffy": "j_duffy_nz",
    "devon conway": "d_conway_nz", "dp conway": "d_conway_nz", "d conway": "d_conway_nz",
    "daryl mitchell": "d_mitchell_nz", "dj mitchell": "d_mitchell_nz", "d mitchell": "d_mitchell_nz",
    "glenn phillips": "g_phillips_nz", "gd phillips": "g_phillips_nz", "g phillips": "g_phillips_nz",
    "michael bracewell": "m_bracewell_nz", "mg bracewell": "m_bracewell_nz", "m bracewell": "m_bracewell_nz",
    "mitchell santner": "m_santner_nz", "mj santner": "m_santner_nz", "m santner": "m_santner_nz",
    "matt henry": "m_henry_nz", "mj henry": "m_henry_nz", "m henry": "m_henry_nz",
    "zakary foulkes": "z_foulkes_nz", "z foulkes": "z_foulkes_nz",
    "tim robinson": "t_robinson_nz", "ta robinson": "t_robinson_nz", "t robinson": "t_robinson_nz",
    "tom latham": "t_latham_nz", "twm latham": "t_latham_nz", "t latham": "t_latham_nz",
    "will young": "w_young_nz", "wa young": "w_young_nz", "w young": "w_young_nz",
    "rachin ravindra": "r_ravindra_nz", "r ravindra": "r_ravindra_nz",
    "kane williamson": "k_williamson_nz", "ks williamson": "k_williamson_nz", "k williamson": "k_williamson_nz",
    "lockie ferguson": "l_ferguson_nz", "lh ferguson": "l_ferguson_nz", "l ferguson": "l_ferguson_nz",
    # Bangladesh
    "shadman islam": "s_islam_ban", "mehidy hasan miraz": "m_hasan_miraz_ban", "m hasan miraz": "m_hasan_miraz_ban", "mehedi hasan miraz": "m_hasan_miraz_ban",
    "taijul islam": "t_islam_ban", "t islam": "t_islam_ban", "litton das": "l_das_ban", "l das": "l_das_ban", "liton das": "l_das_ban",
    "mominul haque": "m_haque_ban", "m haque": "m_haque_ban", "najmul hossain shanto": "n_shanto_ban", "nazmul hossain shanto": "n_shanto_ban", "n shanto": "n_shanto_ban",
    "hasan mahmud": "h_mahmud_ban", "taskin ahmed": "t_ahmed_ban", "nahid rana": "n_rana_ban", "towhid hridoy": "t_hridoy_ban", "t hridoy": "t_hridoy_ban",
    "tanzid hasan": "t_hasan_ban", "t hasan": "t_hasan_ban", "jaker ali": "j_ali_ban", "j ali": "j_ali_ban", "ebadot hossain": "e_hossain_ban", "ebadat hossain": "e_hossain_ban",
    "mahfijul islam": "m_islam_ban", "khaled ahmed": "k_ahmed_ban", "shoriful islam": "shoriful_ban", "mahmudul hasan joy": "m_joy_ban",
    # South Africa
    "aiden markram": "a_markram_sa", "ak markram": "a_markram_sa", "a markram": "a_markram_sa",
    "kagiso rabada": "k_rabada_sa", "k rabada": "k_rabada_sa",
    "anrich nortje": "a_nortje_sa", "a nortje": "a_nortje_sa",
    "marco jansen": "m_jansen_sa", "m jansen": "m_jansen_sa",
    "keshav maharaj": "k_maharaj_sa", "ka maharaj": "k_maharaj_sa", "k maharaj": "k_maharaj_sa",
    "wiaan mulder": "w_mulder_sa", "pwa mulder": "w_mulder_sa", "w mulder": "w_mulder_sa",
    "tony de zorzi": "t_de_zorzi_sa", "t de zorzi": "t_de_zorzi_sa",
    "tristan stubbs": "t_stubbs_sa", "t stubbs": "t_stubbs_sa",
    "david bedingham": "d_bedingham_sa", "dg bedingham": "d_bedingham_sa", "d bedingham": "d_bedingham_sa",
    "ryan rickelton": "r_rickelton_sa", "rd rickelton": "r_rickelton_sa", "r rickelton": "r_rickelton_sa",
    "kyle verreynne": "k_verreynne_sa", "k verreynne": "k_verreynne_sa",
    "simon harmer": "s_harmer_sa", "sr harmer": "s_harmer_sa", "s harmer": "s_harmer_sa",
    "matthew breetzke": "m_breetzke_sa", "mp breetzke": "m_breetzke_sa", "m breetzke": "m_breetzke_sa",
    "marques ackerman": "m_ackerman_sa", "mj ackerman": "m_ackerman_sa",
    "dane paterson": "d_paterson_sa", "d paterson": "d_paterson_sa",
    "senuran muthusamy": "s_muthusamy_sa", "s muthusamy": "s_muthusamy_sa",
    "corbin bosch": "c_bosch_sa", "c bosch": "c_bosch_sa",
    "zubayr hamza": "z_hamza_sa", "z hamza": "z_hamza_sa",
    "gerald coetzee": "g_coetzee_sa", "g coetzee": "g_coetzee_sa",
    "lungi ngidi": "l_ngidi_sa", "l ngidi": "l_ngidi_sa",
    "quinton de kock": "q_dekock_sa", "q de kock": "q_dekock_sa",
    "heinrich klaasen": "h_klaasen_sa", "h klaasen": "h_klaasen_sa",
    "david miller": "d_miller_sa", "da miller": "d_miller_sa", "d miller": "d_miller_sa",
    # West Indies
    "kavem hodge": "k_hodge_wi", "k hodge": "k_hodge_wi", "amir jangoo": "a_jangoo_wi", "a jangoo": "a_jangoo_wi",
    "justin greaves": "j_greaves_wi", "j greaves": "j_greaves_wi", "roston chase": "r_chase_wi", "r chase": "r_chase_wi",
    "shamar joseph": "s_joseph_wi", "s joseph": "s_joseph_wi", "jayden seales": "j_seales_wi", "j seales": "j_seales_wi",
    "jomel warrican": "j_warrican_wi", "j warrican": "j_warrican_wi", "kraigg brathwaite": "k_brathwaite_wi",
    "kemar roach": "k_roach_wi", "alzarri joseph": "a_joseph_wi", "brandon king": "b_king_wi",
    "joshua da silva": "j_dasilva_wi", "kevin wickham": "k_wickham_wi",
    # Afghanistan & Zimbabwe
    "rashid khan": "r_khan_afg", "rashid khan arman": "r_khan_afg", "r khan": "r_khan_afg",
    "rahmat shah": "r_shah_afg", "rahmat shah zurmatai": "r_shah_afg", "r shah": "r_shah_afg",
    "hashmatullah shahidi": "h_shahidi_afg", "h shahidi": "h_shahidi_afg", "ibrahim zadran": "i_zadran_afg",
    "rahmanullah gurbaz": "r_gurbaz_afg", "fazalhaq farooqi": "f_farooqi_afg", "f farooqi": "f_farooqi_afg",
    "allah mohammad ghazanfar": "a_ghazanfar_afg", "am ghazanfar": "a_ghazanfar_afg",
    "sikandar raza": "s_raza_zim", "sikandar raza butt": "s_raza_zim", "s raza": "s_raza_zim",
    "craig ervine": "c_ervine_zim", "cr ervine": "c_ervine_zim", "c ervine": "c_ervine_zim",
    "blessing muzarabani": "b_muzarabani_zim", "b muzarabani": "b_muzarabani_zim",
    "richard ngarava": "r_ngarava_zim", "r ngarava": "r_ngarava_zim",
    "tendai chatara": "t_chatara_zim", "tl chatara": "t_chatara_zim",
    # Pakistan
    "azan awais": "a_awais_pak", "a awais": "a_awais_pak",
    "tayyab arif": "t_arif_pak", "t arif": "t_arif_pak",
    "mohammad taha": "m_taha_pak", "m taha": "m_taha_pak",
    "babar azam": "b_azam_pak", "b azam": "b_azam_pak",
    "saud shakeel": "s_shakeel_pak", "s shakeel": "s_shakeel_pak",
    "salman agha": "s_ali_agha_pak", "salman ali agha": "s_ali_agha_pak", "s agha": "s_ali_agha_pak", "agha salman": "s_ali_agha_pak",
    "ghazi ghori": "g_ghori_pak", "g ghori": "g_ghori_pak",
    "arafat minhas": "a_minhas_pak", "a minhas": "a_minhas_pak",
    "razaullah": "razaullah_pak",
    "mohammad abbas": "m_abbas_pak", "m abbas": "m_abbas_pak",
    "saqib khan": "saqib_khan_pak", "saqib k": "saqib_khan_pak",
    # Sri Lanka
    "pasindu sooriyabandara": "p_sooriyabandara_sl", "p sooriyabandara": "p_sooriyabandara_sl",
    "lahiru udara": "l_udara_sl", "l udara": "l_udara_sl",
    "dinesh chandimal": "d_chandimal_sl", "d chandimal": "d_chandimal_sl",
    "kamindu mendis": "k_mendis_kam_sl", "k mendis kam": "k_mendis_kam_sl",
    "dhananjaya de silva": "d_desilva_sl", "d de silva": "d_desilva_sl", "dhananjaya": "d_desilva_sl",
    "kusal mendis": "k_mendis_sl", "kusal mendis(c & )": "k_mendis_sl", "kusal mendis(c & wk)": "k_mendis_sl", "k mendis": "k_mendis_sl",
    "sonal dinusha": "s_dinusha_sl", "s dinusha": "s_dinusha_sl",
    "milan priyanath rathnayake": "m_rathnayake_sl", "milan rathnayake": "m_rathnayake_sl", "m rathnayake": "m_rathnayake_sl",
    "lahiru kumara": "l_kumara_sl", "l kumara": "l_kumara_sl",
    "asitha fernando": "a_fernando_sl", "a fernando": "a_fernando_sl",
    "dilshan madushanka": "d_madushanka_sl", "d madushanka": "d_madushanka_sl"
}

class EntityResolver:
    def __init__(self, db_conn=None):
        self.db = db_conn

    async def resolve_player(
        self,
        scraped_name: str,
        team_id: str,
        known_role: Optional[str] = None,
        bowling_style: Optional[str] = None,
        all_players: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Maps a scraped name to a player record.
        1. Checks exact alias table in DB.
        2. Direct match or RapidFuzz token_set_ratio with country filter.
        3. If no match (score < 80), creates an Automatic Debutant using franchise priors or role defaults.
        """
        clean_name = scraped_name.strip()
        norm_key = clean_name.lower().replace(".", " ").replace("  ", " ").strip()

        # 0. Fast-path in-memory Cricbuzz/Cricsheet canonical cross-check
        if norm_key in GLOBAL_CRICBUZZ_CRICSHEET_ALIASES:
            canonical_id = GLOBAL_CRICBUZZ_CRICSHEET_ALIASES[norm_key]
            if all_players:
                for p in all_players:
                    if p.get("id") == canonical_id:
                        return p
            if self.db:
                try:
                    async with self.db.execute("SELECT * FROM players WHERE id = ?", (canonical_id,)) as p_cur:
                        p_row = await p_cur.fetchone()
                        if p_row:
                            return dict(p_row)
                except Exception:
                    pass

        # 1. Check exact alias table in DB if DB connection available
        if self.db:
            try:
                async with self.db.execute(
                    "SELECT cricsheet_id FROM player_aliases WHERE LOWER(alias) = LOWER(?)",
                    (clean_name,)
                ) as cursor:
                    alias_row = await cursor.fetchone()
                    if alias_row:
                        target_id = alias_row[0]
                        if all_players:
                            for p in all_players:
                                if p["id"] == target_id:
                                    return p
                        async with self.db.execute("SELECT * FROM players WHERE id = ?", (target_id,)) as p_cur:
                            p_row = await p_cur.fetchone()
                            if p_row:
                                return dict(p_row)
            except Exception as e:
                pass

        # 2. Check exact matches & RapidFuzz among candidate players
        if all_players:
            # Direct exact match
            for p in all_players:
                if p["name"].lower() == clean_name.lower():
                    return p

            # Rapidfuzz search among players for this team or global
            team_players = [p for p in all_players if p.get("team_id") == team_id]
            candidates = team_players if team_players else all_players

            best_match = None
            best_score = 0.0

            for p in candidates:
                # Compare full name and initials
                score1 = fuzz.token_set_ratio(clean_name, p["name"])
                score2 = fuzz.token_sort_ratio(clean_name, p["name"])
                score = max(score1, score2)
                if score > best_score:
                    best_score = score
                    best_match = p

            if best_match and best_score >= 78.0:
                return best_match

            # 2b. Global search if team-specific candidate search yielded no confident match (< 78.0)
            # Prevents allocating a known international player (e.g. Tanzid Hasan) to an opposing team as uncapped
            if candidates is not all_players:
                global_best = None
                global_score = 0.0
                for p in all_players:
                    score1 = fuzz.token_set_ratio(clean_name, p["name"])
                    score2 = fuzz.token_sort_ratio(clean_name, p["name"])
                    score = max(score1, score2)
                    if score > global_score:
                        global_score = score
                        global_best = p
                if global_best and global_score >= 82.0:
                    return global_best

        # 3. Check Franchise database for uncapped / new players
        slug = clean_name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
        for f_key, f_data in FRANCHISE_DATABASE.items():
            if fuzz.token_set_ratio(clean_name, f_data["name"]) >= 85:
                # Apply league translation factor
                prior = compute_debutant_prior(
                    role=f_data["role"],
                    bowling_style=f_data.get("bowling_style", "NONE"),
                    league_tier=f_data.get("league_tier", "tier_1_ipl"),
                    observed_sr=f_data.get("t20_batting_sr"),
                    observed_avg=f_data.get("t20_batting_avg"),
                    observed_econ=f_data.get("t20_economy")
                )
                return {
                    "id": f"deb_{slug}",
                    "name": f_data["name"],
                    "team_id": team_id,
                    "role": f_data["role"],
                    "batting_hand": "right",
                    "bowling_style": f_data.get("bowling_style", "NONE"),
                    "intl_balls_faced": 30,
                    "intl_balls_bowled": 24,
                    "t20_batting_sr": prior["t20_batting_sr"],
                    "t20_batting_avg": prior["t20_batting_avg"],
                    "t20_economy": prior["t20_economy"],
                    "t20_bowling_sr": prior["t20_bowling_sr"],
                    "powerplay_batting_rating": prior["powerplay_batting_rating"],
                    "middle_batting_rating": prior["middle_batting_rating"],
                    "death_batting_rating": prior["death_batting_rating"],
                    "powerplay_bowling_rating": prior["powerplay_bowling_rating"],
                    "middle_bowling_rating": prior["middle_bowling_rating"],
                    "death_bowling_rating": prior["death_bowling_rating"],
                    "foreign_condition_ratio_subcontinent": 1.0,
                    "foreign_condition_ratio_sena": 1.0,
                    "recent_selection_weight": 0.8,
                    "is_debutant": 1,
                    "league_experience_tier": f_data.get("league_tier", "tier_1_ipl")
                }

        # 4. Fallback: Hierarchical Role Prior for completely unrecorded debutant
        detected_role = known_role or "all_rounder"
        detected_style = bowling_style or "NONE"
        prior = compute_debutant_prior(role=detected_role, bowling_style=detected_style, league_tier="domestic_list_a")

        return {
            "id": f"uncapped_{slug}",
            "name": clean_name,
            "team_id": team_id,
            "role": detected_role,
            "batting_hand": "right",
            "bowling_style": detected_style,
            "intl_balls_faced": 0,
            "intl_balls_bowled": 0,
            "t20_batting_sr": prior["t20_batting_sr"],
            "t20_batting_avg": prior["t20_batting_avg"],
            "t20_economy": prior["t20_economy"],
            "t20_bowling_sr": prior["t20_bowling_sr"],
            "powerplay_batting_rating": prior["powerplay_batting_rating"],
            "middle_batting_rating": prior["middle_batting_rating"],
            "death_batting_rating": prior["death_batting_rating"],
            "powerplay_bowling_rating": prior["powerplay_bowling_rating"],
            "middle_bowling_rating": prior["middle_bowling_rating"],
            "death_bowling_rating": prior["death_bowling_rating"],
            "foreign_condition_ratio_subcontinent": 1.0,
            "foreign_condition_ratio_sena": 1.0,
            "recent_selection_weight": 0.7,
            "is_debutant": 1,
            "league_experience_tier": "domestic_list_a"
        }
