-- CricOracle Database Schema (SQLite compatible)

CREATE TABLE IF NOT EXISTS venues (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    region_cluster TEXT NOT NULL, -- 'sena', 'subcontinent', 'caribbean', 'other'
    pitch_type TEXT NOT NULL,     -- 'dusty', 'green', 'bouncy', 'flat', 'balanced'
    spin_wicket_ratio REAL NOT NULL DEFAULT 0.40,
    avg_first_innings_score_t20 INTEGER NOT NULL DEFAULT 165,
    avg_first_innings_score_odi INTEGER NOT NULL DEFAULT 280,
    chase_win_pct REAL NOT NULL DEFAULT 0.52,
    dew_factor REAL NOT NULL DEFAULT 0.35,
    dimensions_avg_boundary_m REAL NOT NULL DEFAULT 72.0
);

CREATE TABLE IF NOT EXISTS teams (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL,
    color TEXT NOT NULL,
    elo_home REAL NOT NULL DEFAULT 1600.0,
    elo_away_subcontinent REAL NOT NULL DEFAULT 1550.0,
    elo_away_sena REAL NOT NULL DEFAULT 1550.0,
    spin_index REAL NOT NULL DEFAULT 50.0,
    pace_index REAL NOT NULL DEFAULT 50.0,
    powerplay_rating REAL NOT NULL DEFAULT 50.0,
    middle_overs_rating REAL NOT NULL DEFAULT 50.0,
    death_overs_rating REAL NOT NULL DEFAULT 50.0
);

CREATE TABLE IF NOT EXISTS players (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    team_id TEXT NOT NULL,
    role TEXT NOT NULL, -- 'top_order_batter', 'middle_order_batter', 'finisher', 'all_rounder', 'wicket_keeper', 'frontline_pacer', 'frontline_spinner'
    batting_hand TEXT NOT NULL DEFAULT 'right', -- 'right', 'left'
    bowling_style TEXT NOT NULL DEFAULT 'NONE', -- 'RF', 'RMF', 'SLA', 'LBG', 'OB', 'LAWS', 'NONE'
    intl_balls_faced INTEGER NOT NULL DEFAULT 500,
    intl_balls_bowled INTEGER NOT NULL DEFAULT 300,
    t20_batting_sr REAL NOT NULL DEFAULT 130.0,
    t20_batting_avg REAL NOT NULL DEFAULT 28.0,
    t20_economy REAL NOT NULL DEFAULT 8.2,
    t20_bowling_sr REAL NOT NULL DEFAULT 18.0,
    powerplay_batting_rating REAL NOT NULL DEFAULT 50.0,
    middle_batting_rating REAL NOT NULL DEFAULT 50.0,
    death_batting_rating REAL NOT NULL DEFAULT 50.0,
    powerplay_bowling_rating REAL NOT NULL DEFAULT 50.0,
    middle_bowling_rating REAL NOT NULL DEFAULT 50.0,
    death_bowling_rating REAL NOT NULL DEFAULT 50.0,
    foreign_condition_ratio_subcontinent REAL NOT NULL DEFAULT 1.0,
    foreign_condition_ratio_sena REAL NOT NULL DEFAULT 1.0,
    recent_selection_weight REAL NOT NULL DEFAULT 0.85,
    is_debutant INTEGER NOT NULL DEFAULT 0,
    league_experience_tier TEXT NOT NULL DEFAULT 'none', -- 'tier_1_ipl', 'tier_2_bbl_psl', 'domestic_list_a', 'none'
    FOREIGN KEY(team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS player_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cricsheet_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    country TEXT,
    FOREIGN KEY(cricsheet_id) REFERENCES players(id)
);

CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,
    series_name TEXT NOT NULL,
    format TEXT NOT NULL DEFAULT 'T20I', -- 'T20I', 'ODI', 'TEST'
    team_a_id TEXT NOT NULL,
    team_b_id TEXT NOT NULL,
    venue_id TEXT NOT NULL,
    scheduled_start TEXT NOT NULL,
    is_day_night INTEGER NOT NULL DEFAULT 1,
    state TEXT NOT NULL DEFAULT 'UPCOMING_NO_SQUAD', -- 'UPCOMING_NO_SQUAD', 'SQUAD_ANNOUNCED', 'XI_AND_TOSS_LOCKED', 'LIVE_BALL_BY_BALL', 'COMPLETED'
    toss_winner_id TEXT,
    toss_decision TEXT, -- 'bat', 'field'
    current_innings INTEGER DEFAULT 1,
    current_score INTEGER DEFAULT 0,
    current_wickets INTEGER DEFAULT 0,
    current_overs INTEGER DEFAULT 0,
    current_balls INTEGER DEFAULT 0,
    target_runs INTEGER,
    current_striker_id TEXT,
    current_non_striker_id TEXT,
    current_bowler_id TEXT,
    p_team_a REAL NOT NULL DEFAULT 50.0,
    p_team_b REAL NOT NULL DEFAULT 50.0,
    p_draw REAL DEFAULT 0.0,
    p1_macro REAL DEFAULT 50.0,
    p2_squad REAL DEFAULT 50.0,
    p0_pre_match REAL DEFAULT 50.0,
    par_score_projected INTEGER DEFAULT 170,
    par_score_min INTEGER DEFAULT 155,
    par_score_max INTEGER DEFAULT 185,
    delta_squad_shift REAL DEFAULT 0.0,
    delta_toss_shift REAL DEFAULT 0.0,
    biggest_matchup_adv TEXT,
    last_wpa_swing REAL DEFAULT 0.0,
    last_event_description TEXT DEFAULT 'Match scheduled',
    weather_condition TEXT DEFAULT 'CLEAR_OPTIMAL',
    rain_probability_pct REAL DEFAULT 15.0,
    overs_lost REAL DEFAULT 0.0,
    dls_target INTEGER,
    dls_par_score INTEGER,
    is_stumps INTEGER DEFAULT 0,
    stumps_day INTEGER DEFAULT 1,
    stumps_label TEXT DEFAULT NULL,
    pitch_report TEXT DEFAULT NULL,
    pitch_type TEXT DEFAULT 'flat',
    active_pitch_type TEXT DEFAULT 'flat',
    ground_pitch_archetype TEXT DEFAULT 'flat',
    batting_tank_pct REAL DEFAULT 50.0,
    bowling_tank_pct REAL DEFAULT 50.0,
    day_1_overs REAL DEFAULT NULL,
    day_2_overs REAL DEFAULT NULL,
    day_3_overs REAL DEFAULT NULL,
    day_4_overs REAL DEFAULT NULL,
    day_start_overs REAL DEFAULT NULL,
    innings_1_declared INTEGER DEFAULT 0,
    innings_2_declared INTEGER DEFAULT 0,
    innings_3_declared INTEGER DEFAULT 0,
    is_declared INTEGER DEFAULT 0,
    overs_left_today REAL DEFAULT NULL,
    min_overs_remaining REAL DEFAULT NULL,
    total_match_overs_remaining REAL DEFAULT NULL,
    follow_on_threshold INTEGER DEFAULT NULL,
    follow_on_avoided INTEGER DEFAULT 0,
    runs_needed_to_avoid_follow_on INTEGER DEFAULT NULL,
    stage3_frozen_json TEXT DEFAULT NULL,
    live_pitch_state_json TEXT DEFAULT NULL,
    FOREIGN KEY(team_a_id) REFERENCES teams(id),
    FOREIGN KEY(team_b_id) REFERENCES teams(id),
    FOREIGN KEY(venue_id) REFERENCES venues(id)
);

CREATE TABLE IF NOT EXISTS venue_pitch_formats (
    venue_id TEXT NOT NULL,
    format TEXT NOT NULL, -- 'TEST', 'ODI', 'T20I'
    pitch_archetype TEXT NOT NULL, -- 'dusty', 'green', 'bouncy', 'flat', 'balanced'
    avg_first_innings_score INTEGER NOT NULL,
    chase_win_pct REAL NOT NULL DEFAULT 0.50,
    pace_wicket_pct REAL NOT NULL DEFAULT 0.60,
    spin_wicket_pct REAL NOT NULL DEFAULT 0.40,
    avg_pace_econ REAL NOT NULL DEFAULT 7.80,
    avg_spin_econ REAL NOT NULL DEFAULT 7.50,
    pitch_wear_rate REAL DEFAULT 5.0,
    dew_factor REAL DEFAULT 0.30,
    dimensions_straight_m REAL DEFAULT 70.0,
    dimensions_square_m REAL DEFAULT 68.0,
    dimensions_avg_m REAL DEFAULT 69.0,
    PRIMARY KEY (venue_id, format),
    FOREIGN KEY(venue_id) REFERENCES venues(id)
);

CREATE TABLE IF NOT EXISTS concussion_substitutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    injured_player_id TEXT NOT NULL,
    substitute_player_id TEXT NOT NULL,
    inning INTEGER NOT NULL DEFAULT 1,
    over_ball TEXT NOT NULL DEFAULT '0.0',
    similarity_pct REAL NOT NULL,
    referee_status TEXT NOT NULL,
    rating_shift_delta REAL NOT NULL,
    reason TEXT DEFAULT 'Struck on helmet by bouncer - ICC PC 1.2.7 approved',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(match_id) REFERENCES matches(id),
    FOREIGN KEY(injured_player_id) REFERENCES players(id),
    FOREIGN KEY(substitute_player_id) REFERENCES players(id)
);

CREATE TABLE IF NOT EXISTS match_squads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    is_in_squad INTEGER NOT NULL DEFAULT 1,
    is_probable_xi INTEGER NOT NULL DEFAULT 0,
    is_confirmed_xi INTEGER NOT NULL DEFAULT 0,
    is_subbed_out INTEGER NOT NULL DEFAULT 0,
    is_concussion_sub INTEGER NOT NULL DEFAULT 0,
    sub_status TEXT DEFAULT 'ACTIVE', -- 'ACTIVE', 'CONCUSSED_OUT', 'CONCUSSION_SUB_IN'
    batting_order INTEGER DEFAULT 99,
    pitch_fit_rating REAL DEFAULT 50.0,
    FOREIGN KEY(match_id) REFERENCES matches(id),
    FOREIGN KEY(player_id) REFERENCES players(id)
);

CREATE TABLE IF NOT EXISTS match_deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    innings INTEGER NOT NULL,
    over INTEGER NOT NULL,
    ball INTEGER NOT NULL,
    striker_id TEXT NOT NULL,
    non_striker_id TEXT NOT NULL,
    bowler_id TEXT NOT NULL,
    runs_batter INTEGER NOT NULL DEFAULT 0,
    extras INTEGER NOT NULL DEFAULT 0,
    extra_type TEXT DEFAULT 'none', -- 'wide', 'noball', 'legbye', 'bye', 'none'
    is_wicket INTEGER NOT NULL DEFAULT 0,
    wicket_type TEXT DEFAULT 'none',
    dismissed_player_id TEXT,
    score_after INTEGER NOT NULL,
    wickets_after INTEGER NOT NULL,
    p_team_a_after REAL NOT NULL,
    wpa REAL NOT NULL DEFAULT 0.0,
    commentary TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(match_id) REFERENCES matches(id)
);
