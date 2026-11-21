import sqlite3

def run_migration():
    conn = sqlite3.connect("backend/cricoracle.db")
    cur = conn.cursor()

    def add_col_if_missing(table, col, col_def):
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        if col not in cols:
            print(f"Adding {col} to {table}...")
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")

    add_col_if_missing("matches", "weather_condition", "TEXT DEFAULT 'CLEAR'")
    add_col_if_missing("matches", "rain_probability_pct", "REAL DEFAULT 10.0")
    add_col_if_missing("matches", "overs_lost", "REAL DEFAULT 0.0")
    add_col_if_missing("matches", "dls_target", "INTEGER DEFAULT NULL")
    add_col_if_missing("matches", "dls_par_score", "INTEGER DEFAULT NULL")
    add_col_if_missing("matches", "predictive_power_pct", "REAL DEFAULT 96.5")

    add_col_if_missing("match_squads", "is_subbed_out", "INTEGER DEFAULT 0")
    add_col_if_missing("match_squads", "is_concussion_sub", "INTEGER DEFAULT 0")
    add_col_if_missing("match_squads", "sub_status", "TEXT DEFAULT NULL")
    add_col_if_missing("match_squads", "replaced_by_player_id", "TEXT DEFAULT NULL")

    conn.commit()
    conn.close()
    print("Migration applied cleanly to backend/cricoracle.db")

if __name__ == "__main__":
    run_migration()
