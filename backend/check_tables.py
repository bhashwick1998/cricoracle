import sqlite3

conn = sqlite3.connect('backend/cricoracle.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print('Tables in cricoracle.db:', tables)

for t in ['players', 'match_squads', 'player_strengths', 'venues']:
    if t in tables:
        c.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in c.fetchall()]
        print(f"\n{t} columns ({len(cols)}):", cols)
        c.execute(f"SELECT * FROM {t} LIMIT 1")
        row = c.fetchone()
        if row:
            print(f"Sample {t} row:")
            for col, val in zip(cols[:15], row[:15]):
                print(f"  {col}: {val}")
