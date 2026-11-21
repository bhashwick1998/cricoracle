# CricOracle — Automated Multi-Stage Cricket Win Predictor & Live Telemetry Platform

CricOracle is a full-stack, real-time international cricket prediction web application (T20I & ODI) that automatically scrapes/ingests match schedules, squads, toss/Playing XIs, and live ball-by-ball commentary from external sources, maps them to historical Cricsheet player/venue vectors, and dynamically updates win probabilities across **4 Automated Match States**.

---

## 1. The 4 Automated Match States & Engines

### STATE 1: `UPCOMING_NO_SQUAD` (Gear 1: Macro Condition-Adjusted Engine)
- **Data Available:** Teams, Format, Venue, Date/Time.
- **Engine Logic (`backend/engine/stage1_macro.py`):**
  - Evaluates Condition-Specific Team Elo (`elo_home` for host, `elo_away_subcontinent` or `elo_away_sena` for visitor based on venue region cluster).
  - Computes pitch-to-team fit differential ($\Delta_{\text{Pitch}}$):
    $$\Delta_{\text{Pitch}} = (\text{TeamA\_SpinIdx} - \text{TeamB\_SpinIdx}) \cdot \text{Venue\_SpinRatio} + (\text{TeamA\_PaceIdx} - \text{TeamB\_PaceIdx}) \cdot (1 - \text{Venue\_SpinRatio})$$
  - Combines $\Delta \text{Elo}_{\text{Cond}}$, $\Delta_{\text{Pitch}}$, and $\Delta \text{TeamPhaseNetRating}$ via a calibrated logistic regression model.
  - Integrates across both toss outcomes using `venue.chase_win_pct` and day/night dew factor.
- **Homepage UI Card:** Displays Team A vs Team B Win % bar with badge `"GEAR 1 • MACRO FIXTURE (NO SQUADS)"`.

---

### STATE 2: `SQUAD_ANNOUNCED` (Gear 2A: Squad Depth & Condition Delta Model)
- **Data Available:** 15–18 member squads extracted for both teams.
- **Engine Logic (`backend/engine/stage2_squad.py`):**
  - **Probable XI Selector Heuristic:** Automatically selects the Optimal 11 from the 15-man squad by balancing recent selection weights, pitch fit (spin vs. pace conditions), and tactical team composition (minimum 1 wicketkeeper, top 5 specialist batters, 20/50 overs of bowling coverage).
  - **Bottom-Up Lineup Aggregation:** Evaluates player vectors including Foreign Condition Deltas and Empirical Bayes shrinkage for uncapped squad call-ups.
  - Blends $0.70 \times P_{\text{Projected\_XI}} + 0.30 \times P_{\text{Stage1\_Macro}}$.
- **Homepage UI Card:** Badges `"GEAR 2A • 15-MAN SQUADS EXTRACTED"`, displays the net squad shift ($\Delta\%$), and provides an expandable drawer showing the 15-man squad with the Probable XI highlighted.

---

### STATE 3: `XI_AND_TOSS_LOCKED` (Gear 2B: Exact 11v11 Aggregation & 5,000 Monte Carlo Sims)
- **Data Available:** Confirmed 11v11 Playing XI + Toss Winner & Decision (`bat`/`field`).
- **Engine Logic (`backend/engine/stage3_xi_locked.py`):**
  - Exact Phase Ratings (Powerplay, Middle, Death) for all 22 confirmed players.
  - Generates the **11x6 Batter-vs-Bowler H2H Matchup Matrix** using Empirical Bayes shrinkage toward bowler archetypes (`RF`, `RMF`, `SLA`, `LBG`, `OB`, `LAWS`).
  - Applies directional **Toss & Dew Modifier** based on who is batting first vs. chasing under lights.
  - Executes **5,000 Vectorized Monte Carlo Simulations** from Ball 0.1 to lock the official Pre-Match Prior ($P_0$) and calculate the Projected 1st Innings Par Score (with 10th–90th percentile bounds).
- **Homepage UI Card:** Badges `"GEAR 2B • PLAYING XI & TOSS CONFIRMED"`, displays the locked Pre-Match Prior ($P_0$), par score range, and highlights the biggest head-to-head matchup advantage.

---

### STATE 4: `LIVE_BALL_BY_BALL` (Gear 3: Live Sub-50ms Multinomial + Monte Carlo Rollout)
- **Data Available:** Live ball-by-ball stream (`over.ball`, `striker`, `non_striker`, `bowler`, `runs`, `extras`, `wicket`).
- **Engine Logic (`backend/engine/stage4_live.py`):**
  - **Micro Next-Ball Multinomial Predictor:** Predicts probabilities for `{0, 1, 2, 3, 4, 6, Extra, Wicket}` using active striker vs bowler matchup, striker Settledness Curve ($1 + \gamma \ln(1 + \text{balls\_faced})$), phase of play, and Required Run Rate pressure.
  - **Macro Resource Tank Evaluation:** Computes dugout batting reserve capacity and bowling quota quality.
  - **Live Vectorized Monte Carlo Rollout (3,000 runs):** Simulates to match finish, smoothly anchored to $P_0$ via Log5 calibration.
  - Computes real-time Win Probability Added (WPA) swing: $\text{WPA} = P_{t} - P_{t-1}$.
- **Homepage & Match Center UI:** Pulsing badge `"GEAR 3 • LIVE BALL-BY-BALL"`, active striker vs bowler HUD, live worm chart anchored at $P_0$, and ball-by-ball commentary with WPA tags.

---

## 2. Multi-Source Ingestion Pipeline (`backend/scraper/`)

- **Adapter 1: Unofficial/Public Cricket JSON Endpoints (`backend/scraper/providers/cric_json_provider.py`)**
  - Scrapes upcoming schedules, squads, Playing XIs, and live scores with graceful high-fidelity international fallback.
- **Adapter 2: Universal URL / HTML Scraper (`backend/scraper/providers/html_scraper.py`)**
  - Allows pasting any ESPNcricinfo or Cricbuzz match link into the UI toolbar to extract metadata, DOM nodes, and `__NEXT_DATA__` JSON payloads.
- **Adapter 3: Live Lifecycle & Replay Simulator (`backend/scraper/providers/replay_simulator.py`)**
  - Steps or streams any fixture through all 4 states (`NO_SQUAD` $\to$ `SQUAD_ANNOUNCED` $\to$ `XI_LOCKED` $\to$ `LIVE_BALL_BY_BALL`) with single-ball triggers and autonomous playback.
- **Entity Resolution & Debutant Auto-Detection (`backend/scraper/entity_resolver.py`):**
  - RapidFuzz token-set matching and country filtering for player aliases (e.g., `"PWH de Silva"` $\to$ `"Wanindu Hasaranga"`).
  - Uncapped / debutant auto-detection with franchise league translation factors ($\alpha_L$) and hierarchical role priors shrunk via Empirical Bayes:
    $$\hat{\theta} = \frac{N}{N + N_0}\bar{x} + \frac{N_0}{N + N_0}\mu_0$$

---

## 3. Technology Stack

- **Frontend:** Next.js 16 (App Router), TypeScript, Tailwind CSS, Lucide Icons, Recharts.
- **Backend & ML Engine:** Python 3.14, FastAPI, NumPy, SciPy, RapidFuzz, APScheduler, Uvicorn, aiosqlite.
- **Real-Time Communication:** WebSockets (`ws://localhost:8000/ws/matches`) with automatic reconnection and fallback polling.
- **Database:** SQLite (`backend/cricoracle.db`) with relational schema for venues, teams, players, aliases, matches, squads, and deliveries.

---

## 4. Running the Application

### A. Start the Backend API & Engine:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API endpoints are available at `http://127.0.0.1:8000/api/matches`.

### B. Start the Next.js Frontend:
```bash
cd frontend
npm run dev
# or for production
npm start -p 3005
```
Open [http://localhost:3005](http://localhost:3005) in your browser.

### C. Run the Comprehensive Verification Test Suite:
```bash
python -m backend.tests.test_cricoracle
```
Runs automated assertions for all 4 prediction engines, entity resolution, debutant shrinkage, and simulator state transitions.
