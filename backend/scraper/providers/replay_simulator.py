"""
Lifecycle & Replay Simulator for CricOracle (Adapter 3)
Allows transitioning any match through the 4 states:
STATE 1: UPCOMING_NO_SQUAD
STATE 2: SQUAD_ANNOUNCED
STATE 3: XI_AND_TOSS_LOCKED
STATE 4: LIVE_BALL_BY_BALL (step-by-step ball trigger or autonomous playback loop)
"""

import asyncio
import random
from typing import Dict, Any, List, Optional
from ...engine.stage1_macro import calculate_stage1_macro
from ...engine.stage2_squad import calculate_stage2_squad
from ...engine.stage3_xi_locked import calculate_stage3_xi_locked
from ...engine.stage4_live import calculate_stage4_live

class ReplaySimulator:
    def __init__(self, broadcast_callback=None):
        self.broadcast_callback = broadcast_callback
        self.running_sims: Dict[str, asyncio.Task] = {}

    async def advance_match_state(self, match_id: str, target_state: str, db_conn) -> Dict[str, Any]:
        """
        Transitions match to target_state and recalculates prediction.
        """
        # Fetch match
        async with db_conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as cursor:
            match = await cursor.fetchone()
            if not match:
                raise ValueError(f"Match {match_id} not found")
            match = dict(match)

        # Fetch teams and venue
        async with db_conn.execute("SELECT * FROM teams WHERE id = ?", (match["team_a_id"],)) as c:
            team_a = dict(await c.fetchone())
        async with db_conn.execute("SELECT * FROM teams WHERE id = ?", (match["team_b_id"],)) as c:
            team_b = dict(await c.fetchone())
        async with db_conn.execute("SELECT * FROM venues WHERE id = ?", (match["venue_id"],)) as c:
            venue = dict(await c.fetchone())

        if target_state == "UPCOMING_NO_SQUAD":
            # Gear 1
            res = calculate_stage1_macro(team_a, team_b, venue, match.get("format", "T20I"))
            await db_conn.execute("""
                UPDATE matches SET
                    state = 'UPCOMING_NO_SQUAD',
                    p_team_a = ?, p_team_b = ?, p1_macro = ?,
                    toss_winner_id = NULL, toss_decision = NULL,
                    last_event_description = 'Match scheduled • Macro prediction calculated'
                WHERE id = ?
            """, (res["p_team_a"], res["p_team_b"], res["p_team_a"], match_id))
            # Clear squad assignments
            await db_conn.execute("UPDATE match_squads SET is_in_squad = 0, is_probable_xi = 0, is_confirmed_xi = 0 WHERE match_id = ?", (match_id,))
            await db_conn.commit()

        elif target_state == "SQUAD_ANNOUNCED":
            # Gear 2A: Set 15-man squad active
            await db_conn.execute("""
                UPDATE match_squads SET is_in_squad = 1 WHERE match_id = ?
            """, (match_id,))
            
            # Fetch squads
            async with db_conn.execute("""
                SELECT p.* FROM match_squads ms
                JOIN players p ON ms.player_id = p.id
                WHERE ms.match_id = ? AND ms.team_id = ? AND ms.is_in_squad = 1
            """, (match_id, match["team_a_id"])) as c:
                squad_a = [dict(r) for r in await c.fetchall()]

            async with db_conn.execute("""
                SELECT p.* FROM match_squads ms
                JOIN players p ON ms.player_id = p.id
                WHERE ms.match_id = ? AND ms.team_id = ? AND ms.is_in_squad = 1
            """, (match_id, match["team_b_id"])) as c:
                squad_b = [dict(r) for r in await c.fetchall()]

            p1_anchor = match.get("p1_macro", 50.0)
            res = calculate_stage2_squad(squad_a, squad_b, venue, p1_anchor)

            # Mark probable XIs in DB
            await db_conn.execute("UPDATE match_squads SET is_probable_xi = 0 WHERE match_id = ?", (match_id,))
            for pid in res["probable_xi_a_ids"]:
                await db_conn.execute("UPDATE match_squads SET is_probable_xi = 1 WHERE match_id = ? AND player_id = ?", (match_id, pid))
            for pid in res["probable_xi_b_ids"]:
                await db_conn.execute("UPDATE match_squads SET is_probable_xi = 1 WHERE match_id = ? AND player_id = ?", (match_id, pid))

            await db_conn.execute("""
                UPDATE matches SET
                    state = 'SQUAD_ANNOUNCED',
                    p_team_a = ?, p_team_b = ?, p2_squad = ?,
                    delta_squad_shift = ?,
                    last_event_description = '15-man squads announced • Gear 2A Probable XI evaluated'
                WHERE id = ?
            """, (res["p_team_a"], res["p_team_b"], res["p_team_a"], res["delta_squad_shift"], match_id))
            await db_conn.commit()

        elif target_state == "XI_AND_TOSS_LOCKED":
            # Gear 2B: Lock confirmed Playing XI (11 players) and set toss
            toss_winner = match["team_a_id"]
            toss_decision = "field"

            # Set confirmed XI to probable XI
            await db_conn.execute("""
                UPDATE match_squads SET is_confirmed_xi = is_probable_xi WHERE match_id = ?
            """, (match_id,))

            async with db_conn.execute("""
                SELECT p.* FROM match_squads ms
                JOIN players p ON ms.player_id = p.id
                WHERE ms.match_id = ? AND ms.team_id = ? AND ms.is_confirmed_xi = 1
            """, (match_id, match["team_a_id"])) as c:
                xi_a = [dict(r) for r in await c.fetchall()]

            async with db_conn.execute("""
                SELECT p.* FROM match_squads ms
                JOIN players p ON ms.player_id = p.id
                WHERE ms.match_id = ? AND ms.team_id = ? AND ms.is_confirmed_xi = 1
            """, (match_id, match["team_b_id"])) as c:
                xi_b = [dict(r) for r in await c.fetchall()]

            p2_anchor = match.get("p2_squad", 50.0)
            res = calculate_stage3_xi_locked(
                confirmed_xi_a=xi_a,
                confirmed_xi_b=xi_b,
                toss_winner_id=toss_winner,
                toss_decision=toss_decision,
                venue=venue,
                team_a_id=match["team_a_id"],
                p2_squad_prob_a=p2_anchor,
                match_format=match.get("format", "T20I"),
                team_b_id=match["team_b_id"],
                pitch_report=match.get("pitch_report"),
                match_date=match.get("scheduled_start")
            )

            # Reset replay queue and deliveries if match has queued replays
            await db_conn.execute("UPDATE match_replay_queue SET is_played = 0 WHERE match_id = ?", (match_id,))
            await db_conn.execute("DELETE FROM match_deliveries WHERE match_id = ?", (match_id,))

            await db_conn.execute("""
                UPDATE matches SET
                    state = 'XI_AND_TOSS_LOCKED',
                    current_innings = 1, current_score = 0, current_wickets = 0, current_overs = 0, current_balls = 0,
                    toss_winner_id = ?, toss_decision = ?,
                    p_team_a = ?, p_team_b = ?, p0_pre_match = ?,
                    par_score_projected = ?, par_score_min = ?, par_score_max = ?,
                    delta_toss_shift = ?, biggest_matchup_adv = ?,
                    last_event_description = ?
                WHERE id = ?
            """, (
                toss_winner, toss_decision,
                res["p_team_a"], res["p_team_b"], res["p0_pre_match"],
                res["par_score_projected"], res["par_score_min"], res["par_score_max"],
                res["delta_toss_shift"], res["biggest_matchup_adv"],
                f"{team_a['short_name']} won toss & chose to bowl • P0 Locked ({res['p_team_a']}%)",
                match_id
            ))
            await db_conn.commit()

        elif target_state == "LIVE_BALL_BY_BALL":
            is_test = match.get("format") == "TEST"
            inn = match.get("current_innings") or 1
            cur_score = match.get("current_score") if match.get("current_score") is not None and match.get("current_score") > 0 else 0
            cur_wickets = match.get("current_wickets") if match.get("current_wickets") is not None else 0
            cur_overs = match.get("current_overs") if match.get("current_overs") is not None else 0
            cur_balls = match.get("current_balls") if match.get("current_balls") is not None else 0
            target_val = match.get("target_runs")
            desc = "Match in progress • Day 1 (1st innings)" if is_test else ("Match in progress • 1st innings" if inn == 1 else "Match in progress • 2nd innings chase")
            await db_conn.execute("""
                UPDATE matches SET
                    state = 'LIVE_BALL_BY_BALL',
                    current_innings = ?,
                    current_score = ?,
                    current_wickets = ?,
                    current_overs = ?,
                    current_balls = ?,
                    target_runs = ?,
                    last_event_description = ?
                WHERE id = ?
            """, (inn, cur_score, cur_wickets, cur_overs, cur_balls, target_val, desc, match_id))
            await db_conn.commit()

        # Fetch updated match
        async with db_conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
            updated = dict(await c.fetchone())

        if self.broadcast_callback:
            await self.broadcast_callback(match_id, {"type": "STATE_CHANGE", "match": updated})

        return updated

    async def step_single_ball(self, match_id: str, db_conn) -> Dict[str, Any]:
        """
        Simulates arrival of next delivery, executes Stage 4 live inference, logs delivery, updates state.
        """
        async with db_conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
            match = dict(await c.fetchone())

        # If match is already marked completed, halt progression immediately
        if match.get("state") == "COMPLETED":
            return {
                "type": "MATCH_COMPLETED",
                "match": match,
                "delivery": {
                    "over": match.get("current_overs", 0),
                    "ball": match.get("current_balls", 0),
                    "score": f"{match.get('current_score', 0)}/{match.get('current_wickets', 0)}",
                    "runs_batter": 0,
                    "extras": 0,
                    "is_wicket": False,
                    "wpa": 0.0,
                    "p_team_a": match.get("p_team_a", 0.0),
                    "commentary": match.get("match_status_text") or "Match completed."
                }
            }

        if match["state"] != "LIVE_BALL_BY_BALL":
            await self.advance_match_state(match_id, "LIVE_BALL_BY_BALL", db_conn)
            async with db_conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
                match = dict(await c.fetchone())

        target = match.get("target_runs")
        if target is None and match.get("current_innings", 1) == 2:
            target = (match.get("innings_1_score") or match.get("current_score") or 168) + 1
        p0 = match.get("p0_pre_match", 50.0)
        prev_p = match.get("p_team_a", 50.0)
        toss_winner = match.get("toss_winner_id") or match["team_a_id"]
        toss_dec = match.get("toss_decision") or "bat"
        team_a_bats_first = (toss_winner == match["team_a_id"] and toss_dec == "bat") or (toss_winner == match["team_b_id"] and toss_dec == "field")

        # Check if match has a queued curated replay delivery sequence
        queued_item = None
        try:
            async with db_conn.execute("""
                SELECT COUNT(*) as unplayed,
                       (SELECT COUNT(*) FROM match_replay_queue WHERE match_id = ?) as total
                FROM match_replay_queue
                WHERE match_id = ? AND is_played = 0
            """, (match_id, match_id)) as q_cnt:
                cnt_row = await q_cnt.fetchone()
                unplayed_cnt = cnt_row[0] if cnt_row else 0
                total_cnt = cnt_row[1] if cnt_row else 0
                if total_cnt > 0 and unplayed_cnt == 0:
                    # All curated replay balls played -> match is complete!
                    await db_conn.execute("UPDATE matches SET state = 'COMPLETED' WHERE id = ?", (match_id,))
                    await db_conn.commit()
                    match["state"] = "COMPLETED"
                    return {
                        "type": "MATCH_COMPLETED",
                        "match": match,
                        "delivery": {
                            "over": match.get("current_overs", 0),
                            "ball": match.get("current_balls", 0),
                            "score": f"{match.get('current_score', 0)}/{match.get('current_wickets', 0)}",
                            "runs_batter": 0,
                            "extras": 0,
                            "is_wicket": False,
                            "wpa": 0.0,
                            "p_team_a": match.get("p_team_a", 0.0),
                            "commentary": match.get("match_status_text") or "Replay completed."
                        }
                    }

            async with db_conn.execute("""
                SELECT * FROM match_replay_queue
                WHERE match_id = ? AND is_played = 0
                ORDER BY seq ASC LIMIT 1
            """, (match_id,)) as q_c:
                q_row = await q_c.fetchone()
                if q_row:
                    queued_item = dict(q_row)
        except Exception:
            queued_item = None

        if queued_item:
            q_id = queued_item["id"]
            inn = queued_item["innings"]
            overs = queued_item["over"]
            balls = queued_item["ball"]
            striker_id = queued_item["striker_id"]
            non_striker_id = queued_item["non_striker_id"]
            bowler_id = queued_item["bowler_id"]
            runs_b = queued_item["runs_batter"]
            extras = queued_item["extras"]
            is_wkt = bool(queued_item["is_wicket"])
            w_type = queued_item["wicket_type"]
            commentary = queued_item["commentary"]

            cur_inn = match.get("current_innings", 1)
            if inn != cur_inn:
                score = 0
                wickets = 0
                target_to_set = match.get("target_runs") or ((match.get("innings_1_score") or match.get("current_score", 168)) + 1)
                await db_conn.execute("""
                    UPDATE matches SET
                        current_innings = ?,
                        current_score = 0,
                        current_wickets = 0,
                        current_overs = 0,
                        current_balls = 0,
                        target_runs = ?
                    WHERE id = ?
                """, (inn, target_to_set, match_id))
                match["current_innings"] = inn
                match["target_runs"] = target_to_set
                target = target_to_set
            else:
                score = match.get("current_score", 0)
                wickets = match.get("current_wickets", 0)
                if inn == 2 and target is None:
                    target = (match.get("innings_1_score") or 168) + 1
                    match["target_runs"] = target

            team_a_batting = team_a_bats_first if inn == 1 else (not team_a_bats_first)
            new_score = score + runs_b + extras
            new_wickets = min(10, wickets + (1 if is_wkt else 0))
            is_queued_replay = True
        else:
            is_queued_replay = False
            # Fallback generic increment logic
            overs = match.get("current_overs", 0)
            balls = match.get("current_balls", 0)
            score = match.get("current_score", 0)
            wickets = match.get("current_wickets", 0)
            inn = match.get("current_innings", 1)
            team_a_batting = team_a_bats_first if inn == 1 else (not team_a_bats_first)

            balls += 1
            if balls >= 6:
                overs += 1
                balls = 0

            striker_id = match.get("current_striker_id") or "j_dasilva_wi"
            non_striker_id = match.get("current_non_striker_id") or "k_wickham_wi"
            bowler_id = match.get("current_bowler_id") or "t_ahmed_ban"

        is_test = match.get("format") == "TEST"
        is_odi = match.get("format") == "ODI"
        max_overs = 50 if is_odi else 20

        # Outcome probabilities
        if is_test:
            outcomes = [
                (0, 0, False, "dot", "Good length in the channel, defended with soft hands towards point."),
                (0, 0, False, "dot", "BEATEN! Sharp seam movement off the deck, whistling past the outside edge."),
                (1, 0, False, "one", "Pushed softly into the gap at cover for a brisk single."),
                (2, 0, False, "two", "Tucked off the pads behind square, comfortable couple taken."),
                (4, 0, False, "four", "GLORIOUS COVER DRIVE! Leans right into the pitch of the ball and strokes it for FOUR!"),
                (0, 0, True, "caught", "OUT! GONE! Edged and caught in the slip cordon!"),
                (1, 1, False, "wide", "Sprayed down leg, called a wide by the umpire."),
            ]
            weights = [0.46, 0.26, 0.15, 0.04, 0.05, 0.025, 0.015]
        elif is_odi:
            outcomes = [
                (0, 0, False, "dot", "Defended back down the pitch."),
                (1, 0, False, "one", "Dabbed into the off-side for a single."),
                (2, 0, False, "two", "Driven firmly into deep extra cover for two."),
                (4, 0, False, "four", "CRUNCHED! One bounce over the ropes for FOUR!"),
                (6, 0, False, "six", "DISPATCHED! Clears long-on with effortless timing for SIX!"),
                (0, 0, True, "bowled", "OUT! Clean bowled through the gate!"),
                (1, 1, False, "wide", "Fired down the leg side, called wide."),
            ]
            weights = [0.48, 0.32, 0.06, 0.07, 0.03, 0.025, 0.015]
        else:
            outcomes = [
                (0, 0, False, "dot", "Defended back to bowler."),
                (1, 0, False, "one", "Pushed into the gap for a quick single."),
                (2, 0, False, "two", "Driven firmly into deep cover, good running for two."),
                (4, 0, False, "four", "CRUNCHED! One bounce over the ropes for FOUR!"),
                (6, 0, False, "six", "DISPATCHED! Huge maximum sailing deep into the stands! SIX!"),
                (0, 0, True, "bowled", "OUT! Castle knocked back! Clean bowled through the gate!"),
                (1, 1, False, "wide", "Fired down the leg side, called a wide."),
            ]
            weights = [0.34, 0.35, 0.08, 0.12, 0.05, 0.04, 0.02]

        if not is_queued_replay:
            runs_b, extras, is_wkt, w_type, commentary = random.choices(outcomes, weights=weights, k=1)[0]
            new_score = score + runs_b + extras
            new_wickets = min(10, wickets + (1 if is_wkt else 0))

        # Enrich Batting Dugout and Bowling Squad
        batting_team_id = match["team_a_id"] if team_a_batting else match["team_b_id"]
        bowling_team_id = match["team_b_id"] if team_a_batting else match["team_a_id"]

        async with db_conn.execute("""
            SELECT p.*, ms.batting_order
            FROM match_squads ms
            JOIN players p ON ms.player_id = p.id
            WHERE ms.match_id = ? AND ms.team_id = ?
            ORDER BY ms.batting_order ASC
        """, (match_id, batting_team_id)) as b_cur:
            raw_bat_squad = [dict(r) for r in await b_cur.fetchall()]

        # Check match completion or innings transition
        is_completed = False
        if not is_test:
            if inn == 1 and (new_wickets >= 10 or (overs >= max_overs and balls == 0)):
                # Innings 1 Concludes -> Record Innings 1 result
                target = new_score + 1
                inn1_ov = round(overs + (balls / 6.0), 1)
                match["innings_1_score"] = new_score
                match["innings_1_wickets"] = new_wickets
                match["innings_1_overs"] = inn1_ov
                match["target_runs"] = target
                status_text = f"Innings Break • Target: {target} runs"
                await db_conn.execute("""
                    UPDATE matches SET
                        innings_1_score = ?,
                        innings_1_wickets = ?,
                        innings_1_overs = ?,
                        target_runs = ?,
                        match_status_text = ?
                    WHERE id = ?
                """, (new_score, new_wickets, inn1_ov, target, status_text, match_id))

                if not is_queued_replay:
                    commentary = f"INNINGS BREAK • 1st Innings concluded at {new_score}/{new_wickets}. Target set: {target} runs."
                    await db_conn.execute("""
                        UPDATE matches SET
                            current_innings = 2,
                            current_score = 0,
                            current_wickets = 0,
                            current_overs = 0,
                            current_balls = 0
                        WHERE id = ?
                    """, (match_id,))
                    match["current_innings"] = 2
                    match["current_score"] = 0
                    match["current_wickets"] = 0
                    match["current_overs"] = 0
                    match["current_balls"] = 0
                    inn = 2
                    score = 0
                    wickets = 0
                    overs = 0
                    balls = 0
                    new_score = 0
                    new_wickets = 0
                    team_a_batting = not team_a_batting
                    batting_team_id = match["team_a_id"] if team_a_batting else match["team_b_id"]
                    bowling_team_id = match["team_b_id"] if team_a_batting else match["team_a_id"]
                    async with db_conn.execute("""
                        SELECT p.*, ms.batting_order
                        FROM match_squads ms
                        JOIN players p ON ms.player_id = p.id
                        WHERE ms.match_id = ? AND ms.team_id = ?
                        ORDER BY ms.batting_order ASC
                    """, (match_id, batting_team_id)) as b_cur2:
                        raw_bat_squad = [dict(r) for r in await b_cur2.fetchall()]
                    if len(raw_bat_squad) >= 2:
                        striker_id = raw_bat_squad[0]["id"]
                        non_striker_id = raw_bat_squad[1]["id"]
            elif inn == 2:
                if target and new_score >= target:
                    is_completed = True
                    commentary = f"VICTORY! Target reached at {overs}.{balls} overs!"
                elif new_wickets >= 10:
                    is_completed = True
                    commentary = f"ALL OUT! Bowled out for {new_score} in {overs}.{balls} overs."
                elif overs >= max_overs and balls == 0:
                    is_completed = True
                    commentary = f"INNINGS FINISHED! Completed 50 overs at {new_score}/{new_wickets}."
        else:
            if target and new_score >= target:
                is_completed = True
                commentary = f"VICTORY! Target reached at {overs}.{balls} overs!"
            elif new_wickets >= 10:
                if match.get("current_innings", 1) < 4:
                    commentary = f"INNINGS CLOSED! All out for {new_score} in {overs}.{balls} overs."
                else:
                    is_completed = True
                    commentary = "Match completed!"

        # Batter & Bowler rotation
        if is_wkt and not is_completed:
            bat_order_ids = [p["id"] for p in raw_bat_squad if p.get("is_confirmed_xi", 1) == 1]
            if not bat_order_ids:
                bat_order_ids = [p["id"] for p in raw_bat_squad]
            next_idx = new_wickets + 1
            if next_idx < len(bat_order_ids):
                striker_id = bat_order_ids[next_idx]
        elif runs_b % 2 == 1:
            striker_id, non_striker_id = non_striker_id, striker_id

        if balls == 0:
            striker_id, non_striker_id = non_striker_id, striker_id
            if is_test:
                bowler_id = "h_mahmud_ban" if bowler_id == "t_ahmed_ban" else "t_ahmed_ban"

        enriched_bat_squad = []
        for p in raw_bat_squad:
            pid = p["id"]
            try:
                from backend.engine.player_strengths import get_player_dossier, calculate_player_pitch_adaptability
                from backend.engine.cricpulse_master_engine import compute_22_master_attributes
                dossier = get_player_dossier(pid)
                adapt = calculate_player_pitch_adaptability(dossier)
                attrs = compute_22_master_attributes(dossier, match.get("format", "T20I"))
                p_copy = {**dossier, **p, **attrs}
                p_copy["pitch_adaptability"] = adapt.get("pitch_adaptability", {})
                p_copy["favored_pitch"] = adapt.get("favored_pitch", "flat")
                p_copy["clutch"] = float(attrs.get("clutch_bat") or 50.0)
                p_copy["b_power"] = float(attrs.get("b_power") or 50.0)
                enriched_bat_squad.append(p_copy)
            except Exception:
                enriched_bat_squad.append(p)
        bat_squad = enriched_bat_squad

        async with db_conn.execute("""
            SELECT p.*, ms.batting_order
            FROM match_squads ms
            JOIN players p ON ms.player_id = p.id
            WHERE ms.match_id = ? AND ms.team_id = ?
        """, (match_id, bowling_team_id)) as bw_cur:
            raw_bowl_squad = [dict(r) for r in await bw_cur.fetchall()]

        enriched_bowl_squad = []
        for b in raw_bowl_squad:
            b_id = b["id"]
            try:
                from backend.engine.player_strengths import get_player_dossier, calculate_player_pitch_adaptability
                from backend.engine.cricpulse_master_engine import compute_22_master_attributes
                dossier = get_player_dossier(b_id)
                adapt = calculate_player_pitch_adaptability(dossier)
                attrs = compute_22_master_attributes(dossier, match.get("format", "T20I"))
                b_copy = {**dossier, **b, **attrs}
                b_copy["pitch_adaptability"] = adapt.get("pitch_adaptability", {})
                b_copy["favored_pitch"] = adapt.get("favored_pitch", "flat")
                b_copy["clutch_death_bowl"] = float(attrs.get("clutch_death_bowl") or 50.0)
                b_copy["spearhead"] = float(attrs.get("spearhead") or 50.0)
                enriched_bowl_squad.append(b_copy)
            except Exception:
                enriched_bowl_squad.append(b)
        bowl_squad = enriched_bowl_squad

        striker = next((p for p in bat_squad if p["id"] == striker_id), None)
        if not striker:
            striker = bat_squad[0] if bat_squad else {"id": striker_id, "name": striker_id.replace("_", " ").title(), "balls_faced": 10}
        non_striker = next((p for p in bat_squad if p["id"] == non_striker_id), None)
        if not non_striker:
            non_striker = bat_squad[1] if len(bat_squad) > 1 else {"id": non_striker_id, "name": non_striker_id.replace("_", " ").title(), "balls_faced": 5}
        bowler = next((b for b in bowl_squad if b["id"] == bowler_id), None)
        if not bowler:
            bowler = bowl_squad[0] if bowl_squad else {"id": bowler_id, "name": bowler_id.replace("_", " ").title(), "overs_bowled": 1.0}

        # Dugout: Batters yet to bat
        dugout_players = [p for p in bat_squad if (p.get("batting_order") or 99) > (new_wickets + 2)]
        if not dugout_players:
            dugout_players = [{"t20_batting_avg": 24.0, "clutch": 50.0}] * max(0, 10 - new_wickets)

        # Track bowler live overs, runs, wickets, and remaining quota
        async with db_conn.execute("""
            SELECT bowler_id, COUNT(*) as balls_bowled, SUM(runs_batter + extras) as runs_conceded, SUM(is_wicket) as wickets_taken
            FROM match_deliveries
            WHERE match_id = ? AND innings = ?
            GROUP BY bowler_id
        """, (match_id, inn)) as d_cur:
            bowler_stats_map = {r["bowler_id"]: dict(r) for r in await d_cur.fetchall()}

        if bowler_id:
            if bowler_id not in bowler_stats_map:
                bowler_stats_map[bowler_id] = {"bowler_id": bowler_id, "balls_bowled": 0, "runs_conceded": 0, "wickets_taken": 0}
            bowler_stats_map[bowler_id]["balls_bowled"] += 1
            bowler_stats_map[bowler_id]["runs_conceded"] += (runs_b + extras)
            if is_wkt:
                bowler_stats_map[bowler_id]["wickets_taken"] += 1

        max_quota_overs = 10.0 if is_odi else (4.0 if not is_test else 999.0)
        bowlers_list = []
        for b in bowl_squad:
            b_id = b["id"]
            st = bowler_stats_map.get(b_id, {"balls_bowled": 0, "runs_conceded": 0, "wickets_taken": 0})
            b_balls = st["balls_bowled"]
            b_ov = b_balls // 6 + (b_balls % 6) / 10.0
            b_quota_rem = max(0.0, max_quota_overs - (b_balls / 6.0))
            live_econ = (st["runs_conceded"] / (b_balls / 6.0)) if b_balls >= 6 else float(b.get("t20_economy") or b.get("bowling_economy") or 7.8)
            b_entry = dict(b)
            b_entry["balls_bowled"] = b_balls
            b_entry["overs_bowled"] = b_ov
            b_entry["quota_remaining"] = b_quota_rem
            b_entry["runs_conceded"] = st["runs_conceded"]
            b_entry["wickets_taken"] = st["wickets_taken"]
            b_entry["live_economy"] = round(live_econ, 2)
            bowlers_list.append(b_entry)

        if not bowlers_list:
            bowlers_list = [{"t20_economy": 8.1, "quota_remaining": max(0.0, 4.0 - (overs / 4.0))}] * 4

        # Track recent deliveries (last 6 balls) for live momentum calculation
        async with db_conn.execute("""
            SELECT runs_batter, extras, is_wicket, over, ball
            FROM match_deliveries
            WHERE match_id = ? AND innings = ?
            ORDER BY id DESC LIMIT 6
        """, (match_id, inn)) as rec_cur:
            recent_rows = [dict(r) for r in await rec_cur.fetchall()]

        if recent_rows:
            r_runs = sum(r["runs_batter"] + r["extras"] for r in recent_rows)
            r_wkts = sum(r["is_wicket"] for r in recent_rows)
            r_dots = sum(1 for r in recent_rows if (r["runs_batter"] + r["extras"]) == 0 and not r["is_wicket"])
            recent_momentum_shift = round(min(5.0, max(-6.0, ((r_runs - 8.0) * 0.35) - (r_wkts * 3.5) - (r_dots * 0.25))), 1)
        else:
            recent_momentum_shift = 0.0

        # Leverage Index on current delivery:
        # A six or boundary on ball 1 heavily shifts pressure (+3.2% to +4.5%)
        # Tight death-over dot ball causes chasing equity to plummet (-3.5%)
        leverage_shift = 0.0
        if balls == 1 and (runs_b + extras) >= 4:
            leverage_shift = 4.2 if (runs_b + extras) == 6 else 3.2
        elif inn == 2 and overs >= 16 and (runs_b + extras) == 0 and not is_wkt:
            leverage_shift = -3.5

        # Run Stage 4 Engine
        res = calculate_stage4_live(
            innings=inn,
            score=new_score,
            wickets=new_wickets,
            overs=overs,
            balls=balls,
            target=target,
            striker=striker,
            non_striker=non_striker,
            bowler=bowler,
            team_a_batting=team_a_batting,
            p0_pre_match=p0,
            previous_p_team_a=prev_p,
            previous_p_draw=float(match.get("p_draw") or 0.0) if is_test else None,
            dugout_players=dugout_players,
            bowlers_list=bowlers_list,
            pitch_spin_ratio=match.get("spin_wicket_ratio", 0.62),
            match_format=match.get("format", "T20I"),
            live_pitch_type=match.get("active_pitch_type") or match.get("pitch_type", "flat"),
            is_day_night=bool(match.get("is_day_night", 0)),
            dew_factor=float(match.get("dew_factor") or 0.35),
            innings_1_score=match.get("innings_1_score"),
            innings_2_score=match.get("innings_2_score"),
            match_day=match.get("stumps_day"),
            is_stumps=bool(match.get("is_stumps")),
            recent_momentum_shift=recent_momentum_shift,
            leverage_shift=leverage_shift,
            historical_venue_par=float(match.get("first_innings_par_score") or 195.0)
        )

        # Match outcome determination
        if inn == 2 and not is_test:
            if target and new_score >= target:
                new_p_a = 100.0 if team_a_batting else 0.0
                is_completed = True
            elif new_wickets >= 10 or (overs >= max_overs and balls == 0):
                if new_score < (target or 9999):
                    new_p_a = 0.0 if team_a_batting else 100.0
                    is_completed = True
            else:
                new_p_a = res["p_team_a"]
        elif is_test:
            if target and new_score >= target:
                new_p_a = 100.0 if team_a_batting else 0.0
                is_completed = True
            elif inn == 4 and (new_wickets >= 10 or (overs >= max_overs and balls == 0)):
                new_p_a = 0.0 if team_a_batting else 100.0
                is_completed = True
            else:
                new_p_a = res["p_team_a"]
        else:
            new_p_a = res["p_team_a"]

        new_p_draw = res.get("p_draw", 0.0) if is_test else 0.0
        if is_test:
            new_p_b = round(max(0.0, 100.0 - new_p_a - new_p_draw), 1)
        else:
            new_p_b = round(100.0 - new_p_a, 1)
        wpa = round(new_p_a - prev_p, 1)

        # Mark queued delivery as played if from curated replay
        if is_queued_replay and queued_item:
            await db_conn.execute("UPDATE match_replay_queue SET is_played = 1 WHERE id = ?", (queued_item["id"],))

        # Log delivery
        await db_conn.execute("""
            INSERT INTO match_deliveries (
                match_id, innings, over, ball, striker_id, non_striker_id, bowler_id,
                runs_batter, extras, extra_type, is_wicket, wicket_type,
                score_after, wickets_after, p_team_a_after, wpa, commentary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match_id, inn, overs, balls, striker["id"], non_striker["id"], bowler["id"],
            runs_b, extras, "wide" if extras > 0 else "none", 1 if is_wkt else 0, w_type if is_wkt else "none",
            new_score, new_wickets, new_p_a, wpa, commentary
        ))

        # Update match table
        new_state = "COMPLETED" if is_completed else "LIVE_BALL_BY_BALL"
        team_a_name = match.get("team_a_id", "PAK")
        team_b_name = match.get("team_b_id", "SL")
        status_text = match.get("match_status_text", "")
        if is_completed:
            if target and new_score >= target:
                winner = team_a_name if team_a_batting else team_b_name
                status_text = f"{winner} won by {10 - new_wickets} wickets"
            elif not is_test:
                winner = team_b_name if team_a_batting else team_a_name
                runs_diff = (target - new_score - 1) if target else 0
                status_text = f"{winner} won by {runs_diff} runs"
        elif inn == 1 and (new_wickets >= 10 or (overs >= max_overs and balls == 0)):
            status_text = f"Innings Break • Target: {target} runs"
        elif inn == 2 and target:
            chasing_team = team_a_name if team_a_batting else team_b_name
            runs_needed = max(0, target - new_score)
            max_b = 300 if is_odi else 120
            balls_left = max(0, max_b - (overs * 6 + balls))
            status_text = f"{chasing_team} need {runs_needed} runs in {balls_left} balls"

        last_desc = status_text if is_completed else (commentary[:180] if len(commentary) > 180 else commentary)

        await db_conn.execute("""
            UPDATE matches SET
                state = ?,
                current_innings = ?,
                current_score = ?,
                current_wickets = ?,
                current_overs = ?,
                current_balls = ?,
                target_runs = ?,
                current_striker_id = ?,
                current_non_striker_id = ?,
                current_bowler_id = ?,
                p_team_a = ?,
                p_team_b = ?,
                p_draw = ?,
                last_wpa_swing = ?,
                last_event_description = ?,
                batting_tank_pct = ?,
                bowling_tank_pct = ?,
                par_score_projected = ?,
                par_score_min = ?,
                par_score_max = ?,
                match_status_text = ?
            WHERE id = ?
        """, (
            new_state, inn, new_score, new_wickets, overs, balls, target,
            striker_id, non_striker_id, bowler_id,
            new_p_a, new_p_b, new_p_draw, wpa, last_desc,
            res.get("batting_tank_pct", 100.0), res.get("bowling_tank_pct", 100.0),
            res.get("par_score_projected", 200), res.get("par_score_min", 180), res.get("par_score_max", 220),
            status_text,
            match_id
        ))
        await db_conn.commit()

        if is_completed:
            try:
                from ...engine.self_learning_engine import process_completed_match_learning
                await process_completed_match_learning(match_id, db_conn)
            except Exception:
                pass

        async with db_conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)) as c:
            updated_match = dict(await c.fetchone())

        delivery_info = {
            "type": "NEW_DELIVERY",
            "match": updated_match,
            "delivery": {
                "over": overs,
                "ball": balls,
                "score": f"{new_score}/{new_wickets}",
                "runs_batter": runs_b,
                "extras": extras,
                "is_wicket": is_wkt,
                "wpa": wpa,
                "p_team_a": new_p_a,
                "commentary": commentary,
                "next_ball_dist": res.get("next_ball_distribution", {}),
                "crease_duel": res.get("crease_duel", {}),
                "quality_tanks": res.get("resource_tanks", {})
            }
        }

        if self.broadcast_callback:
            await self.broadcast_callback(match_id, delivery_info)

        return delivery_info
