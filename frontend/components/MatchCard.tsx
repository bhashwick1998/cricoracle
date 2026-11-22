"use client";

import React, { useState } from "react";
import Link from "next/link";
import { getTeamMeta, formatKickoffCountdown, formatDateTime } from "@/lib/teamUtils";
import {
  Activity,
  ArrowRight,
  Sparkles,
  Zap,
  MapPin,
  Clock,
  Target,
  Trophy,
  Dices,
  Calendar,
  Compass,
  Users2,
  ChevronRight,
  Flame,
  CheckCircle2,
  Sliders,
  Moon,
  Shield,
  Coffee,
  CloudRain
} from "lucide-react";

export interface MatchData {
  id: string;
  series_name: string;
  format: string;
  team_a_id: string;
  team_b_id: string;
  venue_id: string;
  scheduled_start: string;
  is_day_night: number;
  state: "UPCOMING_NO_SQUAD" | "SQUAD_ANNOUNCED" | "XI_AND_TOSS_LOCKED" | "LIVE_BALL_BY_BALL" | "COMPLETED";
  toss_winner_id: string | null;
  toss_decision: string | null;
  current_innings: number;
  current_score: number;
  current_wickets: number;
  current_overs: number;
  current_balls: number;
  target_runs: number | null;
  current_striker_id: string | null;
  current_non_striker_id: string | null;
  current_bowler_id: string | null;
  is_completed?: boolean;
  p_team_a: number;
  p_team_b: number;
  p_draw?: number;
  p1_macro: number;
  p2_squad: number;
  p0_pre_match: number;
  par_score_projected: number;
  par_score_min: number;
  par_score_max: number;
  delta_squad_shift: number;
  delta_toss_shift: number;
  biggest_matchup_adv: string | null;
  last_wpa_swing: number;
  last_event_description: string;
  team_a_name: string;
  team_a_short: string;
  team_a_color: string;
  team_b_name: string;
  team_b_short: string;
  team_b_color: string;
  venue_name: string;
  venue_city: string;
  venue_country: string;
  spin_wicket_ratio: number;
  pitch_type: string;
  active_pitch_type?: string;
  pitch_display_name?: string;
  batting_tank_pct?: number;
  bowling_tank_pct?: number;
  avg_first_innings_score_t20: number;
  chase_win_pct: number;
  dew_factor: number;
  dew_intelligence?: any;
  ground_pitch_archetype?: string;
  team_a_country_pitch_adaptability?: number;
  team_b_country_pitch_adaptability?: number;
  innings_1_team_id?: string | null;
  innings_1_score?: number | null;
  innings_1_wickets?: number | null;
  innings_1_overs?: number | null;
  innings_2_team_id?: string | null;
  innings_2_score?: number | null;
  innings_2_wickets?: number | null;
  innings_2_overs?: number | null;
  match_status_text?: string | null;
  batting_weights?: number[];
  max_quota_per_bowler?: number;
  base_quota_per_bowler?: number;
  quota_consumption_rate_pct?: number;
  quota_summary?: string;
  is_curtailed?: boolean;
  scheduled_overs?: number;
  quality_resource_tanks?: any;
  powerplay_breakdown?: {
    summary?: string;
    p1?: any;
    p2?: any;
    p3?: any;
    format?: string;
    scheduled_overs?: number;
  };
  pitch_sweat_index?: number;
  spin_grip_penalty_pct?: number;
  outfield_friction_pct?: number;
  worm_points?: any[];
}

interface MatchCardProps {
  match: MatchData;
  onStateChange: (matchId: string, newState: string) => Promise<void>;
  onStepDelivery: (matchId: string) => Promise<void>;
  sandboxMode?: boolean;
}

export default function MatchCard({ match, onStateChange, onStepDelivery, sandboxMode = false }: MatchCardProps) {
  const [activeCardTab, setActiveCardTab] = useState<"overview" | "toss" | "squad">("overview");
  const [teamASquad, setTeamASquad] = useState<any[]>([]);
  const [teamBSquad, setTeamBSquad] = useState<any[]>([]);
  const [selectedCardSquadTeam, setSelectedCardSquadTeam] = useState<"team_a" | "team_b">("team_a");
  const [loadingSquad, setLoadingSquad] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [tossSimChoice, setTossSimChoice] = useState<"none" | "a_bowls" | "a_bats">("none");

  const metaA = getTeamMeta(match.team_a_short || match.team_a_id, match.team_a_name);
  const metaB = getTeamMeta(match.team_b_short || match.team_b_id, match.team_b_name);

  // Favored Team
  const isTeamAFavored = match.p_team_a >= match.p_team_b;
  const favoredMeta = isTeamAFavored ? metaA : metaB;
  const favoredProb = isTeamAFavored ? match.p_team_a : match.p_team_b;
  const probDiff = Math.abs(match.p_team_a - match.p_team_b);

  // Confidence
  const confidence =
    probDiff >= 15
      ? { label: "High Confidence Edge", bg: "bg-emerald-950/80 text-emerald-300 border-emerald-700/60" }
      : probDiff >= 6
      ? { label: "Moderate Edge", bg: "bg-cyan-950/80 text-cyan-300 border-cyan-700/60" }
      : { label: "50-50 Toss Up", bg: "bg-amber-950/80 text-amber-300 border-amber-700/60" };

  // Simulated toss adjustments
  const getEffectiveProbs = () => {
    const drawPct = match.format === "TEST" ? (match.p_draw || 0) : 0;
    const maxAvail = 100 - drawPct;
    if (tossSimChoice === "a_bowls") {
      const dewBoost = (match.dew_factor || 0.35) * 4.5;
      const simA = Math.min(maxAvail - 5, Math.max(5, Math.round((match.p_team_a + dewBoost) * 10) / 10));
      const simB = Math.max(0, Math.round((maxAvail - simA) * 10) / 10);
      return { pA: simA, pB: simB };
    }
    if (tossSimChoice === "a_bats") {
      const dewPenalty = (match.dew_factor || 0.35) * 4.5;
      const simA = Math.max(5, Math.min(maxAvail - 5, Math.round((match.p_team_a - dewPenalty) * 10) / 10));
      const simB = Math.max(0, Math.round((maxAvail - simA) * 10) / 10);
      return { pA: simA, pB: simB };
    }
    return { pA: match.p_team_a, pB: match.p_team_b };
  };
  const effective = getEffectiveProbs();

  const isInningsBreak =
    (match.current_innings === 2 && match.current_overs === 0 && match.current_balls === 0 && (match.target_runs || 0) > 0) ||
    (match.match_status_text || "").toLowerCase().includes("innings break");

  const isRainInterrupted =
    Boolean((match as any).weather_condition?.includes("RAIN")) ||
    (match.match_status_text || "").toLowerCase().includes("rain stoppage") ||
    (match.match_status_text || "").toLowerCase().includes("rain interruption");

  const isStumps =
    Boolean((match as any).is_stumps) ||
    (match.match_status_text || "").toLowerCase().includes("stumps");

  const isCompleted = match.state === "COMPLETED";

  // State Badge
  const getStatusBadge = () => {
    switch (match.state) {
      case "LIVE_BALL_BY_BALL":
        if (isInningsBreak) {
          return {
            pill: "☕ INNINGS BREAK",
            gear: `TARGET ${match.target_runs || (match.innings_1_score ? match.innings_1_score + 1 : 200)}`,
            bg: "bg-amber-950/90 text-amber-300 border-amber-500/80 shadow-md animate-pulse",
            dot: "bg-amber-400"
          };
        }
        if (isStumps) {
          return {
            pill: (match as any).stumps_label || "🌙 STUMPS • DAY 1",
            gear: "CLOSE OF PLAY",
            bg: "bg-indigo-950/90 text-indigo-300 border-indigo-500/80 shadow-md",
            dot: "bg-indigo-400"
          };
        }
        if (isRainInterrupted) {
          return {
            pill: "🌧️ RAIN STOPPAGE",
            gear: `DLS PAR: ${(match as any).dls_par_score || 109}`,
            bg: "bg-sky-950/90 text-sky-300 border-sky-500/80 shadow-md animate-pulse",
            dot: "bg-sky-400"
          };
        }
        if ((match as any).is_stopped) {
          return {
            pill: (match as any).stoppage_label || "⏸️ PLAY SUSPENDED",
            gear: "STOPPED",
            bg: "bg-amber-950/90 text-amber-300 border-amber-600/50 shadow-sm",
            dot: "bg-amber-400 animate-pulse"
          };
        }
        return {
          pill: "🔴 LIVE IN-PLAY",
          gear: "BALL-BY-BALL",
          bg: "bg-rose-950/80 text-rose-300 border-rose-500/80 shadow-lg shadow-rose-950/50",
          dot: "bg-rose-500 animate-ping"
        };
      case "COMPLETED":
        return {
          pill: "🏆 MATCH COMPLETED",
          gear: "FINAL RESULT",
          bg: "bg-emerald-950/80 text-emerald-300 border-emerald-500/80 shadow-sm",
          dot: "bg-emerald-400"
        };
      case "XI_AND_TOSS_LOCKED":
        return {
          pill: "🎯 PLAYING XI & TOSS LOCKED",
          gear: "GEAR 2B • P₀",
          bg: "bg-emerald-950/80 text-emerald-300 border-emerald-600/50 glow-emerald",
          dot: "bg-emerald-400"
        };
      case "SQUAD_ANNOUNCED":
        return {
          pill: "👥 SQUADS ANNOUNCED",
          gear: "GEAR 2A • SQUAD FIT",
          bg: "bg-amber-950/80 text-amber-300 border-amber-600/50",
          dot: "bg-amber-400"
        };
      case "UPCOMING_NO_SQUAD":
      default:
        return {
          pill: "⚡ GEAR 1 • STAGE 1 MACRO PRIOR",
          gear: "STAGE 1",
          bg: "bg-blue-950/90 text-cyan-300 border-cyan-700/60 shadow-sm",
          dot: "bg-cyan-400"
        };
    }
  };
  const badge = getStatusBadge();

  const { dateStr, timeStr } = formatDateTime(match.scheduled_start);
  const countdown = formatKickoffCountdown(match.scheduled_start);

  const handleFetchSquad = async () => {
    if (teamASquad.length === 0 && teamBSquad.length === 0) {
      setLoadingSquad(true);
      try {
        const res = await fetch(`/api/matches/${match.id}`);
        const data = await res.json();
        setTeamASquad(data.team_a_squad || []);
        setTeamBSquad(data.team_b_squad || []);
      } catch (err) {
        console.error("Failed to load squad drawer:", err);
      } finally {
        setLoadingSquad(false);
      }
    }
  };

  const handleStateClick = async (newState: string) => {
    setActionLoading(true);
    try {
      await onStateChange(match.id, newState);
      if (activeCardTab === "squad") {
        const res = await fetch(`/api/matches/${match.id}`);
        const data = await res.json();
        setTeamASquad(data.team_a_squad || []);
        setTeamBSquad(data.team_b_squad || []);
      }
    } finally {
      setActionLoading(false);
    }
  };

  const handleBallClick = async () => {
    setActionLoading(true);
    try {
      await onStepDelivery(match.id);
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="rounded-3xl glass-card overflow-hidden hover:border-cyan-500/40 transition-all duration-300 flex flex-col justify-between shadow-2xl relative group border border-slate-800 bg-[#0c1222]/90">
      {/* Top Banner: State Badge & Scheduled Time */}
      <div className="p-4 sm:p-5 pb-3">
        <div className="flex items-center justify-between gap-2 mb-3 flex-wrap">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 sm:px-3 py-1 rounded-full text-[10px] font-black tracking-wider uppercase border ${badge.bg}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${badge.dot}`}></span>
              {badge.pill}
            </span>
            <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded-md bg-slate-800 text-slate-300 border border-slate-700/70">
              {match.format}
            </span>
            {((match as any).is_cricdata_sourced || (match as any).data_source === "CricketData.org" || (match as any).cricapi_id) && (
              <span
                className="text-[9px] font-mono font-bold px-2 py-0.5 rounded-md uppercase flex items-center gap-1 bg-emerald-950/80 text-emerald-300 border border-emerald-600/70 shadow-sm"
                title="Official fixture and playing XI sourced from CricketData.org API"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                CricketData.org
              </span>
            )}
            {(match.active_pitch_type || match.ground_pitch_archetype || match.pitch_type) && (
              <span
                className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-md uppercase flex items-center gap-1 border shadow-sm ${
                  (match as any).is_pitch_override
                    ? "bg-emerald-950 text-emerald-300 border-emerald-600/80 shadow-[0_0_8px_rgba(16,185,129,0.2)]"
                    : (match as any).pitch_detection_source === "NLP_PITCH_REPORT"
                    ? "bg-cyan-950 text-cyan-300 border-cyan-700/80"
                    : "bg-amber-950/80 text-amber-300 border-amber-800/80"
                }`}
                title={match.pitch_display_name || (match as any).pitch_override_reason || `Using ${match.active_pitch_type || match.ground_pitch_archetype || match.pitch_type} pitch archetype`}
              >
                <span className="text-[9px] text-cyan-400 font-black">⚡ PITCH:</span>
                <span className="underline decoration-cyan-400/60 font-black">
                  {match.active_pitch_type || match.ground_pitch_archetype || match.pitch_type}
                </span>
              </span>
            )}

            {((match as any).dew_intelligence?.active_dew_advantage || ((match as any).is_day_night && (match.dew_factor || 0) >= 0.30 && match.format !== 'TEST')) && (
              <span
                className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-md uppercase flex items-center gap-1 border bg-blue-950/90 text-blue-300 border-blue-600/80 shadow-[0_0_8px_rgba(59,130,246,0.3)]"
                title={(match as any).dew_intelligence?.tactical_assessment || `Live Dew Impact: +${(((match as any).dew_intelligence?.chase_equity_boost_pct || (match.dew_factor || 0.30) * 8.5)).toFixed(1)}% Chasing Advantage under lights`}
              >
                <span className="text-blue-400">💧</span>
                <span>{(match as any).dew_intelligence?.status_badge || "DEW ACTIVE"}</span>
                <span className="text-blue-200">
                  (+{(((match as any).dew_intelligence?.chase_equity_boost_pct || (match.dew_factor || 0.30) * 8.5)).toFixed(1)}%)
                </span>
              </span>
            )}
          </div>

          <div className="text-right shrink-0">
            <span className="text-[11px] text-slate-400 font-medium flex items-center gap-1 justify-end">
              <MapPin className="w-3.5 h-3.5 text-cyan-400" />
              {match.venue_city || match.venue_name}
            </span>
          </div>
        </div>

        {/* Series Title & Countdown */}
        <div className="flex items-center justify-between text-xs text-slate-400 mb-3.5 gap-2">
          <span className="truncate font-semibold text-slate-300">{match.series_name}</span>
          <span className="shrink-0 flex items-center gap-1 text-[10px] font-medium text-cyan-300 bg-slate-900/90 px-2 py-0.5 rounded-md border border-slate-800">
            <Calendar className="w-3 h-3 text-cyan-400" />
            {dateStr} {match.state !== "COMPLETED" && match.state !== "LIVE_BALL_BY_BALL" && countdown && countdown !== "In Progress" ? `(${countdown})` : ""}
          </span>
        </div>

        {/* Head to Head Matchup Scoreboard Box */}
        <div className="p-3 sm:p-4 rounded-2xl bg-slate-900/90 border border-slate-800/80 mb-3.5 shadow-inner">
          <div className="grid grid-cols-2 gap-2 sm:gap-3 items-center">
            {/* Team A */}
            <div className="flex items-center gap-2 sm:gap-2.5">
              <div
                className={`w-9 h-9 sm:w-11 sm:h-11 rounded-xl flex items-center justify-center text-lg sm:text-xl shadow-md border ${metaA.borderGlow} bg-gradient-to-br ${metaA.gradient} shrink-0`}
              >
                {metaA.flag}
              </div>
              <div className="overflow-hidden">
                <div className="text-xs sm:text-sm font-black text-white truncate leading-tight">
                  {metaA.name}
                </div>
                <div className="text-base sm:text-lg font-black text-cyan-400 font-mono-numbers mt-0.5">
                  {effective.pA}%
                </div>
              </div>
            </div>

            {/* Team B */}
            <div className="flex items-center gap-2 sm:gap-2.5 justify-end text-right">
              <div className="overflow-hidden">
                <div className="text-xs sm:text-sm font-black text-white truncate leading-tight">
                  {metaB.name}
                </div>
                <div className="text-base sm:text-lg font-black text-amber-400 font-mono-numbers mt-0.5">
                  {effective.pB}%
                </div>
              </div>
              <div
                className={`w-9 h-9 sm:w-11 sm:h-11 rounded-xl flex items-center justify-center text-lg sm:text-xl shadow-md border ${metaB.borderGlow} bg-gradient-to-br ${metaB.gradient} shrink-0`}
              >
                {metaB.flag}
              </div>
            </div>
          </div>

          {/* WinViz Bar */}
          <div className="mt-3 space-y-1">
            <div className="h-2.5 w-full bg-slate-950 rounded-full overflow-hidden flex p-[1px] shadow-inner border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-blue-600 to-cyan-400 rounded-l-full transition-all duration-500 ease-out"
                style={{ width: `${effective.pA}%` }}
              />
              {match.format === "TEST" && (match.p_draw || 0) > 0 && (
                <div
                  className="h-full bg-slate-500 transition-all duration-500 ease-out"
                  style={{ width: `${match.p_draw}%` }}
                />
              )}
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-rose-500 rounded-r-full transition-all duration-500 ease-out"
                style={{ width: `${effective.pB}%` }}
              />
            </div>
            <div className="flex justify-between items-center text-[10px] text-slate-400 px-1 font-mono-numbers font-bold">
              <span className="text-cyan-400">{metaA.short} {effective.pA}%</span>
              {match.format === "TEST" && (match.p_draw || 0) > 0 ? (
                <span className="text-slate-300 font-sans text-[9px] uppercase font-bold tracking-wider">
                  Draw {match.p_draw}%
                </span>
              ) : (
                <span className="text-slate-400 font-sans text-[9px] uppercase font-semibold">
                  {favoredMeta.short} Favored
                </span>
              )}
              <span className="text-amber-400">{effective.pB}% {metaB.short}</span>
            </div>
          </div>
        </div>

        {/* Stage 3 (XI & Toss Locked P₀) and Stage 4 (Live In-Play / Final Result) Probabilities Only */}
        <div className="grid grid-cols-2 gap-2 p-2 rounded-xl bg-slate-950/70 border border-slate-800/80 mb-3 text-center font-mono-numbers">
          <div className="p-2 rounded-lg bg-slate-900/80 border border-emerald-900/40 min-w-0 overflow-hidden flex flex-col justify-between" title="Stage 3: XI & Toss Locked Pre-Match Anchor (P₀)">
            <span className="text-[9px] text-emerald-400 font-bold block uppercase tracking-wider truncate mb-0.5">
              Stage 3 Locked (P₀)
            </span>
            <div className="flex items-center justify-center gap-1 text-[11px] sm:text-xs font-black text-white flex-wrap leading-tight my-0.5">
              <span className="text-cyan-400 font-bold">{metaA.short} {match.p0_pre_match != null ? match.p0_pre_match : match.p_team_a}%</span>
              <span className="text-[10px] text-slate-500 font-normal">vs</span>
              <span className="text-amber-400 font-bold">{Math.round((100 - (match.p0_pre_match != null ? match.p0_pre_match : match.p_team_a)) * 10) / 10}% {metaB.short}</span>
            </div>
            <span className="text-[9px] text-slate-400 block truncate mt-0.5">
              Toss Shift: {match.delta_toss_shift ? (match.delta_toss_shift > 0 ? `+${match.delta_toss_shift}%` : `${match.delta_toss_shift}%`) : "0%"}
            </span>
          </div>

          <div className={`p-2 rounded-lg border min-w-0 overflow-hidden flex flex-col justify-between ${
            match.state === "LIVE_BALL_BY_BALL"
              ? "bg-rose-950/30 border-rose-800/60 shadow-inner"
              : match.state === "COMPLETED"
              ? "bg-emerald-950/30 border-emerald-800/60"
              : "bg-slate-900/80 border-slate-800/60"
          }`} title="Stage 4: Live Ball-by-Ball Dynamic Win Probability">
            <span className={`text-[9px] font-bold block uppercase tracking-wider truncate flex items-center justify-center gap-1 mb-0.5 ${
              match.state === "LIVE_BALL_BY_BALL" ? "text-rose-400" : match.state === "COMPLETED" ? "text-emerald-400" : "text-cyan-400"
            }`}>
              {match.state === "LIVE_BALL_BY_BALL" && (
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-ping shrink-0 inline-block" />
              )}
              <span className="truncate">
                {match.state === "COMPLETED" ? "Stage 4 Result" : match.state === "LIVE_BALL_BY_BALL" ? "Stage 4 Live" : "Stage 4 In-Play"}
              </span>
            </span>
            <div className="flex items-center justify-center gap-1 text-[11px] sm:text-xs font-black text-white flex-wrap leading-tight my-0.5">
              <span className="text-cyan-400 font-bold">{metaA.short} {match.p_team_a}%</span>
              <span className="text-[10px] text-slate-500 font-normal">vs</span>
              <span className="text-amber-400 font-bold">{match.p_team_b}% {metaB.short}</span>
            </div>
            {match.format === "TEST" && (match.p_draw || 0) > 0 && (
              <div className="text-[10px] font-bold text-slate-300 leading-tight">
                Draw <span className="text-slate-200 font-mono-numbers">{match.p_draw}%</span>
              </div>
            )}
            <span className="text-[9px] text-slate-400 block truncate mt-0.5">
              {match.state === "COMPLETED" ? "Match Finalized" : match.state === "LIVE_BALL_BY_BALL" ? "Telemetry In-Play" : "Pre-Match Standby"}
            </span>
          </div>
        </div>

        {/* Country Pitch Adaptability Strip */}
        {match.team_a_country_pitch_adaptability && match.team_b_country_pitch_adaptability && (
          <div className="flex items-center justify-between text-[11px] text-slate-300 px-3 py-1.5 mb-3 bg-slate-950/60 rounded-xl border border-slate-800 font-mono-numbers">
            <span className="text-cyan-400 font-bold">{metaA.short} Adapt: {match.team_a_country_pitch_adaptability}%</span>
            <span className="text-[9px] text-slate-400 uppercase font-sans flex items-center gap-1">
              Deck: <span className={`${(match as any).is_pitch_override ? "text-emerald-300 font-black" : "text-amber-300 font-bold"}`}>{match.ground_pitch_archetype || "Flat"}</span>
              {(match as any).is_pitch_override ? (
                <span className="text-[8px] px-1 py-0.2 rounded bg-emerald-950 text-emerald-300 border border-emerald-700/80 font-mono font-bold" title={(match as any).pitch_override_reason}>
                  REAL NLP DECK
                </span>
              ) : (
                <span className="text-[8px] px-1 py-0.2 rounded bg-slate-800 text-cyan-300 font-mono">65% 1-Yr</span>
              )}
            </span>
            <span className="text-amber-400 font-bold">{metaB.short} Adapt: {match.team_b_country_pitch_adaptability}%</span>
          </div>
        )}

        {/* Live Score Strip if State 4 or Completed */}
        {(match.state === "LIVE_BALL_BY_BALL" || match.state === "COMPLETED") && (() => {
          const battingTeam = (match as any).batting_team_id === match.team_b_id ? metaB : metaA;
          const bowlingTeam = battingTeam === metaA ? metaB : metaA;
          const isStumps = Boolean((match as any).is_stumps);
          const isStopped = Boolean((match as any).is_stopped);
          const isCompleted = match.state === "COMPLETED";
          const isLive = match.state === "LIVE_BALL_BY_BALL";
          const matchStatusText = (match as any).match_status_text;
          const hasInn1 = (match as any).innings_1_score != null;
          const inn1Team = (match as any).innings_1_team_id === match.team_b_id ? metaB : metaA;

          return (
            <div className={`mb-3.5 p-3 rounded-2xl border flex items-center justify-between ${
              isCompleted
                ? "bg-emerald-950/40 border-emerald-700/60 shadow-sm"
                : isInningsBreak
                ? "bg-amber-950/60 border-amber-500/80 shadow-md"
                : isRainInterrupted
                ? "bg-sky-950/60 border-sky-500/80 shadow-md"
                : isStumps
                ? "bg-indigo-950/50 border-indigo-700/60 shadow-sm"
                : isStopped
                ? "bg-amber-950/50 border-amber-700/60 shadow-sm"
                : "bg-rose-950/40 border-rose-800/40"
            }`}>
              <div>
                <div className={`text-[10px] font-black uppercase tracking-wider flex items-center gap-1 ${
                  isCompleted
                    ? "text-emerald-300"
                    : isInningsBreak
                    ? "text-amber-300"
                    : isRainInterrupted
                    ? "text-sky-300"
                    : isStumps
                    ? "text-indigo-300"
                    : isStopped
                    ? "text-amber-300"
                    : "text-rose-400"
                }`}>
                  {isCompleted ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : isInningsBreak ? (
                    <Coffee className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
                  ) : isRainInterrupted ? (
                    <CloudRain className="w-3.5 h-3.5 text-sky-400 animate-pulse" />
                  ) : isStumps ? (
                    <Moon className="w-3.5 h-3.5 text-indigo-400" />
                  ) : isStopped ? (
                    <Clock className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
                  ) : (
                    <Flame className="w-3.5 h-3.5 animate-bounce" />
                  )}
                  {isCompleted
                    ? "MATCH COMPLETED • RESULT"
                    : isInningsBreak
                    ? `☕ INNINGS BREAK • TARGET ${match.target_runs || (match.innings_1_score ? match.innings_1_score + 1 : 200)}`
                    : isRainInterrupted
                    ? `🌧️ RAIN STOPPAGE • DLS PAR ${(match as any).dls_par_score || 109}`
                    : matchStatusText
                    ? matchStatusText
                    : isStumps
                    ? ((match as any).stumps_label || "STUMPS")
                    : isStopped
                    ? ((match as any).stoppage_label || "PLAY STOPPED")
                    : `Live In-Play • ${match.format === "TEST" ? `Innings ${match.current_innings}` : `Innings ${match.current_innings}`}`}
                </div>
                {isCompleted ? (
                  <div className="text-sm font-extrabold text-white mt-1">
                    {(match as any).result_description || match.last_event_description || "Match Completed"}
                  </div>
                ) : isInningsBreak ? (
                  <div className="mt-1">
                    <div className="text-sm font-extrabold text-amber-300">
                      Target: {match.target_runs} runs for {metaA.name}
                    </div>
                    <div className="text-xs text-slate-300 font-mono-numbers mt-0.5">
                      {(() => {
                        const target = Number(match.target_runs || 0);
                        const curScore = Number(match.current_score || 0);
                        const runsNeeded = Math.max(0, target - curScore);
                        const maxOv = match.format === "ODI" ? 50 : 20;
                        const totalBalls = maxOv * 6;
                        const ballsBowled = match.current_innings === 2 ? (Number(match.current_overs || 0) * 6 + Number(match.current_balls || 0)) : 0;
                        const ballsRem = Math.max(1, totalBalls - ballsBowled);
                        const reqRR = (runsNeeded / (ballsRem / 6)).toFixed(2);
                        return (
                          <span>
                            1st Inn: {(match as any).innings_1_score}/{(match as any).innings_1_wickets} ({(match as any).innings_1_overs} ov) • Req RR: {reqRR} RPO
                          </span>
                        );
                      })()}
                    </div>
                  </div>
                ) : isRainInterrupted ? (
                  <div className="mt-1">
                    <div className="text-sm font-extrabold text-sky-200">
                      Play Suspended: Rain Interruption (Covers On)
                    </div>
                    <div className="text-xs text-slate-300 font-mono-numbers mt-0.5 flex items-center gap-2">
                      <span>Current: {match.current_score}/{match.current_wickets} ({match.current_overs} ov)</span>
                      <span className="text-emerald-400 font-bold">
                        DLS Par: {(match as any).dls_par_score || 109} ({Number(match.current_score) - Number((match as any).dls_par_score || 109) >= 0
                          ? `+${Number(match.current_score) - Number((match as any).dls_par_score || 109)} Ahead`
                          : `${Number(match.current_score) - Number((match as any).dls_par_score || 109)} Behind`})
                      </span>
                    </div>
                  </div>
                ) : hasInn1 && match.current_innings >= 2 ? (
                  <div className="space-y-0.5 mt-1">
                    <div className="text-xs font-semibold text-slate-300 font-mono-numbers flex items-center gap-1.5 flex-wrap">
                      <span>{inn1Team.flag}</span>
                      <span className="text-slate-400 font-bold">{inn1Team.name}{match.format === "TEST" ? " (1st Inn):" : ":"}</span>
                      <span className="text-white font-bold">
                        {(match as any).innings_1_score}{(match as any).innings_1_wickets != null ? `/${(match as any).innings_1_wickets}` : ""}
                        {((match as any).innings_1_declared || (match as any).is_declared) ? " d" : ""}
                      </span>
                      {(match as any).innings_1_overs ? (
                        <span className="text-[11px] text-slate-400 font-normal">
                          ({(match as any).innings_1_overs} ov)
                        </span>
                      ) : null}
                    </div>
                    {(match as any).innings_2_score != null && match.current_innings >= 3 && (() => {
                      const inn2Team = inn1Team === metaA ? metaB : metaA;
                      return (
                        <div className="text-xs font-semibold text-slate-300 font-mono-numbers flex items-center gap-1.5 flex-wrap">
                          <span>{inn2Team.flag}</span>
                          <span className="text-slate-400 font-bold">{inn2Team.name}{match.format === "TEST" ? " (1st Inn):" : ":"}</span>
                          <span className="text-white font-bold">
                            {(match as any).innings_2_score}{(match as any).innings_2_wickets != null ? `/${(match as any).innings_2_wickets}` : ""}
                            {(match as any).innings_2_declared ? " d" : ""}
                          </span>
                          {(match as any).innings_2_overs ? (
                            <span className="text-[11px] text-slate-400 font-normal">
                              ({(match as any).innings_2_overs} ov)
                            </span>
                          ) : null}
                        </div>
                      );
                    })()}
                    <div className="text-lg font-black text-white font-mono-numbers flex items-center gap-1.5 flex-wrap">
                      <span>{battingTeam.flag}</span>
                      <span className="text-cyan-300 font-bold">
                        {battingTeam.name}
                        {match.format === "TEST"
                          ? match.current_innings <= 2
                            ? " (1st Inn):"
                            : " (2nd Inn):"
                          : ":"}
                      </span>
                      <span className="text-white">{match.current_score}/{match.current_wickets}</span>
                      <span className="text-xs text-slate-400 font-normal">
                        ({match.current_overs}.{match.current_balls} ov)
                      </span>
                    </div>
                    {match.format === "TEST" && match.current_innings === 2 && (match as any).follow_on_threshold && (
                      <div className="text-[10px] text-amber-300 font-mono flex items-center gap-1">
                        <span>🛡️ Target to avoid follow-on: <strong>{(match as any).follow_on_threshold}</strong></span>
                        <span className="text-slate-400">• Need {Math.max(0, Number((match as any).follow_on_threshold) - Number(match.current_score))} to avoid</span>
                      </div>
                    )}
                    {match.format === "TEST" && ((match as any).min_overs_remaining !== undefined || (match as any).overs_left_today !== undefined) && (
                      <div className="text-[10px] text-amber-300 font-mono flex items-center gap-1.5 flex-wrap">
                        <span>• Min. Ov. Rem: <strong>{isStumps ? 0 : ((match as any).min_overs_remaining ?? (match as any).overs_left_today)}</strong> ov</span>
                        {(match as any).total_match_overs_remaining ? (
                          <span className="text-slate-400">({(match as any).total_match_overs_remaining} total match ov)</span>
                        ) : null}
                      </div>
                    )}
                    <div className="text-[10px] text-slate-400 font-mono flex items-center gap-1">
                      <span>vs</span>
                      <span>{bowlingTeam.flag}</span>
                      <span className="text-slate-300 font-semibold">{bowlingTeam.name}</span>
                      <span>Bowling</span>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="text-lg font-black text-white font-mono-numbers mt-0.5 flex items-center gap-1.5 flex-wrap">
                      <span>{battingTeam.flag}</span>
                      <span className="text-cyan-300 font-bold">{battingTeam.name}:</span>
                      <span className="text-white">{match.current_score}/{match.current_wickets}</span>
                      <span className="text-xs text-slate-400 font-normal">
                        ({match.current_overs}.{match.current_balls} ov)
                      </span>
                    </div>
                    <div className="text-[10px] text-slate-400 font-mono mt-0.5 flex items-center gap-1">
                      <span>vs</span>
                      <span>{bowlingTeam.flag}</span>
                      <span className="text-slate-300 font-semibold">{bowlingTeam.name}</span>
                      <span>Bowling</span>
                    </div>
                  </>
                )}
              </div>

              <div className="text-right">
                {isCompleted ? (
                  <span className="px-2.5 py-1 rounded-lg bg-emerald-900/80 border border-emerald-500/50 text-emerald-200 text-xs font-black uppercase">
                    Ended
                  </span>
                ) : isInningsBreak ? (
                  <div className="text-right">
                    <span className="px-2.5 py-1 rounded-lg bg-amber-900/80 border border-amber-500/50 text-amber-200 text-xs font-black uppercase inline-block">
                      Break
                    </span>
                    <div className="text-xs text-amber-300 font-bold font-mono-numbers mt-1">
                      Target: {match.target_runs}
                    </div>
                  </div>
                ) : isRainInterrupted ? (
                  <div className="text-right">
                    <span className="px-2.5 py-1 rounded-lg bg-sky-900/80 border border-sky-500/50 text-sky-200 text-xs font-black uppercase inline-block">
                      Rain Delay
                    </span>
                    <div className="text-xs text-cyan-300 font-bold font-mono-numbers mt-1">
                      DLS Par: {(match as any).dls_par_score || 109}
                    </div>
                  </div>
                ) : isStumps ? (
                  <>
                    <div className="text-[10px] text-indigo-300 font-semibold uppercase tracking-wider">Day Close</div>
                    <div className="text-xs text-amber-300 font-semibold font-mono-numbers">
                      {match.format === "TEST" && match.current_innings === 2 && (match as any).innings_1_score ? (() => {
                        const curBat = (match as any).batting_team_id === match.team_b_id ? metaB : metaA;
                        const s1 = Number((match as any).innings_1_score || 0);
                        const s2 = Number(match.current_score || 0);
                        const diff = s2 - s1;
                        const leadTrail = diff >= 0 ? `${curBat.short} lead by ${diff}` : `${curBat.short} trail by ${Math.abs(diff)}`;
                        const dayLabel = (match as any).stumps_label || (match.match_status_text?.includes("Stumps") ? match.match_status_text.split("-")[0].trim() : "Day Close");
                        const projScore = Math.max(s2, Number(match.par_score_projected || 0) || Math.round(s2 + Math.max(0, 10 - Number(match.current_wickets || 0)) * 25));
                        return `${dayLabel} • ${leadTrail} • Proj: ~${projScore}`;
                      })() : `Overnight Expected: ~${match.par_score_projected || 290}`}
                    </div>
                  </>
                ) : isStopped ? (
                  <>
                    <div className="text-[10px] text-amber-300 font-semibold uppercase tracking-wider">Interrupted</div>
                    <div className="text-xs text-amber-200 font-semibold font-mono-numbers">
                      Par: ~{match.par_score_projected || 250}
                    </div>
                  </>
                ) : match.format === "TEST" ? (
                  <div className="text-xs text-amber-300 font-semibold font-mono-numbers">
                    {match.current_innings === 3 && (match as any).innings_1_score && (match as any).innings_2_score
                      ? (() => {
                          const s1 = Number((match as any).innings_1_score || 0);
                          const s2 = Number((match as any).innings_2_score || 0);
                          const curBattingId = (match as any).batting_team_id;
                          const inn1TeamId = (match as any).innings_1_team_id || match.team_a_id;
                          const isFollowOn = curBattingId && curBattingId !== inn1TeamId;
                          if (isFollowOn) {
                            const total = s2 + Number(match.current_score || 0);
                            const def = s1 - total;
                            return def > 0 ? `Trail by ${def} runs (F/O)` : `Lead by ${Math.abs(def)} runs`;
                          } else {
                            const total = s1 + Number(match.current_score || 0);
                            const def = total - s2;
                            return def >= 0 ? `Lead by ${def} runs` : `Trail by ${Math.abs(def)} runs`;
                          }
                        })()
                      : match.current_innings === 2 && (match as any).innings_1_score ? (() => {
                        const curBat = (match as any).batting_team_id === match.team_b_id ? metaB : metaA;
                        const s1 = Number((match as any).innings_1_score || 0);
                        const s2 = Number(match.current_score || 0);
                        const diff = s2 - s1;
                        return diff >= 0 ? `${curBat.short} lead by ${diff} runs` : `${curBat.short} trail by ${Math.abs(diff)} runs`;
                      })()
                      : `Par 1st Inn: ~${match.par_score_projected || 250}`}
                  </div>
                ) : (match as any).dls_par_score ? (
                  <div className="text-xs text-cyan-300 font-semibold font-mono-numbers">
                    DLS Par: {(match as any).dls_par_score}
                  </div>
                ) : (match.current_innings > 1 && match.target_runs) ? (
                  <div className="text-xs text-slate-300">
                    Target: <span className="font-bold text-white font-mono-numbers">{match.target_runs}</span>
                  </div>
                ) : (
                  <div className="text-xs text-slate-400 font-mono-numbers">
                    1st Inn Par: ~{match.par_score_projected || 170}
                  </div>
                )}
                {isLive && match.last_wpa_swing !== 0 && !isCompleted && (
                  <div
                    className={`text-[10px] font-black font-mono-numbers mt-0.5 ${
                      match.last_wpa_swing > 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    WPA: {match.last_wpa_swing > 0 ? "+" : ""}
                    {Number(match.last_wpa_swing).toFixed(1)}%
                  </div>
                )}
              </div>
            </div>
          );
        })()}

        {/* 3 Interactive Tabs */}
        <div className="border-t border-slate-800/80 pt-3">
          <div className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800 text-[11px] font-bold mb-3">
            <button
              onClick={() => setActiveCardTab("overview")}
              className={`flex-1 py-1 rounded-lg transition flex items-center justify-center gap-1 ${
                activeCardTab === "overview"
                  ? "bg-cyan-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Target className="w-3 h-3" />
              Insights
            </button>
            <button
              onClick={() => setActiveCardTab("toss")}
              className={`flex-1 py-1 rounded-lg transition flex items-center justify-center gap-1 ${
                activeCardTab === "toss"
                  ? "bg-cyan-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Dices className="w-3 h-3" />
              Toss Sim
            </button>
            <button
              onClick={() => {
                setActiveCardTab("squad");
                handleFetchSquad();
              }}
              className={`flex-1 py-1 rounded-lg transition flex items-center justify-center gap-1 ${
                activeCardTab === "squad"
                  ? "bg-cyan-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Users2 className="w-3 h-3" />
              Squads
            </button>
          </div>

          {/* TAB 1: OVERVIEW & INSIGHTS */}
          {activeCardTab === "overview" && (
            <div className="space-y-2.5 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  {match.state === "COMPLETED" ? (
                    <>
                      <span className="text-[10px] uppercase font-bold text-emerald-400 block mb-0.5">
                        Match Result
                      </span>
                      <div className="text-xs font-black text-white truncate">
                        {match.match_status_text || (match as any).result_description || "Match Completed"}
                      </div>
                      <div className="text-[9px] text-emerald-400 mt-0.5">
                        Official Result Recorded
                      </div>
                    </>
                  ) : match.state === "LIVE_BALL_BY_BALL" ? (() => {
                    const battingTeam = (match as any).batting_team_id === match.team_b_id ? metaB : metaA;
                    const bowlingTeam = battingTeam === metaA ? metaB : metaA;
                    const liveProj = (match as any).scenario_score_projections?.live_projection;
                    const testProg = (match as any).scenario_score_projections?.live_test_innings_progression;
                    const inn3Prog = testProg?.innings_3;
                    const activeInnProg = testProg ? (testProg[`innings_${match.current_innings}`] || testProg.innings_2 || testProg.innings_3) : undefined;

                    const rawProj = activeInnProg?.live_projected_total || match.par_score_projected || liveProj?.live_projected_final_score ||
                      (match.format === "TEST" ? (Number(match.current_score || 0) + Math.max(0, 10 - Number(match.current_wickets || 0)) * 25) : Math.round(Number(match.current_score || 0) + Math.max(0, 10 - Number(match.current_wickets || 0)) * 11));
                    const projectedTotal = Math.max(Number(match.current_score || 0), rawProj);
                    const projRange = activeInnProg?.range || (match.par_score_min && match.par_score_max ? `${match.par_score_min} - ${match.par_score_max}` : (liveProj?.live_probable_range || "511 - 586"));

                    const isChase2ndInn = match.format !== "TEST" && match.current_innings === 2;

                    return (
                      <>
                        <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5">
                          {match.format === "TEST" 
                            ? (match.current_innings === 2 ? `${battingTeam.short} 1st Innings Forecast` : (match.current_innings === 3 ? `${battingTeam.short} 2nd Innings Forecast` : (match.current_innings === 4 ? `${battingTeam.short} 4th Innings Chase` : "Live 1st Innings Forecast")))
                            : (isChase2ndInn ? "Live Win Probability" : "Live 1st Inn Forecast")}
                        </span>
                        <div className="text-xs font-black text-white font-mono-numbers">
                          {isChase2ndInn ? (
                            <>
                              <span className={battingTeam === metaA ? "text-cyan-300" : "text-amber-300"}>
                                {battingTeam.short} {battingTeam === metaA ? match.p_team_a : match.p_team_b}%
                              </span>
                              <span className="text-slate-500 mx-1.5">|</span>
                              <span className={bowlingTeam === metaA ? "text-cyan-300" : "text-amber-300"}>
                                {bowlingTeam.short} {bowlingTeam === metaA ? match.p_team_a : match.p_team_b}%
                              </span>
                            </>
                          ) : (
                            <>
                              <span className={battingTeam === metaA ? "text-cyan-300" : "text-amber-300"}>
                                {battingTeam.short}: ~{projectedTotal} runs
                              </span>
                              <span className="text-[9px] text-slate-400 font-normal ml-1">
                                ({projRange})
                              </span>
                            </>
                          )}
                        </div>
                        <div className="text-[9px] text-slate-400 mt-0.5">
                          {match.format === "TEST" ? (
                            match.current_innings === 3 ? (
                              <span>
                                {battingTeam.short} 2nd Inn: {match.current_score}/{match.current_wickets} •{" "}
                                {Number(inn3Prog?.overall_match_deficit || 0) > 0 || (match.match_status_text || "").toLowerCase().includes("trail")
                                  ? `Trail by ${inn3Prog?.overall_match_deficit || 233} runs • ${bowlingTeam.short} yet to play`
                                  : `Lead by ${inn3Prog?.overall_match_lead || 45} runs • Setting Target`}
                              </span>
                            ) : match.current_innings === 2 && (match as any).innings_1_score ? (() => {
                              const curBat = (match as any).batting_team_id === match.team_b_id ? metaB : metaA;
                              const inn1Team = (match as any).innings_1_team_id === match.team_b_id ? metaB : metaA;
                              const s1 = Number((match as any).innings_1_score || 0);
                              const s2 = Number(match.current_score || 0);
                              const diff = s2 - s1;
                              const leadTrail = diff >= 0 ? `${curBat.short} lead by ${diff} runs` : `${curBat.short} trail by ${Math.abs(diff)} runs`;
                              return (
                                <span>{inn1Team.short} 1st Inn: {s1}/{(match as any).innings_1_wickets || 10}{(match as any).is_declared || (match as any).innings_1_declared ? "d" : ""} • {leadTrail}</span>
                              );
                            })() : (
                              <span>Setting Target • 4th Innings</span>
                            )
                          ) : isChase2ndInn ? (
                            <span>Target: {match.target_runs || 0} • Need {Math.max(0, (match.target_runs || 0) - match.current_score)} off {Math.max(0, (match.format === "T20I" ? 120 : 300) - (Number(match.current_overs) * 6 + Number(match.current_balls)))} balls • CRR: {(Number(match.current_score) / Math.max(1, Number(match.current_overs) + Number(match.current_balls) / 6)).toFixed(2)}</span>
                          ) : (
                            <span>Par Score: ~{match.par_score_projected || 175}</span>
                          )}
                        </div>
                      </>
                    );
                  })() : (
                    <>
                      <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5">
                        Probable 1st Innings Par
                      </span>
                      <div className="text-xs font-black text-white font-mono-numbers">
                        <span className="text-cyan-300">
                          {metaA.short}: ~{(match as any).scenario_score_projections?.scenario_team_1_bats_first?.innings_1_par || (match as any).scenario_score_projections?.scenario_team_1_bats_first?.projected_1st_innings_score || (match as any).scenario_score_projections?.scenario_team_1_bats_first?.probable_score || match.par_score_projected}
                        </span>
                        <span className="text-slate-500 mx-1">|</span>
                        <span className="text-amber-300">
                          {metaB.short}: ~{(match as any).scenario_score_projections?.scenario_team_2_bats_first?.innings_1_par || (match as any).scenario_score_projections?.scenario_team_2_bats_first?.projected_1st_innings_score || (match as any).scenario_score_projections?.scenario_team_2_bats_first?.probable_score || match.par_score_projected}
                        </span>
                      </div>
                      <div className="text-[9px] text-slate-400 mt-0.5">
                        Stage: {(match as any).scenario_score_projections?.stage_name?.split("(")[0] || (match.state === "SQUAD_ANNOUNCED" ? "Squad Announced" : "Pre-Squad Scheduled")}
                      </div>
                    </>
                  )}
                </div>

                <div className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5">
                    Venue Pitch Split
                  </span>
                  <div className="text-xs font-bold text-cyan-300 capitalize truncate">
                    {match.pitch_display_name || `${(match.active_pitch_type || match.ground_pitch_archetype || match.pitch_type)?.replace("_", " ")} Deck`}
                  </div>
                  <div className="text-[10px] text-slate-400 font-mono-numbers">
                    {Math.round((match.spin_wicket_ratio || 0.38) * 100)}% Spin • {Math.round((1 - (match.spin_wicket_ratio || 0.38)) * 100)}% Pace
                  </div>
                </div>
              </div>

              {/* Key Projected Performers: Probable Highest Scoring Batter & Bowler (HIDDEN AS REQUESTED) */}
              {false && (match as any).top_performers && (
                <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800/80 space-y-1.5 text-[11px]">
                  <div className="flex items-center justify-between text-slate-400 font-bold border-b border-slate-800/80 pb-1">
                    <span className="text-amber-300 flex items-center gap-1 uppercase tracking-wider text-[10px]">
                      <Sparkles className="w-3 h-3 text-amber-400" />
                      Key Projected Performers
                    </span>
                    <span className="text-slate-400 font-mono text-[9px] uppercase px-1.5 py-0.2 rounded bg-slate-800">
                      {match.state === "LIVE_BALL_BY_BALL" ? "Live In-Play" : (match.state === "SQUAD_ANNOUNCED" ? "Squad Probable" : "Playing XI")}
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">🏏 Top Batter:</span>
                    <span className="truncate ml-1 font-mono-numbers">
                      <strong className="text-cyan-300">{(match as any).top_performers.team_a_batter?.name?.split(" ").pop()}</strong> {((match as any).top_performers.team_a_batter?.status === "DISMISSED" ? `(${ (match as any).top_performers.team_a_batter?.current_match_runs || (match as any).top_performers.team_a_batter?.expected_runs }r OUT)` : `(~${(match as any).top_performers.team_a_batter?.expected_runs}r)`)} vs <strong className="text-amber-300">{(match as any).top_performers.team_b_batter?.name?.split(" ").pop()}</strong> {((match as any).top_performers.team_b_batter?.status === "DISMISSED" ? `(${ (match as any).top_performers.team_b_batter?.current_match_runs || (match as any).top_performers.team_b_batter?.expected_runs }r OUT)` : `(~${(match as any).top_performers.team_b_batter?.expected_runs}r)`)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">🎯 Top Bowler:</span>
                    <span className="truncate ml-1 font-mono-numbers">
                      <strong className="text-cyan-300">{(match as any).top_performers.team_a_bowler?.name?.split(" ").pop()}</strong> (~{(match as any).top_performers.team_a_bowler?.expected_wickets}w) vs <strong className="text-amber-300">{(match as any).top_performers.team_b_bowler?.name?.split(" ").pop()}</strong> (~{(match as any).top_performers.team_b_bowler?.expected_wickets}w)
                    </span>
                  </div>
                </div>
              )}

              {/* Historical Venue Pitch Diagnostics */}
              <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800/80 space-y-1 text-[11px]">
                <div className="flex justify-between items-center text-slate-400">
                  <span>Historical Venue Sample:</span>
                  <span className="text-slate-200 font-bold font-mono-numbers">
                    {(match as any).sample_matches || 35} Int'l Matches
                  </span>
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>Historical Bowling Econ ({match.format || "T20I"}):</span>
                  <span className="text-cyan-300 font-bold font-mono-numbers">
                    Pace {(match as any).pace_avg_economy ? (match as any).pace_avg_economy.toFixed(2) : (match.format === "TEST" ? "2.92" : (match.format === "ODI" ? "5.35" : "7.60"))} RPO • Spin {(match as any).spin_avg_economy ? (match as any).spin_avg_economy.toFixed(2) : (match.format === "TEST" ? "2.64" : (match.format === "ODI" ? "4.90" : "7.50"))} RPO
                  </span>
                </div>
              </div>

              {/* 12-Pillar Cricket Intelligence Variables */}
              {(match as any).format_specific_variables && (
                <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800/80 space-y-1.5 text-[11px]">
                  <div className="flex items-center justify-between text-slate-400 font-bold border-b border-slate-800/80 pb-1">
                    <span className="text-cyan-400 flex items-center gap-1 uppercase tracking-wider text-[10px]">
                      <Zap className="w-3 h-3 text-cyan-400" />
                      12-Pillar Pre-Match Intelligence
                    </span>
                    <span className="text-emerald-400 font-mono text-[9px] uppercase px-1.5 py-0.2 rounded bg-emerald-950/60 border border-emerald-800/50">
                      {(match as any).predictive_power_pct ? `${(match as any).predictive_power_pct}% Power` : "95.2% Calibrated"}
                    </span>
                  </div>
                  {/* Rolling Form & Bilateral H2H */}
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">🔥 Recent Form (L5):</span>
                    <span className="font-mono-numbers">
                      <strong className="text-cyan-300">{metaA.short} {(match as any).format_specific_variables?.team_a_form?.streak || "W"} ({Math.round(((match as any).format_specific_variables?.team_a_form?.recent_l5_win_pct ?? 0.6) * 100)}%)</strong> vs <strong className="text-amber-300">{metaB.short} {(match as any).format_specific_variables?.team_b_form?.streak || "W"} ({Math.round(((match as any).format_specific_variables?.team_b_form?.recent_l5_win_pct ?? 0.5) * 100)}%)</strong>
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">⚔️ Bilateral H2H (3-Yr):</span>
                    <span className="font-mono-numbers text-slate-200">
                      {metaA.short} {Math.round(((match as any).format_specific_variables?.h2h_bilateral?.last_3yr_win_pct_a ?? 0.52) * 100)}% • {metaB.short} {Math.round(((match as any).format_specific_variables?.h2h_bilateral?.last_3yr_win_pct_b ?? 0.48) * 100)}%
                    </span>
                  </div>
                  {/* Tactical Key */}
                  {((match as any).format_specific_variables?.test_specific?.tactical_key ||
                    (match as any).format_specific_variables?.odi_specific?.tactical_key ||
                    (match as any).format_specific_variables?.t20_specific?.tactical_key) && (
                    <div className="text-[10px] text-slate-300 bg-slate-950/60 p-1.5 rounded-lg border border-slate-800/60 leading-relaxed">
                      {(match as any).format_specific_variables?.test_specific?.tactical_key ||
                       (match as any).format_specific_variables?.odi_specific?.tactical_key ||
                       (match as any).format_specific_variables?.t20_specific?.tactical_key}
                    </div>
                  )}
                </div>
              )}

              {match.biggest_matchup_adv && (
                <div className="p-2.5 rounded-xl bg-cyan-950/40 border border-cyan-800/40 text-[11px] text-slate-300">
                  <span className="text-cyan-400 font-black block mb-0.5 flex items-center gap-1">
                    <Zap className="w-3 h-3 text-cyan-400" /> Crease Duel Matchup Edge
                  </span>
                  {typeof match.biggest_matchup_adv === "object" && (match.biggest_matchup_adv as any)?.description
                    ? (match.biggest_matchup_adv as any).description
                    : String(match.biggest_matchup_adv)}
                </div>
              )}

              {(match.state === "LIVE_BALL_BY_BALL" || match.state === "XI_AND_TOSS_LOCKED") && (match.batting_tank_pct !== undefined || match.bowling_tank_pct !== undefined) && (
                <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800/80 space-y-1.5 text-[11px]">
                  <div className="flex items-center justify-between text-slate-400 font-bold border-b border-slate-800/80 pb-1">
                    <span className="text-cyan-300 flex items-center gap-1 uppercase tracking-wider text-[10px]">
                      <Shield className="w-3 h-3 text-cyan-400" />
                      Quality Resource Tanks (Inference Inputs)
                    </span>
                    <span className="text-amber-300 font-mono text-[9px] uppercase px-1.5 py-0.2 rounded bg-amber-950/60 border border-amber-800/50">
                      ⚡ Dynamic ML
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono px-0.5">
                    <span>
                      {(() => {
                        const w = match.batting_weights || (match.quality_resource_tanks as any)?.batting_weights;
                        const isCurtailed = match.is_curtailed || (match.quality_resource_tanks as any)?.is_curtailed;
                        const schedOv = match.scheduled_overs || (match.quality_resource_tanks as any)?.scheduled_overs;
                        if (w && Array.isArray(w) && w.length === 4) {
                          const wStr = `w=[${w.map((v: number) => Number(v).toFixed(2)).join(", ")}]`;
                          return isCurtailed ? (
                            <span className="text-amber-400 font-semibold" title="Rain-curtailed dynamic resource compression">
                              ⚡ {match.format} ({schedOv} ov) {wStr}
                            </span>
                          ) : (
                            <span>{match.format} {wStr}</span>
                          );
                        }
                        return match.format === "T20I" || match.format === "T20"
                          ? "T20I w=[0.45, 0.30, 0.15, 0.10]"
                          : match.format === "ODI"
                          ? "ODI w=[0.40, 0.30, 0.20, 0.10]"
                          : "Test w=[0.30, 0.35, 0.25, 0.10]";
                      })()}
                    </span>
                    <span>
                      {(() => {
                        if (match.format === "TEST") return <span className="text-emerald-400 font-semibold">80-ov New Ball Cycle</span>;
                        const maxQ = match.max_quota_per_bowler || (match.quality_resource_tanks as any)?.max_quota_per_bowler;
                        const qRate = match.quota_consumption_rate_pct || (match.quality_resource_tanks as any)?.quota_consumption_rate_pct;
                        const isCurtailed = match.is_curtailed || (match.quality_resource_tanks as any)?.is_curtailed;
                        const schedOv = match.scheduled_overs || (match.quality_resource_tanks as any)?.scheduled_overs;
                        if (maxQ && qRate) {
                          return isCurtailed ? (
                            <span className="text-amber-300 font-semibold" title={`ICC Law 13.7: ${match.quota_summary || (match.quality_resource_tanks as any)?.quota_summary || ''}`}>
                              Quota: {maxQ} ov max ({qRate}%/ov) • {schedOv} ov
                            </span>
                          ) : (
                            <span className="text-emerald-400 font-semibold">
                              Quota: {maxQ} ov max ({qRate}%/ov)
                            </span>
                          );
                        }
                        return match.format === "T20I" || match.format === "T20"
                          ? <span className="text-emerald-400 font-semibold">Quota: 4 ov max (20%/ov)</span>
                          : <span className="text-emerald-400 font-semibold">Quota: 10 ov max (10%/ov)</span>;
                      })()}
                    </span>
                  </div>
                  <div className="space-y-1.5 mt-1">
                    <div>
                      <div className="flex justify-between items-center text-slate-300">
                        <span className="text-slate-400">🏏 Batting Dugout Quality:</span>
                        <span className="font-mono font-bold text-cyan-300">
                          {Math.round(match.batting_tank_pct ?? 50)}% Remaining
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-0.5">
                        <div
                          className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(100, Math.max(5, match.batting_tank_pct ?? 50))}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between items-center text-slate-300">
                        <span className="text-slate-400">
                          {match.format === "TEST" ? "🎯 Attack Penetration & Freshness:" : "🎯 Bowling Quota Quality:"}
                        </span>
                        <span className="font-mono font-bold text-emerald-300">
                          {Math.round(match.bowling_tank_pct ?? 50)}% {match.format === "TEST" ? "Readiness" : "Remaining"}
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-0.5">
                        <div
                          className="bg-gradient-to-r from-emerald-500 to-teal-400 h-full rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(100, Math.max(5, match.bowling_tank_pct ?? 50))}%` }}
                        />
                      </div>
                    </div>
                  </div>
                  {(match.powerplay_breakdown || match.is_curtailed || (match.pitch_sweat_index && match.pitch_sweat_index > 0)) && (
                    <div className="pt-1.5 border-t border-slate-800/60 space-y-1 mt-1">
                      {match.powerplay_breakdown?.summary && (
                        <div className="flex items-center justify-between text-[9px] font-mono text-slate-400">
                          <span className="text-slate-500">🛡️ PP Phases:</span>
                          <span className="text-slate-300 font-semibold">{match.powerplay_breakdown.summary}</span>
                        </div>
                      )}
                      {(match.is_curtailed || (match.pitch_sweat_index && match.pitch_sweat_index > 0)) && (
                        <div className="flex items-center justify-between text-[9px] font-mono text-slate-400">
                          <span className="text-slate-500">🌧️ Physics:</span>
                          <span className="flex items-center gap-1.5 text-cyan-300">
                            <span>Seam Sweat +{Math.round((match.pitch_sweat_index || 2.5) * 3)}%</span>
                            <span>•</span>
                            <span>Spin Grip -{Math.round(match.spin_grip_penalty_pct || 20)}%</span>
                            <span>•</span>
                            <span>Boundary Drag +{Math.round(match.outfield_friction_pct || 15)}%</span>
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Live Telemetry Notice */}
              {match.worm_points && match.worm_points.length > 1 && (
                <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800 text-[10px] text-slate-400 font-mono">
                  <span className="flex items-center gap-1.5 text-cyan-400 font-bold">
                    <Activity className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                    <span>{match.worm_points.length} Live Deliveries Logged</span>
                  </span>
                  <span className="text-cyan-400 font-semibold flex items-center gap-1">
                    Telemetry Center <ChevronRight className="w-3 h-3" />
                  </span>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: TOSS SIMULATOR */}
          {activeCardTab === "toss" && (
            <div className="space-y-2.5 text-xs">
              <p className="text-[11px] text-slate-400">
                Simulate how pitch friction and dew impact win probability based on who bowls first:
              </p>

              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setTossSimChoice(tossSimChoice === "a_bowls" ? "none" : "a_bowls")}
                  className={`p-2 rounded-xl text-left border transition ${
                    tossSimChoice === "a_bowls"
                      ? "bg-cyan-950 border-cyan-500 text-white shadow-md shadow-cyan-500/20"
                      : "bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700"
                  }`}
                >
                  <span className="font-bold block text-xs">{metaA.short} Bowls 1st</span>
                  <span className="text-[10px] text-cyan-400 font-mono-numbers">
                    +{(match.dew_factor * 4.5).toFixed(1)}% Chase Advantage
                  </span>
                </button>

                <button
                  onClick={() => setTossSimChoice(tossSimChoice === "a_bats" ? "none" : "a_bats")}
                  className={`p-2 rounded-xl text-left border transition ${
                    tossSimChoice === "a_bats"
                      ? "bg-cyan-950 border-cyan-500 text-white shadow-md shadow-cyan-500/20"
                      : "bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700"
                  }`}
                >
                  <span className="font-bold block text-xs">{metaA.short} Bats 1st</span>
                  <span className="text-[10px] text-amber-400 font-mono-numbers">
                    Defending Under Lights
                  </span>
                </button>
              </div>

              {tossSimChoice !== "none" && (
                <div className="text-[11px] text-slate-300 bg-slate-900/90 p-2 rounded-xl border border-cyan-800/50 flex items-center justify-between">
                  <span>Simulated Win Rate:</span>
                  <span className="font-black text-cyan-400 font-mono-numbers">
                    {metaA.short} {effective.pA}% • {metaB.short} {effective.pB}%
                  </span>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: SQUADS & XI */}
          {activeCardTab === "squad" && (
            <div className="space-y-2.5 text-xs">
              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>
                    {match.state === "XI_AND_TOSS_LOCKED" || match.state === "LIVE_BALL_BY_BALL"
                      ? "Confirmed Playing XI & Reserves"
                      : "Official Board Squads"}
                  </span>
                </span>
                <span className="text-cyan-400 font-bold font-mono">
                  {metaA.short} ({teamASquad.length}) &amp; {metaB.short} ({teamBSquad.length})
                </span>
              </div>

              {/* Team Selector Toggle */}
              <div className="grid grid-cols-2 gap-1.5 p-1 rounded-xl bg-slate-950/80 border border-slate-800">
                <button
                  type="button"
                  onClick={() => setSelectedCardSquadTeam("team_a")}
                  className={`py-1.5 px-2 rounded-lg text-xs font-bold transition flex items-center justify-center gap-1.5 ${
                    selectedCardSquadTeam === "team_a"
                      ? "bg-cyan-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  <span>{metaA.flag}</span>
                  <span>{metaA.short} ({teamASquad.length > 0 ? teamASquad.length : "15"})</span>
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedCardSquadTeam("team_b")}
                  className={`py-1.5 px-2 rounded-lg text-xs font-bold transition flex items-center justify-center gap-1.5 ${
                    selectedCardSquadTeam === "team_b"
                      ? "bg-amber-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  <span>{metaB.flag}</span>
                  <span>{metaB.short} ({teamBSquad.length > 0 ? teamBSquad.length : "15"})</span>
                </button>
              </div>

              {loadingSquad ? (
                <div className="py-6 text-center text-slate-500 animate-pulse text-xs">
                  Loading squad telemetry...
                </div>
              ) : (selectedCardSquadTeam === "team_a" ? teamASquad : teamBSquad).length === 0 ? (
                <div className="py-4 text-center text-slate-500 text-xs">
                  Click to inspect full squad roster and player condition factors.
                </div>
              ) : (
                <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 custom-scroll">
                  {(selectedCardSquadTeam === "team_a" ? teamASquad : teamBSquad).map((p, idx) => (
                    <div
                      key={p.id || p.player_id || idx}
                      className="flex flex-col p-2 rounded-xl bg-slate-900/70 border border-slate-800/80 text-[11px] space-y-0.5"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 truncate">
                          <span className="font-mono text-[10px] text-slate-400 w-4">{idx + 1}.</span>
                          <span className="font-bold text-slate-100 truncate">{p.name}</span>
                          {match.state === "SQUAD_ANNOUNCED" ? (
                            <span className="text-[8px] font-bold px-1 rounded bg-slate-800 text-slate-300 border border-slate-700 shrink-0">
                              {p.is_captain ? "★ C" : "SQUAD"}
                            </span>
                          ) : p.is_confirmed_xi === 1 ? (
                            <span className="text-[8px] font-bold px-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 shrink-0">
                              XI
                            </span>
                          ) : p.sub_status === "BENCH_RESERVE" ? (
                            <span className="text-[8px] font-bold px-1 rounded bg-slate-800 text-slate-400 shrink-0">
                              BENCH
                            </span>
                          ) : null}
                          {p.sub_status === "CONCUSSION_SUB_IN" && (
                            <span className="text-[8px] font-bold px-1 rounded bg-amber-950 text-amber-300 border border-amber-800 shrink-0">
                              SUB IN
                            </span>
                          )}
                        </div>
                        <span className="text-[10px] text-cyan-400 capitalize font-medium shrink-0 ml-2">
                          {p.role?.replace(/_/g, " ")}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono-numbers">
                        <span>{p.batting_hand?.toUpperCase() || "RHB"} • {p.bowling_style || "N/A"} • SR {p.pace_sr || 135}</span>
                        <span className="text-emerald-400 font-sans font-bold text-[9px]">
                          {p.pitch_adaptability?.[match.ground_pitch_archetype || "flat"]
                            ? `${Math.min(99, Math.round(p.pitch_adaptability[match.ground_pitch_archetype || "flat"]))}% Deck Fit`
                            : p.pitch_fit
                            ? `${Math.min(99, Math.round(p.pitch_fit > 1.5 ? p.pitch_fit : p.pitch_fit * 100))}% Deck Fit`
                            : `${Math.min(99, Math.round(82 * (p.venue_affinity || 1.0)))}% Deck Fit`}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Card Action Footer */}
      <div className="p-4 pt-2 border-t border-slate-800/80 bg-slate-900/40">
        <div className="flex items-center justify-between gap-2">
          <Link
            href={`/match/${match.id}`}
            className="flex-1 py-2 px-3 rounded-xl bg-cyan-600/20 hover:bg-cyan-600 text-cyan-300 hover:text-white font-bold text-xs border border-cyan-500/30 transition flex items-center justify-center gap-1.5"
          >
            <span>Live Telemetry Center</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {/* Developer Sandbox Controls (Hidden by default unless sandboxMode is toggled in Navbar) */}
        {sandboxMode && (
          <div className="mt-3 pt-3 border-t border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-[10px] font-black text-amber-400 uppercase tracking-wider">
              <span className="flex items-center gap-1">
                <Sliders className="w-3 h-3 text-amber-400" />
                Dev Sandbox Transition Stepper
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-[10px] font-bold">
              <button
                disabled={actionLoading}
                onClick={() => handleStateClick("XI_AND_TOSS_LOCKED")}
                className={`py-1.5 rounded-lg border transition ${
                  match.state === "XI_AND_TOSS_LOCKED"
                    ? "bg-emerald-900/60 border-emerald-500 text-emerald-300 shadow-sm"
                    : "bg-slate-800/60 border-slate-700 text-slate-400 hover:text-white"
                }`}
              >
                Stage 3: XI &amp; Toss Locked
              </button>
              <button
                disabled={actionLoading}
                onClick={() => handleStateClick("LIVE_BALL_BY_BALL")}
                className={`py-1.5 rounded-lg border transition ${
                  match.state === "LIVE_BALL_BY_BALL"
                    ? "bg-rose-900/60 border-rose-500 text-rose-300 shadow-sm"
                    : "bg-slate-800/60 border-slate-700 text-slate-400 hover:text-white"
                }`}
              >
                Stage 4: Live In-Play
              </button>
            </div>

            {match.state === "LIVE_BALL_BY_BALL" && (
              <button
                disabled={actionLoading}
                onClick={handleBallClick}
                className="w-full py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-black text-xs transition flex items-center justify-center gap-1.5 shadow-md shadow-rose-600/30"
              >
                <Flame className="w-3.5 h-3.5" />
                Stream Next Ball (Simulate Delivery)
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
