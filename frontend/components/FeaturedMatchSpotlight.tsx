"use client";

import React from "react";
import Link from "next/link";
import { MatchData } from "./MatchCard";
import { getTeamMeta, formatKickoffCountdown, formatDateTime } from "@/lib/teamUtils";
import {
  Trophy,
  Sparkles,
  MapPin,
  Target,
  ArrowRight,
  ShieldCheck,
  Flame,
  Clock,
  Compass,
  Zap,
  Activity
} from "lucide-react";

interface FeaturedMatchSpotlightProps {
  match: MatchData;
}

export default function FeaturedMatchSpotlight({ match }: FeaturedMatchSpotlightProps) {
  if (!match) return null;

  const metaA = getTeamMeta(match.team_a_short || match.team_a_id, match.team_a_name);
  const metaB = getTeamMeta(match.team_b_short || match.team_b_id, match.team_b_name);

  const isTeamAFavored = match.p_team_a >= match.p_team_b;
  const favoredMeta = isTeamAFavored ? metaA : metaB;
  const favoredProb = isTeamAFavored ? match.p_team_a : match.p_team_b;

  const probDiff = Math.abs(match.p_team_a - match.p_team_b);
  const confidenceLabel =
    probDiff >= 15
      ? { text: "High Confidence Pick", color: "text-emerald-400 bg-emerald-950/80 border-emerald-800" }
      : probDiff >= 6
      ? { text: "Moderate Model Edge", color: "text-cyan-400 bg-cyan-950/80 border-cyan-800" }
      : { text: "Tight Contest / 50-50", color: "text-amber-400 bg-amber-950/80 border-amber-800" };

  const { dateStr, timeStr } = formatDateTime(match.scheduled_start);
  const countdown = formatKickoffCountdown(match.scheduled_start);

  return (
    <div className="mb-10 rounded-3xl bg-gradient-to-br from-[#0c1324] via-[#0e1930] to-[#0a101f] border border-cyan-500/20 p-6 sm:p-8 shadow-2xl relative overflow-hidden group">
      {/* Dynamic ambient lights */}
      <div className="absolute top-0 left-1/4 -mt-20 w-96 h-96 rounded-full bg-cyan-500/10 blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 -mb-20 w-96 h-96 rounded-full bg-indigo-500/10 blur-3xl pointer-events-none" />

      <div className="relative z-10">
        {/* Header Ribbon */}
        <div className="flex items-center justify-between gap-4 mb-6 flex-wrap pb-4 border-b border-slate-800/80">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 text-white shadow-md shadow-cyan-600/30">
              <Trophy className="w-3.5 h-3.5 text-amber-300" />
              Marquee Matchup of the Week
            </span>

            <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-slate-800/90 text-cyan-300 border border-slate-700 font-mono">
              {match.format}
            </span>

            <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${confidenceLabel.color}`}>
              {confidenceLabel.text}
            </span>

            <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-blue-950/90 text-blue-300 border border-blue-700/70 flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
              Official ICC International Fixture
            </span>

            {((match as any).active_pitch_type || (match as any).pitch_display_name || match.pitch_type) && (
              <span
                className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-emerald-950/90 text-emerald-300 border border-emerald-600/70 flex items-center gap-1.5 shadow-sm"
                title={(match as any).pitch_display_name || `Surface: ${match.pitch_type}`}
              >
                <span className="text-[10px] text-cyan-400 font-black">⚡ PITCH:</span>
                <span className="uppercase tracking-wide font-black">
                  {((match as any).active_pitch_type || match.pitch_type)}
                </span>
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 text-xs text-slate-300 font-medium">
            <span className="flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-cyan-400" />
              {match.venue_name}, {match.venue_city}
            </span>
            <span className="text-slate-600">•</span>
            <span className="flex items-center gap-1.5 text-cyan-300 font-semibold">
              <Clock className="w-3.5 h-3.5 text-cyan-400" />
              {dateStr} {timeStr ? `• ${timeStr}` : ""} {match.state !== "COMPLETED" && match.state !== "LIVE_BALL_BY_BALL" && countdown && countdown !== "In Progress" ? `(${countdown})` : ""}
            </span>
          </div>
        </div>

        {/* Main Content Grid: Duel & Forecast */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left: Team Head to Head & WinViz Gauge */}
          <div className="lg:col-span-7 space-y-6">
            <div>
              <span className="text-[11px] text-cyan-400 font-black tracking-widest uppercase block mb-1">
                {match.series_name}
              </span>
              <h2 className="text-2xl sm:text-4xl font-black text-white tracking-tight flex items-center gap-3">
                <span>{metaA.name}</span>
                <span className="text-slate-500 font-light text-xl sm:text-2xl">vs</span>
                <span>{metaB.name}</span>
              </h2>
            </div>

            {/* Duel Scoreboard Card */}
            <div className="grid grid-cols-2 gap-4 p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-inner">
              {/* Team A */}
              <div className="flex items-center gap-3.5">
                <div
                  className={`w-14 h-14 rounded-2xl flex items-center justify-center text-2xl shadow-lg border ${metaA.borderGlow} bg-gradient-to-br ${metaA.gradient}`}
                >
                  {metaA.flag}
                </div>
                <div>
                  <div className="text-base sm:text-lg font-black text-white leading-tight">
                    {metaA.name}
                  </div>
                  <div className="text-2xl font-black text-cyan-400 font-mono-numbers mt-0.5">
                    {match.p_team_a}%
                    <span className="text-[10px] font-bold text-slate-400 ml-1.5 uppercase font-sans">
                      Win Prob
                    </span>
                  </div>
                </div>
              </div>

              {/* Team B */}
              <div className="flex items-center gap-3.5 justify-end text-right">
                <div>
                  <div className="text-base sm:text-lg font-black text-white leading-tight">
                    {metaB.name}
                  </div>
                  <div className="text-2xl font-black text-amber-400 font-mono-numbers mt-0.5">
                    {match.p_team_b}%
                    <span className="text-[10px] font-bold text-slate-400 ml-1.5 uppercase font-sans">
                      Win Prob
                    </span>
                  </div>
                </div>
                <div
                  className={`w-14 h-14 rounded-2xl flex items-center justify-center text-2xl shadow-lg border ${metaB.borderGlow} bg-gradient-to-br ${metaB.gradient}`}
                >
                  {metaB.flag}
                </div>
              </div>
            </div>

            {/* Glowing WinViz Stacked Bar */}
            <div className="space-y-2">
              <div className="h-4 w-full bg-slate-950 rounded-full overflow-hidden flex p-[2px] shadow-inner border border-slate-800/80">
                <div
                  className="h-full bg-gradient-to-r from-blue-600 via-cyan-500 to-sky-400 rounded-l-full transition-all duration-700 ease-out relative group"
                  style={{ width: `${match.p_team_a}%` }}
                />
                {match.format === "TEST" && (match.p_draw || 0) > 0 && (
                  <div
                    className="h-full bg-slate-500 transition-all duration-700 ease-out"
                    style={{ width: `${match.p_draw}%` }}
                  />
                )}
                <div
                  className="h-full bg-gradient-to-r from-amber-500 via-orange-500 to-rose-500 rounded-r-full transition-all duration-700 ease-out relative group"
                  style={{ width: `${match.p_team_b}%` }}
                />
              </div>

              <div className="flex justify-between items-center text-xs font-bold text-slate-400 px-1 font-mono-numbers">
                <span className="text-cyan-400 flex items-center gap-1">
                  <span>{metaA.short}</span>
                  <span className="text-white font-black">{match.p_team_a}%</span>
                </span>
                <span className="text-[10px] text-slate-400 font-sans uppercase tracking-wider font-semibold">
                  {match.format === "TEST" && (match.p_draw || 0) > 0
                    ? `Draw: ${match.p_draw}% • Live WinViz`
                    : match.state === "LIVE_BALL_BY_BALL"
                    ? "Stage 4 Live WinViz"
                    : match.state === "COMPLETED"
                    ? "Official Final Result"
                    : "Calibrated WinViz Anchor (P₀)"}
                </span>
                <span className="text-amber-400 flex items-center gap-1">
                  <span className="text-white font-black">{match.p_team_b}%</span>
                  <span>{metaB.short}</span>
                </span>
              </div>
            </div>

            {/* Multi-Stage Intelligence Progression Strip */}
            <div className="grid grid-cols-3 gap-2 p-3 rounded-2xl bg-slate-950/80 border border-slate-800 text-center font-mono-numbers">
              <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800/80">
                <span className="text-[9px] text-cyan-400 font-bold block uppercase tracking-wider">Gear 1 Macro Prior</span>
                <span className="text-sm font-black text-white">{match.p1_macro || match.p_team_a}%</span>
                <span className="text-[10px] text-slate-400 block">{Math.round((100 - (match.p1_macro || match.p_team_a)) * 10) / 10}% {metaB.short}</span>
              </div>
              <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800/80">
                <span className="text-[9px] text-indigo-400 font-bold block uppercase tracking-wider">Gear 2A Squad Fit</span>
                <span className="text-sm font-black text-white">{match.p2_squad || match.p_team_a}%</span>
                <span className="text-[10px] text-slate-400 block">Δ {match.delta_squad_shift ? (match.delta_squad_shift > 0 ? `+${match.delta_squad_shift}%` : `${match.delta_squad_shift}%`) : "0%"}</span>
              </div>
              <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800/80">
                <span className="text-[9px] text-emerald-400 font-bold block uppercase tracking-wider">Gear 2B XI Locked</span>
                <span className="text-sm font-black text-white">{match.p0_pre_match || match.p_team_a}%</span>
                <span className="text-[10px] text-slate-400 block">Toss {match.delta_toss_shift ? (match.delta_toss_shift > 0 ? `+${match.delta_toss_shift}%` : `${match.delta_toss_shift}%`) : "0%"}</span>
              </div>
            </div>
          </div>

          {/* Right: AI Predictive Breakdown Panel */}
          <div className="lg:col-span-5 p-6 rounded-2xl bg-[#090f1d] border border-cyan-500/30 space-y-4 shadow-xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-black uppercase tracking-wider text-slate-300">
                  AI Model Verdict
                </span>
              </div>
              <span className="text-xs font-black text-cyan-300 px-2.5 py-1 rounded-full bg-cyan-950/80 border border-cyan-700">
                {favoredMeta.name} Favored ({favoredProb}%)
              </span>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800">
                <span className="text-[10px] uppercase font-bold text-slate-400 block flex items-center gap-1 mb-1">
                  <Target className="w-3.5 h-3.5 text-amber-400" />
                  {match.format === "TEST"
                    ? match.current_innings === 2
                      ? `${(match as any).batting_team_short || "2nd"} Innings Live Forecast`
                      : match.current_innings === 3
                      ? `${(match as any).batting_team_short || match.team_a_short || "3rd"} Innings Live Forecast`
                      : match.current_innings === 4
                      ? `${(match as any).batting_team_short || match.team_b_short || "4th"} Target Chase Forecast`
                      : "1st Innings Par Score"
                    : match.current_innings === 2
                    ? "Target Chase Par"
                    : "1st Innings Par Score"}
                </span>
                <div className="text-lg font-black text-white font-mono-numbers">
                  {(() => {
                    const testProg = (match as any).scenario_score_projections?.live_test_innings_progression;
                    const activeProg = testProg ? (testProg[`innings_${match.current_innings}`] || testProg.innings_2 || testProg.innings_3) : undefined;
                    const projVal = activeProg?.live_projected_total || match.par_score_projected || 546;
                    const rangeVal = activeProg?.range || (match.par_score_min && match.par_score_max ? `${match.par_score_min}–${match.par_score_max}` : "510–585");
                    return (
                      <>
                        ~{projVal}{" "}
                        <span className="text-xs font-normal text-slate-400">
                          ({rangeVal})
                        </span>
                      </>
                    );
                  })()}
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800">
                <span className="text-[10px] uppercase font-bold text-slate-400 block flex items-center gap-1 mb-1">
                  <Compass className="w-3.5 h-3.5 text-cyan-400" />
                  Active Model Pitch Type
                </span>
                <div className="text-sm font-extrabold text-cyan-300 capitalize truncate" title={(match as any).pitch_display_name || undefined}>
                  {(match as any).pitch_display_name || ((match as any).active_pitch_type || match.pitch_type).replace("_", " ")}
                </div>
                <div className="text-[10px] text-emerald-400 font-mono font-bold mt-0.5">
                  ⚡ Model: {((match as any).active_pitch_type || match.pitch_type).toUpperCase()}
                </div>
              </div>
            </div>

            {/* Matchup Highlight */}
            {match.biggest_matchup_adv && (
              <div className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-800/40 text-xs">
                <div className="flex items-center gap-1.5 text-cyan-300 font-black mb-1">
                  <Zap className="w-3.5 h-3.5 text-cyan-400" />
                  Crease Duel Tactical Advantage
                </div>
                <p className="text-slate-300 text-[11px] leading-relaxed">
                  {typeof match.biggest_matchup_adv === "object" && (match.biggest_matchup_adv as any)?.description
                    ? (match.biggest_matchup_adv as any).description
                    : String(match.biggest_matchup_adv)}
                </p>
              </div>
            )}

            {/* Action CTA Link */}
            <Link
              href={`/match/${match.id}`}
              className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-bold text-xs flex items-center justify-center gap-2 transition duration-200 shadow-lg shadow-cyan-600/25"
            >
              <span>Explore Live Telemetry & Ball-by-Ball Center</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
