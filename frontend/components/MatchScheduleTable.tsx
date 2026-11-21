"use client";

import React, { useState } from "react";
import Link from "next/link";
import { MatchData } from "./MatchCard";
import { getTeamMeta, formatKickoffCountdown, formatDateTime } from "@/lib/teamUtils";
import {
  ArrowRight,
  Sparkles,
  MapPin,
  Trophy,
  Flame,
  ChevronDown,
  ChevronUp,
  Activity,
  Moon,
  CheckCircle2,
  Coffee,
  CloudRain,
  ShieldCheck,
  Clock
} from "lucide-react";

interface MatchScheduleTableProps {
  matches: MatchData[];
}

export default function MatchScheduleTable({ matches }: MatchScheduleTableProps) {
  const [expandedMatchId, setExpandedMatchId] = useState<string | null>(null);

  if (matches.length === 0) {
    return (
      <div className="text-center py-12 bg-slate-900/40 rounded-2xl border border-slate-800">
        <p className="text-sm text-slate-400">No matches found matching your filters.</p>
      </div>
    );
  }

  const toggleExpand = (id: string) => {
    setExpandedMatchId(expandedMatchId === id ? null : id);
  };

  return (
    <div className="w-full overflow-hidden rounded-2xl border border-slate-800 bg-[#0a0f1d] shadow-xl">
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-slate-900/90 border-b border-slate-800 text-[11px] font-black uppercase tracking-wider text-slate-400">
              <th className="py-3.5 px-4">Date / Status</th>
              <th className="py-3.5 px-4">Matchup</th>
              <th className="py-3.5 px-3">Format</th>
              <th className="py-3.5 px-4">Venue & Pitch</th>
              <th className="py-3.5 px-4 text-center">AI Win Probability</th>
              <th className="py-3.5 px-4 text-center">Par Score</th>
              <th className="py-3.5 px-4 text-right">Match Center</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {matches.map((m) => {
              const metaA = getTeamMeta(m.team_a_short || m.team_a_id, m.team_a_name);
              const metaB = getTeamMeta(m.team_b_short || m.team_b_id, m.team_b_name);
              const isLive = m.state === "LIVE_BALL_BY_BALL";
              const isTeamAFavored = m.p_team_a >= m.p_team_b;
              const favoredMeta = isTeamAFavored ? metaA : metaB;
              const favoredProb = isTeamAFavored ? m.p_team_a : m.p_team_b;
              const { dateStr, timeStr } = formatDateTime(m.scheduled_start);
              const countdown = formatKickoffCountdown(m.scheduled_start);
              const isExpanded = expandedMatchId === m.id;

              const isCompleted = m.state === "COMPLETED";
              const isStumps = Boolean((m as any).is_stumps) || (m.match_status_text || "").toLowerCase().includes("stumps");
              const isInningsBreak = (m.current_innings === 2 && m.current_overs === 0 && m.current_balls === 0 && (m.target_runs || 0) > 0) || (m.match_status_text || "").toLowerCase().includes("innings break");
              const isRainInterrupted = Boolean((m as any).weather_condition?.includes("RAIN")) || (m.match_status_text || "").toLowerCase().includes("rain");

              return (
                <React.Fragment key={m.id}>
                  <tr
                    className={`hover:bg-slate-800/40 transition duration-150 cursor-pointer ${
                      isLive ? "bg-rose-950/15" : isCompleted ? "bg-emerald-950/10" : ""
                    }`}
                    onClick={() => toggleExpand(m.id)}
                  >
                    {/* Date / Status */}
                    <td className="py-3.5 px-4 whitespace-nowrap">
                      {isCompleted ? (
                        <span className="px-2.5 py-1 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-600/70 text-[10px] font-black flex items-center gap-1.5 w-fit">
                          <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                          <span>COMPLETED</span>
                        </span>
                      ) : isInningsBreak ? (
                        <span className="px-2.5 py-1 rounded-full bg-amber-950/80 text-amber-300 border border-amber-600/70 text-[10px] font-black flex items-center gap-1.5 w-fit animate-pulse">
                          <Coffee className="w-3 h-3 text-amber-400" />
                          <span>INNINGS BREAK</span>
                        </span>
                      ) : isRainInterrupted ? (
                        <span className="px-2.5 py-1 rounded-full bg-sky-950/80 text-sky-300 border border-sky-600/70 text-[10px] font-black flex items-center gap-1.5 w-fit animate-pulse">
                          <CloudRain className="w-3 h-3 text-sky-400" />
                          <span>RAIN STOPPAGE</span>
                        </span>
                      ) : isStumps ? (
                        <span className="px-2.5 py-1 rounded-full bg-indigo-950/80 text-indigo-300 border border-indigo-600/70 text-[10px] font-black flex items-center gap-1.5 w-fit">
                          <Moon className="w-3 h-3 text-indigo-400" />
                          <span>{(m as any).stumps_label || "STUMPS"}</span>
                        </span>
                      ) : isLive ? (
                        <div className="flex items-center gap-1.5 text-rose-400 font-black">
                          <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping inline-block" />
                          <span>LIVE IN-PLAY</span>
                        </div>
                      ) : (
                        <div className="flex flex-col">
                          <span className="font-bold text-slate-200">{dateStr}</span>
                          <span className="text-[10px] text-cyan-400 font-medium">
                            {countdown} {timeStr ? `• ${timeStr}` : ""}
                          </span>
                        </div>
                      )}
                    </td>

                    {/* Matchup */}
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-3">
                        {/* Team A */}
                        <div className="flex items-center gap-1.5 min-w-[90px]">
                          <span className="text-base">{metaA.flag}</span>
                          <span
                            className={`font-extrabold ${
                              isTeamAFavored ? "text-white" : "text-slate-400"
                            }`}
                          >
                            {metaA.name}
                          </span>
                        </div>

                        <span className="text-slate-600 font-bold text-[10px]">vs</span>

                        {/* Team B */}
                        <div className="flex items-center gap-1.5 min-w-[90px]">
                          <span className="text-base">{metaB.flag}</span>
                          <span
                            className={`font-extrabold ${
                              !isTeamAFavored ? "text-white" : "text-slate-400"
                            }`}
                          >
                            {metaB.name}
                          </span>
                        </div>
                      </div>
                      <div className="text-[10px] text-slate-500 truncate max-w-xs mt-0.5">
                        {m.series_name}
                      </div>
                    </td>

                    {/* Format */}
                    <td className="py-3.5 px-3 whitespace-nowrap">
                      <span className="px-2 py-0.5 rounded-md bg-slate-800 text-slate-300 font-mono font-bold text-[10px] border border-slate-700">
                        {m.format}
                      </span>
                    </td>

                    {/* Venue & Pitch */}
                    <td className="py-3.5 px-4 whitespace-nowrap">
                      <div className="flex flex-col">
                        <span className="font-medium text-slate-300 flex items-center gap-1">
                          <MapPin className="w-3 h-3 text-cyan-400 shrink-0" />
                          {m.venue_name}
                        </span>
                        <span className="text-[10px] text-slate-400 capitalize">
                          {m.venue_city} • {m.pitch_type.replace("_", " ")}
                        </span>
                      </div>
                    </td>

                    {/* Win Probability Bar */}
                    <td className="py-3.5 px-4 text-center min-w-[200px]">
                      <div className="flex items-center justify-between text-[11px] font-mono-numbers font-black mb-1 px-1">
                        <span className="text-cyan-400">
                          {metaA.short} {m.p_team_a}%
                        </span>
                        <span className="text-amber-400">
                          {m.p_team_b}% {metaB.short}
                        </span>
                      </div>
                      <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden flex shadow-inner">
                        <div
                          className="h-full bg-gradient-to-r from-blue-600 to-cyan-400 rounded-l-full"
                          style={{ width: `${m.p_team_a}%` }}
                        />
                        <div
                          className="h-full bg-gradient-to-r from-amber-500 to-rose-500 rounded-r-full"
                          style={{ width: `${m.p_team_b}%` }}
                        />
                      </div>
                      <div className="text-[9px] text-slate-400 mt-1 uppercase font-semibold">
                        {favoredMeta.short} favored ({favoredProb}%)
                      </div>
                    </td>

                    {/* Par Score */}
                    <td className="py-3.5 px-4 text-center whitespace-nowrap">
                      <span className="font-black text-white text-sm font-mono-numbers">
                        {m.par_score_projected}
                      </span>
                      <span className="block text-[10px] text-slate-400 font-mono-numbers">
                        {m.par_score_min}–{m.par_score_max}
                      </span>
                    </td>

                    {/* Actions */}
                    <td className="py-3.5 px-4 text-right whitespace-nowrap">
                      <div className="flex items-center justify-end gap-2">
                        <Link
                          href={`/match/${m.id}`}
                          onClick={(e) => e.stopPropagation()}
                          className="px-3 py-1.5 rounded-lg bg-cyan-600/20 hover:bg-cyan-600 text-cyan-300 hover:text-white font-bold text-xs border border-cyan-500/40 transition flex items-center gap-1"
                        >
                          <span>Center</span>
                          <ArrowRight className="w-3 h-3" />
                        </Link>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleExpand(m.id);
                          }}
                          className="p-1 rounded-lg hover:bg-slate-800 text-slate-400"
                        >
                          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        </button>
                      </div>
                    </td>
                  </tr>

                  {/* Expanded Row for Tactical Insight */}
                  {isExpanded && (
                    <tr className="bg-slate-900/90 border-b border-slate-800">
                      <td colSpan={7} className="p-4">
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
                          <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700">
                            <span className="text-[10px] uppercase font-bold text-cyan-400 block mb-1">
                              Stage 1 Macro Prior
                            </span>
                            <div className="text-white font-mono font-bold text-xs">
                              {metaA.short} {m.p1_macro || m.p_team_a}% vs {Math.round((100 - (m.p1_macro || m.p_team_a))*10)/10}% {metaB.short}
                            </div>
                            <p className="text-slate-400 text-[10px] mt-1">
                              Home/Away Elo & Venue Profile
                            </p>
                          </div>

                          <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700">
                            <span className="text-[10px] uppercase font-bold text-amber-400 block mb-1">
                              Tactical Matchup Advantage
                            </span>
                            <p className="text-slate-300 text-[11px] leading-tight">
                              {m.biggest_matchup_adv || "No single matchup imbalance detected; ratings balanced."}
                            </p>
                          </div>

                          <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700">
                            <span className="text-[10px] uppercase font-bold text-emerald-400 block mb-1">
                              Pitch Archetype & Chase Bias
                            </span>
                            <p className="text-slate-300 text-[11px]">
                              Deck: {m.pitch_type.toUpperCase()} • Spin Ratio: {Math.round(m.spin_wicket_ratio * 100)}% • Chase Win: {Math.round(m.chase_win_pct * 100)}%
                            </p>
                          </div>

                          <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700 flex flex-col justify-between">
                            <div>
                              <span className="text-[10px] uppercase font-bold text-indigo-400 block mb-1">
                                Engine Model Stage
                              </span>
                              <p className="text-slate-300 text-[11px] font-mono">
                                {m.state}
                              </p>
                            </div>
                            <Link
                              href={`/match/${m.id}`}
                              className="text-cyan-400 hover:underline font-bold text-[11px] mt-2 flex items-center gap-1"
                            >
                              Open Live Telemetry & Worm Chart →
                            </Link>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
