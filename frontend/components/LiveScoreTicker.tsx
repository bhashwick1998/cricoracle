"use client";

import React from "react";
import Link from "next/link";
import { MatchData } from "./MatchCard";
import TeamFlag from "./TeamFlag";
import { getTeamMeta, formatKickoffCountdown } from "@/lib/teamUtils";
import { Flame, Radio, Sparkles, ChevronRight } from "lucide-react";

interface LiveScoreTickerProps {
  matches: MatchData[];
  loading?: boolean;
  wakingBackend?: boolean;
}

export default function LiveScoreTicker({
  matches,
  loading = false,
  wakingBackend = false,
}: LiveScoreTickerProps) {
  if (!matches || matches.length === 0) {
    const isActiveWait = loading || wakingBackend;
    const message = wakingBackend
      ? "WAKING CRICORACLE TELEMETRY SERVER • AUTO-RETRYING..."
      : loading
      ? "LIVE MATCH ODDS • CONNECTING TO CRICORACLE TELEMETRY STREAM..."
      : "MONITORING CRICAPI • NO LIVE MATCHES IN PLAY";

    return (
      <div className="w-full bg-[#0a0f1d] border-b border-slate-800/80 py-2.5 px-6 text-xs text-slate-500 select-none">
        <div className="flex items-center gap-2">
          <span className="flex h-2 w-2 relative">
            {isActiveWait && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
            )}
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                isActiveWait ? "bg-cyan-500" : "bg-emerald-500"
              }`}
            ></span>
          </span>
          <span
            className={`text-[10px] font-black tracking-wider uppercase ${
              isActiveWait
                ? "bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent"
                : "text-slate-400"
            }`}
          >
            {message}
          </span>
        </div>
      </div>
    );
  }

  // Distinct matches only (never duplicate entries in live odds strip)
  const displayMatches = matches;

  return (
    <div className="w-full bg-[#0a0f1d] border-b border-slate-800/80 overflow-hidden flex items-center py-2 select-none">
      {/* Ticker Header Tag */}
      <div className="shrink-0 bg-[#0a0f1d] z-10 pl-4 pr-4 py-1 flex items-center gap-2 border-r border-slate-800/90">
        <span className="flex h-2 w-2 relative">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
        </span>
        <span className="text-[10px] font-black tracking-wider uppercase bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent whitespace-nowrap">
          LIVE MATCH ODDS
        </span>
      </div>

      {/* Scrolling / Aligned Strip */}
      <div className="flex-1 overflow-hidden relative">
        <div className="flex animate-ticker pl-4 gap-3">
        {displayMatches.map((m, idx) => {
          const metaA = getTeamMeta(m.team_a_short || m.team_a_id, m.team_a_name);
          const metaB = getTeamMeta(m.team_b_short || m.team_b_id, m.team_b_name);
          const isLive = m.state === "LIVE_BALL_BY_BALL";
          const isInningsBreak =
            (m.current_innings === 2 && m.current_overs === 0 && m.current_balls === 0 && (m.target_runs || 0) > 0) ||
            (m.match_status_text || "").toLowerCase().includes("innings break");
          const isRainInterrupted =
            Boolean((m as any).weather_condition?.includes("RAIN")) ||
            (m.match_status_text || "").toLowerCase().includes("rain");

          return (
            <Link
              key={`${m.id}-${idx}`}
              href={`/match/${m.id}`}
              className="group shrink-0 flex items-center gap-3.5 px-3.5 py-1.5 rounded-xl bg-slate-900/90 hover:bg-slate-800/90 border border-slate-800 hover:border-cyan-500/40 transition duration-150 text-xs shadow-sm"
            >
              {/* Match Format & Status */}
              <div className="flex flex-col items-start leading-none gap-0.5">
                <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider">
                  {m.format}
                </span>
                {m.state === "COMPLETED" ? (
                  <span className="text-[9px] font-black text-emerald-400 flex items-center gap-0.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                    RESULT
                  </span>
                ) : isLive ? (
                  isInningsBreak ? (
                    <span className="text-[9px] font-black text-amber-300 flex items-center gap-0.5 animate-pulse">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />
                      INNINGS BREAK
                    </span>
                  ) : isRainInterrupted ? (
                    <span className="text-[9px] font-black text-sky-300 flex items-center gap-0.5 animate-pulse">
                      <span className="w-1.5 h-1.5 rounded-full bg-sky-400 inline-block" />
                      RAIN DELAY
                    </span>
                  ) : (m as any).is_stumps ? (
                    <span className="text-[9px] font-black text-indigo-300 flex items-center gap-0.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 inline-block" />
                      STUMPS
                    </span>
                  ) : (m as any).is_stopped ? (
                    <span className="text-[9px] font-black text-amber-300 flex items-center gap-0.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />
                      STOPPED
                    </span>
                  ) : (
                    <span className="text-[9px] font-black text-rose-400 flex items-center gap-0.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-ping inline-block" />
                      LIVE
                    </span>
                  )
                ) : (
                  <span className="text-[9px] font-medium text-cyan-400">
                    {formatKickoffCountdown(m.scheduled_start)}
                  </span>
                )}
              </div>

              {/* Live or Completed Score with Batting Country */}
              {(isLive || m.state === "COMPLETED") && (
                <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-lg border text-[11px] font-mono font-bold text-white shrink-0 ${
                  m.state === "COMPLETED" ? "bg-emerald-950/80 border-emerald-800 text-emerald-200" :
                  (m as any).is_stumps ? "bg-indigo-950/80 border-indigo-700/80 text-indigo-200" :
                  (m as any).is_stopped ? "bg-amber-950/80 border-amber-800 text-amber-200" :
                  "bg-rose-950/80 border-rose-800"
                }`}>
                  <span className="text-cyan-300">{(m as any).batting_team_short || metaB.short}:</span>
                  <span className="text-white">{m.current_score}/{m.current_wickets}</span>
                  <span className="text-[9px] text-slate-400">({m.current_overs}.{m.current_balls}ov)</span>
                </div>
              )}

              {/* Team Matchup & Win % */}
              <div className="flex items-center gap-2 shrink-0">
                {/* Team A */}
                <div className="flex items-center gap-1.5">
                  <TeamFlag code={metaA.short} size="xs" />
                  <span className="font-extrabold text-slate-200 text-xs group-hover:text-cyan-300 transition">
                    {metaA.short}
                  </span>
                  <span className="font-mono text-[11px] font-black text-cyan-400">
                    {m.p_team_a}%
                  </span>
                </div>

                <span className="text-slate-600 text-[10px] font-bold">vs</span>

                {/* Team B */}
                <div className="flex items-center gap-1.5">
                  <TeamFlag code={metaB.short} size="xs" />
                  <span className="font-extrabold text-slate-200 text-xs group-hover:text-amber-300 transition">
                    {metaB.short}
                  </span>
                  <span className="font-mono text-[11px] font-black text-amber-400">
                    {m.p_team_b}%
                  </span>
                </div>
              </div>

              <ChevronRight className="w-3 h-3 text-slate-600 group-hover:text-cyan-400 transition transform group-hover:translate-x-0.5" />
            </Link>
          );
        })}
        </div>
      </div>
    </div>
  );
}
