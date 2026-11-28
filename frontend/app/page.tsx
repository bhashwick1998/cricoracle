"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import Navbar from "@/components/Navbar";
import MatchCard, { MatchData } from "@/components/MatchCard";
import LiveScoreTicker from "@/components/LiveScoreTicker";
import MatchScheduleTable from "@/components/MatchScheduleTable";
import { apiUrl, getWsUrl, getPollIntervalMs } from "@/lib/apiConfig";
import {
  Layers,
  Radio,
  Flame,
  CheckCircle2,
  Users2,
  Calendar,
  Sparkles,
  Info,
  Search,
  Trophy,
  Target,
  Activity,
  ShieldCheck,
  Cpu,
  BarChart3,
  Sliders,
  LayoutGrid,
  Table as TableIcon,
  ArrowUpDown,
  Compass
} from "lucide-react";

export default function HomePage() {
  const [matches, setMatches] = useState<MatchData[]>([]);
  const [stage3Matches, setStage3Matches] = useState<MatchData[]>([]);
  const [loading, setLoading] = useState(true);
  const [wakingBackend, setWakingBackend] = useState(false);
  const [activeTab, setActiveTab] = useState<string>("LIVE");
  const [formatFilter, setFormatFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [viewMode, setViewMode] = useState<"grid" | "table">("grid");
  const [sortBy, setSortBy] = useState<"date" | "edge" | "par">("date");
  const [isSyncing, setIsSyncing] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [sandboxMode, setSandboxMode] = useState<boolean>(false);
  const wakeRetryRef = React.useRef<{ timer: ReturnType<typeof setTimeout> | null; attempts: number; disposed: boolean }>({
    timer: null,
    attempts: 0,
    disposed: false,
  });

  // Fetch matches from CDN-cacheable backend API (respecting s-maxage=15 + ETag)
  const fetchMatches = useCallback(async () => {
    if (wakeRetryRef.current.timer) {
      clearTimeout(wakeRetryRef.current.timer);
      wakeRetryRef.current.timer = null;
    }
    const primaryUrl = apiUrl("/api/matches");
    const stage3Url = apiUrl("/api/matches?state=XI_AND_TOSS_LOCKED");
    const wsActive = Boolean(getWsUrl());
    let keepWakingSpinner = false;
    try {
      const [resLive, resStage3] = await Promise.all([
        fetch(primaryUrl).then(async (r) => (!r.ok && primaryUrl !== "/api/matches" ? fetch("/api/matches") : r)),
        fetch(stage3Url).catch(() => null),
      ]);
      const res = resLive;
      if (resStage3 && resStage3.ok) {
        const s3Data = await resStage3.json();
        setStage3Matches(Array.isArray(s3Data) ? s3Data : []);
      }
      if (res.ok) {
        const data = await res.json();
        setMatches(Array.isArray(data) ? data : []);
        setWakingBackend(false);
        wakeRetryRef.current.attempts = 0;
        if (!wsActive) setWsConnected(true);
      } else {
        if (!wsActive) setWsConnected(false);
        if (res.status >= 500 && wakeRetryRef.current.attempts < 20 && !wakeRetryRef.current.disposed) {
          keepWakingSpinner = true;
          wakeRetryRef.current.attempts += 1;
          setWakingBackend(true);
          wakeRetryRef.current.timer = setTimeout(fetchMatches, 3000);
        } else {
          setWakingBackend(false);
        }
      }
    } catch (err) {
      console.error("Failed to fetch matches:", err);
      if (!wsActive) setWsConnected(false);
      if (wakeRetryRef.current.attempts < 20 && !wakeRetryRef.current.disposed) {
        keepWakingSpinner = true;
        wakeRetryRef.current.attempts += 1;
        setWakingBackend(true);
        wakeRetryRef.current.timer = setTimeout(fetchMatches, 3000);
      } else {
        setWakingBackend(false);
      }
    } finally {
      if (!keepWakingSpinner) {
        setLoading(false);
      }
    }
  }, []);

  // Optional WebSocket (disabled by default on public crowds; inline-only, zero HTTP refetch) + jittered CDN polling
  useEffect(() => {
    fetchMatches();

    const wsUrl = getWsUrl();
    let socket: WebSocket | null = null;
    let reconnectTimeout: any = null;
    let pollTimeout: any = null;
    let disposed = false;

    const connectWs = () => {
      if (!wsUrl || disposed) return;
      try {
        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
          setWsConnected(true);
        };

        socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (
              (data.type === "STATE_CHANGE" ||
                data.type === "NEW_DELIVERY" ||
                data.type === "LIVE_TELEMETRY_SYNCED" ||
                data.type === "CRICBUZZ_LIVE_INGESTED") &&
              data.match
            ) {
              setMatches((prev) =>
                prev.map((m) => (m.id === data.match.id ? { ...m, ...data.match } : m))
              );
            }
          } catch (e) {
            console.error("Failed to parse ws event:", e);
          }
        };

        socket.onclose = () => {
          setWsConnected(false);
          if (!disposed) reconnectTimeout = setTimeout(connectWs, 5000);
        };

        socket.onerror = () => {
          setWsConnected(false);
          socket?.close();
        };
      } catch {
        setWsConnected(false);
        if (!disposed) reconnectTimeout = setTimeout(connectWs, 5000);
      }
    };

    if (wsUrl) {
      connectWs();
    }

    const scheduleNextPoll = () => {
      if (disposed) return;
      pollTimeout = setTimeout(async () => {
        if (typeof document === "undefined" || !document.hidden) {
          await fetchMatches();
        }
        scheduleNextPoll();
      }, getPollIntervalMs(15000, 3000));
    };
    scheduleNextPoll();

    const onVisibilityChange = () => {
      if (typeof document !== "undefined" && !document.hidden) {
        fetchMatches();
      }
    };
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", onVisibilityChange);
    }

    wakeRetryRef.current.disposed = false;
    return () => {
      disposed = true;
      wakeRetryRef.current.disposed = true;
      if (wakeRetryRef.current.timer) clearTimeout(wakeRetryRef.current.timer);
      if (pollTimeout) clearTimeout(pollTimeout);
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", onVisibilityChange);
      }
      if (socket) socket.close();
    };
  }, [fetchMatches]);

  const handleSyncSources = async () => {
    setIsSyncing(true);
    try {
      await fetch(apiUrl("/api/scraper/sync"), { method: "POST" });
      await fetchMatches();
    } finally {
      setIsSyncing(false);
    }
  };

  const handleResetData = async () => {
    try {
      await fetch(apiUrl("/api/simulator/reset"), { method: "POST" });
      await fetchMatches();
    } catch (e) {
      console.error(e);
    }
  };

  const handleUrlIngest = async (url: string) => {
    await fetch(apiUrl("/api/scraper/url"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    await fetchMatches();
  };

  const handleStateChange = async (matchId: string, newState: string) => {
    try {
      const res = await fetch(apiUrl(`/api/matches/${matchId}/state`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: newState }),
      });
      if (res.ok) {
        const data = await res.json();
        setMatches((prev) =>
          prev.map((m) => (m.id === matchId ? { ...m, ...data.match } : m))
        );
      }
    } catch (err) {
      console.error("Failed to transition match state:", err);
    }
  };

  const handleStepDelivery = async (matchId: string) => {
    try {
      const res = await fetch(apiUrl(`/api/matches/${matchId}/delivery`), {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        if (data.match) {
          setMatches((prev) =>
            prev.map((m) => (m.id === matchId ? { ...m, ...data.match } : m))
          );
        }
      }
    } catch (err) {
      console.error("Failed to step delivery:", err);
    }
  };

  // Marquee featured match (Live in-play matches first, then Stage 3 Confirmed Playing 11 & Toss Locked)
  const featuredMatch =
    matches.find((m) => m.state === "LIVE_BALL_BY_BALL" && !m.is_completed) ||
    matches.find((m) => m.state === "XI_AND_TOSS_LOCKED") ||
    matches[0];

  // Live in-play matches (strictly active matches, excluding completed fixtures)
  const liveMatches = matches.filter((m) => m.state === "LIVE_BALL_BY_BALL" && !m.is_completed);

  // Filtered & Sorted Matches
  const filteredMatches = useMemo(() => {
    return matches
      .filter((m) => {
        // Strictly keep Live In-Play matches only (extracted from CricAPI)
        if (m.state !== "LIVE_BALL_BY_BALL" || m.is_completed) return false;

        // Format Filter
        if (formatFilter !== "ALL" && m.format !== formatFilter) {
          return false;
        }

        // Search Query Filter
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase();
          const matchText = `${m.team_a_name} ${m.team_a_short} ${m.team_b_name} ${m.team_b_short} ${m.series_name} ${m.venue_name} ${m.venue_city}`.toLowerCase();
          if (!matchText.includes(q)) {
            return false;
          }
        }

        return true;
      })
      .sort((a, b) => {
        if (sortBy === "edge") {
          const edgeA = Math.abs(a.p_team_a - a.p_team_b);
          const edgeB = Math.abs(b.p_team_a - b.p_team_b);
          return edgeB - edgeA;
        }
        if (sortBy === "par") {
          return b.par_score_projected - a.par_score_projected;
        }
        // default date
        return new Date(a.scheduled_start).getTime() - new Date(b.scheduled_start).getTime();
      });
  }, [matches, activeTab, formatFilter, searchQuery, sortBy]);

  return (
    <div className="flex-1 flex flex-col min-h-screen">
      {/* Top Navbar */}
      <Navbar
        onSync={handleSyncSources}
        onReset={handleResetData}
        onUrlIngest={handleUrlIngest}
        isSyncing={isSyncing}
        wsConnected={wsConnected}
        sandboxMode={sandboxMode}
        onToggleSandbox={() => setSandboxMode(!sandboxMode)}
        activeTab={activeTab}
        onSelectTab={(tab) => setActiveTab(tab)}
      />

      {/* Broadcast Score & Odds Ticker */}
      <LiveScoreTicker matches={matches} loading={loading} wakingBackend={wakingBackend} />

      <main className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 py-5 sm:py-8 w-full flex-1">

        {/* Mobile Quick-Navigation Category Pill */}
        <div className="lg:hidden flex items-center gap-2 overflow-x-auto pb-3 -mx-3 px-3 sm:mx-0 sm:px-0 scrollbar-none mb-4">
          <button
            onClick={() => setActiveTab("LIVE")}
            className="px-3.5 py-1.5 rounded-full text-xs font-bold whitespace-nowrap transition flex items-center gap-1.5 border shrink-0 bg-rose-600/30 text-rose-300 border-rose-500 shadow-sm"
          >
            <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping inline-block" />
            <span>Live Now ({liveMatches.length})</span>
          </button>
        </div>

        {/* Unified Filter & View Bar + Live Matches Hub */}
        <div className="mb-6 space-y-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              <div className="flex items-center gap-2.5 flex-wrap">
                <span className="relative flex h-3 w-3 shrink-0">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-rose-500"></span>
                </span>
                <h2 className="text-2xl font-black text-white tracking-tight">
                  Live International Matches
                </h2>
                <span className="text-xs font-bold text-cyan-300 bg-cyan-950/80 px-2.5 py-0.5 rounded-full border border-cyan-800">
                  {filteredMatches.length} {filteredMatches.length === 1 ? "Match" : "Matches"} In-Play
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Real-time CricAPI ball-by-ball telemetry, calibrated WinViz probability, DLS par targets &amp; dugout resource analysis.
              </p>
            </div>

            {/* View Switcher & Sort Toolbar */}
            <div className="flex items-center gap-3 flex-wrap">
              {/* Format Filter Chips */}
              <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs font-bold">
                <span className="text-slate-400 px-2 font-medium text-[11px]">Format:</span>
                <button
                  onClick={() => setFormatFilter("ALL")}
                  className={`px-3 py-1 rounded-lg transition ${
                    formatFilter === "ALL"
                      ? "bg-cyan-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  All
                </button>
                <button
                  onClick={() => setFormatFilter("TEST")}
                  className={`px-3 py-1 rounded-lg transition ${
                    formatFilter === "TEST"
                      ? "bg-cyan-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  TEST
                </button>
                <button
                  onClick={() => setFormatFilter("T20I")}
                  className={`px-3 py-1 rounded-lg transition ${
                    formatFilter === "T20I"
                      ? "bg-cyan-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  T20I
                </button>
                <button
                  onClick={() => setFormatFilter("ODI")}
                  className={`px-3 py-1 rounded-lg transition ${
                    formatFilter === "ODI"
                      ? "bg-cyan-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  ODI
                </button>
              </div>

              {/* Sort By Dropdown */}
              <div className="flex items-center gap-1.5 bg-slate-900/90 px-3 py-1.5 rounded-xl border border-slate-800 text-xs font-semibold text-slate-300">
                <ArrowUpDown className="w-3.5 h-3.5 text-cyan-400" />
                <span className="text-slate-400 text-[11px]">Sort:</span>
                <select
                  value={sortBy}
                  onChange={(e: any) => setSortBy(e.target.value)}
                  className="bg-transparent text-white font-bold focus:outline-none cursor-pointer text-xs"
                >
                  <option value="date" className="bg-slate-900 text-white">
                    Date (Soonest)
                  </option>
                  <option value="edge" className="bg-slate-900 text-white">
                    Highest AI Edge
                  </option>
                  <option value="par" className="bg-slate-900 text-white">
                    Par Score (High-Low)
                  </option>
                </select>
              </div>

              {/* View Mode Toggle: Cards vs Schedule Table */}
              <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs font-bold">
                <button
                  onClick={() => setViewMode("grid")}
                  title="Card Grid View"
                  className={`p-1.5 rounded-lg transition ${
                    viewMode === "grid"
                      ? "bg-cyan-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  <LayoutGrid className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode("table")}
                  title="Compact Schedule Table View"
                  className={`p-1.5 rounded-lg transition ${
                    viewMode === "table"
                      ? "bg-cyan-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  <TableIcon className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Search Input Bar */}
          <div className="relative w-full max-w-xl">
            <Search className="w-4 h-4 text-slate-400 absolute left-4 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search matches by country or venue (e.g. India, New Zealand, Christchurch)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-11 pr-16 py-2.5 bg-slate-900/80 border border-slate-800 rounded-2xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition shadow-inner"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-3.5 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-white font-medium"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Matches Display (Adaptive Grid vs Table — zero duplicate cards) */}
        {wakingBackend && matches.length === 0 ? (
          <div className="text-center py-16 bg-slate-900/40 rounded-3xl border border-cyan-500/30">
            <Radio className="w-12 h-12 text-cyan-400 mx-auto mb-3 animate-pulse" />
            <h3 className="text-base font-bold text-cyan-300">
              Waking CricOracle telemetry server...
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Initial wake-up after idle sleep takes ~30 seconds. Retrying automatically every 3 seconds.
            </p>
          </div>
        ) : loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-96 rounded-3xl bg-slate-900/50 animate-pulse border border-slate-800"
              />
            ))}
          </div>
        ) : filteredMatches.length === 0 ? (
          <div className="text-center py-16 bg-slate-900/40 rounded-3xl border border-slate-800">
            <Radio className="w-12 h-12 text-slate-600 mx-auto mb-3" />
            <h3 className="text-base font-bold text-slate-300">
              No live international matches match your filter right now
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Monitoring CricAPI for live senior international matches (TEST, ODI, T20I).
            </p>
          </div>
        ) : viewMode === "table" ? (
          <MatchScheduleTable matches={filteredMatches} />
        ) : (
          <div className={`grid gap-6 ${
            filteredMatches.length === 1
              ? "grid-cols-1 max-w-3xl mx-auto"
              : filteredMatches.length === 2
              ? "grid-cols-1 md:grid-cols-2"
              : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
          }`}>
            {filteredMatches.map((match) => (
              <MatchCard
                key={match.id}
                match={match}
                onStateChange={handleStateChange}
                onStepDelivery={handleStepDelivery}
                sandboxMode={sandboxMode}
              />
            ))}
          </div>
        )}

        {/* Stage 3 Pre-Match Confirmed Playing 11 & Venue (CricAPI Schedule + 11v11 + Venue) */}
        {stage3Matches.length > 0 && activeTab !== "COMPLETED" && (
          <div className="mt-10 p-4 sm:p-6 rounded-2xl sm:rounded-3xl bg-gradient-to-r from-emerald-950/35 via-slate-900/90 to-cyan-950/30 border border-emerald-500/30 shadow-2xl">
            <div className="flex items-center justify-between gap-4 mb-5 flex-wrap">
              <div>
                <h2 className="text-xl font-black text-white tracking-tight flex items-center gap-2">
                  <span>Stage 3 • Confirmed Playing 11 &amp; Venue Locked</span>
                  <span className="text-xs font-bold text-emerald-300 bg-emerald-950/90 px-2.5 py-0.5 rounded-full border border-emerald-700">
                    {stage3Matches.length} Match{stage3Matches.length > 1 ? "es" : ""} Ready
                  </span>
                </h2>
                <p className="text-xs text-slate-300 mt-0.5">
                  Calibrated 6-signal Stage 3 1st-innings projections shown for both batting-first scenarios upon Playing 11 &amp; venue lock.
                </p>
              </div>
            </div>

            <div className={`grid gap-6 ${
              stage3Matches.length === 1
                ? "grid-cols-1 max-w-3xl mx-auto"
                : stage3Matches.length === 2
                ? "grid-cols-1 md:grid-cols-2"
                : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
            }`}>
              {stage3Matches.map((match) => (
                <MatchCard
                  key={match.id}
                  match={match}
                  onStateChange={handleStateChange}
                  onStepDelivery={handleStepDelivery}
                  sandboxMode={sandboxMode}
                />
              ))}
            </div>
          </div>
        )}

        {/* 4-Stage Prediction Architecture Infographic Banner */}
        <section className="mt-16 p-8 rounded-3xl bg-gradient-to-br from-slate-900/90 via-[#0e172a] to-[#0a1020] border border-slate-800/80 shadow-2xl">
          <div className="max-w-3xl mb-6">
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5 mb-1.5">
              <Cpu className="w-4 h-4 text-cyan-400" />
              Machine Learning Architecture
            </span>
            <h3 className="text-2xl font-black text-white tracking-tight">
              The 4-Stage Multi-Gear Prediction Pipeline
            </h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              CricOracle transitions progressively through 4 automated stages as information is revealed: from long-range macro team ratings to confirmed 11v11 matchups and sub-35ms live ball-by-ball rollouts.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 hover:border-cyan-500/40 transition">
              <span className="font-extrabold text-cyan-400 block text-xs">
                Stage 1: Dirichlet Bayesian Prior & 318 Venues
              </span>
              <p className="text-slate-300 text-[11px] mt-1.5 leading-relaxed">
                Blends Cricsheet format profiles with Render.com 318 verified venues and calendar-month seasonal moisture matrices into 5 Dirichlet posterior probabilities.
              </p>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 hover:border-amber-500/40 transition">
              <span className="font-extrabold text-amber-400 block text-xs">
                Stage 2: Render.com 22 Master Attributes
              </span>
              <p className="text-slate-300 text-[11px] mt-1.5 leading-relaxed">
                Extracts player-level power ratings, clutch death bowling (clutch_death_bowl), and expected wicket values (xw_bowl) to parameterize phase transitions.
              </p>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 hover:border-emerald-500/40 transition">
              <span className="font-extrabold text-emerald-400 block text-xs">
                Stage 3: Crease Duel Tensors & Pre-Match P₀
              </span>
              <p className="text-slate-300 text-[11px] mt-1.5 leading-relaxed">
                Simulates 11x6 pairwise matchups, foreign pitch adaptability, and toss equity to establish calibrated pre-match prior baseline P₀.
              </p>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 hover:border-rose-500/40 transition">
              <span className="font-extrabold text-rose-400 block text-xs">
                Stage 4: Adaptive Kalman Drift & 2-Innings MC
              </span>
              <p className="text-slate-300 text-[11px] mt-1.5 leading-relaxed">
                Kalman filter tracks scoring velocity drift (Flat ──► Dusty), running 3,000 vectorized 2-innings Monte Carlo chase rollouts in &lt; 80ms per delivery.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Public Footer */}
      <footer className="border-t border-slate-800/80 py-8 mt-16 bg-[#070b14] text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <span className="font-black text-slate-100 text-sm">CricOracle</span>
              <span className="text-slate-500">•</span>
              <span>International Cricket Predictive Intelligence Platform</span>
            </div>
            <div className="flex items-center gap-3 text-[11px] text-slate-400 font-medium flex-wrap">
              <span>ICC World Cup 2026</span>
              <span>•</span>
              <span>The Ashes</span>
              <span>•</span>
              <span>Asia Super Cup</span>
              <span>•</span>
              <span>Cricsheet Historical Vectors</span>
            </div>
          </div>

          <div className="border-t border-slate-800/60 pt-4 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] text-slate-500">
            <div>
              © 2026 CricOracle. Sub-35ms live telemetry & vectorized Monte Carlo predictive engine.
            </div>
            <div className="text-slate-400 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>Status: <strong className="text-emerald-400">All ML Workers Operational</strong></span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
