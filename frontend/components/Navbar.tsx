"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Radio,
  RotateCcw,
  Sparkles,
  Layers,
  CheckCircle2,
  Menu,
  X
} from "lucide-react";

interface NavbarProps {
  onSync?: () => Promise<void>;
  onReset: () => Promise<void>;
  onUrlIngest?: (url: string) => Promise<void>;
  isSyncing?: boolean;
  wsConnected: boolean;
  sandboxMode?: boolean;
  onToggleSandbox?: () => void;
  activeTab: string;
  onSelectTab: (tab: string) => void;
}

export default function Navbar({
  onReset,
  wsConnected,
  activeTab,
  onSelectTab
}: NavbarProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleSelectTab = (tab: string) => {
    onSelectTab(tab);
    setMobileMenuOpen(false);
  };

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-slate-800/80 bg-[#090d16]/95 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-3">
          {/* Logo & Platform Title */}
          <Link href="/" className="flex items-center gap-2.5 sm:gap-3 group shrink-0">
            <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-gradient-to-tr from-cyan-600 via-blue-600 to-indigo-600 p-[1px] shadow-lg shadow-cyan-500/20 group-hover:shadow-cyan-500/40 transition">
              <div className="w-full h-full bg-[#0d1322] rounded-[11px] flex items-center justify-center">
                <Radio className="w-4 h-4 text-cyan-400 animate-pulse" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-1.5 sm:gap-2">
                <span className="font-black text-base sm:text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent">
                  CricOracle
                </span>
                <span className="text-[9px] uppercase font-bold tracking-wider px-1.5 sm:px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
                  AI Predictor
                </span>
              </div>
              <p className="text-[10px] sm:text-[11px] text-slate-400 font-medium hidden sm:block">
                International Cricket Predictive Intelligence
              </p>
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1.5 text-xs font-semibold">
            <button
              onClick={() => handleSelectTab("LIVE")}
              className="px-3 py-1.5 rounded-xl transition flex items-center gap-1.5 bg-rose-600/25 text-rose-300 border border-rose-600 shadow-sm"
            >
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
              </span>
              Live Matches
            </button>
          </nav>

          {/* Action Toolbar */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Live Telemetry Pulse */}
            <div
              className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-xl text-xs font-semibold border ${
                wsConnected
                  ? "bg-emerald-950/60 text-emerald-300 border-emerald-800/60"
                  : "bg-slate-800/60 text-slate-400 border-slate-700/60"
              }`}
              title={wsConnected ? "Real-time ball-by-ball telemetry active" : "Connecting to telemetry stream..."}
            >
              <span className="relative flex h-2 w-2">
                {wsConnected && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                )}
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    wsConnected ? "bg-emerald-500" : "bg-slate-500"
                  }`}
                ></span>
              </span>
              <span className="hidden sm:inline">
                {wsConnected ? "Live Telemetry" : "Connecting"}
              </span>
            </div>

            <button
              onClick={onReset}
              title="Reset matches to benchmark initial state"
              className="p-2 rounded-xl bg-slate-800/60 hover:bg-slate-700 text-slate-400 hover:text-slate-200 border border-slate-700/50 transition shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>

            {/* Mobile Hamburger Toggle Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 lg:hidden border border-slate-700/60 transition shrink-0"
              aria-label="Toggle navigation menu"
            >
              {mobileMenuOpen ? (
                <X className="w-4 h-4 text-cyan-400" />
              ) : (
                <Menu className="w-4 h-4 text-slate-300" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Dropdown Drawer */}
        {mobileMenuOpen && (
          <div className="lg:hidden border-t border-slate-800/90 bg-[#090d16]/98 px-4 py-3 space-y-2 animate-fadeIn shadow-2xl">
            <button
              onClick={() => handleSelectTab("LIVE")}
              className="w-full px-3 py-2.5 rounded-xl transition flex items-center justify-between text-xs font-semibold bg-rose-600/25 text-rose-300 border border-rose-600"
            >
              <div className="flex items-center gap-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
                </span>
                <span>Live Matches</span>
              </div>
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
            </button>
          </div>
        )}
      </header>
    </>
  );
}
