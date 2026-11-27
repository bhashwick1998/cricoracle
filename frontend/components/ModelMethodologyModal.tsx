"use client";

import React from "react";
import { X, ShieldCheck, Cpu, Database, Activity, CheckCircle2, Sparkles, TrendingUp, Layers, Compass } from "lucide-react";

interface ModelMethodologyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ModelMethodologyModal({ isOpen, onClose }: ModelMethodologyModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-fade-in overflow-y-auto">
      <div className="w-full max-w-4xl bg-[#0d1322] border border-slate-700 rounded-3xl p-6 sm:p-8 shadow-2xl relative my-8">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 rounded-xl bg-slate-800 text-slate-400 hover:text-white transition"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-teal-600 to-blue-600 flex items-center justify-center text-white shadow-lg">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-black text-white tracking-tight">
              CricOracle Physics-Informed ML & Monte Carlo Architecture
            </h2>
            <p className="text-xs text-slate-400">
              Conjugate Dirichlet Pitch Priors, Bayesian Log-Odds P₀ Fusion, Conformal Uncertainty Bands &amp; 3,000-Rollout Monte Carlo
            </p>
          </div>
        </div>

        {/* Accuracy Benchmark Metric Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-6">
          <div className="p-3 rounded-2xl bg-slate-900/90 border border-slate-800 text-center">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Stage 4 Inn-2 Chase</span>
            <span className="text-xl font-black text-emerald-400 mt-0.5 block">88.1% / 76.1%</span>
            <span className="text-[10px] text-slate-400">T20I 88.1% (93.7% Dew) | ODI 76.1%</span>
          </div>

          <div className="p-3 rounded-2xl bg-slate-900/90 border border-slate-800 text-center">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">20% Match Win Acc</span>
            <span className="text-xl font-black text-cyan-400 mt-0.5 block">75% / 74%</span>
            <span className="text-[10px] text-slate-400">T20I 74.9% | TEST 73.8% | ODI 66.9%</span>
          </div>

          <div className="p-3 rounded-2xl bg-slate-900/90 border border-slate-800 text-center">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Monte Carlo Rollouts</span>
            <span className="text-xl font-black text-amber-400 mt-0.5 block">3,000</span>
            <span className="text-[10px] text-slate-400">Vectorized 2-Innings Sims</span>
          </div>

          <div className="p-3 rounded-2xl bg-slate-900/90 border border-slate-800 text-center">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">90% Conformal Band</span>
            <span className="text-xl font-black text-indigo-400 mt-0.5 block">84% - 95%</span>
            <span className="text-[10px] text-slate-400">Live Telemetry &amp; Dew Aware</span>
          </div>
        </div>

        {/* The 4 Core Pillars of the Architecture */}
        <div className="space-y-4 text-xs">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            Core Analytical &amp; ML Pillars
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {/* Pillar 1 */}
            <div className="p-3.5 rounded-2xl bg-slate-900/60 border border-slate-800">
              <div className="flex items-center justify-between font-bold text-cyan-400 mb-1">
                <span className="flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-cyan-400" />
                  1. Live Pitch Kalman Drift &amp; Weather/Dew
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800">Atmosphere</span>
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                Tracks 5-way pitch probabilities with hysteresis-damped Kalman drift alongside live Open-Meteo dew-point spread, venue dew priors, and wet-ball spin grip penalties in Day/Night chases.
              </p>
            </div>

            {/* Pillar 2 */}
            <div className="p-3.5 rounded-2xl bg-slate-900/60 border border-slate-800">
              <div className="flex items-center justify-between font-bold text-teal-400 mb-1">
                <span className="flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-teal-400" />
                  2. Phase-Blended Score &amp; Conformal Bands
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-teal-950 border border-teal-800">Stage 4 v2</span>
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                Combines Huber, Ridge, XGBoost, and ExtraTrees regressors over Stern DLS &amp; wicket-era anchors with live 3-over pace, dot/boundary pressure telemetry, and horizon conformal intervals (84%–95% OOS coverage).
              </p>
            </div>

            {/* Pillar 3 */}
            <div className="p-3.5 rounded-2xl bg-slate-900/60 border border-slate-800">
              <div className="flex items-center justify-between font-bold text-amber-400 mb-1">
                <span className="flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-amber-400" />
                  3. Render 22 Master Attributes &amp; Duel Matrix
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-amber-950 border border-amber-800">Attributes</span>
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                Integrates remaining batter dependability, 5-surface pitch averages, death-bowling ratings, and striker-bowler duel expected RPO/hazard directly into live phase rates &amp; resource tanks.
              </p>
            </div>

            {/* Pillar 4 */}
            <div className="p-3.5 rounded-2xl bg-slate-900/60 border border-slate-800">
              <div className="flex items-center justify-between font-bold text-rose-400 mb-1">
                <span className="flex items-center gap-1.5">
                  <TrendingUp className="w-3.5 h-3.5 text-rose-400" />
                  4. Stern DLS Chase Fusion &amp; Monte Carlo
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-rose-950 border border-rose-800">Rollouts</span>
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                Fuses calibrated Innings-1 log-odds and Stern DLS / dew-adjusted Innings-2 chase models with 3,000 SHA-256 deterministic Monte Carlo rollouts and pitch-held counterfactual WPA.
              </p>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2 text-[11px] text-slate-400">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Trained &amp; calibrated strictly chronologically across 1,680+ international matches with zero future leakage
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold transition"
          >
            Got It
          </button>
        </div>
      </div>
    </div>
  );
}
