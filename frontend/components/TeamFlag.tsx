import React from "react";

interface TeamFlagProps {
  code: string;
  size?: "xs" | "sm" | "md" | "lg";
  className?: string;
}

const SIZE_CLASSES: Record<NonNullable<TeamFlagProps["size"]>, string> = {
  xs: "w-4 h-3 rounded-[2px]",
  sm: "w-5 h-3.5 rounded-[3px]",
  md: "w-7 h-5 rounded",
  lg: "w-9 h-6 rounded-md",
};

/**
 * Crisp inline SVG national team flag badges (consistent across Windows/macOS/Linux
 * instead of relying on OS emoji font regional indicator rendering).
 */
export default function TeamFlag({ code, size = "sm", className = "" }: TeamFlagProps) {
  const c = (code || "").toUpperCase().trim();
  const cls = `${SIZE_CLASSES[size]} overflow-hidden shadow-sm border border-white/15 shrink-0 inline-block align-middle ${className}`;

  switch (c) {
    case "IND":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="India flag">
          <rect width="24" height="5.33" y="0" fill="#FF9933" />
          <rect width="24" height="5.34" y="5.33" fill="#FFFFFF" />
          <rect width="24" height="5.33" y="10.67" fill="#138808" />
          <circle cx="12" cy="8" r="1.9" fill="none" stroke="#000080" strokeWidth="0.7" />
          <circle cx="12" cy="8" r="0.5" fill="#000080" />
        </svg>
      );
    case "NZ":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="New Zealand flag">
          <rect width="24" height="16" fill="#012169" />
          <rect width="10" height="7" fill="#012169" />
          <path d="M0 0 L10 7 M10 0 L0 7" stroke="#FFFFFF" strokeWidth="1.2" />
          <path d="M0 0 L10 7 M10 0 L0 7" stroke="#C8102E" strokeWidth="0.55" />
          <path d="M5 0 V7 M0 3.5 H10" stroke="#FFFFFF" strokeWidth="1.8" />
          <path d="M5 0 V7 M0 3.5 H10" stroke="#C8102E" strokeWidth="1.0" />
          <circle cx="18" cy="4.2" r="1.0" fill="#C8102E" stroke="#FFFFFF" strokeWidth="0.35" />
          <circle cx="20.8" cy="7.2" r="0.9" fill="#C8102E" stroke="#FFFFFF" strokeWidth="0.35" />
          <circle cx="15.4" cy="7.5" r="0.9" fill="#C8102E" stroke="#FFFFFF" strokeWidth="0.35" />
          <circle cx="18" cy="12.0" r="1.05" fill="#C8102E" stroke="#FFFFFF" strokeWidth="0.35" />
        </svg>
      );
    case "AUS":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="Australia flag">
          <rect width="24" height="16" fill="#012169" />
          <path d="M0 0 L10 7 M10 0 L0 7" stroke="#FFFFFF" strokeWidth="1.2" />
          <path d="M5 0 V7 M0 3.5 H10" stroke="#FFFFFF" strokeWidth="1.8" />
          <path d="M5 0 V7 M0 3.5 H10" stroke="#E4002B" strokeWidth="1.0" />
          <circle cx="5" cy="11.5" r="1.5" fill="#FFFFFF" />
          <circle cx="18" cy="4" r="0.8" fill="#FFFFFF" />
          <circle cx="20.5" cy="7" r="0.8" fill="#FFFFFF" />
          <circle cx="16" cy="8" r="0.7" fill="#FFFFFF" />
          <circle cx="18" cy="12" r="0.9" fill="#FFFFFF" />
        </svg>
      );
    case "ENG":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="England flag">
          <rect width="24" height="16" fill="#FFFFFF" />
          <rect x="10.2" y="0" width="3.6" height="16" fill="#CE1124" />
          <rect x="0" y="6.2" width="24" height="3.6" fill="#CE1124" />
        </svg>
      );
    case "SA":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="South Africa flag">
          <rect width="24" height="8" y="0" fill="#E03C31" />
          <rect width="24" height="8" y="8" fill="#001489" />
          <polygon points="0,0 10,8 0,16" fill="#000000" stroke="#FFB81C" strokeWidth="1.2" />
          <path d="M0,0 L10,8 L24,8 M0,16 L10,8" fill="none" stroke="#FFFFFF" strokeWidth="3.2" />
          <path d="M0,0 L10,8 L24,8 M0,16 L10,8" fill="none" stroke="#007749" strokeWidth="2.0" />
        </svg>
      );
    case "PAK":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="Pakistan flag">
          <rect width="24" height="16" fill="#01411C" />
          <rect width="6.5" height="16" fill="#FFFFFF" />
          <circle cx="15.2" cy="8" r="4.0" fill="#FFFFFF" />
          <circle cx="16.3" cy="7.2" r="3.4" fill="#01411C" />
          <circle cx="16.8" cy="6.5" r="0.9" fill="#FFFFFF" />
        </svg>
      );
    case "SL":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="Sri Lanka flag">
          <rect width="24" height="16" fill="#FFB81C" />
          <rect x="1.5" y="1.5" width="3.2" height="13" fill="#00534E" />
          <rect x="4.7" y="1.5" width="3.2" height="13" fill="#EB7400" />
          <rect x="9" y="1.5" width="13.5" height="13" fill="#8D153A" />
          <circle cx="15.7" cy="8" r="2.6" fill="#FFB81C" />
        </svg>
      );
    case "BAN":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="Bangladesh flag">
          <rect width="24" height="16" fill="#006A4E" />
          <circle cx="10.8" cy="8" r="4.6" fill="#F42A41" />
        </svg>
      );
    case "AFG":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="Afghanistan flag">
          <rect width="8" height="16" x="0" fill="#18181B" />
          <rect width="8" height="16" x="8" fill="#BF0000" />
          <rect width="8" height="16" x="16" fill="#007A36" />
          <circle cx="12" cy="8" r="2.2" fill="#FFFFFF" opacity="0.85" />
        </svg>
      );
    case "WI":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="West Indies emblem">
          <rect width="24" height="16" fill="#7B0041" />
          <circle cx="12" cy="8" r="4.2" fill="#FACC15" opacity="0.9" />
          <rect x="11.2" y="6.2" width="1.6" height="5.2" fill="#7B0041" />
        </svg>
      );
    case "ZIM":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="Zimbabwe flag">
          <rect width="24" height="16" fill="#006400" />
          <rect y="2.3" width="24" height="11.4" fill="#FFD200" />
          <rect y="4.6" width="24" height="6.8" fill="#D40000" />
          <rect y="6.8" width="24" height="2.4" fill="#000000" />
          <polygon points="0,0 8.5,8 0,16" fill="#FFFFFF" />
        </svg>
      );
    case "IRE":
      return (
        <svg viewBox="0 0 24 16" className={cls} aria-label="Ireland flag">
          <rect width="8" height="16" x="0" fill="#169B62" />
          <rect width="8" height="16" x="8" fill="#FFFFFF" />
          <rect width="8" height="16" x="16" fill="#FF883E" />
        </svg>
      );
    default:
      return (
        <span
          className={`${SIZE_CLASSES[size]} inline-flex items-center justify-center bg-slate-800 border border-slate-600 text-[9px] font-black text-slate-200 tracking-tighter shrink-0 ${className}`}
        >
          {c.slice(0, 3) || "XI"}
        </span>
      );
  }
}
