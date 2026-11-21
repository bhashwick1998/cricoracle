export interface TeamMetadata {
  id: string;
  name: string;
  short: string;
  flag: string;
  primaryColor: string;
  secondaryColor: string;
  gradient: string;
  borderGlow: string;
  textAccent: string;
}

export const TEAM_REGISTRY: Record<string, TeamMetadata> = {
  IND: {
    id: "IND",
    name: "India",
    short: "IND",
    flag: "🇮🇳",
    primaryColor: "#0284c7", // Sky/India Blue
    secondaryColor: "#f97316", // Saffron
    gradient: "from-sky-600 via-blue-700 to-indigo-900",
    borderGlow: "border-sky-500/40 shadow-sky-500/20",
    textAccent: "text-sky-400"
  },
  AUS: {
    id: "AUS",
    name: "Australia",
    short: "AUS",
    flag: "🇦🇺",
    primaryColor: "#eab308", // Canary Gold
    secondaryColor: "#15803d", // Bottle Green
    gradient: "from-amber-500 via-yellow-600 to-green-800",
    borderGlow: "border-amber-500/40 shadow-amber-500/20",
    textAccent: "text-amber-400"
  },
  ENG: {
    id: "ENG",
    name: "England",
    short: "ENG",
    flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    primaryColor: "#dc2626", // Lions Red
    secondaryColor: "#1e3a8a", // Navy
    gradient: "from-red-600 via-rose-700 to-blue-900",
    borderGlow: "border-red-500/40 shadow-red-500/20",
    textAccent: "text-rose-400"
  },
  SA: {
    id: "SA",
    name: "South Africa",
    short: "SA",
    flag: "🇿🇦",
    primaryColor: "#16a34a", // Proteas Green
    secondaryColor: "#facc15", // Gold
    gradient: "from-emerald-600 via-green-700 to-yellow-600",
    borderGlow: "border-emerald-500/40 shadow-emerald-500/20",
    textAccent: "text-emerald-400"
  },
  PAK: {
    id: "PAK",
    name: "Pakistan",
    short: "PAK",
    flag: "🇵🇰",
    primaryColor: "#059669", // Emerald Green
    secondaryColor: "#f3f4f6", // White/Gold
    gradient: "from-emerald-600 via-teal-700 to-green-900",
    borderGlow: "border-emerald-500/40 shadow-emerald-500/20",
    textAccent: "text-emerald-400"
  },
  NZ: {
    id: "NZ",
    name: "New Zealand",
    short: "NZ",
    flag: "🇳🇿",
    primaryColor: "#27272a", // Blackcaps
    secondaryColor: "#38bdf8", // Teal/Silver
    gradient: "from-zinc-700 via-zinc-800 to-black",
    borderGlow: "border-cyan-400/30 shadow-cyan-500/10",
    textAccent: "text-slate-200"
  },
  WI: {
    id: "WI",
    name: "West Indies",
    short: "WI",
    flag: "🌴",
    primaryColor: "#831843", // Maroon
    secondaryColor: "#eab308", // Yellow
    gradient: "from-pink-900 via-rose-900 to-amber-700",
    borderGlow: "border-pink-600/40 shadow-pink-600/20",
    textAccent: "text-pink-400"
  },
  SL: {
    id: "SL",
    name: "Sri Lanka",
    short: "SL",
    flag: "🇱🇰",
    primaryColor: "#1d4ed8", // Lion Blue
    secondaryColor: "#eab308", // Gold
    gradient: "from-blue-700 via-indigo-800 to-amber-600",
    borderGlow: "border-blue-500/40 shadow-blue-500/20",
    textAccent: "text-blue-400"
  },
  BAN: {
    id: "BAN",
    name: "Bangladesh",
    short: "BAN",
    flag: "🇧🇩",
    primaryColor: "#047857", // Tigers Green
    secondaryColor: "#dc2626", // Red
    gradient: "from-emerald-700 via-green-800 to-red-800",
    borderGlow: "border-emerald-600/40 shadow-emerald-600/20",
    textAccent: "text-emerald-400"
  },
  AFG: {
    id: "AFG",
    name: "Afghanistan",
    short: "AFG",
    flag: "🇦🇫",
    primaryColor: "#0056b3", // Afghan Blue
    secondaryColor: "#dc2626", // Red
    gradient: "from-blue-700 via-sky-800 to-emerald-900",
    borderGlow: "border-blue-500/40 shadow-blue-500/20",
    textAccent: "text-blue-400"
  },
  ZIM: {
    id: "ZIM",
    name: "Zimbabwe",
    short: "ZIM",
    flag: "🇿🇼",
    primaryColor: "#dc2626", // Zimbabwe Red
    secondaryColor: "#eab308", // Yellow
    gradient: "from-red-700 via-amber-700 to-green-900",
    borderGlow: "border-red-500/40 shadow-red-500/20",
    textAccent: "text-amber-400"
  }
};

export function getTeamMeta(teamIdOrShort?: string, fallbackName?: string): TeamMetadata {
  const key = (teamIdOrShort || "").toUpperCase();
  if (TEAM_REGISTRY[key]) {
    return TEAM_REGISTRY[key];
  }

  // Check by name
  for (const t of Object.values(TEAM_REGISTRY)) {
    if (fallbackName && t.name.toLowerCase() === fallbackName.toLowerCase()) {
      return t;
    }
  }

  return {
    id: key || "UNK",
    name: fallbackName || key || "Unknown",
    short: key.slice(0, 3).toUpperCase() || "TBD",
    flag: "🏏",
    primaryColor: "#475569",
    secondaryColor: "#94a3b8",
    gradient: "from-slate-700 to-slate-900",
    borderGlow: "border-slate-700 shadow-none",
    textAccent: "text-slate-300"
  };
}

export function formatKickoffCountdown(dateString: string): string {
  try {
    const target = new Date(dateString).getTime();
    const now = Date.now();
    const diff = target - now;

    if (diff <= 0) {
      return "In Progress";
    }

    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);

    if (days >= 2) {
      return `Starts in ${days} days`;
    }
    if (days === 1) {
      return "Starts tomorrow";
    }
    if (hours >= 1) {
      return `Starts in ${hours}h`;
    }
    const mins = Math.floor(diff / (1000 * 60));
    return `Starts in ${Math.max(5, mins)} mins`;
  } catch {
    return dateString;
  }
}

export function formatDateTime(dateString: string): { dateStr: string; timeStr: string } {
  try {
    const d = new Date(dateString);
    const dateStr = d.toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric"
    });
    const timeStr = d.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true
    });
    return { dateStr, timeStr };
  } catch {
    return { dateStr: dateString, timeStr: "" };
  }
}
