function normalizeOrigin(raw?: string): string {
  if (!raw) return "";
  let url = raw.trim();
  if (!url) return "";
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    const isLocalHost = url.startsWith("localhost") || url.startsWith("127.0.0.1");
    url = `${isLocalHost ? "http" : "https"}://${url}`;
  }
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") {
      return `http://${host}:8000`;
    }
  }
  const configured = normalizeOrigin(process.env.NEXT_PUBLIC_API_URL);
  if (configured) return configured;
  return "";
}

export function apiUrl(path: string): string {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  const base = getApiBaseUrl();
  return base ? `${base}${cleanPath}` : cleanPath;
}

export function isWsEnabled(): boolean {
  return process.env.NEXT_PUBLIC_ENABLE_WS === "true";
}

export function getWsUrl(): string | null {
  if (!isWsEnabled() || typeof window === "undefined") {
    return null;
  }
  const apiBase = getApiBaseUrl();
  if (apiBase) {
    const wsBase = apiBase.replace(/^http:\/\//i, "ws://").replace(/^https:\/\//i, "wss://");
    return `${wsBase}/ws/matches`;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/matches`;
}

export function getPollIntervalMs(baseMs = 15000, maxJitterMs = 3000): number {
  return baseMs + Math.floor(Math.random() * maxJitterMs);
}

export function mergeMatchLiveDelta(prev: any, delta: any): any {
  if (!prev || delta?.view === "full") {
    return delta;
  }

  const prevScoreProj = prev.score_projections || {};
  const deltaScoreProj = delta.score_projections;
  const mergedScoreProj = deltaScoreProj
    ? {
        ...prevScoreProj,
        ...deltaScoreProj,
        four_stage_progression: {
          ...(prevScoreProj.four_stage_progression || {}),
          ...(deltaScoreProj.four_stage_progression || {}),
        },
        stage_performers: {
          ...(prevScoreProj.stage_performers || {}),
          ...(deltaScoreProj.stage_performers || {}),
        },
      }
    : prevScoreProj;

  return {
    ...prev,
    ...delta,
    match: {
      ...(prev.match || {}),
      ...(delta.match || {}),
    },
    score_projections: mergedScoreProj,
  };
}
