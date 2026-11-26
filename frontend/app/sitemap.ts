import { MetadataRoute } from "next";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const baseUrl =
    process.env.NEXT_PUBLIC_SITE_URL ||
    (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "http://localhost:3000");
  const apiUrl = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  let matchRoutes: MetadataRoute.Sitemap = [];

  try {
    const res = await fetch(`${apiUrl}/api/matches`, { next: { revalidate: 3600 } });
    if (res.ok) {
      const matches = await res.json();
      if (Array.isArray(matches)) {
        matchRoutes = matches.map((m: any) => ({
          url: `${baseUrl}/match/${m.id}`,
          lastModified: new Date(),
          changeFrequency: m.state === "LIVE_BALL_BY_BALL" ? "always" : "hourly",
          priority: m.state === "LIVE_BALL_BY_BALL" ? 0.9 : 0.8,
        }));
      }
    }
  } catch {
    // Graceful fallback if backend is offline during build
    matchRoutes = [
      {
        url: `${baseUrl}/match/m_nz_ind_2026_test_1`,
        lastModified: new Date(),
        changeFrequency: "always",
        priority: 0.9,
      },
      {
        url: `${baseUrl}/match/m_aus_eng_2026_t20i_1`,
        lastModified: new Date(),
        changeFrequency: "daily",
        priority: 0.8,
      },
    ];
  }

  return [
    {
      url: baseUrl,
      lastModified: new Date(),
      changeFrequency: "always",
      priority: 1.0,
    },
    ...matchRoutes,
  ];
}
