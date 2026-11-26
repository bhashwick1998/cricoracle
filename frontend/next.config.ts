import type { NextConfig } from "next";

let backendUrl =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

if (!backendUrl.startsWith("http://") && !backendUrl.startsWith("https://")) {
  const isLocalHost = backendUrl.startsWith("localhost") || backendUrl.startsWith("127.0.0.1");
  backendUrl = `${isLocalHost ? "http" : "https"}://${backendUrl}`;
}

if (backendUrl.endsWith("/")) {
  backendUrl = backendUrl.slice(0, -1);
}

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
