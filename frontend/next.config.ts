import type { NextConfig } from "next";

let backendUrl =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "https://cricoracle-backend.onrender.com";

// If bare internal service name or missing protocol/domain, route to public onrender host
if (!backendUrl.startsWith("http://") && !backendUrl.startsWith("https://")) {
  if (backendUrl.includes(".onrender.com")) {
    backendUrl = `https://${backendUrl}`;
  } else if (backendUrl === "cricoracle-backend" || !backendUrl.includes(".")) {
    backendUrl = "https://cricoracle-backend.onrender.com";
  } else {
    backendUrl = `https://${backendUrl}`;
  }
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
