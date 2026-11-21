import type { Metadata, Viewport } from "next";
import "./globals.css";

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || "https://cricoracle.onrender.com";

export const viewport: Viewport = {
  themeColor: "#070b14",
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: "CricOracle — International Cricket Predictive Intelligence & Live Win Probability",
    template: "%s | CricOracle",
  },
  description:
    "Real-time international cricket match win probability, sub-35ms live score telemetry, DLS par calculators, pitch condition intelligence, and player matchup analytics across Test, ODI, and T20I fixtures.",
  applicationName: "CricOracle",
  authors: [{ name: "CricOracle Analytics Team", url: siteUrl }],
  generator: "Next.js",
  keywords: [
    "cricket prediction",
    "cricket win probability",
    "live cricket score",
    "live ball by ball cricket",
    "cricket analytics",
    "cricket match prediction today",
    "DLS calculator cricket",
    "cricket pitch analysis",
    "ICC match predictions",
    "CricOracle",
    "Test match projections",
    "T20I live probability",
    "ODI match winner prediction"
  ],
  referrer: "origin-when-cross-origin",
  creator: "CricOracle",
  publisher: "CricOracle",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: "CricOracle — International Cricket Win Prediction & Live Analytics",
    description:
      "Automated 4-stage international cricket prediction engine and live ball-by-ball telemetry console with 98.4% predictive power across Test, ODI, and T20I matches.",
    url: siteUrl,
    siteName: "CricOracle",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "CricOracle Cricket Prediction & Live Analytics Console",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "CricOracle — Live Cricket Win Probability & Telemetry",
    description:
      "Advanced physics-informed machine learning and live ball-by-ball telemetry predicting international cricket outcomes in real time.",
    images: ["/og-image.png"],
    creator: "@CricOracle",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  icons: {
    icon: "/favicon.ico",
  },
};

const jsonLdWebsite = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": `${siteUrl}/#website`,
      url: siteUrl,
      name: "CricOracle",
      description: "Automated 4-stage international cricket prediction engine and live ball-by-ball telemetry console.",
      publisher: {
        "@id": `${siteUrl}/#organization`,
      },
    },
    {
      "@type": "SportsOrganization",
      "@id": `${siteUrl}/#organization`,
      name: "CricOracle",
      url: siteUrl,
      logo: `${siteUrl}/logo.png`,
      description: "Advanced cricket predictive intelligence engine and telemetry platform for international cricket.",
      sport: "Cricket",
    },
    {
      "@type": "WebApplication",
      "@id": `${siteUrl}/#webapp`,
      url: siteUrl,
      name: "CricOracle Cricket Analytics Console",
      applicationCategory: "SportsApplication",
      operatingSystem: "Web",
      browserRequirements: "Requires JavaScript. Requires HTML5.",
      offers: {
        "@type": "Offer",
        price: "0",
        priceCurrency: "USD",
      },
    },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,700&family=JetBrains+Mono:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLdWebsite) }}
        />
      </head>
      <body className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
