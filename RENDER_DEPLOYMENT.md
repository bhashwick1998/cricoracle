# CricOracle — Free-Tier Public Crowd Deployment Guide

This setup is tuned to stay within **Render Free (`750` instance hours/month, `5 GB` outbound bandwidth/month)** even under public match traffic:

- **Backend only on Render Free (`cricoracle-backend`)**: `1` service running 24/7 = `744 hrs/month` (`<= 750 hrs`).
- **Frontend on Vercel Hobby or Cloudflare Pages**: Eliminates the second Node service (halving Render hours) and keeps all JS/CSS bundles off Render's `5 GB` quota.
- **Cloudflare Free CDN in front of Backend (`s-maxage=15`)**: Absorbs live crowd polling (`GET /api/matches`, `GET /api/matches/{id}/live`) at the edge so Render only serves ~1 refresh per `15s` regardless of concurrent viewers.
- **Payload Split + GZip + `ETag` (`304`)**:
  - Initial open loads deduplicated `GET /api/matches/{id}` (`score_projections` sent once, duplicate `concussion_info` bench copies removed).
  - Ongoing polls fetch only `GET /api/matches/{id}/live` (omits static squads, matchup tensors, 22-attribute breakdowns, and pre-match stage trees).
  - `GZipMiddleware` compresses JSON `>10x` and `ETag` returns `304 Not Modified` when state hasn't changed.
- **No WebSocket Refetch / Opt-In WS**: `NEXT_PUBLIC_ENABLE_WS=false` by default so public viewers use CDN-cached `15s–18s` HTTP polling (paused when tabs are hidden).

---

## Step 1: Push Code to Your New GitHub Account
1. Create a new repository on your new GitHub account (e.g., `cricoracle`).
2. Add the new remote locally and push `main`:
   ```bash
   git remote add new-origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO>.git
   git push -u new-origin main
   ```

## Step 2: Deploy Backend to Your New Render.com Account
1. Log in to [dashboard.render.com](https://dashboard.render.com) with your new Render account (connect your new GitHub account when prompted).
2. Click **New +** -> **Blueprint**, select your new GitHub repo.
3. Render reads [`render.yaml`](render.yaml) and configures **one** free service: **`cricoracle-backend`**.
4. Enter your secret value for **`CRICAPI_KEY_PAID`** when prompted, then click **Apply**.
5. Once live, note your Render service hostname (e.g., `cricoracle-backend-xxxx.onrender.com`) and verify `https://<your-render-host>/health` returns `{"status": "online", ...}`.

## Step 3: Choose Your Free CDN Layer (`s-maxage=15`)

### Option A — No Custom Domain (Vercel Edge CDN Proxy)
Use this if you want 100% free `*.vercel.app` + `*.onrender.com` URLs with no custom domain purchase:
1. Leave **`NEXT_PUBLIC_API_URL` UNSET** in Vercel so browsers request same-origin `/api/*` paths.
2. Set **`BACKEND_URL=https://cricoracle-backend-xzcw.onrender.com`** (server-side only in Vercel).
3. [`frontend/app/api/[...path]/route.ts`](frontend/app/api/%5B...path%5D/route.ts) proxies `/api/*` server-side and returns `Vercel-CDN-Cache-Control: public, s-maxage=15, stale-while-revalidate=15` (`300s` for analytics/static endpoints).
   - Vercel Hobby (`100 GB` Fast Data Transfer, `1M` CDN requests/month free) serves repeat crowd polls from edge cache (`x-vercel-cache: HIT`) so Render sees ~1 poll every 15s–30s instead of 1 per viewer.

### Option B — Custom Domain with Cloudflare Free CDN
*(Required if hitting backend directly instead of via Vercel Edge — requests straight to `*.onrender.com` bypass edge caching.)*
1. Add your custom domain (e.g., `your-domain.com`) to **Cloudflare Free** and update your domain registrar's nameservers to Cloudflare.
2. In **Render Dashboard -> `cricoracle-backend` -> Settings -> Custom Domains**, add `api.your-domain.com` and verify the CNAME target shown by Render.
3. In **Cloudflare DNS**:
   - Add a **CNAME** record: `api` -> `<your-render-host>.onrender.com`, Proxy status: **Proxied (Orange Cloud)**.
4. In **Cloudflare SSL/TLS**:
   - Set mode to **Full (strict)**.
5. In **Cloudflare Caching -> Configuration**:
   - Set Caching Level to **Standard**.
6. In **Cloudflare Caching -> Cache Rules**, create a rule **`Respect CricOracle API Origin Cache Headers`**:
   - Match: **Hostname equals `api.your-domain.com`** AND **URI Path starts with `/api/`**.
   - **Cache eligibility**: **Eligible for cache**.
   - **Edge TTL**: **Use cache-control header if present, bypass cache if not** *(respects FastAPI's `s-maxage=15` for live match feeds, `s-maxage=300` for analytics/static data, and `no-store` for `POST` & `/api/system/*`)*.
   - **Browser TTL**: **Respect origin**.

## Step 4: Deploy Frontend (`frontend/`) on Vercel Hobby (or Cloudflare Pages)
1. On [vercel.com](https://vercel.com), import `bhashwick1998/cricoracle` from your GitHub account.
2. Set **Root Directory** to `frontend` (Build command `npm run build`).
3. Set Environment Variables (see [`frontend/.env.example`](frontend/.env.example)):
   - **Option A (No custom domain / Vercel Edge CDN):**
     - `BACKEND_URL` = `https://cricoracle-backend-xzcw.onrender.com`
     - `NEXT_PUBLIC_SITE_URL` = `https://<your-vercel-app>.vercel.app`
     - `NEXT_PUBLIC_ENABLE_WS` = `false`
     - *(Leave `NEXT_PUBLIC_API_URL` empty/unset!)*
   - **Option B (Cloudflare custom domain):**
     - `NEXT_PUBLIC_API_URL` = `https://api.your-domain.com`
     - `NEXT_PUBLIC_SITE_URL` = `https://your-domain.com`
     - `NEXT_PUBLIC_ENABLE_WS` = `false`
4. Deploy.
5. *(Optional hardening)*: On Render **`cricoracle-backend` -> Environment**, set `CORS_ALLOWED_ORIGINS` to your deployed frontend origin (e.g. `https://<your-vercel-app>.vercel.app`).

## Step 5: Verify Production Caching & Bandwidth Behavior
1. Check CDN cache headers via curl or browser DevTools:
   - **Option A (Vercel):** Request `https://<your-vercel-app>.vercel.app/api/matches` twice within 15 seconds -> confirm `x-vercel-cache: HIT` on the second call.
   - **Option B (Cloudflare):** Request `https://api.your-domain.com/api/matches` twice within 15 seconds -> confirm `cf-cache-status: HIT` and `content-encoding: gzip`.
2. Open a match page on your frontend `/match/<id>` with DevTools **Network** open:
   - Initial request hits `GET /api/matches/<id>` (`view: "full"`).
   - Background updates every `~15–18s` hit `GET /api/matches/<id>/live` (`view: "live"`, ~10.6 KB gzipped) and pause automatically when the tab is hidden or the match finishes.
   - Filter **WS** in DevTools: zero WebSocket connections should be opened when `NEXT_PUBLIC_ENABLE_WS=false`.
3. Note on ephemeral Free Tier disk: Free Render instances do not persist `backend/cricoracle.db` across restarts/sleeps. Automated startup sync and cache refreshers ([`render_cache_updater.py`](backend/engine/render_cache_updater.py)) rebuild stale local caches automatically on boot.
