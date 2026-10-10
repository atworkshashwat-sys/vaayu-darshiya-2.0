// Live web-context client — used ONLY by the dashboard sections that render it
// (Active Alerts now; Best Time and Delhi's Drastic Case next).
//
// Talks to the backend's isolated context routes (GET /api/context/*). The
// TinyFish API key lives on the backend only and never reaches the browser.
// Every payload carries `fetched_at`, `stale`, `ttl_seconds` and `sources` so
// the UI can render fresh / stale / unavailable states without guessing.
// Every failure resolves to `null` — components then keep showing their
// existing (model) content, never a blank card or a stack trace.

export type ContextStatus = "ok" | "stale" | "unavailable";
export type AlertSeverity = "high" | "medium" | "low";

export interface ContextSource {
  name: string;
  url: string;
  fetched_at: string;
}

export interface ContextAlert {
  id: string;
  title: string;
  severity: AlertSeverity;
  summary: string; // ≤ 200 chars, verbatim from the source
  source: string; // display name, e.g. "CAQM"
  url: string;
  official: boolean; // published by an official body (badged "Official")
  published_at: string | null; // null when the source states no date
  fetched_at: string;
}

export interface AlertsContext {
  status: ContextStatus;
  stale: boolean;
  fetched_at: string | null; // null only for the "unavailable" state
  age_seconds: number | null;
  ttl_seconds: number;
  items: ContextAlert[];
  sources: ContextSource[];
  message?: string;
}

/** Fetches Active-Alerts context, or `null` when unreachable/invalid. Never throws.
 *
 * 45 s: a *cold* cache fill runs 3 searches + up to 5 page fetches, each with
 * one retry (6–8 s budget) — past runs took 10–50 s. Warm loads hit the cache
 * in milliseconds; only the first request after a backend restart waits. */
export async function fetchContextAlerts(
  force = false,
  timeoutMs = 45000,
): Promise<AlertsContext | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(`/api/context/alerts${force ? "?force=true" : ""}`, {
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    const data = (await res.json()) as AlertsContext;
    if (!data || typeof data !== "object" || !Array.isArray(data.items)) return null;
    return data;
  } catch {
    return null;
  }
}

// ── Best Time to Go Outside ────────────────────────────────────────────────

export interface WeatherDay {
  source: string;
  url: string;
  fetched_at: string;
  text: string; // the source's own forecast wording
  wind_min_kmph: number | null;
  wind_max_kmph: number | null;
  humidity_pct: number | null;
  temp_c: number | null;
  fog: boolean;
  rain: boolean;
}

export interface WeatherNotice {
  id: string;
  kind: "crop-burning" | "traffic" | "notice";
  title: string;
  summary: string;
  source: string;
  url: string;
  published_at: string | null;
  fetched_at: string;
}

export interface WeatherSlot {
  period: string;
  hours: string;
  ml_status: string; // the ML verdict we sent
  status: string; // ML verdict, or one step down when the forecast triggers it
  adjusted: boolean;
  reasons: string[];
}

export interface SlotInput {
  period: string;
  hours: string;
  status: string;
}

export interface WeatherWindows {
  status: ContextStatus;
  stale: boolean;
  fetched_at: string | null;
  age_seconds: number | null;
  ttl_seconds: number;
  weather: WeatherDay | null;
  notices: WeatherNotice[];
  slots: WeatherSlot[];
  sources: ContextSource[];
  message?: string;
}

/**
 * Sends the ML windows to the backend, which attaches the cached official
 * forecast and a reasoned (never silent) adjustment. `null` on any failure —
 * the component then renders the ML windows exactly as before.
 *
 * 45 s for the same cold-cache reason as the alerts call (search + fetches,
 * one retry each); warm loads return from the 6 h TTL cache immediately.
 */
export async function fetchWeatherWindows(
  slots: SlotInput[],
  force = false,
  timeoutMs = 45000,
): Promise<WeatherWindows | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(`/api/context/weather-windows${force ? "?force=true" : ""}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slots }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    const data = (await res.json()) as WeatherWindows;
    if (!data || typeof data !== "object" || !Array.isArray(data.slots)) return null;
    return data;
  } catch {
    return null;
  }
}

// ── shared display helpers ────────────────────────────────────────────────

/** "1:32 PM" from an ISO timestamp, or null when unparseable. */
export const formatClock = (iso?: string | null): string | null => {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
};

/** "08 Oct" (UTC) from an ISO date/timestamp, or null. */
export const formatDate = (iso: string): string | null => {
  const date = new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleDateString("en-GB", { day: "2-digit", month: "short", timeZone: "UTC" });
};

/** "just now" / "12m" / "3h 20m" — for "last updated X ago" markers. */
export const formatAge = (seconds?: number | null): string => {
  if (seconds === null || seconds === undefined) return "some time";
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const hours = Math.floor(seconds / 3600);
  return `${hours}h ${Math.round((seconds % 3600) / 60)}m`;
};

// ── Case study figures ─────────────────────────────────

/** One figure from job D, each carrying its own source and as-of date. */
export interface CaseStat {
  stat_key: string;
  label: string;
  value: string;
  unit: string | null;
  as_of: string | null;
  source: string;
  url: string;
  fetched_at: string;
}

export interface CaseStudyData {
  stats: CaseStat[];
  fetched_at?: string | null;
  server_time?: string;
  active?: boolean;
}

/**
 * Fetches the stored "Delhi's Drastic Case" figures, or `null` on any
 * failure. Never throws.
 *
 * The route only reads the store — job D refreshes it once a day in the
 * background — so opening the page can never spend a call against the daily
 * TinyFish cap. A figure the sources never stated is simply absent from the
 * array, which is why the caller renders only what it receives rather than
 * reserving a slot for each one (a reserved slot would be a blank card).
 */
export async function fetchCaseStudy(
  timeoutMs = 15000,
): Promise<CaseStudyData | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(`/api/case-study`, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return null;
    const data = (await res.json()) as CaseStudyData;
    if (!data || typeof data !== "object" || !Array.isArray(data.stats)) return null;
    return data;
  } catch {
    return null;
  }
}
