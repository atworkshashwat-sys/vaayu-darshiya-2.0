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

/** Fetches Active-Alerts context, or `null` when unreachable/invalid. Never throws. */
export async function fetchContextAlerts(
  force = false,
  timeoutMs = 15000,
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
