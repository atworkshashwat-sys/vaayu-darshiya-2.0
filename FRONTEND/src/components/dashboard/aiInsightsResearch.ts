// TinyFish live-research client — used ONLY by the AIInsights component.
//
// Talks to the backend's isolated AI Insights route
// (GET /api/ai-insights/research). The TinyFish API key lives on the backend
// only and never reaches the browser. Every failure resolves to `null` so the
// ML insight cards keep rendering exactly as before.

export interface ResearchItem {
  title: string;
  url: string;
  snippet: string;
  site_name?: string;
  publisher?: string;
  date?: string;
}

export interface ResearchExcerpt {
  title: string;
  url: string;
  site_name?: string;
  text: string;
}

export interface AiInsightsResearch {
  status: "ok" | "unavailable";
  provider: string;
  city: string;
  query: string;
  retrieved_at: string;
  cached: boolean;
  items: ResearchItem[];
  excerpt?: ResearchExcerpt | null;
  message?: string | null;
}

/** Fetches live research, or `null` when unreachable/invalid — never throws. */
export async function fetchAiInsightsResearch(
  city: string,
  timeoutMs = 22000,
): Promise<AiInsightsResearch | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(`/api/ai-insights/research?city=${encodeURIComponent(city)}`, {
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    const data = (await res.json()) as AiInsightsResearch;
    if (!data || typeof data !== "object" || !Array.isArray(data.items)) return null;
    return data;
  } catch {
    return null;
  }
}
