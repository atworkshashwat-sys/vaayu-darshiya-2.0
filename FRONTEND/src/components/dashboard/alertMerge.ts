// Pure merge logic for the Active Alerts section — kept out of Alerts.tsx so
// the component file exports only components (react-refresh rule) and so the
// ML / live-advisory separation can be unit-tested without rendering.
//
// Live (web-sourced) advisories always sort above model alerts, duplicates are
// dropped on both sides, and model text is never rewritten by web data.

import { AlertTriangle, Info, ShieldAlert } from "lucide-react";
import type { ContextAlert } from "./contextApi";

export interface Alert {
  id: string;
  severity: "high" | "medium" | "low";
  title: string;
  message: string;
  icon: typeof AlertTriangle;
}

export type DisplayAlert = {
  id: string;
  severity: "high" | "medium" | "low";
  title: string;
  body: string;
  icon: typeof AlertTriangle;
  origin: "live" | "model";
  source?: string;
  url?: string;
  official?: boolean;
  publishedAt?: string | null;
  fetchedAt?: string;
};

const tokenize = (text: string) => new Set((text || "").toLowerCase().match(/[a-z0-9]{3,}/g) ?? []);

/** Overlap coefficient ≥ 0.6 ⇒ same story told twice. */
export const titleOverlaps = (a: string, b: string) => {
  const ta = tokenize(a);
  const tb = tokenize(b);
  if (!ta.size || !tb.size) return false;
  let shared = 0;
  ta.forEach((token) => {
    if (tb.has(token)) shared += 1;
  });
  return shared / Math.min(ta.size, tb.size) >= 0.6;
};

/** Official/press advisories first, then model alerts with duplicates dropped. */
export function mergeAlerts(live: ContextAlert[], model: Alert[]): DisplayAlert[] {
  const seen = new Set<string>();
  const merged: DisplayAlert[] = [];

  for (const item of live) {
    if (!item?.id || seen.has(item.id)) continue;
    seen.add(item.id);
    merged.push({
      id: `live-${item.id}`,
      severity: item.severity,
      title: item.title,
      body: item.summary,
      icon:
        item.severity === "high" ? ShieldAlert : item.severity === "medium" ? AlertTriangle : Info,
      origin: "live",
      source: item.source,
      url: item.url,
      official: item.official,
      publishedAt: item.published_at,
      fetchedAt: item.fetched_at,
    });
  }

  const liveTitles = merged.map((item) => item.title);
  for (const alert of model) {
    if (liveTitles.some((title) => titleOverlaps(title, alert.title))) continue; // de-duplicate
    merged.push({
      id: `model-${alert.id}`,
      severity: alert.severity,
      title: alert.title,
      body: alert.message,
      icon: alert.icon,
      origin: "model",
    });
  }
  return merged;
}
