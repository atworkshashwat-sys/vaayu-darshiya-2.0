/**
 * palette.ts — the JS bridge to the colour tokens in src/styles.css.
 *
 * styles.css is the single source of truth for the warm palette. This module
 * reads those CSS custom properties at runtime so MapLibre paint props,
 * Recharts stroke/fill values and jsPDF colours all follow the same tokens.
 * The embedded fallbacks mirror the CSS values and are only used where no
 * stylesheet is present (unit tests, SSR) — keep them in sync with styles.css.
 */

const FALLBACK: Record<string, string> = {
  paper: "#f6f1e7",
  "paper-2": "#fbf8f1",
  ink: "#1f1b16",
  "ink-muted": "#6b6257",
  rule: "#d9d0bf",
  accent: "#a8432a",
  "sev-good": "#5e8c61",
  "sev-satisfactory": "#a9b44a",
  "sev-moderate": "#e2b33c",
  "sev-poor": "#e07b39",
  "sev-very-poor": "#b33a2e",
  "sev-severe": "#6e1f2b",
  "hcho-1": "#f3e9d2",
  "hcho-2": "#e8c98a",
  "hcho-3": "#d98e4e",
  "hcho-4": "#b4472b",
  "hcho-5": "#5c1f1a",
};

/** Read a token from :root; falls back to the mirrored hex when absent. */
export function token(name: string): string {
  if (typeof document !== "undefined" && typeof getComputedStyle === "function") {
    const v = getComputedStyle(document.documentElement)
      .getPropertyValue(`--${name}`)
      .trim();
    if (v) return v;
  }
  return FALLBACK[name] ?? "#000000";
}

export type Rgb = [number, number, number];

/** "#rrggbb" (or "rgb(r,g,b)") → [r, g, b] 0-255. */
export function toRgb(color: string): Rgb {
  const rgbMatch = color.match(/rgba?\(([^)]+)\)/i);
  if (rgbMatch) {
    const parts = rgbMatch[1].split(/[\s,/]+/).filter(Boolean).slice(0, 3);
    return parts.map((p) => Math.round(Number(p))) as Rgb;
  }
  const hex = color.replace("#", "");
  const full =
    hex.length === 3
      ? hex
          .split("")
          .map((c) => c + c)
          .join("")
      : hex.slice(0, 6);
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)) as Rgb;
}

/** Token as an [r, g, b] triple (0-255) for MapLibre / jsPDF consumers. */
export function tokenRgb(name: string): Rgb {
  return toRgb(token(name));
}

/* ── core palette ─────────────────────────────────────────────────────── */

export const paper = () => token("paper");
export const paper2 = () => token("paper-2");
export const ink = () => token("ink");
export const inkMuted = () => token("ink-muted");
export const rule = () => token("rule");
export const accent = () => token("accent");

export const paperRgb = () => tokenRgb("paper");
export const paper2Rgb = () => tokenRgb("paper-2");
export const inkRgb = () => tokenRgb("ink");
export const inkMutedRgb = () => tokenRgb("ink-muted");
export const ruleRgb = () => tokenRgb("rule");
export const accentRgb = () => tokenRgb("accent");

/* ── severity scale (single source of truth) ──────────────────────────── */

/** Token names in severity order: good → severe. */
export const SEVERITY_TOKENS = [
  "sev-good",
  "sev-satisfactory",
  "sev-moderate",
  "sev-poor",
  "sev-very-poor",
  "sev-severe",
] as const;

export const SEVERITY_LABELS = [
  "Good",
  "Satisfactory",
  "Moderate",
  "Poor",
  "Very Poor",
  "Severe",
] as const;

/** Clamp a level (0-5, from any banded scale) and return its fill colour. */
export function severityColor(level: number): string {
  const i = Math.min(SEVERITY_TOKENS.length - 1, Math.max(0, Math.round(level)));
  return token(SEVERITY_TOKENS[i]);
}

export function severityColorRgb(level: number): Rgb {
  const i = Math.min(SEVERITY_TOKENS.length - 1, Math.max(0, Math.round(level)));
  return tokenRgb(SEVERITY_TOKENS[i]);
}

/**
 * Text colour on a severity fill — verified for WCAG AA:
 *   ink on good (4.40:1, use ≥14px semibold), satisfactory (7.59:1),
 *   moderate (8.76:1) and poor (5.76:1); paper-2 on very-poor (5.56:1)
 *   and severe (10.45:1).
 *
 * Contrast-audit correction: white on poor #E07B39 fails AA (2.80:1), so
 * `poor` takes ink like the lighter severity fills instead of white.
 */
export function severityTextColor(level: number): string {
  const i = Math.min(SEVERITY_TOKENS.length - 1, Math.max(0, Math.round(level)));
  return i >= 4 ? paper2() : ink();
}

export function severityTextRgb(level: number): Rgb {
  const i = Math.min(SEVERITY_TOKENS.length - 1, Math.max(0, Math.round(level)));
  return i >= 4 ? paper2Rgb() : inkRgb();
}

/* ── HCHO sequential ramp (5 stops, interpolated linearly) ────────────── */

export const HCHO_RAMP_TOKENS = [
  "hcho-1",
  "hcho-2",
  "hcho-3",
  "hcho-4",
  "hcho-5",
] as const;

/** Ramp stops, low → high, as CSS colours. */
export function hchoRampStops(): string[] {
  return HCHO_RAMP_TOKENS.map(token);
}

export function hchoRampStopRgb(i: number): Rgb {
  const clamped = Math.min(HCHO_RAMP_TOKENS.length - 1, Math.max(0, i));
  return tokenRgb(HCHO_RAMP_TOKENS[clamped]);
}
