/**
 * HCHO hotspot layer — data, colour ramp and the IDW field renderer.
 *
 * HCHO here is the Sentinel-5P satellite VERTICAL COLUMN in µmol/m² (the
 * database stores mol/m²; scripts/build_hcho_hotspots.py scales ×1e6 before
 * writing data/hcho_hotspots.json). It is not ground-level air and it has no
 * WHO/CPCB limit, so hotspots are defined RELATIVE to the region on the pass
 * date — never against a statutory band.
 *
 * The map draws a smooth inverse-distance-weighted field (power 2, 15 km
 * influence) rather than cells, so nothing about the picture suggests a
 * resolution the ~5 × 3.5 km satellite pixel cannot support.
 */

export interface HchoStation {
  id: string;
  name: string;
  lat: number;
  lon: number;
  /** Daily mean of the latest pass, µmol/m². */
  value: number;
  /** Date that value was observed ("YYYY-MM-DD"). */
  date: string;
  /** Mean over the 7-day window, µmol/m². */
  mean_7d: number;
  /** Days inside the window that carry HCHO. */
  days: number;
  /** True when the station's value is from the latest pass itself. */
  latest_pass: boolean;
}

export interface HchoClusterStation {
  id: string;
  name: string;
  value: number;
}

export interface HchoHotspot {
  rank: number;
  peak: number;
  ratio_to_median: number | null;
  /** Last-N-days count this cluster was a hotspot ("5 of 7 days"). */
  persistence: number;
  days: number;
  stations: HchoClusterStation[];
  /** Closed [lon, lat] ring outlining the cluster. */
  polygon: [number, number][];
  center: [number, number];
}

export interface HchoRegionStats {
  as_of: string;
  n_stations: number;
  n_stations_window: number;
  median: number | null;
  p75: number | null;
  p90: number | null;
  threshold: number | null;
  elevated_floor: number;
}

export interface HchoHotspotsPayload {
  available: boolean;
  reason?: string;
  as_of?: string;
  age_days?: number | null;
  archived?: boolean;
  generated_at?: string;
  region?: string;
  unit?: string;
  window_days?: number;
  rule?: string;
  notes?: Record<string, string>;
  region_stats?: HchoRegionStats;
  stations: HchoStation[];
  no_recent_data: Array<{ id: string; name: string; lat: number; lon: number; reason?: string }>;
  hotspots: HchoHotspot[];
}

export class HchoHotspotsError extends Error {}

/**
 * Fetch the offline-built layer. Rejects with HchoHotspotsError carrying a
 * one-line, user-presentable reason — never a stack trace.
 */
export async function fetchHchoHotspots(signal?: AbortSignal): Promise<HchoHotspotsPayload> {
  let res: Response;
  try {
    res = await fetch("/api/hcho-hotspots", { signal });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") throw err;
    throw new HchoHotspotsError("HCHO layer unreachable — the backend is not responding");
  }
  if (!res.ok) throw new HchoHotspotsError(`HCHO layer unavailable (HTTP ${res.status})`);
  let payload: HchoHotspotsPayload;
  try {
    payload = (await res.json()) as HchoHotspotsPayload;
  } catch {
    throw new HchoHotspotsError("HCHO layer returned an unreadable response");
  }
  if (!payload.available) {
    throw new HchoHotspotsError(payload.reason ?? "HCHO hotspot data is not available");
  }
  return payload;
}

// --- colour ramp: one sequential pale → deep ramp, shared with the legend ----

// The five sequential stops defined in src/styles.css (--hcho-1 … --hcho-5):
// pale sand → amber → warm orange → burnt red → deep ember, interpolated
// linearly with no step bands.
export const HCHO_RAMP: ReadonlyArray<readonly [number, readonly [number, number, number]]> = [
  [0, [243, 233, 210]], // #F3E9D2 hcho-1 pale sand
  [22.5, [232, 201, 138]], // #E8C98A hcho-2
  [45, [217, 142, 78]], // #D98E4E hcho-3
  [67.5, [180, 71, 43]], // #B4472B hcho-4
  [90, [92, 31, 26]], // #5C1F1A hcho-5 deep ember
];

/** Highest value the ramp stops at — above this, the deepest colour. */
export const HCHO_RAMP_MAX = HCHO_RAMP[HCHO_RAMP.length - 1][0];

/** Continuous value → RGB (linear between stops). */
export function rampColor(value: number): [number, number, number] {
  const v = Math.max(0, Math.min(HCHO_RAMP_MAX, value));
  for (let i = 1; i < HCHO_RAMP.length; i++) {
    const [v1, c1] = HCHO_RAMP[i];
    if (v <= v1) {
      const [v0, c0] = HCHO_RAMP[i - 1];
      const t = v1 === v0 ? 0 : (v - v0) / (v1 - v0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * t),
        Math.round(c0[1] + (c1[1] - c0[1]) * t),
        Math.round(c0[2] + (c1[2] - c0[2]) * t),
      ];
    }
  }
  const last = HCHO_RAMP[HCHO_RAMP.length - 1][1];
  return [last[0], last[1], last[2]];
}

/** CSS gradient for the legend, built from the same stops. */
export function rampCssGradient(): string {
  return `linear-gradient(90deg, ${HCHO_RAMP.map(
    ([v, c]) => `rgb(${c[0]},${c[1]},${c[2]}) ${Math.round((v / HCHO_RAMP_MAX) * 100)}%`,
  ).join(", ")})`;
}

/**
 * Ramp colour at the middle of each label band (edges live in data.ts:
 * 15/30/50/70 µmol/m²). Legends that show the band *labels* sample the same
 * continuous ramp, so the field itself is never drawn in step bands.
 */
export function hchoBandSwatch(band: number): [number, number, number] {
  const mids = [7.5, 23, 40.5, 60.5, 85];
  const i = Math.min(mids.length - 1, Math.max(0, Math.round(band)));
  return rampColor(mids[i]);
}

// --- geometry ---------------------------------------------------------------

const EARTH_KM = 6371.0088;

/** Great-circle distance in km. */
export function distanceKm(
  aLat: number,
  aLon: number,
  bLat: number,
  bLon: number,
): number {
  const toRad = Math.PI / 180;
  const p1 = aLat * toRad;
  const p2 = bLat * toRad;
  const dp = p2 - p1;
  const dl = (bLon - aLon) * toRad;
  const h =
    Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * EARTH_KM * Math.asin(Math.min(1, Math.sqrt(h)));
}

// --- inverse-distance weighting --------------------------------------------

export interface IdwStation {
  lat: number;
  lon: number;
  value: number;
}

export interface IdwBounds {
  north: number;
  south: number;
  east: number;
  west: number;
}

export const IDW_POWER = 2;
export const IDW_RADIUS_KM = 15;

/**
 * RGBA byte raster. The buffer is pinned to `ArrayBuffer` (not the wider
 * `ArrayBufferLike`) because the DOM's `ImageData` constructor rejects
 * `SharedArrayBuffer`-backed views — `new Uint8ClampedArray(n)` already
 * returns this, the annotation just stops it widening.
 */
export interface HchoRaster {
  data: Uint8ClampedArray<ArrayBuffer>;
  width: number;
  height: number;
}

/**
 * Smooth IDW field → RGBA bytes.
 *
 * weight_i = K(d_i) / d_i^p with the falloff kernel K(d) = (1 − (d/R)²)² for
 * d < R and 0 beyond: the field is continuous, has no cells, and fades to
 * transparent at the 15 km influence radius — so no strong colour ever
 * appears far from a station. Alpha uses the same kernel, which is what makes
 * the edge fade rather than stop dead.
 */
export function idwRaster(
  stations: IdwStation[],
  bounds: IdwBounds,
  width: number,
  height: number,
  radiusKm: number = IDW_RADIUS_KM,
  power: number = IDW_POWER,
): HchoRaster {
  const data = new Uint8ClampedArray(width * height * 4);
  if (!stations.length || width <= 0 || height <= 0) {
    return { data, width, height };
  }

  const latSpan = Math.max(1e-6, bounds.north - bounds.south);
  const lonSpan = Math.max(1e-6, bounds.east - bounds.west);
  const midLat = (bounds.north + bounds.south) / 2;
  const kmPerLat = 111.32;
  const kmPerLon = 111.32 * Math.cos((midLat * Math.PI) / 180);

  for (let py = 0; py < height; py++) {
    // Rows run north → south in the image.
    const lat = bounds.north - ((py + 0.5) / height) * latSpan;
    for (let px = 0; px < width; px++) {
      const lon = bounds.west + ((px + 0.5) / width) * lonSpan;
      let num = 0;
      let den = 0;
      let nearest = Infinity;
      for (const s of stations) {
        const dy = (s.lat - lat) * kmPerLat;
        const dx = (s.lon - lon) * kmPerLon;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < nearest) nearest = d;
        if (d >= radiusKm) continue;
        const k = 1 - (d / radiusKm) ** 2;
        const w = (k * k) / Math.max(d, 0.5) ** power;
        num += w * s.value;
        den += w;
      }
      const o = (py * width + px) * 4;
      if (den <= 0 || nearest >= radiusKm) {
        data[o + 3] = 0; // fully transparent — outside every influence radius
        continue;
      }
      const [r, g, b] = rampColor(num / den);
      const kf = 1 - (nearest / radiusKm) ** 2;
      // Alpha = the kernel of the nearest station, capped so overlapping
      // stations never exceed the layer's design opacity.
      const alpha = Math.round(235 * kf * kf);
      data[o] = r;
      data[o + 1] = g;
      data[o + 2] = b;
      data[o + 3] = alpha;
    }
  }
  return { data, width, height };
}

/** Raster → PNG data URL through an offscreen canvas (MapLibre image source). */
export function rasterToDataUrl(raster: HchoRaster): string {
  const canvas = document.createElement("canvas");
  canvas.width = raster.width;
  canvas.height = raster.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  // Smoothing on: MapLibre stretches the quad and the browser's bilinear
  // filter plus this keep the field soft at every zoom.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  const img = ctx.createImageData(raster.width, raster.height);
  img.data.set(raster.data);
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

/** Stations that feed the field: latest pass only. */
export function interpolationStations(payload: HchoHotspotsPayload): IdwStation[] {
  return (payload.stations ?? [])
    .filter((s) => s.latest_pass && Number.isFinite(s.value))
    .map((s) => ({ lat: s.lat, lon: s.lon, value: s.value }));
}

/** "2026-06-01" → "1 Jun 2026" (no timezone tricks — it is a date, not a time). */
export function formatPassDate(isoDate?: string | null): string {
  if (!isoDate) return "unknown date";
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(isoDate);
  if (!m) return isoDate;
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  // Groups are (year, month, day) — m[2] is the month, m[3] the day.
  return `${Number(m[3])} ${months[Number(m[2]) - 1]} ${m[1]}`;
}
