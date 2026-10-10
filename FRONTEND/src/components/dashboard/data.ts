export const TIMES = [
  "12 AM", "1 AM", "2 AM", "3 AM", "4 AM", "5 AM",
  "6 AM", "7 AM", "8 AM", "9 AM", "10 AM", "11 AM",
  "12 PM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM",
  "6 PM", "7 PM", "8 PM", "9 PM", "10 PM", "11 PM",
] as const;

export type TimeLabel = (typeof TIMES)[number];

export function formatTime(timeIdx: number, minute: number): string {
  const h12 = timeIdx === 0 ? 12 : timeIdx > 12 ? timeIdx - 12 : timeIdx;
  const ampm = timeIdx >= 12 ? "PM" : "AM";
  return `${h12}:${String(minute).padStart(2, "0")} ${ampm}`;
}

const API_BASE = "";

async function fetchWithFallback<T>(apiPath: string, _staticPath: string, fallback: T): Promise<T> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 45000);
    const res = await fetch(`${API_BASE}${apiPath}`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return fallback;
  }
}

export interface CityForecast {
  aqi: number[];
  no2: number[];
  o3: number[];
  /** 24h HCHO forecast (µmol/m², EXP_11). Absent when the API serves no HCHO model. */
  hcho?: number[];
  label: string;
  healthAdvisory: string;
  bestHours: number[];
  matchedDate?: string;
  currentHour?: number;
}

export interface TimeData {
  matched_date: string;
  current_hour: number;
  current_minute: number;
  real_now: string;
}

export interface WeeklyTrend {
  days: string[];
  no2: number[];
  o3: number[];
  /** Daily mean PM2.5 (µg/m³) from the project database. */
  pm25?: number[];
  /** Daily mean HCHO (µmol/m²) from the project database. Null on gap days. */
  hcho?: (number | null)[];
}

export interface StationData {
  name: string;
  no2: number;
  o3: number;
  lat: number;
  lon: number;
  temp: number;
  humidity: number;
  wind: number;
  visibility: number;
  // Added by the project database (data/vaayu_final_v3.csv).
  // Optional so cached/static JSON without them still type-checks.
  windDir?: number;
  pm25?: number | null;
  aqi?: number | null;
  aqiCategory?: string | null;
  district?: string | null;
  season?: string | null;
  observedAt?: string;
  /** Observed NO2 / O3 at this station, straight from the database. */
  obsNo2?: number | null;
  obsO3?: number | null;
  /** Model HCHO forecast (µmol/m²) and observed satellite column, EXP_11. */
  hcho?: number | null;
  obsHcho?: number | null;
  /** Issue time of the model run behind `no2` / `o3`. */
  modelRunAt?: string;
  site_id?: number;
  // ── Live layer (spec §3) ────────────────────────────────────────────────
  // Present only when the store has a current web reading for this site.
  // `no2` / `o3` above remain ML output: live values live under `live` so a
  // web-sourced number can never be mistaken for a forecast.
  live?: {
    NO2?: number;
    O3?: number;
    PM25?: number;
    AQI?: number;
    HCHO?: number;
  };
  /** Units for each key in `live` (e.g. { NO2: "ppb", AQI: "index" }). */
  liveUnits?: Record<string, string>;
  /** "live" | "stale" | "offline" — recomputed from the reading's own age. */
  status?: "live" | "stale" | "offline";
  /** Source's own observation time, IST (field `observedAt` above). */
  source?: string;
  /** Whole minutes since `observedAt`. */
  ageMinutes?: number;
  /** How strongly sources disagreed; >0 means the value is not clean live. */
  conflict?: number;
  /** True when the station stopped reporting permanently (e.g. retired 2023). */
  retired?: boolean;
  /** Human note for a non-live status, e.g. "Offline — no live source". */
  statusNote?: string;
}

export interface ScoreBlock {
  n: number;
  r2: number | null;
  mae: number | null;
  rmse: number | null;
  bias: number | null;
}

export interface DatabaseValidation {
  generated_at: string;
  database: { extract: string; rows: number; min_date: string; max_date: string };
  targets: Record<
    string,
    {
      database_column: string;
      model_test: Record<"t1h" | "t24h" | "all", ScoreBlock>;
      database: Record<"t1h" | "t24h" | "all", ScoreBlock>;
      source_agreement: ScoreBlock & {
        correlation: number | null;
        lag_correlation: Record<string, number | null>;
      };
      per_site_t1h: Array<{ site_id: number; site_name: string } & ScoreBlock>;
      overlap: { start: string | null; end: string | null; rows: number };
      verdict: string;
    }
  >;
}

export interface ModelMetricsData {
  model: string;
  metrics: {
    t1h: { NO2: { r2: number; mae: number; rmse: number; best_iter?: number; val_r2?: number; val_mae?: number; val_rmse?: number }; O3: { r2: number; mae: number; rmse: number; best_iter?: number; val_r2?: number; val_mae?: number; val_rmse?: number }; HCHO?: { r2: number; mae: number; rmse: number; best_iter?: number; val_r2?: number; val_mae?: number; val_rmse?: number } };
    t6h: { NO2: { r2: number; mae: number; rmse: number }; O3: { r2: number; mae: number; rmse: number }; HCHO?: { r2: number; mae: number; rmse: number } };
    t12h: { NO2: { r2: number; mae: number; rmse: number }; O3: { r2: number; mae: number; rmse: number }; HCHO?: { r2: number; mae: number; rmse: number } };
    t24h: { NO2: { r2: number; mae: number; rmse: number }; O3: { r2: number; mae: number; rmse: number }; HCHO?: { r2: number; mae: number; rmse: number } };
    all_horizons: {
      NO2: Array<{ horizon: number; r2: number; mae: number; rmse: number }>;
      O3: Array<{ horizon: number; r2: number; mae: number; rmse: number }>;
      /** EXP_11 HCHO — absent in deployments without the HCHO artefacts. */
      HCHO?: Array<{ horizon: number; r2: number; mae: number; rmse: number }>;
    };
    validation_comparison?: {
      NO2: { test: { r2: number; mae: number; rmse: number }; val: { r2: number; mae: number; rmse: number } };
      O3: { test: { r2: number; mae: number; rmse: number }; val: { r2: number; mae: number; rmse: number } };
      HCHO?: { test: { r2: number; mae: number; rmse: number }; val: { r2: number; mae: number; rmse: number } };
    };
    /** Scores the model's predictions against data/vaayu_final_v3.csv.
     *  Produced by scripts/evaluate_accuracy.py. Absent when that has not run. */
    database_validation?: DatabaseValidation;
  };
  feature_importance: {
    NO2: Array<{ feature: string; importance: number }>;
    O3: Array<{ feature: string; importance: number }>;
    HCHO?: Array<{ feature: string; importance: number }>;
  };
  seasonal?: {
    NO2: Array<{ season: string; horizon: number; r2: number; mae: number; rmse: number }>;
    O3: Array<{ season: string; horizon: number; r2: number; mae: number; rmse: number }>;
    HCHO?: Array<{ season: string; horizon: number; r2: number; mae: number; rmse: number }>;
  };
  site_errors?: {
    NO2: Array<{ site_id: number; site_name: string; horizon: number; r2: number; mae: number; rmse: number }>;
    O3: Array<{ site_id: number; site_name: string; horizon: number; r2: number; mae: number; rmse: number }>;
    HCHO?: Array<{ site_id: number; site_name: string; horizon: number; r2: number; mae: number; rmse: number }>;
  };
  hyperparameters: Record<string, number>;
  training: {
    train_years: string;
    val_years: string;
    test_years: string;
    sites: number;
    targets: string[];
    horizons: number;
    features: number;
    /** EXP_11 HCHO uses its own feature set and chronological split. */
    features_hcho?: number;
    hcho_train_years?: string;
    hcho_val_years?: string;
    hcho_test_years?: string;
  };
}

const FALLBACK_FORECAST: CityForecast = {
  aqi: [
    180, 175, 168, 160, 155, 148, 140, 155, 175, 200, 225, 248, 265, 280, 290, 295, 288, 275, 260,
    240, 220, 205, 195, 188,
  ],
  no2: [
    42, 40, 38, 35, 33, 30, 28, 35, 45, 58, 72, 85, 92, 98, 102, 105, 100, 92, 82, 72, 62, 55, 48,
    44,
  ],
  o3: [
    18, 16, 14, 12, 10, 12, 18, 28, 42, 58, 72, 82, 88, 92, 95, 90, 80, 65, 48, 35, 28, 24, 22, 20,
  ],
  // HCHO (satellite column) in µmol/m² — afternoon photochemical peak.
  hcho: [
    14, 13, 13, 12, 12, 13, 15, 17, 19, 21, 23, 24, 25, 25, 24, 23, 21, 19, 18, 17, 16, 15, 15, 14,
  ],
  label: "Delhi NCR",
  healthAdvisory: "Avoid outdoor activities between 12 PM – 5 PM.",
  bestHours: [5, 6, 7],
};

const SITE_MAP: Record<string, number> = {
  "Ashok Vihar": 1,
  Dwarka: 2,
  "Jawaharlal Nehru": 3,
  "Northwest Delhi": 4,
  "East Delhi": 5,
  "Northwest Delhi 2": 6,
  "Northeast Delhi": 7,
  Delhi: 1,
};

// --- HCHO (formaldehyde, satellite column) bands ------------------------------
// HCHO has no WHO/CPCB ambient breakpoint (it is a satellite column in
// µmol/m², not a ground concentration), so the app uses one shared,
// distribution-based scale everywhere: the map layer, Metrics, Best Time,
// Alerts, AI Insights and the HCHO panel all read these edges, so a value
// always gets the same label on every feature.
export const HCHO_BAND_EDGES = [15, 30, 50, 70] as const;
export const HCHO_BAND_LABELS = ["Good", "Moderate", "Elevated", "High", "Very High"] as const;
export const HCHO_BAND_COLORS = ["#00E400", "#FFD300", "#FF9800", "#FF0000", "#9C27B0"] as const;
export type HchoLevel = (typeof HCHO_BAND_LABELS)[number];

/** Band index 0 (best) – 4 (worst) for an HCHO column value in µmol/m². */
export function hchoBand(value: number): number {
  if (value <= HCHO_BAND_EDGES[0]) return 0;
  if (value <= HCHO_BAND_EDGES[1]) return 1;
  if (value <= HCHO_BAND_EDGES[2]) return 2;
  if (value <= HCHO_BAND_EDGES[3]) return 3;
  return 4;
}

/** Shared display label for an HCHO value ("Good" … "Very High"). */
export function hchoLevel(value: number): HchoLevel {
  return HCHO_BAND_LABELS[hchoBand(value)];
}

/** Peak HCHO of a series, ignoring gaps; `null` when the series has no data. */
export function peakHcho(hcho?: number[] | null): { value: number; hour: number } | null {
  if (!Array.isArray(hcho)) return null;
  let value = -Infinity;
  let hour = -1;
  hcho.forEach((v, i) => {
    if (v != null && v > value) {
      value = v;
      hour = i;
    }
  });
  return hour === -1 ? null : { value, hour };
}

export function deriveHealthAdvisory(no2: number[], o3: number[], hcho?: number[]): string {
  const peakNo2 = Math.max(...no2);
  const peakO3 = Math.max(...o3);
  // Fold HCHO in on the same band edges as the map: >50 is the "High" band
  // (red), >30 the "Elevated" band. Absent HCHO keeps the NO₂/O₃-only rules.
  const worstHcho = peakHcho(hcho)?.value ?? -Infinity;
  if (peakNo2 > 100 || peakO3 > 90 || worstHcho > 50)
    return "Avoid outdoor activities during afternoon peak hours.";
  if (peakNo2 > 53 || peakO3 > 70 || worstHcho > 30)
    return "Moderate pollution expected. Reduce prolonged outdoor exertion.";
  return "Air quality is acceptable. Enjoy outdoor activities in early morning.";
}

export function deriveBestHours(no2: number[], o3: number[], hcho?: number[]): number[] {
  return no2
    .map((v, i) => ({
      i,
      // "Safe" hour = NO₂/O₃ below their thresholds AND HCHO not above the
      // Moderate band (≤30 µmol/m²) when HCHO data is available.
      safe: v < 40 && o3[i] < 30 && (hcho == null || hcho[i] == null || hcho[i] <= 30),
    }))
    .filter((x) => x.safe)
    .map((x) => x.i)
    .slice(0, 3);
}

export function deriveAqi(no2: number[], o3: number[]): number[] {
  return no2.map((v, i) => Math.round(v * 2.1 + o3[i] * 1.8));
}

export async function fetchForecast(city: string): Promise<CityForecast> {
  const siteId = SITE_MAP[city.trim()] ?? 1;
  const staticRes = await fetchWithFallback<{ no2: number[]; o3: number[]; hcho?: number[]; label: string; matched_date?: string; current_hour?: number }>(
    `/api/forecast?site=${siteId}`, `api/forecast-${siteId}.json`, { no2: FALLBACK_FORECAST.no2, o3: FALLBACK_FORECAST.o3, hcho: FALLBACK_FORECAST.hcho, label: city }
  );
  return {
    no2: staticRes.no2,
    o3: staticRes.o3,
    hcho: staticRes.hcho,
    aqi: deriveAqi(staticRes.no2, staticRes.o3),
    label: staticRes.label,
    healthAdvisory: deriveHealthAdvisory(staticRes.no2, staticRes.o3, staticRes.hcho),
    bestHours: deriveBestHours(staticRes.no2, staticRes.o3, staticRes.hcho),
    matchedDate: staticRes.matched_date,
    currentHour: staticRes.current_hour,
  };
}

export async function fetchTime(): Promise<TimeData> {
  return fetchWithFallback<TimeData>("/api/time", "api/time.json", {
    matched_date: new Date().toISOString().slice(0, 10),
    current_hour: new Date().getHours(),
    current_minute: new Date().getMinutes(),
    real_now: new Date().toISOString(),
  });
}

export async function fetchStations(): Promise<StationData[]> {
  return fetchWithFallback<StationData[]>("/api/stations", "api/stations.json", []);
}

/** Per-source health from `/api/status` — drives the live/expired badge. */
export interface LiveStatusData {
  enabled: boolean;
  active: boolean;
  expired: boolean;
  expires_at?: string;
  remaining_seconds?: number | null;
  /** "Live context ended - last updated HH: MM IST" once the window closes. */
  message?: string | null;
  scheduler_running?: boolean;
  jobs?: Record<
    string,
    {
      status?: string;
      last_success_at?: string | null;
      last_error?: string | null;
      interval_seconds?: number;
    }
  >;
  tinyfish?: { cap?: number; used?: number; remaining?: number; active_key?: string };
  server_time?: string;
}

export async function fetchLiveStatus(): Promise<LiveStatusData> {
  // No static fallback: an unreachable backend is reported as `active: false`
  // rather than as a live window that does not exist.
  return fetchWithFallback<LiveStatusData>("/api/status", "api/status.json", {
    enabled: false,
    active: false,
    expired: false,
    message: null,
  });
}

export async function fetchWeeklyTrend(city: string): Promise<WeeklyTrend> {
  const siteId = SITE_MAP[city.trim()] ?? 1;
  return fetchWithFallback<WeeklyTrend>(`/api/weekly-trend?site=${siteId}`, `api/weekly-trend-${siteId}.json`, { days: [], no2: [], o3: [], pm25: [] });
}

export async function fetchMetrics(): Promise<ModelMetricsData> {
  return fetchWithFallback<ModelMetricsData>("/api/metrics", "api/metrics.json", {
    model: "XGBoost EXP_12 (database retrain)",
    metrics: {
      t1h: {
        NO2: { r2: 0.6854, mae: 11.16, rmse: 16.14, val_r2: 0.8, val_mae: 9.37, val_rmse: 13.15, best_iter: 809 },
        O3: { r2: 0.8105, mae: 16.47, rmse: 23.23, val_r2: 0.8587, val_mae: 10.85, val_rmse: 15.54, best_iter: 947 },
        HCHO: { r2: 0.5413, mae: 4.51, rmse: 5.88, val_r2: 0.5481, val_mae: 4.21, val_rmse: 5.35, best_iter: 874 },
      },
      t6h: { NO2: { r2: 0.6181, mae: 12.61, rmse: 17.87 }, O3: { r2: 0.768, mae: 18.62, rmse: 25.62 }, HCHO: { r2: 0.4952, mae: 4.77, rmse: 6.19 } },
      t12h: { NO2: { r2: 0.6499, mae: 12.05, rmse: 17.07 }, O3: { r2: 0.7797, mae: 18.17, rmse: 25.21 }, HCHO: { r2: 0.4928, mae: 4.76, rmse: 6.19 } },
      t24h: { NO2: { r2: 0.6504, mae: 11.99, rmse: 17.15 }, O3: { r2: 0.7741, mae: 18.4, rmse: 25.84 }, HCHO: { r2: 0.4849, mae: 4.72, rmse: 6.14 } },
      all_horizons: { NO2: [], O3: [], HCHO: [] },
      validation_comparison: {
        NO2: { test: { r2: 0.6854, mae: 11.16, rmse: 16.14 }, val: { r2: 0.8, mae: 9.37, rmse: 13.15 } },
        O3: { test: { r2: 0.8105, mae: 16.47, rmse: 23.23 }, val: { r2: 0.8587, mae: 10.85, rmse: 15.54 } },
        HCHO: { test: { r2: 0.5413, mae: 4.51, rmse: 5.88 }, val: { r2: 0.5481, mae: 4.21, rmse: 5.35 } },
      },
    },
    feature_importance: { NO2: [], O3: [], HCHO: [] },
    seasonal: {
      NO2: [
        { season: "Summer", horizon: 0, r2: 0.6896, mae: 11.18, rmse: 16.1 },
        { season: "Monsoon", horizon: 0, r2: 0.6631, mae: 11.09, rmse: 16.28 },
      ],
      O3: [
        { season: "Summer", horizon: 0, r2: 0.8303, mae: 15.49, rmse: 21.53 },
        { season: "Monsoon", horizon: 0, r2: 0.7421, mae: 20.22, rmse: 28.81 },
      ],
      HCHO: [
        { season: "Summer", horizon: 0, r2: 0.5254, mae: 4.69, rmse: 6.12 },
        { season: "Monsoon", horizon: 0, r2: 0.5473, mae: 3.69, rmse: 4.7 },
      ],
    },
    site_errors: { NO2: [], O3: [], HCHO: [] },
    hyperparameters: {
      n_estimators: 2500,
      learning_rate: 0.03,
      max_depth: 6,
      min_child_weight: 20,
      gamma: 0.1,
      subsample: 0.6,
      colsample_bytree: 0.5,
      reg_alpha: 0.5,
      reg_lambda: 2.0,
      early_stopping_rounds: 100,
    },
    training: {
      train_years: "2018 – Dec 2023 (< 2025-01-01)",
      val_years: "Jan – Mar 2025 (real)",
      test_years: "Apr – Jun 2025 (real)",
      sites: 7,
      targets: ["NO2_target", "O3_target", "HCHO_target"],
      horizons: 24,
      features: 199,
    },
  });
}

export function getCityData(city: string): CityForecast {
  return { ...FALLBACK_FORECAST, label: city || "Delhi NCR" };
}

export const FALLBACK = FALLBACK_FORECAST;

export const FORECAST: Record<TimeLabel, { aqi: number; no2: number; o3: number }> = {} as Record<
  TimeLabel,
  { aqi: number; no2: number; o3: number }
>;

TIMES.forEach((t, i) => {
  (FORECAST as Record<string, { aqi: number; no2: number; o3: number }>)[t] = {
    aqi: FALLBACK_FORECAST.aqi[i],
    no2: FALLBACK_FORECAST.no2[i],
    o3: FALLBACK_FORECAST.o3[i],
  };
});
