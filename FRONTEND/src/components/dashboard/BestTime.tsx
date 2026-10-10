import { useMemo, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Sun, Sunrise, Moon, Clock, ExternalLink } from "lucide-react";
import { CityForecast, hchoBand } from "./data";
import {
  WeatherDay,
  WeatherWindows,
  fetchWeatherWindows,
  formatAge,
  formatClock,
  formatDate,
} from "./contextApi";

function getSafetyLabel(no2: number, o3: number, hcho?: number | null): string {
  const worst = Math.max(
    no2 > 53 ? 1 : 0,
    o3 > 70 ? 1 : 0,
    no2 > 100 ? 2 : 0,
    o3 > 90 ? 2 : 0,
    // HCHO rides on the shared band scale so this card always agrees with the
    // map's composite colouring: Elevated (band 2) counts as Moderate,
    // High / Very High (band 3-4) as Avoid Outdoor. Absent HCHO = NO2/O3 only.
    hcho != null ? Math.min(2, Math.max(0, hchoBand(hcho) - 1)) : 0,
  );
  if (worst === 0) return "Safe";
  if (worst === 1) return "Moderate";
  return "Avoid Outdoor";
}

type SafetyStatus = "Safe" | "Moderate" | "Avoid Outdoor";

function getSafetyStatus(no2: number, o3: number, hcho?: number | null): SafetyStatus {
  return getSafetyLabel(no2, o3, hcho) as SafetyStatus;
}

const STATUS_STYLES: Record<SafetyStatus, { badge: string; border: string; tint: string }> = {
  Safe: {
    badge: "bg-blue-500 text-white border-transparent",
    border: "border-l-blue-500",
    tint: "bg-blue-50",
  },
  Moderate: {
    badge: "bg-amber-500 text-black border-transparent",
    border: "border-l-amber-500",
    tint: "bg-amber-50",
  },
  "Avoid Outdoor": {
    badge: "bg-red-500 text-white border-transparent",
    border: "border-l-red-500",
    tint: "bg-red-50",
  },
};

function getSafetyIconColor(status: SafetyStatus): string {
  switch (status) {
    case "Safe":
      return "text-blue-500";
    case "Moderate":
      return "text-amber-500";
    case "Avoid Outdoor":
      return "text-red-500";
  }
}

interface BestTimeProps {
  forecast: CityForecast;
}

interface TimeSlot {
  period: string;
  hours: string;
  no2: number;
  o3: number;
  hcho: number | null;
  status: SafetyStatus;
  recommended: boolean;
  icon: typeof Sun;
}

// Hour indices (forecast[h] is the value at hour h) for each time slot.
const SLOT_HOURS: number[][] = [
  [5, 6, 7],
  [8, 9, 10, 11],
  [12, 13, 14, 15, 16, 17],
  [18, 19, 20, 21],
  [22, 23, 0, 1, 2, 3, 4],
];

const SLOT_META = [
  { period: "Early Morning", hours: "5:00 AM – 7:00 AM", icon: Sunrise },
  { period: "Morning Rush", hours: "8:00 AM – 11:00 AM", icon: Sun },
  { period: "Afternoon Peak", hours: "12:00 PM – 5:00 PM", icon: Sun },
  { period: "Evening", hours: "6:00 PM – 9:00 PM", icon: Moon },
  { period: "Late Night", hours: "10:00 PM – 4:00 AM", icon: Moon },
];

/** Mean of a series over the given hours; null when HCHO data is missing. */
function avgAt(series: number[] | undefined, hours: number[]): number | null {
  if (!Array.isArray(series)) return null;
  let sum = 0;
  let n = 0;
  for (const h of hours) {
    const v = series[h];
    if (v != null) {
      sum += v;
      n++;
    }
  }
  return n === 0 ? null : Math.round(sum / n);
}

// Rough "pollution load" used only to pick the least-bad slot when nothing
// earns a Safe status. Scaled so each pollutant contributes comparably.
function slotScore(slot: { no2: number; o3: number; hcho: number | null }): number {
  return slot.no2 / 100 + slot.o3 / 70 + (slot.hcho ?? 0) / 50;
}

function buildTimeSlots(forecast: CityForecast): TimeSlot[] {
  const slots = SLOT_META.map((meta, i) => {
    const hours = SLOT_HOURS[i];
    const no2 = Math.round(hours.reduce((s, h) => s + forecast.no2[h], 0) / hours.length);
    const o3 = Math.round(hours.reduce((s, h) => s + forecast.o3[h], 0) / hours.length);
    const hcho = avgAt(forecast.hcho, hours);
    return { ...meta, no2, o3, hcho, status: getSafetyStatus(no2, o3, hcho) };
  });

  // Recommendation follows the forecast instead of being hardcoded: every Safe
  // slot is recommended, and when none is Safe the least-polluted slot wins so
  // there is always an actionable answer.
  const safe = slots.filter((s) => s.status === "Safe");
  const recommendedPeriods = new Set(
    safe.length > 0
      ? safe.map((s) => s.period)
      : [slots.reduce((a, b) => (slotScore(a) <= slotScore(b) ? a : b)).period],
  );

  return slots.map((s) => ({ ...s, recommended: recommendedPeriods.has(s.period) }));
}

/** "10–15 km/h" / "15 km/h" from the backend's parsed wind values, or null. */
function windPhrase(weather: WeatherDay | null): string | null {
  if (!weather) return null;
  const { wind_min_kmph: lo, wind_max_kmph: hi } = weather;
  if (lo == null && hi == null) return null;
  if (lo != null && hi != null && lo !== hi) return `${lo}–${hi} km/h`;
  return `${(hi ?? lo)!} km/h`;
}

const NOTICE_LABEL: Record<string, string> = {
  "crop-burning": "Crop burning",
  traffic: "Traffic",
  notice: "Notice",
};

export function BestTime({ forecast }: BestTimeProps) {
  const TIME_SLOTS = useMemo(() => buildTimeSlots(forecast), [forecast]);

  // Live context from /api/context/weather-windows: the ML windows above stay
  // the source of truth for the numbers; this only adds a forecast strip,
  // per-window weather values and (when the official forecast says fog or low
  // wind) a one-step downgrade with a visible reason.
  //
  // Keyed ["weatherWindows", slots] because the POST body is part of the
  // request, and because the SSE `weather` event needs a handle to invalidate:
  // as a bare useState the broadcast reached nothing, so this strip kept the
  // values from page load while the backend had already moved on.
  // `retry: false` — one attempt, as before; a retry would be a second backend
  // refresh, and the endpoint's own TTL is what decides when data is refetched.
  const { data: live = null, isLoading: loading } = useQuery<WeatherWindows | null>({
    queryKey: ["weatherWindows", TIME_SLOTS],
    queryFn: () => fetchWeatherWindows(TIME_SLOTS),
    staleTime: 60_000,
    refetchOnWindowFocus: false,
    retry: false,
  });

  const weather = live?.weather ?? null;
  const notices = live?.notices ?? [];
  const contextByPeriod = new Map((live?.slots ?? []).map((s) => [s.period, s]));
  const weatherClock = formatClock(weather?.fetched_at ?? null);

  // Status line: loading / fresh / stale / unavailable (never a stack trace).
  let statusLine: ReactNode;
  if (loading) {
    statusLine = (
      <span className="flex items-center gap-1.5 text-muted-foreground">
        <Clock className="h-3 w-3 animate-pulse" />
        Checking the IMD forecast…
      </span>
    );
  } else if (!live || live.status === "unavailable") {
    statusLine = (
      <span className="text-muted-foreground">
        Live forecast unavailable — showing ML windows unchanged.
      </span>
    );
  } else if (live.stale) {
    statusLine = (
      <span className="text-amber-600">
        Forecast last updated {formatAge(live.age_seconds)} ago — refresh failed.
      </span>
    );
  } else {
    statusLine = (
      <span className="text-muted-foreground">
        {weather ? `${weather.source} forecast` : "Forecast"} retrieved
        {weatherClock ? ` at ${weatherClock}` : ""}
        {notices.length > 0 ? ` · ${notices.length} live notice(s)` : ""}
      </span>
    );
  }

  return (
    <section className="mx-4 mt-8 md:mx-8">
      <div className="mb-4 flex items-center gap-2">
        <div className="rounded-lg bg-green-600/10 p-2">
          <Clock className="h-5 w-5 text-green-600" />
        </div>
        <div>
          <h2 className="font-display text-xl font-bold">Best Time to Go Outside</h2>
          <p className="text-xs text-muted-foreground">
            Optimal windows based on AI-forecasted NO₂, O₃ &amp; HCHO concentrations
          </p>
        </div>
      </div>

      <p className="mb-3 text-xs" data-testid="weather-status">
        {statusLine}
      </p>

      {/* Live advisory: the source's own forecast wording, clearly separate
          from the ML numbers below. Stays empty when nothing was fetched. */}
      {weather && (
        <div className="mb-3 rounded-lg border border-green-600/20 bg-green-50/40 p-3 text-xs">
          <p className="mb-1">
            <span className="font-semibold">Today&apos;s official forecast:</span> {weather.text}
          </p>
          <a
            href={weather.url}
            target="_blank"
            rel="noreferrer noopener"
            className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
          >
            <ExternalLink className="h-3 w-3" />
            Source: {weather.source}
            {weatherClock ? ` · ${weatherClock}` : ""}
          </a>
        </div>
      )}

      {/* Crop-burning / traffic notices: day-level context with their own
          source and fetched-at label. */}
      {notices.length > 0 && (
        <div className="mb-3 space-y-1.5" data-testid="weather-notices">
          {notices.map((notice) => {
            const published = formatDate(notice.published_at ?? "");
            const clock = formatClock(notice.fetched_at);
            return (
              <div key={notice.id} className="rounded-lg border p-2 text-xs">
                <div className="flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className="text-[9px] uppercase tracking-wide text-muted-foreground"
                  >
                    {NOTICE_LABEL[notice.kind] ?? notice.kind}
                  </Badge>
                  <span className="font-medium">{notice.title}</span>
                </div>
                <p className="mt-0.5 text-[11px] text-secondary">{notice.summary}</p>
                <a
                  href={notice.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="mt-0.5 inline-flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground"
                >
                  <ExternalLink className="h-3 w-3" />
                  Source: {notice.source}
                  {clock ? ` · ${clock}` : ""}
                  {published ? ` · published ${published}` : ""}
                </a>
              </div>
            );
          })}
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-5">
        {TIME_SLOTS.map((slot) => {
          const ctx = contextByPeriod.get(slot.period);
          // Backend status wins only if it is a verdict we understand.
          const status = (
            ctx && ctx.status in STATUS_STYLES ? ctx.status : slot.status
          ) as SafetyStatus;
          const styles = STATUS_STYLES[status];
          const iconColor = getSafetyIconColor(status);
          const adjusted = Boolean(ctx?.adjusted);
          const wind = windPhrase(weather);
          return (
            <Card
              key={slot.period}
              className={`border-l-4 ${styles.border} ${styles.tint} transition-all hover:shadow-lg ${
                slot.recommended ? "ring-2 ring-green-600/40" : ""
              }`}
            >
              <CardHeader className="flex flex-row items-center gap-2 pb-2">
                <slot.icon className={`h-4 w-4 ${iconColor}`} />
                <CardTitle className="text-xs">{slot.period}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-xs text-secondary">{slot.hours}</p>
                <div className="space-y-1">
                  <div className="flex items-baseline gap-1">
                    <span className="text-sm font-bold text-cyan-600">NO₂: {slot.no2}</span>
                    <span className="text-[10px] text-muted-foreground">ppb</span>
                  </div>
                  <div className="flex items-baseline gap-1">
                    <span className="text-sm font-bold text-amber-500">O₃: {slot.o3}</span>
                    <span className="text-[10px] text-muted-foreground">ppb</span>
                  </div>
                  <div className="flex items-baseline gap-1">
                    <span className="text-sm font-bold text-violet-500">
                      HCHO: {slot.hcho ?? "—"}
                    </span>
                    <span className="text-[10px] text-muted-foreground">µmol/m²</span>
                  </div>
                </div>

                {/* ML forecast above · live advisory below — separate blocks,
                    the web values never replace the model's numbers. */}
                {weather && (
                  <div className="space-y-0.5 rounded-md bg-background/50 p-1.5 text-[10px]">
                    <span className="block font-semibold uppercase tracking-wide text-muted-foreground">
                      Live advisory · {weather.source}
                    </span>
                    {wind && <span className="block text-secondary">Wind: {wind}</span>}
                    {weather.humidity_pct != null && (
                      <span className="block text-secondary">
                        Humidity: {weather.humidity_pct}%
                      </span>
                    )}
                    {weather.temp_c != null && (
                      <span className="block text-secondary">Temp: {weather.temp_c}°C</span>
                    )}
                    {weather.fog && (
                      <span className="block font-medium text-amber-700">Fog expected</span>
                    )}
                    {weather.rain && (
                      <span className="block font-medium text-sky-700">Rain likely</span>
                    )}
                  </div>
                )}

                <div className="flex items-center gap-2">
                  <Badge
                    className={`text-[10px] px-2.5 py-0.5 font-semibold rounded-md border ${styles.badge}`}
                  >
                    {status}
                  </Badge>
                  <span className="text-[9px] uppercase tracking-wide text-muted-foreground">
                    ML forecast
                  </span>
                </div>

                {/* The downgrade is never silent: original verdict + reason. */}
                {adjusted && (
                  <div className="space-y-0.5" data-testid={`reason-${slot.period}`}>
                    <p className="text-[10px] text-muted-foreground">
                      ML forecast: {ctx?.ml_status} → {status}
                    </p>
                    {(ctx?.reasons ?? []).map((reason) => (
                      <p key={reason} className="text-[10px] font-medium text-amber-700">
                        {reason}
                      </p>
                    ))}
                  </div>
                )}

                {slot.recommended && (
                  <p className="text-[10px] font-medium text-green-600">Recommended</p>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </section>
  );
}
