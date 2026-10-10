import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FlaskConical, TrendingUp, TrendingDown, Activity } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  CityForecast,
  TIMES,
  formatTime,
  hchoBand,
  hchoLevel,
  HCHO_BAND_COLORS,
  HCHO_BAND_EDGES,
  HCHO_BAND_LABELS,
} from "./data";

const HCHO_COLOR = "#8B5CF6";

// Human ranges for each band, derived from the shared edges so the legend can
// never drift from the scale everything else classifies against.
const BAND_RANGES = [
  `≤ ${HCHO_BAND_EDGES[0]}`,
  `${HCHO_BAND_EDGES[0] + 1} – ${HCHO_BAND_EDGES[1]}`,
  `${HCHO_BAND_EDGES[1] + 1} – ${HCHO_BAND_EDGES[2]}`,
  `${HCHO_BAND_EDGES[2] + 1} – ${HCHO_BAND_EDGES[3]}`,
  `> ${HCHO_BAND_EDGES[3]}`,
];

interface HchoPanelProps {
  forecast: CityForecast;
  timeIdx: number;
  currentMinute: number;
}

/**
 * Dedicated HCHO panel: nowcast value + band, the 24-hour forecast profile
 * and the shared band scale. Renders nothing when the API serves no HCHO
 * (deployments without the HCHO artefacts).
 */
export function HchoPanel({ forecast, timeIdx, currentMinute }: HchoPanelProps) {
  const series = forecast.hcho;
  if (!Array.isArray(series) || !series.some((v) => v != null)) return null;

  const current = series[timeIdx] ?? null;
  const defined = series.filter((v): v is number => v != null);
  const peak = Math.max(...defined);
  const peakAt = series.indexOf(peak);
  const low = Math.min(...defined);
  const lowAt = series.indexOf(low);
  const mean = Math.round((defined.reduce((a, b) => a + b, 0) / defined.length) * 10) / 10;

  const chartData = series.map((v, i) => ({ time: TIMES[i], value: v }));
  const bandIdx = current != null ? hchoBand(current) : null;

  // Scale the Y axis to the data (HCHO sits around 12–24 µmol/m² by day),
  // rounded out to multiples of 5 — same rule the 24-hour chart uses.
  const yDomain: [number, number] = (() => {
    const pad = Math.max((peak - low) * 0.25, 1);
    const min = Math.max(0, Math.floor((low - pad) / 5) * 5);
    const max = Math.ceil((peak + pad) / 5) * 5;
    return max > min ? [min, max] : [min, min + 5];
  })();

  return (
    <section className="mx-4 mt-8 md:mx-8">
      <div className="mb-4 flex items-center gap-2">
        <div className="rounded-lg bg-violet-500/10 p-2">
          <FlaskConical className="h-5 w-5 text-violet-500" />
        </div>
        <div>
          <h2 className="font-display text-xl font-bold">Formaldehyde (HCHO) Panel</h2>
          <p className="text-xs text-muted-foreground">
            Satellite vertical column · µmol/m² · 24-hour ML forecast
          </p>
        </div>
      </div>

      <Card className="glass">
        <CardContent className="pt-6">
          <div className="grid gap-6 md:grid-cols-3">
            {/* Now + day statistics */}
            <div className="space-y-3">
              <div>
                <p className="text-xs text-muted-foreground">
                  At {formatTime(timeIdx, currentMinute)} · {forecast.label}
                </p>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold text-violet-500">{current ?? "—"}</span>
                  <span className="text-sm text-muted-foreground">µmol/m²</span>
                </div>
                {bandIdx != null && (
                  <Badge
                    className="mt-1 text-[10px] font-semibold text-white"
                    style={{ backgroundColor: HCHO_BAND_COLORS[bandIdx] }}
                  >
                    {hchoLevel(current!)}
                  </Badge>
                )}
              </div>
              <div className="space-y-1.5 rounded-lg bg-muted/40 p-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1.5 text-muted-foreground">
                    <TrendingUp className="h-3.5 w-3.5 text-violet-500" /> Peak
                  </span>
                  <span className="font-semibold">
                    {peak} µmol/m² · {TIMES[peakAt]}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1.5 text-muted-foreground">
                    <TrendingDown className="h-3.5 w-3.5 text-emerald-500" /> Low
                  </span>
                  <span className="font-semibold">
                    {low} µmol/m² · {TIMES[lowAt]}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1.5 text-muted-foreground">
                    <Activity className="h-3.5 w-3.5 text-muted-foreground" /> Day mean
                  </span>
                  <span className="font-semibold">{mean} µmol/m²</span>
                </div>
              </div>
            </div>

            {/* 24-hour forecast profile */}
            <div className="md:col-span-1">
              <p className="mb-2 text-xs font-medium text-muted-foreground">24-Hour Forecast</p>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="hchoPanelGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={HCHO_COLOR} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={HCHO_COLOR} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    dataKey="time"
                    tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }}
                    interval={5}
                  />
                  <YAxis
                    domain={yDomain}
                    tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }}
                    width={34}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--color-card)",
                      border: "1px solid var(--color-border)",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(v: number) => [`${v} µmol/m²`, "HCHO"]}
                  />
                  <ReferenceLine
                    y={mean}
                    stroke={HCHO_COLOR}
                    strokeDasharray="4 4"
                    strokeOpacity={0.6}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke={HCHO_COLOR}
                    strokeWidth={2}
                    fill="url(#hchoPanelGrad)"
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Shared band scale + explainer */}
            <div className="space-y-3">
              <p className="text-xs font-medium text-muted-foreground">Band Scale</p>
              <div className="space-y-1">
                {HCHO_BAND_LABELS.map((label, i) => (
                  <div
                    key={label}
                    className={`flex items-center justify-between rounded-md px-2 py-1 text-xs ${
                      bandIdx === i ? "bg-muted/60 font-semibold" : ""
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <span
                        className="inline-block h-2.5 w-2.5 rounded-full"
                        style={{ backgroundColor: HCHO_BAND_COLORS[i] }}
                      />
                      {label}
                    </span>
                    <span className="text-muted-foreground">{BAND_RANGES[i]} µmol/m²</span>
                  </div>
                ))}
              </div>
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                HCHO has no WHO/CPCB ambient breakpoint — these bands are distribution-based over
                the project's 3–90 µmol/m² range (mean ≈ 15). The value is a satellite total-column
                measure, not ground-level concentration. Source: 24-hour ML forecast (EXP_12).
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
