import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Download, FileSpreadsheet, FileJson, FileText, Loader2 } from "lucide-react";
import { downloadPollutantForecast } from "@/lib/export";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ReferenceLine,
  Cell,
} from "recharts";
import { TIMES, CityForecast, StationData, WeeklyTrend } from "./data";

const NO2_COLOR = "#06B6D4";
const O3_COLOR = "#F59E0B";
const HCHO_COLOR = "#8B5CF6";

// Segmented control for the 24-hour forecast chart: show every series at once
// or drill into one pollutant (each with its own unit).
type FcKey = "ALL" | "NO2" | "O3" | "HCHO";
const FC_OPTIONS: { key: FcKey; label: string }[] = [
  { key: "ALL", label: "All" },
  { key: "NO2", label: "NO\u2082" },
  { key: "O3", label: "O\u2083" },
  { key: "HCHO", label: "HCHO" },
];

// The standalone 24-hour card: strictly a 24-hour forecast, with a toggle that
// swaps the whole graph between the three pollutants (one at a time).
type Fc24Key = "NO2" | "O3" | "HCHO";
const FC24_OPTIONS: { key: Fc24Key; label: string }[] = [
  { key: "NO2", label: "NO\u2082" },
  { key: "O3", label: "O\u2083" },
  { key: "HCHO", label: "HCHO" },
];
const FC24_META: Record<
  Fc24Key,
  { title: string; sub: string; unit: string; name: string; color: string }
> = {
  NO2: {
    title: "24-Hour NO\u2082 (Nitrogen Dioxide) Forecast",
    sub: "Hourly model concentration forecast in ppb",
    unit: "ppb",
    name: "NO\u2082",
    color: NO2_COLOR,
  },
  O3: {
    title: "24-Hour O\u2083 (Ozone) Forecast",
    sub: "Hourly model concentration forecast in ppb",
    unit: "ppb",
    name: "O\u2083",
    color: O3_COLOR,
  },
  HCHO: {
    title: "24-Hour HCHO (Formaldehyde) Forecast",
    sub: "Satellite vertical column forecast in \u00B5mol/m\u00B2",
    unit: "\u00B5mol/m\u00B2",
    name: "HCHO",
    color: HCHO_COLOR,
  },
};

interface ChartsProps {
  city: string;
  timeIdx: number;
  forecast: CityForecast;
  apiStations: StationData[];
  weeklyTrend: WeeklyTrend;
}

const TOOLTIP_STYLE = {
  contentStyle: {
    background: "var(--color-card)",
    border: "1px solid var(--color-border)",
    borderRadius: 8,
    fontSize: 12,
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
  },
  cursor: { stroke: "var(--color-muted-foreground)", strokeWidth: 1, strokeDasharray: "4 4" },
};

function getNo2Color(v: number): string {
  if (v <= 30) return "#00E400";
  if (v <= 53) return "#FFD300";
  if (v <= 100) return "#FF9800";
  if (v <= 200) return "#FF0000";
  return "#9C27B0";
}

function getO3Color(v: number): string {
  if (v <= 40) return "#00E400";
  if (v <= 70) return "#FFD300";
  if (v <= 100) return "#FF9800";
  if (v <= 150) return "#FF0000";
  return "#9C27B0";
}

export function Charts({ city, timeIdx, forecast, apiStations, weeklyTrend }: ChartsProps) {
  const hasHcho = Array.isArray(forecast.hcho) && forecast.hcho.some((v) => v != null);
  const [fcPollutant, setFcPollutant] = useState<FcKey>("ALL");
  const [fc24Pollutant, setFc24Pollutant] = useState<Fc24Key>("HCHO");
  const [fc24Downloading, setFc24Downloading] = useState(false);

  // What the 24-hour chart draws for the selected segment.
  const fcShowNo2 = fcPollutant === "ALL" || fcPollutant === "NO2";
  const fcShowO3 = fcPollutant === "ALL" || fcPollutant === "O3";
  const fcShowHcho = hasHcho && (fcPollutant === "ALL" || fcPollutant === "HCHO");
  const fcShowPpbAxis = fcPollutant !== "HCHO"; // left axis (NO₂/O₃)
  const fcShowHchoAxis = fcPollutant === "HCHO" || (fcPollutant === "ALL" && hasHcho);
  const fcSubtitle =
    fcPollutant === "HCHO"
      ? `Satellite vertical column forecast in µmol/m² — ${forecast.label}`
      : fcPollutant === "ALL"
        ? `NO₂/O₃ in ppb (left axis), HCHO in µmol/m² (right axis) — ${forecast.label}`
        : `${fcPollutant === "NO2" ? "Nitrogen dioxide" : "Ozone"} hourly concentration in ppb from CPCB monitors — ${forecast.label}`;

  const hourlyData = TIMES.map((t, i) => ({
    time: t,
    NO2: forecast.no2[i],
    O3: forecast.o3[i],
    HCHO: forecast.hcho?.[i] ?? null,
    isCurrent: i === timeIdx,
  }));

  const trendData = weeklyTrend.days.map((day, i) => {
    const d = new Date(day + "T00:00:00");
    const label = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    return {
      day: label,
      NO2: weeklyTrend.no2[i],
      O3: weeklyTrend.o3[i],
      HCHO: weeklyTrend.hcho?.[i] ?? null,
    };
  });

  const stationData = apiStations.length > 0
    ? apiStations.map((s) => ({ name: s.name, no2: s.no2, o3: s.o3, hcho: s.hcho ?? null }))
    : [
        { name: "Ashok Vihar", no2: 72, o3: 85, hcho: 21.9 },
        { name: "Dwarka", no2: 58, o3: 68, hcho: 16.1 },
        { name: "Jawaharlal Nehru", no2: 45, o3: 72, hcho: 12.6 },
        { name: "Northwest Delhi", no2: 85, o3: 92, hcho: 14.3 },
        { name: "East Delhi", no2: 68, o3: 78, hcho: 13.1 },
        { name: "NW Delhi 2", no2: 92, o3: 95, hcho: 21.6 },
        { name: "NE Delhi", no2: 105, o3: 88, hcho: 17.4 },
      ];

  const currentNo2 = forecast.no2[timeIdx];
  const currentO3 = forecast.o3[timeIdx];
  const currentHcho = forecast.hcho?.[timeIdx] ?? null;

  // Standalone 24-hour card — strictly a 24-hour forecast, for whichever
  // pollutant the toggle has selected: current hour, daily peak and low (with
  // the hour they land on), and the 24-hour mean.
  const fc24Key: Fc24Key = fc24Pollutant === "HCHO" && !hasHcho ? "NO2" : fc24Pollutant;
  const fc24Meta = FC24_META[fc24Key];
  const fc24Series: number[] =
    fc24Key === "NO2" ? forecast.no2 : fc24Key === "O3" ? forecast.o3 : (forecast.hcho ?? []);

  const fc24Stats = (() => {
    let peak = -Infinity;
    let peakAt = 0;
    let low = Infinity;
    let lowAt = 0;
    let sum = 0;
    let n = 0;
    fc24Series.forEach((v, i) => {
      if (v == null) return;
      n += 1;
      sum += v;
      if (v > peak) {
        peak = v;
        peakAt = i;
      }
      if (v < low) {
        low = v;
        lowAt = i;
      }
    });
    if (n === 0) return null;
    return { peak, peakAt, low, lowAt, mean: sum / n, now: fc24Series[timeIdx] ?? null };
  })();

  // A narrow series (HCHO sits around 12–24 µmol/m²) on a fixed 0–100 axis
  // lands in the bottom quarter and leaves the peak/low annotations in dead
  // space. Scale to the data, rounded out to multiples of 5.
  const fc24Domain: [number, number] = (() => {
    if (!fc24Stats) return [0, 100];
    const pad = Math.max((fc24Stats.peak - fc24Stats.low) * 0.25, 1);
    const min = Math.max(0, Math.floor((fc24Stats.low - pad) / 5) * 5);
    const max = Math.ceil((fc24Stats.peak + pad) / 5) * 5;
    return max > min ? [min, max] : [min, min + 5];
  })();

  // Download whichever pollutant the 24-hour card is currently showing, for
  // the selected site (CSV / JSON / PDF).
  const handleFc24Download = async (fmt: "csv" | "json" | "pdf") => {
    setFc24Downloading(true);
    try {
      await downloadPollutantForecast(forecast, forecast.label, timeIdx, fc24Key, fmt);
    } catch (err) {
      console.error("24-hour download failed:", err);
      alert("Download failed. Please try again in a few seconds.");
    } finally {
      setFc24Downloading(false);
    }
  };

  const pollutantBreakdown = [
    { name: "NO₂", value: currentNo2, hcho: null, fill: NO2_COLOR, primary: true },
    { name: "O₃", value: currentO3, hcho: null, fill: O3_COLOR, primary: true },
    { name: "PM2.5", value: Math.round(currentNo2 * 2.2), hcho: null, fill: "#94A3B8", primary: false },
    { name: "PM10", value: Math.round(currentNo2 * 3.6), hcho: null, fill: "#CBD5E1", primary: false },
    { name: "SO₂", value: Math.round(currentNo2 * 0.15), hcho: null, fill: "#A3E635", primary: false },
    { name: "CO", value: Math.round(currentNo2 * 0.2), hcho: null, fill: "#F87171", primary: false },
    ...(currentHcho != null
      ? [{ name: "HCHO", value: null, hcho: currentHcho, fill: HCHO_COLOR, primary: true }]
      : []),
  ];

  return (
    <div className="mx-4 mt-8 grid gap-6 md:mx-8 md:grid-cols-2">
      {/* 24-Hour Forecast — All / NO₂ / O₃ / HCHO toggle */}
      <Card className="glass">
        <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-2">
          <div className="space-y-1">
            <CardTitle className="text-base">24-Hour Forecast</CardTitle>
            <p className="text-xs text-secondary">{fcSubtitle}</p>
          </div>
          <div className="flex items-center gap-1">
            {FC_OPTIONS.map(({ key, label }) => (
              <Button
                key={key}
                variant={fcPollutant === key ? "default" : "ghost"}
                size="sm"
                disabled={key === "HCHO" && !hasHcho}
                onClick={() => setFcPollutant(key)}
                className="h-7 gap-1 px-2 text-xs"
              >
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{
                    background:
                      key === "NO2" ? NO2_COLOR : key === "O3" ? O3_COLOR : key === "HCHO" ? HCHO_COLOR : "currentColor",
                  }}
                />
                {label}
              </Button>
            ))}
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={hourlyData}>
              <defs>
                <linearGradient id="no2AreaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={NO2_COLOR} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={NO2_COLOR} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="o3AreaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={O3_COLOR} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={O3_COLOR} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="hchoHourlyGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={HCHO_COLOR} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={HCHO_COLOR} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                interval={3}
              />
              <YAxis
                yAxisId="left"
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                domain={[0, 120]}
                tickCount={7}
                label={{ value: "ppb", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-muted-foreground)" }}
                hide={!fcShowPpbAxis}
              />
              {fcShowHchoAxis && (
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fontSize: 10, fill: HCHO_COLOR }}
                  domain={[0, 100]}
                  tickCount={6}
                  label={{ value: "µmol/m²", angle: 90, position: "insideRight", fontSize: 10, fill: HCHO_COLOR }}
                />
              )}
              <Tooltip
                {...TOOLTIP_STYLE}
                labelFormatter={(label) => `Time: ${label}`}
                formatter={(value: number, name: string) => [
                  name === "HCHO" ? `${value} µmol/m²` : `${value} ppb`,
                  name === "NO2" ? "NO₂" : name === "HCHO" ? "HCHO" : "O₃",
                ]}
              />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(value: string) =>
                  value === "NO2" ? "NO₂ (ppb)" : value === "HCHO" ? "HCHO (µmol/m²)" : "O₃ (ppb)"
                }
              />
              <ReferenceLine
                x={TIMES[timeIdx]}
                yAxisId={fcShowPpbAxis ? "left" : "right"}
                stroke="#EF4444"
                strokeWidth={2}
                strokeDasharray="4 4"
                label={{
                  value: "Now",
                  position: "top",
                  fill: "#EF4444",
                  fontSize: 10,
                  fontWeight: 700,
                }}
              />
              {fcShowNo2 && (
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="NO2"
                  stroke={NO2_COLOR}
                  fill="url(#no2AreaGrad)"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 5, strokeWidth: 2 }}
                />
              )}
              {fcShowO3 && (
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="O3"
                  stroke={O3_COLOR}
                  fill="url(#o3AreaGrad)"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 5, strokeWidth: 2 }}
                />
              )}
              {fcShowHcho && (
                <Area
                  yAxisId="right"
                  type="monotone"
                  dataKey="HCHO"
                  stroke={HCHO_COLOR}
                  fill="url(#hchoHourlyGrad)"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                  dot={false}
                  activeDot={{ r: 5, strokeWidth: 2 }}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Pollutant Breakdown */}
      <Card className="glass">
        <CardHeader>
          <CardTitle className="text-base">Pollutant Breakdown</CardTitle>
          <p className="text-xs text-secondary">
            Current pollutant levels at {TIMES[timeIdx]} — gases in ppb, HCHO in µmol/m² (right axis)
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={pollutantBreakdown}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
              />
              <YAxis
                yAxisId="left"
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                label={{ value: "ppb", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-muted-foreground)" }}
              />
              {hasHcho && (
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fontSize: 10, fill: HCHO_COLOR }}
                  domain={[0, 100]}
                  label={{ value: "µmol/m²", angle: 90, position: "insideRight", fontSize: 10, fill: HCHO_COLOR }}
                />
              )}
              <Tooltip
                {...TOOLTIP_STYLE}
                formatter={(value: number, name: string) => [
                  name === "hcho" ? `${value} µmol/m²` : `${value} ppb`,
                  name === "hcho" ? "HCHO" : name,
                ]}
              />
              <Bar yAxisId="left" dataKey="value" radius={[4, 4, 0, 0]}>
                {pollutantBreakdown.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} opacity={entry.primary ? 1 : 0.55} />
                ))}
              </Bar>
              {hasHcho && (
                <Bar yAxisId="right" dataKey="hcho" radius={[4, 4, 0, 0]}>
                  {pollutantBreakdown.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} opacity={entry.primary ? 1 : 0.55} />
                  ))}
                </Bar>
              )}
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Station-wise NO₂, O₃ & HCHO Comparison */}
      <Card className="glass md:col-span-2">
        <CardHeader>
          <CardTitle className="text-base">Station-wise NO₂, O₃ &amp; HCHO</CardTitle>
          <p className="text-xs text-secondary">
            Current levels across all 7 monitoring stations — NO₂/O₃ in ppb (bottom axis), HCHO in µmol/m² (top axis)
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={stationData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                xAxisId="left"
                type="number"
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                domain={[0, 120]}
                tickCount={7}
              />
              {hasHcho && (
                <XAxis
                  xAxisId="right"
                  orientation="top"
                  type="number"
                  tick={{ fontSize: 10, fill: HCHO_COLOR }}
                  domain={[0, 100]}
                  tickCount={6}
                />
              )}
              <YAxis
                dataKey="name"
                type="category"
                width={120}
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
              />
              <Tooltip
                {...TOOLTIP_STYLE}
                formatter={(value: number, name: string) => [
                  name === "hcho" ? `${value} µmol/m²` : `${value} ppb`,
                  name === "no2" ? "NO₂" : name === "hcho" ? "HCHO" : "O₃",
                ]}
              />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(value: string) =>
                  value === "no2" ? "NO₂ (ppb)" : value === "hcho" ? "HCHO (µmol/m²)" : "O₃ (ppb)"
                }
              />
              <Bar xAxisId="left" dataKey="no2" fill={NO2_COLOR} radius={[0, 6, 6, 0]} barSize={11} />
              <Bar xAxisId="left" dataKey="o3" fill={O3_COLOR} radius={[0, 6, 6, 0]} barSize={11} />
              {hasHcho && (
                <Bar xAxisId="right" dataKey="hcho" fill={HCHO_COLOR} radius={[0, 6, 6, 0]} barSize={11} />
              )}
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* 14-Day Pollutant Trend */}
      <Card className="glass md:col-span-2">
        <CardHeader>
          <CardTitle className="text-base">14-Day NO₂, O₃ &amp; HCHO Trend</CardTitle>
          <p className="text-xs text-secondary">
            Daily averages over the past two weeks — NO₂/O₃ in ppb, HCHO in µmol/m² (right axis) — {forecast.label}
          </p>
        </CardHeader>
        <CardContent>
          {trendData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="no2Grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={NO2_COLOR} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={NO2_COLOR} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="o3Grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={O3_COLOR} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={O3_COLOR} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="hchoGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={HCHO_COLOR} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={HCHO_COLOR} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis
                  dataKey="day"
                  tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                  interval={1}
                  angle={-35}
                  textAnchor="end"
                  height={50}
                />
                <YAxis
                  yAxisId="left"
                  tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                  domain={[0, 120]}
                  tickCount={7}
                  label={{ value: "ppb", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-muted-foreground)" }}
                />
                {hasHcho && (
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    tick={{ fontSize: 10, fill: HCHO_COLOR }}
                    domain={[0, 100]}
                    tickCount={6}
                    label={{ value: "µmol/m²", angle: 90, position: "insideRight", fontSize: 10, fill: HCHO_COLOR }}
                  />
                )}
                <Tooltip
                  {...TOOLTIP_STYLE}
                  formatter={(value: number, name: string) => [
                    name === "HCHO" ? `${value} µmol/m²` : `${value} ppb`,
                    name === "NO2" ? "NO₂" : name === "HCHO" ? "HCHO" : "O₃",
                  ]}
                />
                <Legend
                  wrapperStyle={{ fontSize: 12 }}
                  formatter={(value: string) => (value === "NO2" ? "NO₂" : value === "HCHO" ? "HCHO (µmol/m²)" : "O₃")}
                />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="NO2"
                  stroke={NO2_COLOR}
                  fill="url(#no2Grad)"
                  strokeWidth={2}
                  dot={{ r: 4, strokeWidth: 2 }}
                  activeDot={{ r: 6, strokeWidth: 2 }}
                />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="O3"
                  stroke={O3_COLOR}
                  fill="url(#o3Grad)"
                  strokeWidth={2}
                  dot={{ r: 4, strokeWidth: 2 }}
                  activeDot={{ r: 6, strokeWidth: 2 }}
                />
                {hasHcho && (
                  <Area
                    yAxisId="right"
                    type="monotone"
                    dataKey="HCHO"
                    stroke={HCHO_COLOR}
                    fill="url(#hchoGrad)"
                    strokeWidth={2}
                    strokeDasharray="5 3"
                    dot={{ r: 4, strokeWidth: 2 }}
                    activeDot={{ r: 6, strokeWidth: 2 }}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[280px] items-center justify-center text-sm text-muted-foreground">
              Loading weekly trend data...
            </div>
          )}
        </CardContent>
      </Card>

      {/* 24-Hour Forecast — strictly 24 h, toggle NO₂ / O₃ / HCHO to redraw the graph */}
      <Card className="glass md:col-span-2">
        <CardHeader>
          <div className="flex flex-row flex-wrap items-start justify-between gap-2">
            <div>
              <CardTitle className="text-base">{fc24Meta.title}</CardTitle>
              <p className="text-xs text-secondary">
                {fc24Meta.sub} — {forecast.label}
              </p>
            </div>
            <div className="flex items-center gap-1">
              {FC24_OPTIONS.map(({ key, label }) => (
                <Button
                  key={key}
                  variant={fc24Key === key ? "default" : "ghost"}
                  size="sm"
                  disabled={key === "HCHO" && !hasHcho}
                  onClick={() => setFc24Pollutant(key)}
                  className="h-7 gap-1.5 px-2 text-xs"
                >
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ background: FC24_META[key].color }}
                  />
                  {label}
                </Button>
              ))}
              <div className="mx-1 h-4 w-px bg-border" />
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={fc24Downloading}
                    className="h-7 gap-1.5 px-2 text-xs text-muted-foreground"
                    title={`Download the 24-hour ${fc24Meta.name} forecast for ${forecast.label}`}
                  >
                    {fc24Downloading ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Download className="h-3.5 w-3.5" />
                    )}
                    Download
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-32">
                  <DropdownMenuLabel>Format</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className="gap-2" onClick={() => handleFc24Download("csv")}>
                    <FileSpreadsheet className="h-3.5 w-3.5 text-green-500" />
                    CSV
                  </DropdownMenuItem>
                  <DropdownMenuItem className="gap-2" onClick={() => handleFc24Download("json")}>
                    <FileJson className="h-3.5 w-3.5 text-blue-500" />
                    JSON
                  </DropdownMenuItem>
                  <DropdownMenuItem className="gap-2" onClick={() => handleFc24Download("pdf")}>
                    <FileText className="h-3.5 w-3.5 text-red-500" />
                    PDF
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
          {fc24Stats && (
            <div className="mt-2 flex flex-wrap gap-2">
              {[
                { k: "Now", v: fc24Stats.now, s: TIMES[timeIdx] },
                { k: "Peak", v: fc24Stats.peak, s: TIMES[fc24Stats.peakAt] },
                { k: "Low", v: fc24Stats.low, s: TIMES[fc24Stats.lowAt] },
                { k: "Average", v: fc24Stats.mean, s: "24 h" },
              ].map(({ k, v, s }) => (
                <div key={k} className="rounded-md border border-border/60 bg-muted/40 px-2 py-1">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    {k}
                  </div>
                  <div className="text-sm font-semibold" style={{ color: fc24Meta.color }}>
                    {v == null ? "—" : v.toFixed(1)}
                    <span className="ml-1 text-[10px] font-normal text-muted-foreground">
                      {fc24Meta.unit}
                    </span>
                    <span className="ml-1.5 text-[10px] font-normal text-muted-foreground">
                      {s}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={hourlyData}>
              <defs>
                <linearGradient id={`fc24AreaGrad-${fc24Key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={fc24Meta.color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={fc24Meta.color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                interval={3}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                domain={fc24Domain}
                tickCount={5}
                label={{
                  value: fc24Meta.unit,
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 10,
                  fill: fc24Meta.color,
                }}
              />
              <Tooltip
                {...TOOLTIP_STYLE}
                labelFormatter={(label) => `Time: ${label}`}
                formatter={(value: number) => [`${value} ${fc24Meta.unit}`, fc24Meta.name]}
              />
              <ReferenceLine
                x={TIMES[timeIdx]}
                stroke="#EF4444"
                strokeWidth={2}
                strokeDasharray="4 4"
                label={{
                  value: "Now",
                  position: "top",
                  fill: "#EF4444",
                  fontSize: 10,
                  fontWeight: 700,
                }}
              />
              {fc24Stats && (
                <ReferenceLine
                  y={fc24Stats.peak}
                  stroke={fc24Meta.color}
                  strokeDasharray="4 4"
                  label={{
                    value: `Peak ${fc24Stats.peak.toFixed(1)} · ${TIMES[fc24Stats.peakAt]}`,
                    position: "insideTopRight",
                    fill: fc24Meta.color,
                    fontSize: 10,
                    fontWeight: 700,
                  }}
                />
              )}
              {fc24Stats && (
                <ReferenceLine
                  y={fc24Stats.low}
                  stroke="var(--color-muted-foreground)"
                  strokeDasharray="2 4"
                  label={{
                    value: `Low ${fc24Stats.low.toFixed(1)} · ${TIMES[fc24Stats.lowAt]}`,
                    position: "insideBottomRight",
                    fill: "var(--color-muted-foreground)",
                    fontSize: 10,
                  }}
                />
              )}
              <Area
                type="monotone"
                dataKey={fc24Key}
                stroke={fc24Meta.color}
                fill={`url(#fc24AreaGrad-${fc24Key})`}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 5, strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
