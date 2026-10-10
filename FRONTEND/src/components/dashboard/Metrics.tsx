import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Droplets, CloudSun, TrendingUp, TrendingDown, Minus, Radio } from "lucide-react";
import { TIMES, CityForecast, formatTime, hchoLevel } from "./data";

function TrendIcon({ current, previous }: { current: number; previous: number }) {
  if (current > previous + 3) return <TrendingUp className="h-4 w-4 text-destructive" />;
  if (current < previous - 3) return <TrendingDown className="h-4 w-4 text-aqi-good" />;
  return <Minus className="h-4 w-4 text-muted-foreground" />;
}

interface MetricsProps {
  aqi: { aqi: number; no2: number; o3: number };
  city: string;
  timeIdx: number;
  currentMinute: number;
  forecast: CityForecast;
}

export function Metrics({ aqi, city, timeIdx, currentMinute, forecast }: MetricsProps) {
  const prevIdx = Math.max(0, timeIdx - 1);

  const metrics = [
    {
      title: "Nitrogen Dioxide (NO₂)",
      value: aqi.no2,
      unit: "ppb",
      icon: Droplets,
      color: "text-blue-500",
      bg: "bg-blue-500/10",
      label: aqi.no2 > 53 ? "Elevated" : "Normal",
      trend: aqi.no2,
      previous: forecast.no2[prevIdx],
      source: "model" as const,
    },
    {
      title: "Ozone (O₃)",
      value: aqi.o3,
      unit: "ppb",
      icon: CloudSun,
      color: "text-amber-500",
      bg: "bg-amber-500/10",
      label: aqi.o3 > 70 ? "High" : "Moderate",
      trend: aqi.o3,
      previous: forecast.o3[prevIdx],
      source: "model" as const,
    },
    ...(forecast.hcho && forecast.hcho[timeIdx] != null
      ? [
          {
            title: "Formaldehyde (HCHO)",
            value: forecast.hcho![timeIdx]!,
            unit: "µmol/m²",
            icon: Droplets,
            color: "text-violet-500",
            bg: "bg-violet-500/10",
            // Shared band labels — the card and the map always agree.
            label: hchoLevel(forecast.hcho![timeIdx]!),
            trend: forecast.hcho![timeIdx]!,
            previous: forecast.hcho![prevIdx] ?? forecast.hcho![timeIdx]!,
            source: "model" as const,
          },
        ]
      : []),
  ];

  return (
    <div className="mx-4 mt-6 grid gap-4 md:mx-8 md:grid-cols-2 lg:grid-cols-3">
      {metrics.map((m) => (
        <Card key={m.title} className="glass overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">{m.title}</CardTitle>
            <div className={`rounded-lg p-2 ${m.bg}`}>
              <m.icon className={`h-4 w-4 ${m.color}`} />
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              <span className={`text-3xl font-bold ${m.color}`}>{m.value}</span>
              <span className="text-sm text-muted-foreground">{m.unit}</span>
              <TrendIcon current={m.trend} previous={m.previous} />
            </div>
            <div className="mt-2 flex items-center gap-2">
              <Badge variant="secondary" className="text-xs">
                {m.label}
              </Badge>
              <Badge
                variant="default"
                className="text-[10px] gap-1"
              >
                <Radio className="h-2.5 w-2.5 animate-pulse" />
                ML forecast
              </Badge>
              <span className="text-xs text-muted-foreground">
                {formatTime(timeIdx, currentMinute)} &middot; {forecast.label}
              </span>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
