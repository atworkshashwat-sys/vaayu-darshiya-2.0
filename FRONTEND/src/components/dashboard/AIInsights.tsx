import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, Sparkles, Shield, AlertTriangle, Radio, FlaskConical } from "lucide-react";
import { CityForecast, hchoLevel, peakHcho } from "./data";

function getNo2Level(ppb: number): string {
  if (ppb <= 28) return "Safe";
  if (ppb <= 53) return "Moderate";
  if (ppb <= 100) return "Elevated";
  return "High";
}

function getO3Level(ppb: number): string {
  if (ppb <= 40) return "Safe";
  if (ppb <= 70) return "Moderate";
  if (ppb <= 100) return "Elevated";
  return "High";
}

function formatHour(idx: number): string {
  const h = idx > 12 ? idx - 12 : idx;
  return `${h} ${idx >= 12 ? "PM" : "AM"}`;
}

interface AIInsightsProps {
  city: string;
  forecast: CityForecast;
}

interface Insight {
  icon: typeof Brain;
  title: string;
  description: string;
  confidence: number;
  source: "live" | "model";
}

function generateInsights(city: string, forecast: CityForecast): Insight[] {
  const currentNo2 = forecast.no2[6];
  const currentO3 = forecast.o3[6];
  const peakNo2 = Math.max(...forecast.no2);
  const peakNo2Hour = forecast.no2.indexOf(peakNo2);
  const peakO3 = Math.max(...forecast.o3);
  const peakO3Hour = forecast.o3.indexOf(peakO3);

  const no2Level = getNo2Level(currentNo2);
  const o3Level = getO3Level(currentO3);

  // HCHO is optional (deployments without the HCHO artefacts serve none).
  const hcho = peakHcho(forecast.hcho);
  const currentHcho = forecast.hcho?.[6] ?? null;
  const hchoNow =
    currentHcho != null ? `HCHO ${currentHcho} µmol/m² (${hchoLevel(currentHcho)})` : null;
  const hchoPeakTxt = hcho
    ? `HCHO, the formaldehyde satellite column, peaks at ${hcho.value} µmol/m² (${hchoLevel(
        hcho.value,
      )}) around ${formatHour(hcho.hour)} — tracking the same sunlight-driven chemistry`
    : null;

  const isElevated = currentNo2 > 53 || currentO3 > 70 || (currentHcho != null && currentHcho > 30);

  // Every series here comes from the 24-hour ML forecast (the dashboard shows
  // the model run's date separately), so the cards are labelled "ML model".
  const insights: Insight[] = [
    {
      icon: Brain,
      title: "Pollution Pattern",
      description: `${city} shows a typical diurnal pattern — NO₂ rises sharply after 8 AM due to traffic emissions, peaking at ${peakNo2} ppb between ${formatHour(
        peakNo2Hour,
      )} and ${formatHour(peakNo2Hour + 1)}. O₃ follows a photochemical cycle, reaching ${peakO3} ppb around ${formatHour(
        peakO3Hour,
      )} as sunlight drives ozone formation.${hchoPeakTxt ? ` ${hchoPeakTxt}.` : ""} NO₂ and O₃ decline after sunset. (Based on the 24-hour ML forecast)`,
      confidence: 92,
      source: "model" as const,
    },
    {
      icon: Sparkles,
      title: "AI Forecast — NO₂, O₃ & HCHO",
      description: `Our ML model predicts NO₂ will reach ${peakNo2} ppb (${getNo2Level(
        peakNo2,
      )}) around ${formatHour(peakNo2Hour)} and O₃ will reach ${peakO3} ppb (${getO3Level(
        peakO3,
      )}) around ${formatHour(peakO3Hour)}${
        hcho
          ? `; HCHO will reach ${hcho.value} µmol/m² (${hchoLevel(hcho.value)}) around ${formatHour(hcho.hour)}`
          : ""
      }. Forecast now: NO₂ is ${currentNo2} ppb (${no2Level}), O₃ is ${currentO3} ppb (${o3Level})${
        hchoNow ? `, ${hchoNow}` : ""
      }. Consider indoor workouts during afternoon peak hours when pollutants are elevated.`,
      confidence: 88,
      source: "model" as const,
    },
    {
      icon: Shield,
      title: "Health Risk Guidance",
      description: isElevated
        ? `Forecast: NO₂ (${currentNo2} ppb) and O₃ (${currentO3} ppb)${
            currentHcho != null ? `, HCHO ${currentHcho} µmol/m² (${hchoLevel(currentHcho)})` : ""
          }${
            currentNo2 > 53 || currentO3 > 70
              ? " exceed safe thresholds for prolonged outdoor exposure. WHO recommends limiting outdoor exertion when NO₂ exceeds 53 ppb or O₃ exceeds 70 ppb. Sensitive groups should move activities indoors until levels drop."
              : " sit above our moderate band — mainly on the HCHO side. There is no WHO/CPCB limit for the HCHO column; treat it as a marker of active photochemical pollution and reduce prolonged afternoon exertion."
          } (Based on the 24-hour ML forecast)`
        : `Forecast: NO₂ (${currentNo2} ppb) and O₃ (${currentO3} ppb)${
            hchoNow ? `, ${hchoNow}` : ""
          } are within acceptable ranges. Air quality is suitable for outdoor activities. Continue monitoring if planning extended time outside. (Based on the 24-hour ML forecast)`,
      confidence: 95,
      source: "model" as const,
    },
    {
      icon: AlertTriangle,
      title: "Vulnerable Groups",
      description: isElevated
        ? `Forecast now: NO₂ ${currentNo2} ppb (${no2Level}), O₃ ${currentO3} ppb (${o3Level})${
            hchoNow ? `, ${hchoNow}` : ""
          }. Children, elderly, and people with respiratory conditions should avoid outdoor exertion, especially near traffic-heavy roads. Keep windows closed and use air purifiers if available.`
        : `Forecast now: NO₂ ${currentNo2} ppb (${no2Level}), O₃ ${currentO3} ppb (${o3Level})${
            hchoNow ? `, ${hchoNow}` : ""
          }. Air quality is currently acceptable for most people, including vulnerable groups. No special precautions needed.`,
      confidence: 90,
      source: "model" as const,
    },
  ];

  // Dedicated HCHO card — only when the deployment actually serves HCHO.
  if (hcho && currentHcho != null) {
    insights.push({
      icon: FlaskConical,
      title: "HCHO — What It Tells Us",
      description: `${city}'s formaldehyde column is forecast at ${currentHcho} µmol/m² right now (${hchoLevel(
        currentHcho,
      )}) and peaks at ${hcho.value} µmol/m² (${hchoLevel(hcho.value)}) around ${formatHour(
        hcho.hour,
      )}. HCHO is emitted by combustion and forms when VOCs break down in sunlight, so a rising column flags active photochemical pollution even before O₃ climbs. It is a satellite total-column measure in µmol/m² with no WHO/CPCB ambient limit — we band it on the project's own distribution: Good ≤15, Moderate ≤30, Elevated ≤50, High ≤70, Very High >70.`,
      confidence: 85,
      source: "model" as const,
    });
  }

  return insights;
}

export function AIInsights({ city, forecast }: AIInsightsProps) {
  const insights = generateInsights(city, forecast);

  return (
    <section id="insights" className="mx-4 mt-8 md:mx-8">
      <div className="mb-4 flex items-center gap-2">
        <div className="rounded-lg bg-primary/10 p-2">
          <Brain className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h2 className="font-display text-xl font-bold">AI Insights</h2>
          <p className="text-xs text-muted-foreground">Powered by atmospheric ML models</p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {insights.map((insight, i) => (
          <Card
            key={insight.title}
            className={`glass transition-all hover:shadow-lg${
              // Odd card count: let the last card span the row instead of
              // leaving a half-empty grid cell.
              i === insights.length - 1 && insights.length % 2 === 1 ? " md:col-span-2" : ""
            }`}
          >
            <CardHeader className="flex flex-row items-start gap-3 pb-2">
              <div className="rounded-lg bg-primary/10 p-2">
                <insight.icon className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1">
                <CardTitle className="text-sm">{insight.title}</CardTitle>
              </div>
              <div className="flex items-center gap-1.5">
                <Badge
                  variant={insight.source === "model" ? "default" : "secondary"}
                  className="text-[10px] gap-1"
                >
                  {insight.source === "live" && <Radio className="h-2.5 w-2.5 animate-pulse" />}
                  {insight.source === "live" ? "Live data" : "ML model"}
                </Badge>
                <Badge variant="secondary" className="text-[10px]">
                  {insight.confidence}% conf.
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-relaxed text-secondary">{insight.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
