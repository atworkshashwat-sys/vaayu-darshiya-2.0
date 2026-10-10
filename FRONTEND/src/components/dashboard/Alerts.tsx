import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Info, ShieldAlert } from "lucide-react";
import { CityForecast, hchoLevel, peakHcho } from "./data";

interface Alert {
  id: string;
  severity: "high" | "medium" | "low";
  title: string;
  message: string;
  icon: typeof AlertTriangle;
}

function generateAlerts(forecast: CityForecast): Alert[] {
  const peakNo2 = Math.max(...forecast.no2);
  const peakO3 = Math.max(...forecast.o3);
  const avgNo2 = forecast.no2.reduce((a, b) => a + b, 0) / forecast.no2.length;
  const avgO3 = forecast.o3.reduce((a, b) => a + b, 0) / forecast.o3.length;
  const peakNo2Hour = forecast.no2.indexOf(peakNo2);
  const peakO3Hour = forecast.o3.indexOf(peakO3);
  const hoursAboveNo2_50 = forecast.no2.filter((v) => v > 50).length;
  const hoursAboveO3_60 = forecast.o3.filter((v) => v > 60).length;

  const formatHour = (h: number) => {
    const hr = h > 12 ? h - 12 : h;
    return `${hr} ${h >= 12 ? "PM" : "AM"}`;
  };

  const alerts: Alert[] = [];

  // High severity alerts
  if (peakNo2 > 100 || peakO3 > 90) {
    alerts.push({
      id: "high-exceed",
      severity: "high",
      title: "NO₂ & O₃ Exceed Safe Limits",
      message: `NO₂ is forecast to reach ${peakNo2} ppb around ${formatHour(peakNo2Hour)} and O₃ to reach ${peakO3} ppb around ${formatHour(peakO3Hour)}. WHO 1-hour guideline for NO₂ is 100 ppb; for O₃ it is 80 ppb. Avoid prolonged outdoor exposure, especially near traffic-heavy roads.`,
      icon: ShieldAlert,
    });
  } else if (peakNo2 > 70) {
    alerts.push({
      id: "high-no2",
      severity: "high",
      title: "Elevated NO₂ Levels Detected",
      message: `NO₂ is forecast to peak at ${peakNo2} ppb around ${formatHour(peakNo2Hour)}. While below the WHO 1-hour limit (100 ppb), prolonged exposure may cause respiratory irritation. Consider reducing outdoor activity during peak hours.`,
      icon: ShieldAlert,
    });
  }

  // Medium severity alerts
  if (peakO3 > 70) {
    alerts.push({
      id: "med-ozone",
      severity: "medium",
      title: "Afternoon Ozone Spike Advisory",
      message: `Photochemical ozone production peaks in the afternoon. O₃ may reach ${peakO3} ppb around ${formatHour(peakO3Hour)}, exceeding the WHO 8-hour guideline of 50 ppb. Sensitive groups should limit outdoor exertion during peak hours.`,
      icon: AlertTriangle,
    });
  } else if (peakO3 > 50) {
    alerts.push({
      id: "med-ozone-mod",
      severity: "medium",
      title: "Moderate Ozone Levels",
      message: `O₃ is forecast to reach ${peakO3} ppb around ${formatHour(peakO3Hour)}. While within WHO limits, sensitive individuals (children, elderly, those with asthma) should be cautious during afternoon hours.`,
      icon: AlertTriangle,
    });
  }

  if (hoursAboveNo2_50 >= 6) {
    alerts.push({
      id: "med-prolonged-no2",
      severity: "medium",
      title: "Prolonged NO₂ Exposure Period",
      message: `NO₂ is expected to remain above 50 ppb for ${hoursAboveNo2_50} hours today. Extended exposure at this level can aggravate respiratory conditions. Limit strenuous outdoor activities.`,
      icon: AlertTriangle,
    });
  }

  // Low severity alerts
  if (hoursAboveO3_60 >= 4) {
    alerts.push({
      id: "low-ozone-duration",
      severity: "low",
      title: "Extended Ozone Window",
      message: `O₃ levels above 60 ppb are expected for ${hoursAboveO3_60} hours. Plan outdoor exercise for early morning or evening when ozone levels are lower.`,
      icon: Info,
    });
  }

  const morningNo2 = forecast.no2.slice(6, 10);
  const peakMorningNo2 = Math.max(...morningNo2);
  if (peakMorningNo2 > 40) {
    alerts.push({
      id: "low-morning",
      severity: "low",
      title: "Morning Rush Hour Impact",
      message: `NO₂ peaks at ${peakMorningNo2} ppb during morning commute (6–10 AM). Consider indoor exercise or later outdoor activities for cleaner air.`,
      icon: Info,
    });
  }

  // HCHO advisory — folded in on the same band edges as the map (Elevated
  // >30, High >50 µmol/m²). The column has no WHO/CPCB limit, so the wording
  // frames it as a photochemistry marker, not an exposure threshold.
  const hchoPeak = peakHcho(forecast.hcho);
  if (hchoPeak && hchoPeak.value > 50) {
    alerts.push({
      id: "med-hcho-high",
      severity: "medium",
      title: "Elevated Formaldehyde (HCHO) Column",
      message: `HCHO is forecast to reach ${hchoPeak.value} µmol/m² (${hchoLevel(hchoPeak.value)}) around ${formatHour(hchoPeak.hour)}. The satellite column has no WHO/CPCB limit, but a High reading flags strong VOC loading and photochemistry — the conditions under which secondary pollutants like O₃ build up. Limit prolonged afternoon outdoor exertion, and keep windows shut if you live near heavy traffic.`,
      icon: AlertTriangle,
    });
  } else if (hchoPeak && hchoPeak.value > 30) {
    alerts.push({
      id: "low-hcho",
      severity: "low",
      title: "Formaldehyde Column Above Baseline",
      message: `HCHO peaks at ${hchoPeak.value} µmol/m² (${hchoLevel(hchoPeak.value)}) around ${formatHour(hchoPeak.hour)} — normal sunny-afternoon photochemistry (Good ≤15, Moderate ≤30 µmol/m²). No action needed for most people; check O₃ at the same hour before planning long outdoor sessions.`,
      icon: Info,
    });
  }

  // Always show at least one alert
  if (alerts.length === 0) {
    alerts.push({
      id: "safe",
      severity: "low",
      title: "Air Quality Within Safe Limits",
      message: `Current forecast shows NO₂ peaking at ${peakNo2} ppb, O₃ at ${peakO3} ppb${hchoPeak ? ` and HCHO at ${hchoPeak.value} µmol/m² (${hchoLevel(hchoPeak.value)})` : ""}. Air quality is suitable for outdoor activities. Continue monitoring if planning extended time outside.`,
      icon: Info,
    });
  }

  return alerts;
}

function getSeverityStyles(severity: Alert["severity"]) {
  switch (severity) {
    case "high":
      return {
        border: "border-l-destructive",
        badge: "destructive" as const,
        icon: "text-destructive",
      };
    case "medium":
      return {
        border: "border-l-amber-500",
        badge: "secondary" as const,
        icon: "text-amber-500",
      };
    case "low":
      return {
        border: "border-l-green-600",
        badge: "outline" as const,
        icon: "text-green-600",
      };
  }
}

interface AlertsProps {
  forecast: CityForecast;
}

export function Alerts({ forecast }: AlertsProps) {
  const ALERTS = generateAlerts(forecast);
  return (
    <section className="mx-4 mt-8 md:mx-8">
      <div className="mb-4 flex items-center gap-2">
        <div className="rounded-lg bg-destructive/10 p-2">
          <AlertTriangle className="h-5 w-5 text-destructive" />
        </div>
        <div>
          <h2 className="font-display text-xl font-bold">Active Alerts</h2>
          <p className="text-xs text-muted-foreground">
            Model-forecast NO₂, O₃ &amp; HCHO pollution advisories
          </p>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        {ALERTS.map((alert) => {
          const styles = getSeverityStyles(alert.severity);
          return (
            <Card
              key={alert.id}
              className={`glass border-l-4 ${styles.border} transition-all hover:shadow-lg`}
            >
              <CardHeader className="flex flex-row items-start gap-3 pb-2">
                <alert.icon className={`mt-0.5 h-5 w-5 shrink-0 ${styles.icon}`} />
                <div className="flex-1">
                  <CardTitle className="text-sm">{alert.title}</CardTitle>
                </div>
                <Badge variant={styles.badge} className="text-[10px] uppercase">
                  {alert.severity}
                </Badge>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed text-secondary">{alert.message}</p>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </section>
  );
}
