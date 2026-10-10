import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Sun, Sunrise, Moon, Clock } from "lucide-react";
import { CityForecast, hchoBand } from "./data";

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

export function BestTime({ forecast }: BestTimeProps) {
  const TIME_SLOTS = buildTimeSlots(forecast);
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

      <div className="grid gap-3 md:grid-cols-5">
        {TIME_SLOTS.map((slot) => {
          const styles = STATUS_STYLES[slot.status];
          const iconColor = getSafetyIconColor(slot.status);
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
                <Badge
                  className={`text-[10px] px-2.5 py-0.5 font-semibold rounded-md border ${styles.badge}`}
                >
                  {slot.status}
                </Badge>
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
