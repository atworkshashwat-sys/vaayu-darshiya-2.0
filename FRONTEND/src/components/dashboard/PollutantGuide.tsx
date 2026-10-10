import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Heart, AlertTriangle, Factory, Sun, BookOpen, Info } from "lucide-react";

interface InfoCard {
  icon: typeof Heart;
  title: string;
  description: string;
  badge: string;
  badgeVariant: "default" | "secondary" | "destructive" | "outline";
}

const HEALTH_EFFECTS: InfoCard[] = [
  {
    icon: Heart,
    title: "NO₂ — Health Effects",
    description:
      "Nitrogen dioxide irritates the airways and aggravates respiratory diseases. Short-term exposure can cause coughing, wheezing, and increased susceptibility to infections. Long-term exposure is linked to reduced lung function, development of asthma in children, and higher risk of respiratory hospital admissions.",
    badge: "Respiratory",
    badgeVariant: "destructive",
  },
  {
    icon: Heart,
    title: "O₃ — Health Effects",
    description:
      "Ground-level ozone causes chest pain, coughing, and throat irritation. It worsens bronchitis, emphysema, and asthma. Even healthy adults exercising outdoors may experience reduced lung function and airway inflammation. Children and the elderly are particularly vulnerable to ozone-related health impacts.",
    badge: "Respiratory",
    badgeVariant: "destructive",
  },
  {
    icon: Heart,
    title: "HCHO — Health Effects",
    description:
      "Formaldehyde is a pungent volatile organic compound that irritates the eyes, nose and throat and can provoke asthma-like reactions; IARC classifies it as a carcinogen on the evidence of occupational exposure. Note what this app shows: the satellite VERTICAL COLUMN over Delhi (µmol/m²), not ground-level air, so treat it as a marker of VOC loading and photochemical activity rather than a direct breathing concentration.",
    badge: "VOC / Irritant",
    badgeVariant: "secondary",
  },
];

const SAFE_LIMITS: InfoCard[] = [
  {
    icon: AlertTriangle,
    title: "NO₂ — Safe Limits",
    description:
      "WHO guideline (2021): 10 µg/m³ annual mean, 25 µg/m³ 24-hour mean (~13 ppb annual, ~28 ppb 24-hr). CPCB NAAQS standard: 40 µg/m³ annual (~21 ppb), 80 µg/m³ 24-hour (~42 ppb). Indian short-term limit for industrial areas: 150 µg/m³/hour (~79 ppb). Levels above these thresholds indicate potential health risks with prolonged exposure.",
    badge: "WHO / CPCB",
    badgeVariant: "default",
  },
  {
    icon: AlertTriangle,
    title: "O₃ — Safe Limits",
    description:
      "WHO guideline (2021): 100 µg/m³ 8-hour mean (~50 ppb), 160 µg/m³ peak season (~80 ppb). CPCB NAAQS standard: 100 µg/m³ 8-hour (~50 ppb). Levels above 50 ppb over 8 hours are considered unhealthy for sensitive groups; above 70 ppb is unhealthy for the general population during prolonged outdoor activity.",
    badge: "WHO / CPCB",
    badgeVariant: "default",
  },
  {
    icon: AlertTriangle,
    title: "HCHO — How We Band It",
    description:
      "There is no WHO ambient-air guideline and no CPCB NAAQS standard for formaldehyde, and none exists for a satellite column at all. The app therefore bands HCHO on the project's own distribution over its 3–90 µmol/m² range (mean ≈ 15): Good ≤15, Moderate ≤30, Elevated ≤50, High ≤70, Very High >70 µmol/m². For reference, WHO's indoor-air guideline is 0.1 mg/m³ averaged over 30 minutes — a ground-level measure that is not directly comparable to the column.",
    badge: "Distribution-based",
    badgeVariant: "outline",
  },
];

const SOURCES: InfoCard[] = [
  {
    icon: Factory,
    title: "NO₂ — Common Sources",
    description:
      "NO₂ is primarily emitted from combustion processes. The main sources are vehicle exhaust (especially diesel engines), power plants, industrial boilers, and residential heating. Traffic-heavy urban areas consistently show the highest NO₂ levels, with concentrations peaking during morning and evening rush hours.",
    badge: "Primary Pollutant",
    badgeVariant: "secondary",
  },
  {
    icon: Sun,
    title: "O₃ — Common Sources",
    description:
      "Ozone is a secondary pollutant — it is not directly emitted. It forms when nitrogen oxides (NOx) and volatile organic compounds (VOCs) react in the presence of sunlight. This is why O₃ levels peak in the afternoon on hot, sunny days. Reducing vehicle and industrial NOx emissions is the key strategy for lowering ground-level ozone.",
    badge: "Secondary Pollutant",
    badgeVariant: "secondary",
  },
  {
    icon: Sun,
    title: "HCHO — Common Sources",
    description:
      "HCHO is emitted directly by combustion — vehicles, biomass and crop-residue burning, cooking and incense — and forms secondarily when VOCs (isoprene, terpenes, aromatics) oxidise in sunlight. Over Delhi the column therefore rises through the morning, peaks on hot sunny afternoons roughly tracking O₃, and spikes during biomass-burning episodes.",
    badge: "Primary + Secondary",
    badgeVariant: "secondary",
  },
];

function InfoSection({
  title,
  icon: Icon,
  cards,
}: {
  title: string;
  icon: typeof Heart;
  cards: InfoCard[];
}) {
  return (
    <div className="mt-6">
      <div className="mb-3 flex items-center gap-2">
        <Icon className="h-4 w-4 text-primary" />
        <h3 className="font-display text-base font-semibold">{title}</h3>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {cards.map((card, i) => (
          <Card
            key={card.title}
            className={`glass transition-all hover:shadow-lg${
              // Odd card count: the HCHO card spans the row rather than
              // leaving a half-empty grid cell.
              i === cards.length - 1 && cards.length % 2 === 1 ? " md:col-span-2" : ""
            }`}
          >
            <CardHeader className="flex flex-row items-start gap-3 pb-2">
              <div className="rounded-lg bg-primary/10 p-2">
                <card.icon className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1">
                <CardTitle className="text-sm">{card.title}</CardTitle>
              </div>
              <Badge variant={card.badgeVariant} className="text-[10px]">
                {card.badge}
              </Badge>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-relaxed text-secondary">{card.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

export function PollutantGuide() {
  return (
    <section className="mx-4 mt-8 md:mx-8">
      <div className="mb-4 flex items-center gap-2">
        <div className="rounded-lg bg-primary/10 p-2">
          <BookOpen className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h2 className="font-display text-xl font-bold">Understanding NO₂, O₃ &amp; HCHO</h2>
          <p className="text-xs text-muted-foreground">
            Educational guide to the pollutants this app monitors
          </p>
        </div>
      </div>

      <InfoSection title="Health Effects" icon={Heart} cards={HEALTH_EFFECTS} />
      <InfoSection title="WHO & CPCB Safe Limits" icon={AlertTriangle} cards={SAFE_LIMITS} />
      <InfoSection title="Common Sources" icon={Factory} cards={SOURCES} />

      <div className="mt-4 flex items-start gap-2 rounded-lg border border-border/50 bg-muted/30 px-4 py-3">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        <p className="text-xs leading-relaxed text-muted-foreground">
          This information is provided for general educational purposes only and is not medical
          advice. For personal health guidance, please consult a qualified healthcare professional.
          Thresholds reference WHO Global Air Quality Guidelines (2021) and India's National Ambient
          Air Quality Standards (CPCB/NAAQS).
        </p>
      </div>
    </section>
  );
}
