import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  MapPin, Activity, Users, TrendingUp, AlertTriangle, Building2, ExternalLink,
} from "lucide-react";
import { StationData } from "./data";
import { fetchCaseStudy, formatDate, type CaseStat, type CaseStudyData } from "./contextApi";

interface Zone {
  name: string;
  area: string;
  population: string;
  avgNo2: number;
  avgPm25: number;
  riskLevel: "Critical" | "High" | "Moderate";
  keySources: string[];
  lat: number;
  lon: number;
}

const CRITICALLY_AFFECTED_ZONES: Zone[] = [
  {
    name: "Anand Vihar ISBT",
    area: "East Delhi",
    population: "~3.2 Lakh",
    avgNo2: 118,
    avgPm25: 285,
    riskLevel: "Critical",
    keySources: ["Inter-state bus terminal", "Industrial units", "Traffic congestion"],
    lat: 28.6393,
    lon: 77.3115,
  },
  {
    name: "Mundka Industrial Area",
    area: "West Delhi",
    population: "~2.8 Lakh",
    avgNo2: 105,
    avgPm25: 260,
    riskLevel: "Critical",
    keySources: ["Factory emissions", "Waste burning", "Construction dust"],
    lat: 28.6820,
    lon: 77.0930,
  },
  {
    name: "ITO / Bahadur Shah Zafar Marg",
    area: "Central Delhi",
    population: "~4.1 Lakh",
    avgNo2: 112,
    avgPm25: 270,
    riskLevel: "Critical",
    keySources: ["Heavy traffic", "Commercial activity", "Road dust"],
    lat: 28.6280,
    lon: 77.2410,
  },
  {
    name: "Wazirpur",
    area: "North Delhi",
    population: "~2.5 Lakh",
    avgNo2: 98,
    avgPm25: 245,
    riskLevel: "High",
    keySources: ["Small-scale industries", "Residential biomass burning", "Traffic"],
    lat: 28.7050,
    lon: 77.1860,
  },
  {
    name: "Rohini Sector 5",
    area: "Northwest Delhi",
    population: "~3.8 Lakh",
    avgNo2: 92,
    avgPm25: 235,
    riskLevel: "High",
    keySources: ["Construction activity", "Local traffic", "Open burning"],
    lat: 28.7480,
    lon: 77.0650,
  },
  {
    name: "Nehru Nagar",
    area: "South Delhi",
    population: "~2.1 Lakh",
    avgNo2: 88,
    avgPm25: 220,
    riskLevel: "High",
    keySources: ["Traffic corridors", "Commercial complexes", "Road dust"],
    lat: 28.5610,
    lon: 77.2580,
  },
];

const DELHI_STATS = [
  { label: "Annual avg PM2.5", value: "99.7 µg/m³", subtext: "WHO limit: 5 µg/m³", icon: Activity, color: "text-red-500" },
  { label: "Annual avg NO₂", value: "52.3 ppb", subtext: "WHO limit: 10 ppb", icon: TrendingUp, color: "text-amber-500" },
  { label: "Population at risk", value: "~2 Crore", subtext: "Entire NCT population", icon: Users, color: "text-purple-500" },
  { label: "Avg winter AQI", value: "350+", category: "Very Poor to Severe", icon: AlertTriangle, color: "text-red-600" },
];

const MONITORING_SITES_ZONES: Record<string, string> = {
  "Ashok Vihar": "Northwest",
  "Dwarka": "Southwest",
  "Jawaharlal Nehru": "South",
  "Northwest Delhi": "Northwest",
  "East Delhi": "East",
  "Northwest Delhi 2": "Northwest",
  "Northeast Delhi": "Northeast",
};

function getRiskBadge(level: Zone["riskLevel"]) {
  switch (level) {
    case "Critical":
      return "bg-red-600 text-white";
    case "High":
      return "bg-amber-500 text-black";
    case "Moderate":
      return "bg-yellow-400 text-black";
  }
}

function getRiskBorder(level: Zone["riskLevel"]) {
  switch (level) {
    case "Critical":
      return "border-l-red-600";
    case "High":
      return "border-l-amber-500";
    case "Moderate":
      return "border-l-yellow-400";
  }
}

function getPollutantColor(val: number, threshold: number): string {
  if (val >= threshold * 1.2) return "text-red-600 font-bold";
  if (val >= threshold) return "text-amber-500 font-semibold";
  return "text-green-600";
}

/** A figure exactly as its source states it: number plus unit, never invented. */
function liveValue(stat: CaseStat): string {
  const unit = stat.unit && stat.unit !== "index" ? ` ${stat.unit}` : "";
  return `${stat.value}${unit}`;
}

export function CaseStudy({ stations }: { stations?: StationData[] }) {
  // Job D's figures, each carrying the source and date it came from. The
  // route only reads the store — the job itself runs once a day in the
  // background — so loading this panel can never spend a TinyFish call.
  // A figure the sources never stated is absent from the array, and only
  // what arrives is drawn: no reserved slot, so no blank card.
  const { data: liveCase } = useQuery<CaseStudyData | null>({
    queryKey: ["caseStudy"],
    queryFn: () => fetchCaseStudy(),
    staleTime: 30 * 60_000,
    refetchOnWindowFocus: false,
    retry: false,
  });
  const liveStats = liveCase?.stats ?? [];
  const figures = liveStats.filter((s) => s.stat_key !== "news_strip");
  const report = liveStats.find((s) => s.stat_key === "news_strip");

  return (
    <section className="mx-4 mt-8 md:mx-8">
      {/* Header */}
      <div className="mb-6 flex items-center gap-2">
        <div className="rounded-lg bg-red-600/10 p-2">
          <Building2 className="h-5 w-5 text-red-600" />
        </div>
        <div>
          <h2 className="font-display text-xl font-bold">Delhi Air Pollution — Case Study</h2>
          <p className="text-xs text-muted-foreground">
            Critical pollution hotspots and model-monitored zones across the National Capital Territory
          </p>
        </div>
      </div>

      {/* Delhi Overview Stats */}
      <div className="mb-6">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Delhi Air Quality Overview
        </h3>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {DELHI_STATS.map((stat) => (
            <Card key={stat.label} className="glass">
              <CardContent className="flex items-start gap-3 p-4">
                <div className="rounded-lg bg-muted/50 p-2">
                  <stat.icon className={`h-4 w-4 ${stat.color}`} />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">{stat.label}</p>
                  <p className="text-lg font-bold">{stat.value}</p>
                  <p className="text-[10px] text-muted-foreground">
                    {stat.subtext || stat.category}
                  </p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Job D's live figures. Kept apart from the overview above on purpose:
          the overview and the narrative below it quote research figures
          (99.7 µg/m³, "350+"), so overwriting those cards would contradict
          the paragraph right beneath them. Everything here is labelled with
          the source and the date that source gave. */}
      {figures.length > 0 && (
        <div className="mb-6">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Live from official sources
          </h3>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {figures.map((stat) => (
              <Card key={stat.stat_key} className="glass">
                <CardContent className="flex items-start gap-3 p-4">
                  <div className="rounded-lg bg-emerald-600/10 p-2">
                    <Activity className="h-4 w-4 text-emerald-500" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs text-muted-foreground">{stat.label}</p>
                    <p className="text-lg font-bold">{liveValue(stat)}</p>
                    <p className="text-[10px] text-muted-foreground">
                      as of {formatDate(stat.as_of ?? stat.fetched_at) ?? "unknown date"}
                    </p>
                    <a
                      href={stat.url}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="mt-1 inline-flex items-center gap-1 text-[10px] text-primary hover:underline"
                    >
                      {stat.source}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {report && (
        <Card className="glass mb-6 border-l-4 border-l-emerald-600">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">{report.label}</p>
            <p className="mt-1 text-sm leading-relaxed text-secondary">{report.value}</p>
            <p className="mt-2 text-[10px] text-muted-foreground">
              {report.source} · as of {formatDate(report.as_of ?? report.fetched_at) ?? "unknown date"}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Narratives */}
      <Card className="glass mb-6 border-l-4 border-l-red-600">
        <CardContent className="p-5">
          <p className="text-sm leading-relaxed text-secondary">
            <span className="font-semibold text-foreground">Delhi consistently ranks among the world's most polluted cities.</span>{" "}
            With an annual average PM2.5 concentration of <span className="text-red-600 font-bold">99.7 µg/m³</span> — nearly
            20 times the WHO guideline of 5 µg/m³ — the city faces a severe public health crisis. Winter months
            (October–February) routinely see AQI levels exceeding 400, classified as "Severe" or "Hazardous." The
            combination of vehicular emissions, industrial discharge, construction dust, crop residue burning in
            neighbouring states, and unfavourable meteorological conditions creates a toxic blanket over the NCR.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-secondary">
            This XGBoost model monitors <span className="font-semibold text-foreground">7 critical zones</span> across
            Delhi, tracking NO₂ and O₃ concentrations at 1-hour resolution with 24-hour forecasts. The model's
            test R² of <span className="font-semibold text-foreground">0.796 (NO₂)</span> and{" "}
            <span className="font-semibold text-foreground">0.867 (O₃)</span> demonstrates strong predictive
            capability for identifying pollution peaks and advising residents.
          </p>
        </CardContent>
      </Card>

      {/* Critically Affected Zones */}
      <div className="mb-6">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Critically Affected Zones in Delhi
        </h3>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {CRITICALLY_AFFECTED_ZONES.map((zone) => (
            <Card
              key={zone.name}
              className={`glass border-l-4 ${getRiskBorder(zone.riskLevel)} transition-all hover:shadow-lg`}
            >
              <CardHeader className="flex flex-row items-start gap-3 pb-2">
                <MapPin className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                <div className="flex-1">
                  <CardTitle className="text-sm">{zone.name}</CardTitle>
                  <p className="text-[10px] text-muted-foreground">
                    {zone.area} · Pop. {zone.population}
                  </p>
                </div>
                <Badge className={`text-[10px] px-2 py-0.5 rounded-md ${getRiskBadge(zone.riskLevel)}`}>
                  {zone.riskLevel}
                </Badge>
              </CardHeader>
              <CardContent className="space-y-3">
                {/* Pollutant levels */}
                <div className="flex gap-4">
                  <div>
                    <p className="text-[10px] text-muted-foreground">Avg NO₂</p>
                    <p className={`text-sm ${getPollutantColor(zone.avgNo2, 80)}`}>
                      {zone.avgNo2} ppb
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] text-muted-foreground">Avg PM2.5</p>
                    <p className={`text-sm ${getPollutantColor(zone.avgPm25, 200)}`}>
                      {zone.avgPm25} µg/m³
                    </p>
                  </div>
                </div>
                {/* Key sources */}
                <div>
                  <p className="mb-1 text-[10px] font-medium text-muted-foreground">Key Sources</p>
                  <div className="flex flex-wrap gap-1">
                    {zone.keySources.map((src) => (
                      <Badge key={src} variant="outline" className="text-[9px] px-1.5 py-0">
                        {src}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Our Monitoring Sites */}
      <div>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Model-Monitored Sites — Current Readings
        </h3>
        <Card className="glass overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/50">
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">Site</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">Zone</th>
                  <th className="px-4 py-2.5 text-right text-xs font-medium text-muted-foreground">NO₂ (ppb)</th>
                  <th className="px-4 py-2.5 text-right text-xs font-medium text-muted-foreground">O₃ (ppb)</th>
                  <th className="px-4 py-2.5 text-center text-xs font-medium text-muted-foreground">Status</th>
                </tr>
              </thead>
              <tbody>
                {(stations && stations.length > 0 ? stations : []).map((site) => {
                  const no2High = site.no2 > 53;
                  const o3High = site.o3 > 70;
                  const no2VeryHigh = site.no2 > 100;
                  const o3VeryHigh = site.o3 > 100;
                  const status = no2VeryHigh || o3VeryHigh ? "Critical" : no2High || o3High ? "Elevated" : "Normal";
                  return (
                    <tr key={site.name} className="border-b border-border/30 transition-colors hover:bg-muted/30">
                      <td className="px-4 py-2.5 font-medium">{site.name}</td>
                      <td className="px-4 py-2.5 text-muted-foreground">{MONITORING_SITES_ZONES[site.name] || "—"}</td>
                      <td className={`px-4 py-2.5 text-right font-mono ${no2VeryHigh ? "text-red-600 font-bold" : no2High ? "text-amber-500 font-semibold" : "text-green-600"}`}>
                        {site.no2}
                      </td>
                      <td className={`px-4 py-2.5 text-right font-mono ${o3VeryHigh ? "text-red-600 font-bold" : o3High ? "text-amber-500 font-semibold" : "text-green-600"}`}>
                        {site.o3}
                      </td>
                      <td className="px-4 py-2.5 text-center">
                        <Badge
                          variant={status === "Critical" ? "destructive" : status === "Elevated" ? "secondary" : "outline"}
                          className="text-[10px]"
                        >
                          {status}
                        </Badge>
                      </td>
                    </tr>
                  );
                })}
                {(!stations || stations.length === 0) && (
                  <tr>
                    <td colSpan={5} className="px-4 py-6 text-center text-sm text-muted-foreground">
                      Loading station data...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </section>
  );
}
