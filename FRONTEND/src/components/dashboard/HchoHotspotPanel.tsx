import { X, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  distanceKm,
  formatPassDate,
  type HchoHotspot,
  type HchoStation,
} from "./hchoHotspots";

/**
 * Detail panel for one hotspot cluster.
 *
 * Desktop: a small card in the top-right of the map. Mobile (Tailwind `md:`):
 * a bottom sheet with ≥44 px tap targets. Colour is never the only signal —
 * the rank is written as a number and every value is labelled in text.
 *
 * Attribution is deliberately neutral: HCHO has no source we can honestly
 * name beyond the satellite product, so the only explanatory line is
 * "HCHO rises with combustion, solvents and sunlight-driven chemistry".
 */

interface HchoHotspotPanelProps {
  hotspot: HchoHotspot;
  region: string;
  median: number | null;
  asOf?: string;
  /** Every station with data in the window — used for "nearest stations". */
  stations: HchoStation[];
  onClose: () => void;
}

export function HchoHotspotPanel({
  hotspot,
  region,
  median,
  asOf,
  stations,
  onClose,
}: HchoHotspotPanelProps) {
  const [centerLon, centerLat] = hotspot.center;

  // Nearest stations to the cluster centre that are NOT members of it, so the
  // panel can place the cluster in context ("3.2 km — Anand Lok, 13.0").
  const memberIds = new Set(hotspot.stations.map((s) => s.id));
  const nearest = stations
    .filter((s) => !memberIds.has(s.id))
    .map((s) => ({ ...s, km: distanceKm(centerLat, centerLon, s.lat, s.lon) }))
    .sort((a, b) => a.km - b.km)
    .slice(0, 3);

  return (
    <div
      role="dialog"
      aria-label={`Hotspot ${hotspot.rank} details`}
      className="pointer-events-auto absolute inset-x-2 bottom-2 z-30 max-h-[62vh] overflow-y-auto rounded-xl border bg-background/95 p-4 backdrop-blur md:inset-x-auto md:bottom-auto md:right-4 md:top-4 md:max-h-none md:w-[330px]"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-sm font-bold text-paper-2"
            style={{ background: "var(--accent)" }}
            aria-hidden="true"
          >
            {hotspot.rank}
          </span>
          <div>
            <div className="text-sm font-semibold leading-tight">
              Hotspot {hotspot.rank}
            </div>
            <div className="text-[11px] text-muted-foreground">
              {hotspot.stations.length === 1
                ? "1 station in cluster"
                : `${hotspot.stations.length} stations in cluster`}
            </div>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          aria-label="Close hotspot details"
          className="h-11 w-11 shrink-0"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="mt-3 rounded-lg border bg-muted/40 p-3">
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold leading-none">{hotspot.peak}</span>
          <span className="text-xs font-medium text-muted-foreground">
            µmol/m² peak
          </span>
        </div>
        <div className="mt-1 text-xs text-muted-foreground">
          {median != null && hotspot.ratio_to_median != null
            ? `${hotspot.ratio_to_median}× the ${region} median (${median} µmol/m²)`
            : `Relative to the ${region} median`}
        </div>
        <div className="mt-1 flex items-center gap-1.5 text-xs">
          <TrendingUp className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
          <span>
            Hotspot on <span className="font-semibold">{hotspot.persistence}</span> of
            the last {hotspot.days} days
          </span>
        </div>
      </div>

      <div className="mt-3 text-[11px] uppercase tracking-wide text-muted-foreground">
        Stations in this cluster
      </div>
      <ul className="mt-1 space-y-1">
        {hotspot.stations.map((s) => (
          <li key={s.id} className="flex items-baseline justify-between gap-2 text-xs">
            <span className="truncate">{s.name}</span>
            <span className="shrink-0 font-semibold">{s.value} µmol/m²</span>
          </li>
        ))}
      </ul>

      {nearest.length > 0 && (
        <>
          <div className="mt-3 text-[11px] uppercase tracking-wide text-muted-foreground">
            Nearest other stations
          </div>
          <ul className="mt-1 space-y-1">
            {nearest.map((s) => (
              <li key={s.id} className="flex items-baseline justify-between gap-2 text-xs">
                <span className="truncate text-muted-foreground">
                  {s.name}
                  <span className="ml-1 opacity-70">({s.km.toFixed(1)} km)</span>
                </span>
                <span className="shrink-0 font-semibold">{s.value} µmol/m²</span>
              </li>
            ))}
          </ul>
        </>
      )}

      <div className="mt-3 border-t pt-2 text-[11px] leading-relaxed text-muted-foreground">
        <div>
          Satellite pass: <span className="font-medium text-foreground">{formatPassDate(asOf)}</span>
        </div>
        <div className="mt-1">Satellite column, not ground level.</div>
        <div className="mt-1">
          HCHO rises with combustion, solvents and sunlight-driven chemistry.
        </div>
      </div>
    </div>
  );
}
