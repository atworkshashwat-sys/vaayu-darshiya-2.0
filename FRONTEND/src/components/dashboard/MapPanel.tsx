import { useEffect, useRef, useState, useCallback } from "react";
import maplibregl from "maplibre-gl";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { formatTime, hchoBand } from "./data";
import { severityColor, severityTextColor, toRgb, paper2, accent, inkMuted } from "@/lib/palette";
import {
  fetchHchoHotspots,
  idwRaster,
  interpolationStations,
  rampColor,
  rampCssGradient,
  rasterToDataUrl,
  formatPassDate,
  HCHO_RAMP_MAX,
  type HchoHotspot,
  type HchoHotspotsPayload,
} from "./hchoHotspots";
import { HchoHotspotPanel } from "./HchoHotspotPanel";
import {
  Wind,
  CloudSun,
  Satellite,
  MapIcon,
  LocateFixed,
  Clock,
  Radio,
  RefreshCw,
  Layers,
} from "lucide-react";

// --- NO₂, O₃ and HCHO color thresholds (ppb for NO₂/O₃, µmol/m² for HCHO) ---
// "ALL" is the combined view: every pollutant is banded 0-4 against its own
// thresholds and the worst band wins — the same rule AQI uses to fold its
// sub-indices into one number — so a single colour ramp can carry all three.
type LayerKey = "NO2" | "O3" | "HCHO" | "ALL";
type SingleLayer = Exclude<LayerKey, "ALL">;

// Five-step band scales sit on steps 0/2/3/4/5 of the shared six-step
// severity scale (good · moderate · poor · very-poor · severe); thresholds
// and labels are unchanged — only the colours are the tokens from styles.css.
const BAND_SEVERITY = [0, 2, 3, 4, 5];
const BAND_COLORS = BAND_SEVERITY.map(severityColor);

const BAND_LABELS: Record<SingleLayer, readonly string[]> = {
  NO2: ["Good", "Moderate", "Elevated", "High", "Very High"],
  O3: ["Good", "Moderate", "High", "Very High", "Hazardous"],
  // HCHO has no statutory breakpoints — these are distribution-based bands
  // over the 3–90 µmol/m² range of sat_hcho_column (mean ≈ 15).
  HCHO: ["Good", "Moderate", "Elevated", "High", "Very High"],
};

/** What the composite (ALL) band index is called in the popup and legend. */
const COMPOSITE_LABELS = BAND_LABELS.NO2;

/** Band index 0 (best) – 4 (worst) for one pollutant against its own scale. */
function band(value: number, layer: SingleLayer): number {
  if (layer === "NO2") {
    if (value <= 30) return 0;
    if (value <= 53) return 1;
    if (value <= 100) return 2;
    if (value <= 200) return 3;
    return 4;
  }
  if (layer === "O3") {
    if (value <= 40) return 0;
    if (value <= 70) return 1;
    if (value <= 100) return 2;
    if (value <= 150) return 3;
    return 4;
  }
  // HCHO — the shared distribution-based bands from data.ts, so the map can
  // never drift from the Metrics card, Best Time, Alerts or the HCHO panel.
  return hchoBand(value);
}

function bandColor(b: number): string {
  return BAND_COLORS[Math.min(BAND_COLORS.length - 1, Math.max(0, Math.round(b)))];
}

/** WCAG-checked text colour for a label sitting on a band fill. */
function bandTextColor(b: number): string {
  const i = Math.min(BAND_SEVERITY.length - 1, Math.max(0, Math.round(b)));
  return severityTextColor(BAND_SEVERITY[i]);
}

/** Token colour + alpha, for MapLibre paint props that need rgba. */
function rgba(color: string, a: number): string {
  const [r, g, b] = toRgb(color);
  return `rgba(${r},${g},${b},${a})`;
}

function getPollutantColor(value: number, layer: SingleLayer): string {
  return bandColor(band(value, layer));
}

function getPollutantLabel(value: number, layer: SingleLayer): string {
  return BAND_LABELS[layer][band(value, layer)];
}

function getLayerUnit(layer: LayerKey): string {
  if (layer === "ALL") return "";
  return layer === "HCHO" ? "µmol/m²" : "ppb";
}

/** Composite band for a station — worst band across the pollutants it has. */
function compositeBand(s: LiveStation): number {
  const bands = [band(s.no2, "NO2"), band(s.o3, "O3")];
  // HCHO is optional (deployments without the EXP_11 artefacts serve none).
  if (s.hcho != null) bands.push(band(s.hcho, "HCHO"));
  return Math.max(...bands);
}

/**
 * The value and colour a marker / popup shows for the active layer.
 * Single layers report the raw concentration; ALL reports the composite band.
 */
function stationStyle(s: LiveStation, layer: LayerKey): { value: number; color: string } {
  if (layer === "ALL") {
    const b = compositeBand(s);
    return { value: b, color: bandColor(b) };
  }
  const value = layer === "NO2" ? s.no2 : layer === "O3" ? s.o3 : (s.hcho ?? 0);
  return { value, color: getPollutantColor(value, layer) };
}

// --- 7 Delhi monitoring stations (coordinates only — NO fake data) ---
interface Station {
  name: string;
  coords: [number, number];
}

const STATIONS: Station[] = [
  { name: "Ashok Vihar",       coords: [77.18169, 28.69536] },
  { name: "Dwarka",            coords: [77.07125, 28.5718] },
  { name: "Jawaharlal Nehru",  coords: [77.23441, 28.58278] },
  { name: "Northwest Delhi",   coords: [77.10197, 28.82286] },
  { name: "East Delhi",        coords: [77.27123, 28.53077] },
  { name: "Northwest Delhi 2", coords: [77.09601, 28.72954] },
  { name: "Northeast Delhi",   coords: [77.24951, 28.71052] },
];

interface LiveStation extends Station {
  no2: number;
  o3: number;
  hcho?: number | null;
  temp: number;
  humidity: number;
  wind: number;
  visibility: number;
  // From the project database — optional, may be absent in cached JSON.
  windDir?: number;
  pm25?: number | null;
  aqi?: number | null;
  aqiCategory?: string | null;
  district?: string | null;
  obsNo2?: number | null;
  obsO3?: number | null;
  obsHcho?: number | null;
  observedAt?: string;
  modelRunAt?: string;
}

type ApiStation = {
  name: string;
  no2: number;
  o3: number;
  hcho?: number | null;
  lat: number;
  lon: number;
  temp: number;
  humidity: number;
  wind: number;
  visibility: number;
  windDir?: number;
  pm25?: number | null;
  aqi?: number | null;
  aqiCategory?: string | null;
  district?: string | null;
  obsNo2?: number | null;
  obsO3?: number | null;
  obsHcho?: number | null;
  observedAt?: string;
  modelRunAt?: string;
};

const EMPTY_LIVE: Omit<LiveStation, "name" | "coords"> = {
  no2: 0, o3: 0, temp: 0, humidity: 0, wind: 0, visibility: 0,
};

function buildLiveStations(base: Station[], api: ApiStation[] | undefined): LiveStation[] {
  // Always return one entry per known station. Returning [] while the API was
  // still loading (or unreachable) meant no markers were created at all, which
  // left the gray "Loading…" marker branch below as dead code and rendered an
  // empty heatmap.
  if (!api || api.length === 0) {
    return base.map((s) => ({ ...s, ...EMPTY_LIVE }));
  }
  return base.map((s) => {
    const found = api.find((a) => a.name === s.name);
    if (!found) return { ...s, ...EMPTY_LIVE };
    return {
      ...s,
      no2: found.no2,
      o3: found.o3,
      hcho: found.hcho,
      temp: found.temp,
      humidity: found.humidity,
      wind: found.wind,
      visibility: found.visibility,
      windDir: found.windDir,
      pm25: found.pm25,
      aqi: found.aqi,
      aqiCategory: found.aqiCategory,
      district: found.district,
      obsNo2: found.obsNo2,
      obsO3: found.obsO3,
      obsHcho: found.obsHcho,
      observedAt: found.observedAt,
      modelRunAt: found.modelRunAt,
    };
  });
}
const DELHI_CENTER: [number, number] = [77.15, 28.68];
const DELHI_ZOOM = 11;

/** Bounds enclosing all 7 stations, used to frame them on load. */
function stationBounds(): maplibregl.LngLatBounds {
  const b = new maplibregl.LngLatBounds();
  for (const s of STATIONS) b.extend(s.coords);
  return b;
}

/**
 * Bounds enclosing everything the HCHO layer draws: every station with a
 * current satellite value plus the hotspot centroids. This is wider than the
 * 7 curated surface sites — on a phone the IDW field and its ranked hotspots
 * sat east of the station frame (Indirapuram at 77.4 N projected off-screen),
 * so the tap target could not be reached.
 */
function hchoBounds(data: HchoHotspotsPayload): maplibregl.LngLatBounds | null {
  const b = new maplibregl.LngLatBounds();
  let any = false;
  for (const s of data.stations ?? []) {
    if (typeof s.lat !== "number" || typeof s.lon !== "number") continue;
    b.extend([s.lon, s.lat]);
    any = true;
  }
  for (const h of data.hotspots ?? []) {
    if (h.center) {
      b.extend(h.center);
      any = true;
    }
  }
  return any ? b : null;
}

/**
 * Frame the whole network instead of a fixed centre/zoom.
 * The stations span 28.53–28.82 N while the panel is 1429x560, so at a fixed
 * zoom 4 of the 7 markers rendered outside the viewport (Dwarka, Jawaharlal
 * Nehru, Northwest Delhi, East Delhi) and simply could not be clicked.
 */
function fitStations(map: maplibregl.Map, animate: boolean) {
  map.fitBounds(stationBounds(), {
    padding: { top: 56, bottom: 56, left: 72, right: 72 },
    maxZoom: 12,
    duration: animate ? 900 : 0,
    essential: true,
  });
}

const LAYERS = [
  { key: "ALL" as const, label: "All", icon: Layers },
  { key: "NO2" as const, label: "NO\u2082", icon: Wind },
  { key: "O3" as const, label: "O\u2083", icon: CloudSun },
  { key: "HCHO" as const, label: "HCHO", icon: Satellite },
];

// --- HCHO satellite hotspot layer -------------------------------------------
// Built offline by scripts/build_hcho_hotspots.py and served by
// GET /api/hcho-hotspots. Replaces the old 7-point HCHO heatmap: the HCHO
// toggle now shows a smooth IDW field of the satellite column plus ranked
// hotspot outlines, and the band heatmap/circles are hidden for that view only.
const HCHO_IDW_SOURCE = "hcho-idw";
const HCHO_IDW_LAYER = "hcho-idw";
const HCHO_HOTSPOT_SOURCE = "hcho-hotspots";
const HCHO_HOTSPOT_FILL = "hcho-hotspot-fill";
const HCHO_HOTSPOT_LINE = "hcho-hotspot-line";
const HCHO_HOTSPOT_RANK = "hcho-hotspot-rank";

/** 1×1 transparent PNG — placeholder until the first IDW raster is ready. */
const TRANSPARENT_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

const EMPTY_FC = { type: "FeatureCollection" as const, features: [] };

/** Small round marker used for HCHO-layer stations (recent = ramp colour). */
function hchoDot(color: string, size: number, faded = false): HTMLDivElement {
  const dot = document.createElement("div");
  dot.style.cssText = `width:${size}px;height:${size}px;border-radius:50%;background:${color};` +
    `border:2px solid var(--paper-2);box-shadow:none;` +
    `${faded ? "opacity:0.65;" : ""}transition:transform 0.15s ease;`;
  return dot;
}

function hchoMarkerLabel(text: string, muted: boolean): HTMLDivElement {
  const label = document.createElement("div");
  label.style.cssText =
    "margin-top:3px;padding:1px 5px;background:var(--paper-2);border:1px solid var(--rule);" +
    "border-radius:4px;font-family:Inter,system-ui,sans-serif;font-size:9px;font-weight:600;" +
    `white-space:nowrap;box-shadow:none;line-height:1.3;pointer-events:none;color:${muted ? "var(--ink-muted)" : "var(--ink)"};`;
  label.textContent = text;
  return label;
}

function buildStationPopup(s: LiveStation, layer: LayerKey): string {
  const isAll = layer === "ALL";
  const { value, color } = stationStyle(s, layer);
  const unit = getLayerUnit(layer);
  const label = isAll ? COMPOSITE_LABELS[value] : getPollutantLabel(value, layer as SingleLayer);
  // In the composite view each pollutant keeps its own band colour, so the
  // values below can be read pollutant-by-pollutant instead of one merged number.
  const dot = (c: string) =>
    isAll
      ? `<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${c};margin-right:5px;vertical-align:middle;"></span>`
      : "";
  const no2Dot = dot(getPollutantColor(s.no2, "NO2"));
  const o3Dot = dot(getPollutantColor(s.o3, "O3"));
  const hchoDot = s.hcho != null ? dot(getPollutantColor(s.hcho, "HCHO")) : "";
  const obsNo2 = s.obsNo2 ?? null;
  const obsO3 = s.obsO3 ?? null;
  const obsHcho = s.obsHcho ?? null;
  const observedAt = s.observedAt ? s.observedAt.replace("T", " ").slice(0, 16) : null;
  const modelRun = s.modelRunAt ? s.modelRunAt.replace("T", " ").slice(0, 16) : null;
  const fmt = (v: number | null) => (v == null ? "—" : v);
  return `
    <div style="padding:14px 16px;font-family:Inter,system-ui,sans-serif;min-width:230px;background:var(--paper-2);border-radius:12px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
        <span style="width:10px;height:10px;border-radius:50%;background:${color};flex-shrink:0;box-shadow:0 0 8px ${color}90;"></span>
        <div>
          <div style="font-weight:700;font-size:14px;line-height:1.2;color:var(--ink);">${s.name}</div>
          <div style="font-size:10px;color:var(--ink-muted);margin-top:1px;">Delhi · CPCB Monitor · <span style="color:var(--sev-good);">●</span> Live</div>
        </div>
      </div>

      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:12px;padding:8px 10px;background:${color}18;border-radius:8px;border:1px solid ${color}30;">
        <span style="font-weight:800;font-size:26px;color:${color};line-height:1;text-shadow:0 0 12px${color}40;">${isAll ? label : value}</span>
        <span style="font-size:12px;color:var(--ink);font-weight:600;">${isAll ? "Composite · worst of NO₂ · O₃ · HCHO" : `${unit} ${label}`}</span>
      </div>

      <div style="font-size:10px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink);opacity:0.65;margin-bottom:4px;">Model forecast</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;font-size:11px;">
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">${no2Dot}NO₂</span><span style="color:var(--ink);font-weight:600;">${s.no2} <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">ppb</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">${o3Dot}O₃</span><span style="color:var(--ink);font-weight:600;">${s.o3} <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">ppb</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">${hchoDot}HCHO</span><span style="color:var(--ink);font-weight:600;">${s.hcho != null ? s.hcho : "—"} <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">µmol/m²</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">Temp</span><span style="color:var(--ink);font-weight:600;">${s.temp}°C</span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">Humidity</span><span style="color:var(--ink);font-weight:600;">${s.humidity}%</span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">Wind</span><span style="color:var(--ink);font-weight:600;">${s.wind} <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">km/h</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">Vis.</span><span style="color:var(--ink);font-weight:600;">${s.visibility} <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">km</span></span></div>
      </div>

      <div style="font-size:10px;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink);opacity:0.65;margin:10px 0 4px;">Database observation</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;font-size:11px;">
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">NO₂</span><span style="color:var(--ink);font-weight:600;">${fmt(obsNo2)} <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">ppb</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">O₃</span><span style="color:var(--ink);font-weight:600;">${fmt(obsO3)} <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">ppb</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">HCHO</span><span style="color:var(--ink);font-weight:600;">${fmt(obsHcho)} <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">µmol/m²</span></span></div>
        ${
          s.pm25 != null
            ? `<div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">PM2.5</span><span style="color:var(--ink);font-weight:600;">${s.pm25} <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">µg/m³</span></span></div>`
            : ""
        }
        ${
          s.aqi != null
            ? `<div style="display:flex;justify-content:space-between;"><span style="color:var(--ink);">AQI</span><span style="color:var(--ink);font-weight:600;">${Math.round(s.aqi)}${s.aqiCategory ? ` <span style="color:var(--ink);opacity:0.65;font-weight:400;font-size:10px;">${s.aqiCategory.replace("_", " ")}</span>` : ""}</span></div>`
            : ""
        }
      </div>
      ${
        s.district
          ? `<div style="margin-top:8px;font-size:10px;color:var(--ink);opacity:0.65;">${s.district}</div>`
          : ""
      }

      <div style="margin-top:10px;padding-top:6px;border-top:1px solid var(--rule);font-size:10px;color:var(--ink);text-align:right;opacity:0.65;">
        ${observedAt ? `Observed ${observedAt}` : ""}
        ${observedAt && modelRun ? " · " : ""}
        ${modelRun ? `Model run ${modelRun}` : ""}
      </div>
    </div>
  `;
}

function buildLoadingPopup(name: string): string {
  return `
    <div style="padding:14px 16px;font-family:Inter,system-ui,sans-serif;min-width:200px;background:var(--paper-2);border-radius:12px;text-align:center;">
      <div style="font-weight:700;font-size:14px;color:var(--ink);margin-bottom:6px;">${name}</div>
      <div style="font-size:12px;color:var(--ink-muted);">⏳ Loading model data…</div>
    </div>
  `;
}

// CARTO now returns an "API KEY REQUIRED" placeholder for every *raster* tile
// (a valid-looking 256x256 PNG that is really a watermark), which is why the
// basemap was blank behind a repeating watermark. Their vector styles — and
// the tiles they pull from — are still keyless, so streets/dark use those.
const CARTO_VOYAGER = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json";
const CARTO_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

const MAP_STYLES = {
  streets: CARTO_VOYAGER,
  dark: CARTO_DARK,
  satellite: {
    version: 8 as const,
    sources: {
      satellite: {
        type: "raster" as const,
        tiles: [
          "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        attribution: "&copy; Esri, Maxar, Earthstar Geographics",
        maxzoom: 19,
      },
      // Esri's keyless place-name overlay — CARTO's raster label tiles are
      // watermarked too, so they cannot be layered on the imagery.
      labels: {
        type: "raster" as const,
        tiles: [
          "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        maxzoom: 19,
      },
    },
    layers: [
      { id: "satellite", type: "raster" as const, source: "satellite", minzoom: 0, maxzoom: 19 },
      { id: "labels", type: "raster" as const, source: "labels", minzoom: 0, maxzoom: 19 },
    ],
  },
} as const;

interface MapPanelProps {
  layer: LayerKey;
  setLayer: (l: LayerKey) => void;
  timeIdx: number;
  setTimeIdx: (i: number) => void;
  onStationSelect?: (station: { name: string; city: string; value: number } | null) => void;
  onRefresh?: () => void;
  stations?: Array<{
    name: string;
    no2: number;
    o3: number;
    hcho?: number | null;
    lat: number;
    lon: number;
    temp: number;
    humidity: number;
    wind: number;
    visibility: number;
    windDir?: number;
    pm25?: number | null;
    aqi?: number | null;
    aqiCategory?: string | null;
    district?: string | null;
    obsNo2?: number | null;
    obsO3?: number | null;
    obsHcho?: number | null;
    observedAt?: string;
    modelRunAt?: string;
  }>;
}


export function MapPanel({ layer, setLayer, onStationSelect, onRefresh, stations: apiStations }: MapPanelProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);
  const popupsRef = useRef<maplibregl.Popup[]>([]);
  const mapLoadedRef = useRef(false);
  const apiStationsRef = useRef(apiStations);
  const layerRef = useRef(layer);
  // Previous layer, so the effect that re-frames the camera can tell an HCHO
  // activation from a paint refresh.
  const prevLayerRef = useRef(layer);
  const onStationSelectRef = useRef(onStationSelect);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [mapStyle, setMapStyle] = useState<"streets" | "satellite" | "dark">("streets");
  const [mapReady, setMapReady] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  apiStationsRef.current = apiStations;
  layerRef.current = layer;
  onStationSelectRef.current = onStationSelect;
  const hasRealData = apiStations && apiStations.length > 0;

  const clearMarkers = useCallback(() => {
    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];
    popupsRef.current.forEach((p) => p.remove());
    popupsRef.current = [];
  }, []);

  // --- HCHO satellite hotspot layer ----------------------------------------
  const [hcho, setHcho] = useState<{
    status: "loading" | "ready" | "error";
    data?: HchoHotspotsPayload;
    error?: string;
  }>({ status: "loading" });
  const [selectedHotspot, setSelectedHotspot] = useState<HchoHotspot | null>(null);
  const hchoDataRef = useRef<HchoHotspotsPayload | null>(null);
  const hchoMarkersRef = useRef<maplibregl.Marker[]>([]);
  const rasterRafRef = useRef(0);
  // MapLibre's image source swaps asynchronously: calling updateImage again
  // before the previous swap settles aborts the in-flight load, which shows
  // up in the console as "InvalidStateError: The source image could not be
  // decoded." So at most one swap is in flight — map moves coalesce and the
  // field re-renders once when the map comes to rest.
  const rasterInFlightRef = useRef(false);
  const rasterDirtyRef = useRef(false);
  const rasterSettleTimerRef = useRef(0);
  /** The pixels the source currently shows — used to skip no-op swaps. */
  const rasterUrlRef = useRef<string | null>(null);

  const clearHchoMarkers = useCallback(() => {
    hchoMarkersRef.current.forEach((m) => m.remove());
    hchoMarkersRef.current = [];
  }, []);

  /** Markers for the HCHO view: the 7 curated sites greyed (no recent HCHO
   *  for any of them) plus a small dot per station that has current data. */
  const buildHchoMarkers = useCallback(
    (map: maplibregl.Map, data: HchoHotspotsPayload | null) => {
      clearHchoMarkers();
      if (!data) return;
      const currentIds = new Set(
        (data.stations ?? []).filter((s) => s.latest_pass).map((s) => s.id),
      );

      // Greyed curated stations — kept for orientation, excluded from the field.
      STATIONS.forEach((station) => {
        const wrapper = document.createElement("div");
        wrapper.style.cssText = "display:flex;flex-direction:column;align-items:center;cursor:default;z-index:5;";
        wrapper.appendChild(hchoDot(inkMuted(), 10, true));
        wrapper.appendChild(hchoMarkerLabel(station.name, true));
        const popup = new maplibregl.Popup({
          offset: 12, closeButton: false, closeOnClick: false, maxWidth: "260px",
        }).setHTML(
          `<div style="padding:10px 12px;font-family:Inter,system-ui,sans-serif;background:var(--paper-2);border-radius:10px;min-width:180px;">
             <div style="font-weight:700;font-size:13px;color:var(--ink);">${station.name}</div>
             <div style="font-size:11px;color:var(--ink-muted);margin-top:3px;">No recent satellite HCHO — excluded from the field</div>
           </div>`,
        );
        popupsRef.current.push(popup);
        wrapper.addEventListener("mouseenter", () => popup.setLngLat(station.coords).addTo(map));
        wrapper.addEventListener("mouseleave", () => popup.remove());
        hchoMarkersRef.current.push(
          new maplibregl.Marker({ element: wrapper, anchor: "center" }).setLngLat(station.coords).addTo(map),
        );
      });

      // Stations with current HCHO — small dot coloured by the shared ramp.
      (data.stations ?? [])
        .filter((s) => s.latest_pass && currentIds.has(s.id))
        .forEach((s) => {
          const wrapper = document.createElement("div");
          wrapper.style.cssText = "display:flex;flex-direction:column;align-items:center;cursor:pointer;z-index:6;";
          wrapper.appendChild(hchoDot(`rgb(${rampColor(s.value).join(",")})`, 12));
          wrapper.appendChild(hchoMarkerLabel(`${s.value} µmol/m²`, false));
          const popup = new maplibregl.Popup({
            offset: 12, closeButton: false, closeOnClick: false, maxWidth: "260px",
          }).setHTML(
            `<div style="padding:10px 12px;font-family:Inter,system-ui,sans-serif;background:var(--paper-2);border-radius:10px;min-width:190px;">
               <div style="font-weight:700;font-size:13px;color:var(--ink);">${s.name}</div>
               <div style="font-size:11px;color:var(--ink-muted);margin-top:2px;">Satellite pass ${formatPassDate(s.date)}</div>
               <div style="margin-top:8px;display:flex;align-items:baseline;gap:6px;">
                 <span style="font-weight:800;font-size:20px;color:var(--ink);">${s.value}</span>
                 <span style="font-size:11px;color:var(--ink-muted);">µmol/m² · daily mean</span>
               </div>
               <div style="font-size:11px;color:var(--ink-muted);margin-top:4px;">7-day mean ${s.mean_7d} · ${s.days}/7 days with data</div>
               <div style="font-size:10px;color:var(--ink-muted);margin-top:6px;">Satellite column, not ground level.</div>
             </div>`,
          );
          popupsRef.current.push(popup);
          wrapper.addEventListener("mouseenter", () => popup.setLngLat([s.lon, s.lat]).addTo(map));
          wrapper.addEventListener("mouseleave", () => popup.remove());
          hchoMarkersRef.current.push(
            new maplibregl.Marker({ element: wrapper, anchor: "center" }).setLngLat([s.lon, s.lat]).addTo(map),
          );
        });
    },
    [clearHchoMarkers],
  );

  /** IDW field for the current viewport: the data URL plus the bounds it is
   *  anchored to. TRANSPARENT_PNG when there is nothing to interpolate. */
  const hchoRasterFrame = useCallback((map: maplibregl.Map) => {
    const b = map.getBounds();
    const west = b.getWest();
    const east = b.getEast();
    const north = b.getNorth();
    const south = b.getSouth();
    const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [
      [west, north], [east, north], [east, south], [west, south],
    ];
    const stations = hchoDataRef.current ? interpolationStations(hchoDataRef.current) : [];
    if (!stations.length) return { coordinates, url: TRANSPARENT_PNG };
    // Grid sized to the viewport (not the world): at any zoom the field is a
    // smooth canvas, so no cell edges can show at zoom 8–12 or anywhere else.
    const width = 240;
    const spanLat = Math.max(1e-6, north - south);
    const spanLon = Math.max(1e-6, east - west);
    const height = Math.max(64, Math.min(240, Math.round((width * spanLat) / spanLon)));
    const url = rasterToDataUrl(idwRaster(stations, { north, south, east, west }, width, height));
    return { coordinates, url: url || TRANSPARENT_PNG };
  }, []);

  /** Repaint the smooth IDW field for the current viewport. One swap in
   *  flight at a time; a trailing update runs when the map settles. */
  const updateHchoRaster = useCallback((map: maplibregl.Map) => {
    const src = map.getSource(HCHO_IDW_SOURCE) as maplibregl.ImageSource | undefined;
    if (!src) return;
    if (rasterInFlightRef.current) {
      rasterDirtyRef.current = true;
      return;
    }
    const { coordinates, url } = hchoRasterFrame(map);
    // Same pixels the source already shows (e.g. the frame drawn when the
    // source was just created with this data) — nothing to swap, and a swap
    // here would only abort the load that is already running.
    if (url === rasterUrlRef.current) return;
    rasterUrlRef.current = url;
    rasterInFlightRef.current = true;
    src.updateImage({ coordinates, url });
    let settled = false;
    const settle = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(rasterSettleTimerRef.current);
      rasterInFlightRef.current = false;
      if (rasterDirtyRef.current) {
        rasterDirtyRef.current = false;
        if (mapRef.current === map && layerRef.current === "HCHO" && hchoDataRef.current) {
          updateHchoRasterRef.current?.(map);
        }
      }
    };
    // 'idle' fires once the swapped image has been uploaded; the timeout
    // guards the rare case where no repaint ever gets scheduled.
    map.once("idle", settle);
    rasterSettleTimerRef.current = window.setTimeout(settle, 1500);
  }, [hchoRasterFrame]);
  const updateHchoRasterRef = useRef(updateHchoRaster);
  updateHchoRasterRef.current = updateHchoRaster;

  const scheduleHchoRaster = useCallback(() => {
    const map = mapRef.current;
    if (!map || rasterRafRef.current) return;
    rasterRafRef.current = requestAnimationFrame(() => {
      rasterRafRef.current = 0;
      if (layerRef.current === "HCHO" && hchoDataRef.current) updateHchoRaster(map);
    });
  }, [updateHchoRaster]);

  /** Add the HCHO sources/layers once (idempotent — style switches wipe them). */
  const ensureHchoLayers = useCallback((map: maplibregl.Map) => {
    if (!map.getSource(HCHO_IDW_SOURCE)) {
      // When the payload is already in, create the source with the real
      // first frame — swapping right after creation would abort the
      // placeholder load and log a decode error.
      const frame = hchoRasterFrame(map);
      rasterUrlRef.current = frame.url;
      map.addSource(HCHO_IDW_SOURCE, {
        type: "image",
        coordinates: frame.coordinates,
        url: frame.url,
      });
      map.addLayer({
        id: HCHO_IDW_LAYER,
        type: "raster",
        source: HCHO_IDW_SOURCE,
        // Below the hotspot outlines and rank labels, above the basemap.
        paint: { "raster-opacity": 0.65, "raster-fade-duration": 150 },
      });
    }
    if (!map.getSource(HCHO_HOTSPOT_SOURCE)) {
      map.addSource(HCHO_HOTSPOT_SOURCE, { type: "geojson", data: EMPTY_FC as any });
      // Outline only — the field underneath must stay visible.
      map.addLayer({
        id: HCHO_HOTSPOT_FILL,
        type: "fill",
        source: HCHO_HOTSPOT_SOURCE,
        paint: { "fill-color": accent(), "fill-opacity": 0.06 },
      });
      map.addLayer({
        id: HCHO_HOTSPOT_LINE,
        type: "line",
        source: HCHO_HOTSPOT_SOURCE,
        paint: { "line-color": accent(), "line-width": 2, "line-opacity": 0.7 },
      });
      // Rank as a number, not just a colour — the non-colour signal.
      map.addLayer({
        id: HCHO_HOTSPOT_RANK,
        type: "symbol",
        source: HCHO_HOTSPOT_SOURCE,
        layout: {
          "text-field": ["get", "rankLabel"],
          "text-size": 12,
          "text-allow-overlap": true,
          "text-ignore-placement": true,
        },
        paint: { "text-color": paper2(), "text-halo-color": accent(), "text-halo-width": 1.6 },
      });
    }
  }, [hchoRasterFrame]);

  /** Show/hide everything HCHO for the active layer + data state. */
  const applyHcho = useCallback(
    (map: maplibregl.Map, currentLayer: LayerKey, data: HchoHotspotsPayload | null, status: string) => {
      ensureHchoLayers(map);
      const show = currentLayer === "HCHO" && status === "ready" && !!data;
      const setVis = (id: string, visible: boolean) => {
        if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
      };
      // The old 7-point band heatmap is NO2/O3/ALL's view now; the HCHO view
      // is the satellite field, so the two never overlay.
      setVis("pollutant-heatmap", !show);
      setVis("pollutant-circles", !show);
      setVis(HCHO_IDW_LAYER, show);
      setVis(HCHO_HOTSPOT_FILL, show);
      setVis(HCHO_HOTSPOT_LINE, show);
      setVis(HCHO_HOTSPOT_RANK, show);

      if (!show) {
        clearHchoMarkers();
        setSelectedHotspot(null);
        return;
      }
      const payload = data as HchoHotspotsPayload;
      const src = map.getSource(HCHO_HOTSPOT_SOURCE) as maplibregl.GeoJSONSource | undefined;
      src?.setData({
        type: "FeatureCollection",
        features: (payload.hotspots ?? []).map((h) => ({
          type: "Feature" as const,
          geometry: { type: "Polygon" as const, coordinates: [h.polygon] },
          properties: {
            rank: h.rank,
            rankLabel: `${h.rank}`,
            peak: h.peak,
            persistence: h.persistence,
            ratio: h.ratio_to_median,
          },
        })),
      } as any);
      updateHchoRaster(map);
      buildHchoMarkers(map, payload);
    },
    [buildHchoMarkers, clearHchoMarkers, ensureHchoLayers, updateHchoRaster],
  );

  // Fetch the offline-built layer once per mount.
  useEffect(() => {
    const ac = new AbortController();
    setHcho({ status: "loading" });
    fetchHchoHotspots(ac.signal)
      .then((data) => {
        hchoDataRef.current = data;
        setHcho({ status: "ready", data });
      })
      .catch((err: Error) => {
        if (err?.name === "AbortError") return;
        // One honest line — the layer hides itself; no stack traces anywhere.
        hchoDataRef.current = null;
        setHcho({ status: "error", error: err?.message ?? "HCHO layer unavailable" });
      });
    return () => ac.abort();
  }, []);

  const buildAndAddMarkers = useCallback(
    (map: maplibregl.Map, currentLayer: LayerKey, stationsData: typeof apiStations) => {
      clearMarkers();
      // The HCHO view owns its markers (greyed curated sites + stations with
      // current satellite data) — the band-coloured set would duplicate them.
      if (currentLayer === "HCHO") return;
      const liveStations = buildLiveStations(STATIONS, stationsData);

      liveStations.forEach((station) => {
        const wrapper = document.createElement("div");
        wrapper.style.cssText = "display:flex;flex-direction:column;align-items:center;cursor:pointer;z-index:10;";

        const dot = document.createElement("div");
        const label = document.createElement("div");

        if (!stationsData || stationsData.length === 0) {
          dot.style.cssText = "width:22px;height:22px;border-radius:50%;background:var(--ink-muted);border:3px solid var(--paper-2);box-shadow:0 0 0 3px color-mix(in srgb, var(--ink-muted) 25%, transparent);position:relative;z-index:2;";
          label.style.cssText = "margin-top:4px;padding:2px 6px;background:var(--paper-2);border:1px solid var(--rule);border-radius:4px;font-family:Inter,system-ui,sans-serif;font-size:10px;font-weight:600;color:var(--ink-muted);white-space:nowrap;box-shadow:none;line-height:1.3;text-align:center;pointer-events:none;";
          label.textContent = station.name + " · Loading…";
        } else {
          const { value, color } = stationStyle(station, currentLayer);

          dot.style.cssText = `width:22px;height:22px;border-radius:50%;background:${color};border:3px solid var(--paper-2);box-shadow:0 0 0 3px ${color}40;transition:transform 0.2s ease,box-shadow 0.2s ease;position:relative;z-index:2;`;

          const pulse = document.createElement("div");
          pulse.style.cssText = `position:absolute;width:22px;height:22px;border-radius:50%;border:2px solid ${color};top:0;left:0;animation:station-pulse-ring 2s ease-out infinite;pointer-events:none;`;
          dot.appendChild(pulse);

          label.style.cssText = "margin-top:4px;padding:2px 6px;background:var(--paper-2);border:1px solid var(--rule);border-radius:4px;font-family:Inter,system-ui,sans-serif;font-size:10px;font-weight:600;color:var(--ink);white-space:nowrap;box-shadow:none;line-height:1.3;text-align:center;pointer-events:none;";
          label.textContent = station.name;

          const popup = new maplibregl.Popup({
            offset: 24, closeButton: false, closeOnClick: false,
            className: "station-popup", maxWidth: "300px",
          }).setHTML(buildStationPopup(station, currentLayer));
          popupsRef.current.push(popup);

          wrapper.addEventListener("mouseenter", () => {
            dot.style.transform = "scale(1.4)";
            dot.style.boxShadow = `0 0 0 5px ${color}50`;
            label.style.background = color;
            label.style.color = bandTextColor(band(value, currentLayer));
            label.style.borderColor = color;
            popup.setLngLat(station.coords).addTo(map);
          });
          wrapper.addEventListener("mouseleave", () => {
            dot.style.transform = "scale(1)";
            dot.style.boxShadow = `0 0 0 3px ${color}40`;
            label.style.background = "var(--paper-2)";
            label.style.color = "var(--ink)";
            label.style.borderColor = "var(--rule)";
            popup.remove();
          });
          wrapper.addEventListener("click", () => {
            onStationSelectRef.current?.({ name: station.name, city: "Delhi", value });
          });
        }

        wrapper.appendChild(dot);
        wrapper.appendChild(label);

        const marker = new maplibregl.Marker({ element: wrapper, anchor: "center" })
          .setLngLat(station.coords)
          .addTo(map);
        markersRef.current.push(marker);
      });
    },
    [clearMarkers],
  );

  const addPollutantLayers = useCallback((map: maplibregl.Map, currentLayer: LayerKey, stationsData: typeof apiStations) => {
    const liveStations = buildLiveStations(STATIONS, stationsData);
    const geoJSON = generatePollutantGeoJSON(liveStations);

    if (map.getSource("pollutant-points")) {
      (map.getSource("pollutant-points") as maplibregl.GeoJSONSource).setData(geoJSON as any);
    } else {
      map.addSource("pollutant-points", { type: "geojson", data: geoJSON });
    }

    if (!map.getLayer("pollutant-heatmap")) {
      map.addLayer({
        id: "pollutant-heatmap",
        type: "heatmap",
        source: "pollutant-points",
        maxzoom: 15,
        paint: {
          "heatmap-weight": ["interpolate", ["linear"], ["get", "no2"], 0, 0, 150, 1],
          "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 0, 0.6, 15, 2],
          "heatmap-color": [
            "interpolate", ["linear"], ["heatmap-density"],
            0, rgba(BAND_COLORS[0], 0), 0.2, rgba(BAND_COLORS[0], 0.15), 0.4, rgba(BAND_COLORS[1], 0.2),
            0.6, rgba(BAND_COLORS[2], 0.2), 0.8, rgba(BAND_COLORS[3], 0.18), 1, rgba(BAND_COLORS[4], 0.15),
          ],
          "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 0, 8, 15, 28],
          "heatmap-opacity": ["interpolate", ["linear"], ["zoom"], 7, 0.6, 15, 0.08],
        },
      });
    }

    if (!map.getLayer("pollutant-circles")) {
      map.addLayer({
        id: "pollutant-circles",
        type: "circle",
        source: "pollutant-points",
        minzoom: 10,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 10, 3, 16, 10],
          "circle-color": [
            "interpolate", ["linear"], ["get", "no2"],
            0, BAND_COLORS[0], 30, BAND_COLORS[0], 53, BAND_COLORS[1], 100, BAND_COLORS[2], 200, BAND_COLORS[3], 300, BAND_COLORS[4],
          ],
          "circle-stroke-color": `rgba(${toRgb(paper2()).join(",")},0.6)`,
          "circle-stroke-width": 1,
          "circle-opacity": 0.65,
          "circle-blur": 0.4,
        },
      });
    }

    updatePaintProperties(map, currentLayer);
  }, []);

  const updatePaintProperties = useCallback((map: maplibregl.Map, currentLayer: LayerKey) => {
    if (!map.getLayer("pollutant-heatmap") || !map.getLayer("pollutant-circles")) return;

    // HCHO is no longer drawn from the 7-point band heatmap — it has the
    // satellite field + hotspot outlines (see applyHcho). The two layers are
    // hidden on that view, so there is nothing to repaint here.
    if (currentLayer === "HCHO") return;

    if (currentLayer === "ALL") {
      // Composite view: colour by the station's worst band (0-4). The band is
      // always an integer, so the interpolated stops land exactly on the
      // palette colours the markers and legend use. Same shape as the
      // per-pollutant `thresholds` below.
      const severity = {
        w: [0, 0, 1, 0.2, 2, 0.5, 3, 0.8, 4, 1],
        c: [0, BAND_COLORS[0], 1, BAND_COLORS[1], 2, BAND_COLORS[2], 3, BAND_COLORS[3], 4, BAND_COLORS[4]],
      };
      map.setPaintProperty("pollutant-heatmap", "heatmap-weight", [
        "interpolate", ["linear"], ["get", "severity"], ...severity.w,
      ]);
      map.setPaintProperty("pollutant-circles", "circle-color", [
        "interpolate", ["linear"], ["get", "severity"], ...severity.c,
      ]);
      return;
    }

    const prop = currentLayer === "NO2" ? "no2" : "o3";
    const thresholds = currentLayer === "NO2"
      ? { w: [0, 0, 30, 0.2, 53, 0.5, 100, 0.8, 200, 1], c: [0, BAND_COLORS[0], 30, BAND_COLORS[0], 53, BAND_COLORS[1], 100, BAND_COLORS[2], 200, BAND_COLORS[3], 300, BAND_COLORS[4]] }
      : { w: [0, 0, 40, 0.2, 70, 0.5, 100, 0.8, 150, 1], c: [0, BAND_COLORS[0], 40, BAND_COLORS[0], 70, BAND_COLORS[1], 100, BAND_COLORS[2], 150, BAND_COLORS[3], 200, BAND_COLORS[4]] };

    map.setPaintProperty("pollutant-heatmap", "heatmap-weight", [
      "interpolate", ["linear"], ["get", prop], ...thresholds.w,
    ]);
    map.setPaintProperty("pollutant-circles", "circle-color", [
      "interpolate", ["linear"], ["get", prop], ...thresholds.c,
    ]);
  }, []);

  // Create map once — never destroyed on style change
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: MAP_STYLES[mapStyle] as any,
      center: DELHI_CENTER,
      zoom: DELHI_ZOOM,
      minZoom: 2,
      maxZoom: 20,
      attributionControl: false,
      pitchWithRotate: true,
      touchZoomRotate: true,
    });

    map.addControl(
      new maplibregl.NavigationControl({ showCompass: true, showZoom: true, visualizePitch: true }),
      "bottom-right",
    );
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 200, unit: "metric" }), "bottom-left");

    map.on("load", () => {
      mapLoadedRef.current = true;
      setMapReady(true);
      // Frame all 7 stations first so the markers below are placed correctly.
      fitStations(map, false);
      addPollutantLayers(map, layerRef.current, apiStationsRef.current);
      buildAndAddMarkers(map, layerRef.current, apiStationsRef.current);
      // HCHO satellite field + hotspot outlines (visible only on the HCHO layer).
      ensureHchoLayers(map);
      applyHcho(map, layerRef.current, hchoDataRef.current, hchoDataRef.current ? "ready" : "loading");
      map.on("move", scheduleHchoRaster);

      // Hotspot click/tap → detail panel. The fill layer is deliberately the
      // hit target (a 2px line is too thin to tap).
      map.on("click", HCHO_HOTSPOT_FILL, (e) => {
        const rank = Number(e.features?.[0]?.properties?.rank);
        const found = hchoDataRef.current?.hotspots.find((h) => h.rank === rank) ?? null;
        setSelectedHotspot(found);
      });
      map.on("click", (e) => {
        // Tapping the bare map dismisses the panel.
        const hits = map.queryRenderedFeatures(e.point, { layers: [HCHO_HOTSPOT_FILL] });
        if (!hits.length) setSelectedHotspot(null);
      });
      map.on("mouseenter", HCHO_HOTSPOT_FILL, () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", HCHO_HOTSPOT_FILL, () => {
        map.getCanvas().style.cursor = "";
      });
    });

    mapRef.current = map;
    // Dev-only handle: scripts/shots_hcho.mjs projects hotspot centres into
    // screen space to tap them. Production builds strip this branch.
    if (import.meta.env.DEV) {
      (window as unknown as { __vdMap?: maplibregl.Map }).__vdMap = map;
    }

    return () => {
      clearMarkers();
      clearHchoMarkers();
      if (rasterRafRef.current) cancelAnimationFrame(rasterRafRef.current);
      window.clearTimeout(rasterSettleTimerRef.current);
      rasterInFlightRef.current = false;
      rasterDirtyRef.current = false;
      mapLoadedRef.current = false;
      setMapReady(false);
      map.remove();
      mapRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Update style without destroying the map
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    map.setStyle(MAP_STYLES[mapStyle] as any);
    // After setStyle, sources/layers are wiped — re-add on next idle
    const onIdle = () => {
      map.off("idle", onIdle);
      if (!mapLoadedRef.current) return;
      addPollutantLayers(map, layerRef.current, apiStationsRef.current);
      // addPollutantLayers writes NO2 paint literals; re-apply whichever layer
      // is active so switching basemap doesn't silently revert O3 → NO2.
      updatePaintProperties(map, layerRef.current);
      buildAndAddMarkers(map, layerRef.current, apiStationsRef.current);
      // setStyle wiped the HCHO sources too — rebuild and reapply.
      ensureHchoLayers(map);
      applyHcho(map, layerRef.current, hchoDataRef.current, hchoDataRef.current ? "ready" : "loading");
    };
    map.on("idle", onIdle);
  }, [mapStyle, addPollutantLayers, buildAndAddMarkers, updatePaintProperties, ensureHchoLayers, applyHcho]);

  // Update GeoJSON data + paint properties when apiStations or layer changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoadedRef.current) return;
    const source = map.getSource("pollutant-points") as maplibregl.GeoJSONSource | undefined;
    if (source) {
      const liveStations = buildLiveStations(STATIONS, apiStations);
      source.setData(generatePollutantGeoJSON(liveStations) as any);
    }
    updatePaintProperties(map, layer);
    // HCHO layer visibility, field and markers follow the toggle + fetch state.
    applyHcho(map, layer, hcho.data ?? null, hcho.status);

    // Frame the camera to whatever the active layer actually draws. The 7
    // curated surface sites are narrower than the HCHO satellite network, so
    // on entering HCHO we fit the satellite stations + hotspots; on leaving we
    // restore the surface-station frame. Without this the hotspots sat east of
    // the viewport on a phone and could not be tapped.
    const prev = prevLayerRef.current;
    prevLayerRef.current = layer;
    if (prev === layer) return;
    if (layer === "HCHO") {
      const b = hcho.data ? hchoBounds(hcho.data) : null;
      if (b && !b.isEmpty()) {
        map.fitBounds(b, {
          padding: { top: 56, bottom: 56, left: 64, right: 64 },
          maxZoom: 12,
          duration: 700,
          essential: true,
        });
      }
    } else if (prev === "HCHO") {
      fitStations(map, true);
    }
  }, [apiStations, layer, updatePaintProperties, applyHcho, hcho]);

  // Update station markers when apiStations or layer changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoadedRef.current) return;
    buildAndAddMarkers(map, layer, apiStations);
  }, [apiStations, layer, buildAndAddMarkers, mapReady]);

  const cycleStyle = useCallback(() => {
    setMapStyle((prev) => {
      if (prev === "streets") return "satellite";
      if (prev === "satellite") return "dark";
      return "streets";
    });
  }, []);

  const handleRefresh = useCallback(() => {
    const map = mapRef.current;
    if (!map || !mapLoadedRef.current) return;
    setRefreshing(true);

    onRefresh?.();

    const source = map.getSource("pollutant-points") as maplibregl.GeoJSONSource | undefined;
    if (source) {
      const liveStations = buildLiveStations(STATIONS, apiStationsRef.current);
      const newGeoJSON = generatePollutantGeoJSON(liveStations);
      source.setData(newGeoJSON as any);
    }

    buildAndAddMarkers(map, layerRef.current, apiStationsRef.current);
    map.triggerRepaint();

    setTimeout(() => setRefreshing(false), 600);
  }, [buildAndAddMarkers, onRefresh]);

  const getStyleIcon = () => {
    switch (mapStyle) {
      case "satellite":
        return <Satellite className="h-3.5 w-3.5" />;
      case "dark":
        return <CloudSun className="h-3.5 w-3.5" />;
      default:
        return <MapIcon className="h-3.5 w-3.5" />;
    }
  };

  /** Heading above the colour swatches — the ALL view bands are a composite. */
  const legendTitle =
    layer === "ALL"
      ? "Composite (worst of NO₂/O₃/HCHO)"
      : layer === "NO2"
        ? "NO₂ (ppb)"
        : layer === "O3"
          ? "O₃ (ppb)"
          : "HCHO (µmol/m²)";

  return (
    <Card className="mx-4 overflow-hidden md:mx-8">
      <CardContent className="p-0">
        {/* Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b px-4 py-2.5">
          <div className="flex flex-wrap items-center gap-1.5">
            {LAYERS.map(({ key, label, icon: Icon }) => (
              <Button
                key={key}
                variant={layer === key ? "default" : "ghost"}
                size="sm"
                onClick={() => setLayer(key)}
                className="gap-1.5 text-xs"
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </Button>
            ))}
            <div className="mx-1 h-4 w-px bg-border" />
            <Button variant="ghost" size="sm" onClick={cycleStyle} className="gap-1.5 text-xs">
              {getStyleIcon()}
              {mapStyle === "streets" ? "Voyager" : mapStyle === "satellite" ? "Satellite" : "Dark"}
            </Button>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="mr-2 flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <Radio className={`h-3 w-3 ${hasRealData ? "animate-pulse text-aqi-good" : "text-muted-foreground"}`} />
              <span>{hasRealData ? `${apiStations.length} live stations` : "Connecting to model…"}</span>
              <span className="mx-1">·</span>
              <Clock className="h-3 w-3" />
              <span>
                {formatTime(now.getHours(), now.getMinutes())}
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={handleRefresh}
              title="Refresh map data"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => {
                if (mapRef.current) fitStations(mapRef.current, true);
              }}
            >
              <LocateFixed className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        {/* Map */}
        <div className="relative">
          <div
            ref={mapContainer}
            className="w-full h-[420px] md:h-[560px] transition-all duration-300"
          />

          {/* HCHO layer status chip — loading, error and pass header (Step 5) */}
          {layer === "HCHO" && (
            <div className="pointer-events-none absolute left-3 top-3 z-20 max-w-[min(92%,340px)] rounded-lg border bg-background/90 px-3 py-2 backdrop-blur">
              {hcho.status === "loading" && (
                <div className="text-xs text-muted-foreground">Loading satellite pass…</div>
              )}
              {hcho.status === "error" && (
                <div className="text-xs font-medium text-destructive">
                  HCHO layer hidden — {hcho.error}
                </div>
              )}
              {hcho.status === "ready" && hcho.data && (
                <>
                  <div className="flex flex-wrap items-center gap-1.5 text-xs font-semibold">
                    <span>
                      Satellite pass: {formatPassDate(hcho.data.as_of)} (
                      {hcho.data.age_days ?? "?"} days old)
                    </span>
                    {hcho.data.archived && (
                      <span className="rounded border px-1 py-px text-[10px] font-medium text-muted-foreground">
                        Archived
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
                    Daily satellite pass — not hourly; the time slider does not move this layer.
                    <br />
                    Estimated between stations.
                    {(hcho.data.no_recent_data?.length ?? 0) > 0 && (
                      <>
                        <br />
                        {hcho.data.no_recent_data.length} of{" "}
                        {hcho.data.no_recent_data.length + (hcho.data.stations?.length ?? 0)}{" "}
                        stations have no recent HCHO and are greyed.
                      </>
                    )}
                    {(hcho.data.hotspots?.length ?? 0) === 0 && (
                      <div className="mt-1 font-semibold text-foreground">
                        No clear hotspot on {formatPassDate(hcho.data.as_of)}
                      </div>
                    )}
                    {(hcho.data.hotspots?.length ?? 0) > 0 && (
                      <div className="mt-1 font-medium text-foreground">
                        {hcho.data.hotspots.length} hotspot
                        {hcho.data.hotspots.length > 1 ? "s" : ""} — tap an outline for details
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Hotspot detail panel — desktop card / mobile bottom sheet */}
          {layer === "HCHO" && hcho.status === "ready" && hcho.data && selectedHotspot && (
            <HchoHotspotPanel
              hotspot={selectedHotspot}
              region={hcho.data.region ?? "Delhi-NCR"}
              median={hcho.data.region_stats?.median ?? null}
              asOf={hcho.data.as_of}
              stations={hcho.data.stations ?? []}
              onClose={() => setSelectedHotspot(null)}
            />
          )}
        </div>

        {/* Bottom bar with legend + stats */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t px-4 py-2.5">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-[10px] font-medium text-muted-foreground">{legendTitle}:</span>
            {layer === "HCHO" ? (
              <>
                {/* One sequential ramp, shared with the field — with numeric
                    stops so colour is never the only signal. */}
                <div className="relative h-5 w-44">
                  <div
                    className="h-2 w-full rounded-sm border"
                    style={{ background: rampCssGradient() }}
                  />
                  {[0, 10, 30, 60, 90].map((v) => (
                    <span
                      key={v}
                      className="absolute top-2 -translate-x-1/2 text-[9px] tabular-nums text-muted-foreground"
                      style={{ left: `${Math.min(100, Math.max(0, (v / HCHO_RAMP_MAX) * 100))}%` }}
                    >
                      {v}
                    </span>
                  ))}
                </div>
                <div className="flex items-center gap-1">
                  <span
                    className="inline-block h-2.5 w-3.5 rounded-sm"
                    style={{ border: "2px solid var(--accent)", background: "color-mix(in srgb, var(--accent) 6%, transparent)" }}
                  />
                  <span className="text-[10px] text-muted-foreground">Hotspot cluster (ranked)</span>
                </div>
                {hcho.status === "ready" && hcho.data && (
                  <span className="text-[10px] text-muted-foreground">
                    Relative to {hcho.data.region ?? "Delhi-NCR"} on {formatPassDate(hcho.data.as_of)}.
                    Satellite column, not ground level.
                  </span>
                )}
              </>
            ) : (
              (
                [
                  ...(layer === "NO2"
                    ? [
                        { label: "Good (0–30)", color: BAND_COLORS[0] },
                        { label: "Moderate (31–53)", color: BAND_COLORS[1] },
                        { label: "Elevated (54–100)", color: BAND_COLORS[2] },
                        { label: "High (101–200)", color: BAND_COLORS[3] },
                        { label: "Very High (200+)", color: BAND_COLORS[4] },
                      ]
                    : layer === "ALL"
                      ? [
                          { label: "Good", color: BAND_COLORS[0] },
                          { label: "Moderate", color: BAND_COLORS[1] },
                          { label: "Elevated", color: BAND_COLORS[2] },
                          { label: "High", color: BAND_COLORS[3] },
                          { label: "Very High", color: BAND_COLORS[4] },
                        ]
                      : [
                          { label: "Good (0–40)", color: BAND_COLORS[0] },
                          { label: "Moderate (41–70)", color: BAND_COLORS[1] },
                          { label: "High (71–100)", color: BAND_COLORS[2] },
                          { label: "Very High (101–150)", color: BAND_COLORS[3] },
                          { label: "Hazardous (150+)", color: BAND_COLORS[4] },
                        ])
                ]
              ).map((item) => (
                <div key={item.label} className="flex items-center gap-1">
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ background: item.color }}
                  />
                  <span className="text-[10px] text-muted-foreground">{item.label}</span>
                </div>
              ))
            )}
          </div>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            <span>Zoom 2–20</span>
            <span>·</span>
            <span>Scroll: zoom</span>
            <span>·</span>
            <span>Drag: pan</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function generatePollutantGeoJSON(stations: LiveStation[]) {
  const features: Array<{
    type: "Feature";
    geometry: { type: "Point"; coordinates: [number, number] };
    properties: { no2: number; o3: number; hcho: number; severity: number; name: string };
  }> = [];

  stations.forEach((s) => {
    features.push({
      type: "Feature",
      geometry: { type: "Point", coordinates: s.coords },
      // `severity` is the composite band (0-4) the ALL layer colours by; the
      // raw pollutant properties stay for the single-pollutant views.
      properties: {
        no2: s.no2,
        o3: s.o3,
        hcho: s.hcho ?? 0,
        severity: compositeBand(s),
        name: s.name,
      },
    });
  });

  return { type: "FeatureCollection", features };
}
