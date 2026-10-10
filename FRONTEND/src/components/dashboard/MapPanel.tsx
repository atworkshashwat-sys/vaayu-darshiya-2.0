import { useEffect, useRef, useState, useCallback } from "react";
import maplibregl from "maplibre-gl";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { formatTime, hchoBand } from "./data";
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

const BAND_COLORS = ["#00E400", "#FFD300", "#FF9800", "#FF0000", "#9C27B0"];

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
    <div style="padding:14px 16px;font-family:Inter,system-ui,sans-serif;min-width:230px;background:#0c1529;border-radius:12px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
        <span style="width:10px;height:10px;border-radius:50%;background:${color};flex-shrink:0;box-shadow:0 0 8px ${color}90;"></span>
        <div>
          <div style="font-weight:700;font-size:14px;line-height:1.2;color:#ffffff;">${s.name}</div>
          <div style="font-size:10px;color:#ffffff;margin-top:1px;opacity:0.7;">Delhi · CPCB Monitor · <span style="color:#4ade80;">● Live</span></div>
        </div>
      </div>

      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:12px;padding:8px 10px;background:${color}18;border-radius:8px;border:1px solid ${color}30;">
        <span style="font-weight:800;font-size:26px;color:${color};line-height:1;text-shadow:0 0 12px${color}40;">${isAll ? label : value}</span>
        <span style="font-size:12px;color:#ffffff;font-weight:600;">${isAll ? "Composite · worst of NO₂ · O₃ · HCHO" : `${unit} ${label}`}</span>
      </div>

      <div style="font-size:10px;letter-spacing:0.06em;text-transform:uppercase;color:#ffffff;opacity:0.45;margin-bottom:4px;">Model forecast</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;font-size:11px;">
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">${no2Dot}NO₂</span><span style="color:#ffffff;font-weight:600;">${s.no2} <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">ppb</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">${o3Dot}O₃</span><span style="color:#ffffff;font-weight:600;">${s.o3} <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">ppb</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">${hchoDot}HCHO</span><span style="color:#ffffff;font-weight:600;">${s.hcho != null ? s.hcho : "—"} <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">µmol/m²</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">Temp</span><span style="color:#ffffff;font-weight:600;">${s.temp}°C</span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">Humidity</span><span style="color:#ffffff;font-weight:600;">${s.humidity}%</span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">Wind</span><span style="color:#ffffff;font-weight:600;">${s.wind} <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">km/h</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">Vis.</span><span style="color:#ffffff;font-weight:600;">${s.visibility} <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">km</span></span></div>
      </div>

      <div style="font-size:10px;letter-spacing:0.06em;text-transform:uppercase;color:#ffffff;opacity:0.45;margin:10px 0 4px;">Database observation</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;font-size:11px;">
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">NO₂</span><span style="color:#ffffff;font-weight:600;">${fmt(obsNo2)} <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">ppb</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">O₃</span><span style="color:#ffffff;font-weight:600;">${fmt(obsO3)} <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">ppb</span></span></div>
        <div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">HCHO</span><span style="color:#ffffff;font-weight:600;">${fmt(obsHcho)} <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">µmol/m²</span></span></div>
        ${
          s.pm25 != null
            ? `<div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">PM2.5</span><span style="color:#ffffff;font-weight:600;">${s.pm25} <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">µg/m³</span></span></div>`
            : ""
        }
        ${
          s.aqi != null
            ? `<div style="display:flex;justify-content:space-between;"><span style="color:#ffffff;">AQI</span><span style="color:#ffffff;font-weight:600;">${Math.round(s.aqi)}${s.aqiCategory ? ` <span style="color:#ffffff;opacity:0.6;font-weight:400;font-size:10px;">${s.aqiCategory.replace("_", " ")}</span>` : ""}</span></div>`
            : ""
        }
      </div>
      ${
        s.district
          ? `<div style="margin-top:8px;font-size:10px;color:#ffffff;opacity:0.55;">${s.district}</div>`
          : ""
      }

      <div style="margin-top:10px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.08);font-size:10px;color:#ffffff;text-align:right;opacity:0.5;">
        ${observedAt ? `Observed ${observedAt}` : ""}
        ${observedAt && modelRun ? " · " : ""}
        ${modelRun ? `Model run ${modelRun}` : ""}
      </div>
    </div>
  `;
}

function buildLoadingPopup(name: string): string {
  return `
    <div style="padding:14px 16px;font-family:Inter,system-ui,sans-serif;min-width:200px;background:#0c1529;border-radius:12px;text-align:center;">
      <div style="font-weight:700;font-size:14px;color:#ffffff;margin-bottom:6px;">${name}</div>
      <div style="font-size:12px;color:#94a3b8;">⏳ Loading model data…</div>
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

  const buildAndAddMarkers = useCallback(
    (map: maplibregl.Map, currentLayer: LayerKey, stationsData: typeof apiStations) => {
      clearMarkers();
      const liveStations = buildLiveStations(STATIONS, stationsData);

      liveStations.forEach((station) => {
        const wrapper = document.createElement("div");
        wrapper.style.cssText = "display:flex;flex-direction:column;align-items:center;cursor:pointer;z-index:10;";

        const dot = document.createElement("div");
        const label = document.createElement("div");

        if (!stationsData || stationsData.length === 0) {
          dot.style.cssText = "width:22px;height:22px;border-radius:50%;background:#64748b;border:3px solid rgba(255,255,255,0.95);box-shadow:0 0 0 3px #64748b40,0 2px 10px rgba(0,0,0,0.3);position:relative;z-index:2;";
          label.style.cssText = "margin-top:4px;padding:2px 6px;background:rgba(255,255,255,0.92);border:1px solid rgba(0,0,0,0.1);border-radius:4px;font-family:Inter,system-ui,sans-serif;font-size:10px;font-weight:600;color:#64748b;white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,0.12);line-height:1.3;text-align:center;pointer-events:none;";
          label.textContent = station.name + " · Loading…";
        } else {
          const { value, color } = stationStyle(station, currentLayer);

          dot.style.cssText = `width:22px;height:22px;border-radius:50%;background:${color};border:3px solid rgba(255,255,255,0.95);box-shadow:0 0 0 3px ${color}40,0 2px 10px rgba(0,0,0,0.3);transition:transform 0.2s ease,box-shadow 0.2s ease;position:relative;z-index:2;`;

          const pulse = document.createElement("div");
          pulse.style.cssText = `position:absolute;width:22px;height:22px;border-radius:50%;border:2px solid ${color};top:0;left:0;animation:station-pulse-ring 2s ease-out infinite;pointer-events:none;`;
          dot.appendChild(pulse);

          label.style.cssText = "margin-top:4px;padding:2px 6px;background:rgba(255,255,255,0.92);border:1px solid rgba(0,0,0,0.1);border-radius:4px;font-family:Inter,system-ui,sans-serif;font-size:10px;font-weight:600;color:#111;white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,0.12);line-height:1.3;text-align:center;pointer-events:none;";
          label.textContent = station.name;

          const popup = new maplibregl.Popup({
            offset: 24, closeButton: false, closeOnClick: false,
            className: "station-popup", maxWidth: "300px",
          }).setHTML(buildStationPopup(station, currentLayer));
          popupsRef.current.push(popup);

          wrapper.addEventListener("mouseenter", () => {
            dot.style.transform = "scale(1.4)";
            dot.style.boxShadow = `0 0 0 5px ${color}50,0 4px 24px ${color}70`;
            label.style.background = color;
            label.style.color = "#fff";
            label.style.borderColor = color;
            popup.setLngLat(station.coords).addTo(map);
          });
          wrapper.addEventListener("mouseleave", () => {
            dot.style.transform = "scale(1)";
            dot.style.boxShadow = `0 0 0 3px ${color}40,0 2px 10px rgba(0,0,0,0.3)`;
            label.style.background = "rgba(255,255,255,0.92)";
            label.style.color = "#111";
            label.style.borderColor = "rgba(0,0,0,0.1)";
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
            0, "rgba(0,228,0,0)", 0.2, "rgba(0,228,0,0.15)", 0.4, "rgba(255,211,0,0.2)",
            0.6, "rgba(255,126,0,0.2)", 0.8, "rgba(255,0,0,0.18)", 1, "rgba(126,0,35,0.15)",
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
            0, "#00E400", 30, "#00E400", 53, "#FFD300", 100, "#FF9800", 200, "#FF0000", 300, "#9C27B0",
          ],
          "circle-stroke-color": "rgba(255,255,255,0.6)",
          "circle-stroke-width": 1,
          "circle-opacity": 0.55,
          "circle-blur": 0.4,
        },
      });
    }

    updatePaintProperties(map, currentLayer);
  }, []);

  const updatePaintProperties = useCallback((map: maplibregl.Map, currentLayer: LayerKey) => {
    if (!map.getLayer("pollutant-heatmap") || !map.getLayer("pollutant-circles")) return;

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

    const prop = currentLayer === "NO2" ? "no2" : currentLayer === "O3" ? "o3" : "hcho";
    const thresholds = currentLayer === "NO2"
      ? { w: [0, 0, 30, 0.2, 53, 0.5, 100, 0.8, 200, 1], c: [0, "#00E400", 30, "#00E400", 53, "#FFD300", 100, "#FF9800", 200, "#FF0000", 300, "#9C27B0"] }
      : currentLayer === "HCHO"
        ? { w: [0, 0, 15, 0.2, 30, 0.5, 50, 0.8, 70, 1], c: [0, "#00E400", 15, "#00E400", 30, "#FFD300", 50, "#FF9800", 70, "#FF0000", 100, "#9C27B0"] }
        : { w: [0, 0, 40, 0.2, 70, 0.5, 100, 0.8, 150, 1], c: [0, "#00E400", 40, "#00E400", 70, "#FFD300", 100, "#FF9800", 150, "#FF0000", 200, "#9C27B0"] };

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
    });

    mapRef.current = map;

    return () => {
      clearMarkers();
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
    };
    map.on("idle", onIdle);
  }, [mapStyle, addPollutantLayers, buildAndAddMarkers, updatePaintProperties]);

  // Update GeoJSON data + paint properties when apiStations or layer changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoadedRef.current) return;
    const source = map.getSource("pollutant-points") as maplibregl.GeoJSONSource | undefined;
    if (!source) return;

    const liveStations = buildLiveStations(STATIONS, apiStations);
    const newGeoJSON = generatePollutantGeoJSON(liveStations);
    source.setData(newGeoJSON as any);
    updatePaintProperties(map, layer);
  }, [apiStations, layer, updatePaintProperties]);

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
        <div
          ref={mapContainer}
          className="w-full h-[420px] md:h-[560px] transition-all duration-300"
        />

        {/* Bottom bar with legend + stats */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t px-4 py-2.5">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-[10px] font-medium text-muted-foreground">{legendTitle}:</span>
            {(layer === "NO2"
              ? [
                  { label: "Good (0–30)", color: "#00E400" },
                  { label: "Moderate (31–53)", color: "#FFD300" },
                  { label: "Elevated (54–100)", color: "#FF9800" },
                  { label: "High (101–200)", color: "#FF0000" },
                  { label: "Very High (200+)", color: "#9C27B0" },
                ]
              : layer === "HCHO"
                ? [
                    { label: "Good (≤15)", color: "#00E400" },
                    { label: "Moderate (16–30)", color: "#FFD300" },
                    { label: "Elevated (31–50)", color: "#FF9800" },
                    { label: "High (51–70)", color: "#FF0000" },
                    { label: "Very High (70+)", color: "#9C27B0" },
                  ]
                : layer === "ALL"
                  ? [
                      { label: "Good", color: "#00E400" },
                      { label: "Moderate", color: "#FFD300" },
                      { label: "Elevated", color: "#FF9800" },
                      { label: "High", color: "#FF0000" },
                      { label: "Very High", color: "#9C27B0" },
                    ]
                  : [
                      { label: "Good (0–40)", color: "#00E400" },
                      { label: "Moderate (41–70)", color: "#FFD300" },
                      { label: "High (71–100)", color: "#FF9800" },
                      { label: "Very High (101–150)", color: "#FF0000" },
                      { label: "Hazardous (150+)", color: "#9C27B0" },
                    ]
            ).map((item) => (
              <div key={item.label} className="flex items-center gap-1">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: item.color }}
                />
                <span className="text-[10px] text-muted-foreground">{item.label}</span>
              </div>
            ))}
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
