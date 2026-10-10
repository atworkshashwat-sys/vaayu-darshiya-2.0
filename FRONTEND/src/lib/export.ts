import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import { StationData, WeeklyTrend, CityForecast, TIMES } from "@/components/dashboard/data";

const API_BASE = "";

function escapeCSV(value: string | number): string {
  const str = String(value);
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function triggerDownload(content: string, filename: string, mimeType: string) {
  const bom = "\uFEFF";
  const blob = new Blob([bom + content], { type: `${mimeType};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function siteSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/_+$/, "");
}

function pdfDoc(title: string, subtitle: string): jsPDF {
  const doc = new jsPDF({ orientation: "landscape", unit: "mm", format: "a4" });
  doc.setFontSize(16);
  doc.text(title, 14, 15);
  doc.setFontSize(10);
  doc.setTextColor(100);
  doc.text(subtitle, 14, 22);
  doc.setTextColor(0);
  return doc;
}

function savePdf(doc: jsPDF, filename: string) {
  doc.save(filename);
}

const SITE_MAP: Record<string, number> = {
  "Ashok Vihar": 1, Dwarka: 2, "Jawaharlal Nehru": 3,
  "Northwest Delhi": 4, "East Delhi": 5, "Northwest Delhi 2": 6,
  "Northeast Delhi": 7, Delhi: 1,
};

const ALL_STATIONS = [
  { id: 1, name: "Ashok Vihar" },
  { id: 2, name: "Dwarka" },
  { id: 3, name: "Jawaharlal Nehru" },
  { id: 4, name: "Northwest Delhi" },
  { id: 5, name: "East Delhi" },
  { id: 6, name: "Northwest Delhi 2" },
  { id: 7, name: "Northeast Delhi" },
];

export async function downloadStationData(stations: StationData[], timeIdx: number, format: "csv" | "json" | "pdf") {
  const timeLabel = TIMES[timeIdx];

  if (format === "pdf") {
    const doc = pdfDoc("VAAYU DARSHIYA — Site-Wise Data", `Time: ${timeLabel}  |  Stations: ${stations.length}`);
    autoTable(doc, {
      startY: 28,
      head: [["Station", "Lat", "Lon", "NO₂ (ppb)", "O₃ (ppb)", "HCHO (µmol/m²)", "PM2.5 (µg/m³)", "AQI", "Temp (°C)", "Humidity (%)", "Wind (km/h)", "Vis (km)"]],
      body: stations.map((s) => [
        s.name, s.lat, s.lon, s.no2, s.o3,
        s.hcho ?? "—",
        s.pm25 ?? "—", s.aqi != null ? Math.round(s.aqi) : "—",
        s.temp, s.humidity, s.wind, s.visibility,
      ]),
      styles: { fontSize: 8 },
      headStyles: { fillColor: [34, 197, 94] },
    });
    savePdf(doc, `vaayu_site_data_${siteSlug(timeLabel)}.pdf`);
  } else if (format === "json") {
    const payload = {
      title: "VAAYU DARSHIYA — Site-Wise Data",
      time: timeLabel,
      site_count: stations.length,
      sites: stations.map((s) => ({
        name: s.name,
        coordinates: { lat: s.lat, lon: s.lon },
        pollutants: {
          no2_ppb: s.no2,
          o3_ppb: s.o3,
          hcho_umol_m2: s.hcho ?? null,
          pm25_ugm3: s.pm25 ?? null,
          aqi: s.aqi ?? null,
          aqi_category: s.aqiCategory ?? null,
        },
        weather: {
          temp_c: s.temp,
          humidity_pct: s.humidity,
          wind_kmh: s.wind,
          wind_dir_deg: s.windDir ?? null,
          visibility_km: s.visibility,
        },
        district: s.district ?? null,
        observed_at: s.observedAt ?? null,
      })),
    };
    triggerDownload(JSON.stringify(payload, null, 2), `vaayu_site_data_${siteSlug(timeLabel)}.json`, "application/json");
  } else {
    const headers = [
      "Station", "Latitude", "Longitude",
      "NO2 (ppb)", "O3 (ppb)", "HCHO (umol/m2)", "PM2.5 (ug/m3)", "AQI", "AQI Category",
      "Temp (C)", "Humidity (%)", "Wind (km/h)", "Wind Dir (deg)", "Visibility (km)",
      "District", "Observed At",
    ];
    const rows = stations.map((s) =>
      [
        s.name, s.lat, s.lon,
        s.no2, s.o3, s.hcho ?? "",
        s.pm25 ?? "", s.aqi != null ? Math.round(s.aqi) : "", s.aqiCategory ?? "",
        s.temp, s.humidity, s.wind, s.windDir ?? "", s.visibility,
        s.district ?? "", s.observedAt ?? "",
      ].map(escapeCSV).join(",")
    );
    const csv = [headers.join(","), ...rows].join("\n");
    triggerDownload(csv, `vaayu_site_data_${siteSlug(timeLabel)}.csv`, "text/csv");
  }
}

async function fetchWeeklyTrendForSite(siteId: number): Promise<WeeklyTrend> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 45000);
  try {
    const res = await fetch(`${API_BASE}/api/weekly-trend?site=${siteId}`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    clearTimeout(timeout);
    console.error(`Failed to fetch weekly trend for site ${siteId}:`, err);
    throw err;
  }
}

async function fetchForecastForSite(siteId: number): Promise<CityForecast> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 45000);
  try {
    const res = await fetch(`${API_BASE}/api/forecast?site=${siteId}`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return {
      no2: data.no2,
      o3: data.o3,
      hcho: data.hcho,
      aqi: Array(24).fill(0),
      label: data.label,
      healthAdvisory: "",
      bestHours: [],
    };
  } catch (err) {
    clearTimeout(timeout);
    console.error(`Failed to fetch forecast for site ${siteId}:`, err);
    throw err;
  }
}

export async function downloadWeeklyTrend(_trend: WeeklyTrend, siteName: string, format: "csv" | "json" | "pdf") {
  // Fetch data for all 7 stations
  const allData = await Promise.all(
    ALL_STATIONS.map(async (s) => ({
      ...s,
      trend: await fetchWeeklyTrendForSite(s.id),
    }))
  );

  if (format === "pdf") {
    const doc = pdfDoc("VAAYU DARSHIYA — 14-Day Trend", "Daily average NO₂, O₃ & HCHO across all monitoring stations");
    const headers = ["Date", ...allData.flatMap((s) => [`${s.name} NO₂`, `${s.name} O₃`, `${s.name} HCHO`])];
    const maxDays = Math.max(...allData.map((s) => s.trend.days.length));
    const rows: (string | number)[][] = [];
    for (let i = 0; i < maxDays; i++) {
      const firstTrend = allData.find((s) => s.trend.days.length > i);
      const date = firstTrend ? firstTrend.trend.days[i] : "";
      const vals = allData.flatMap((s) => [
        s.trend.no2[i] ?? "",
        s.trend.o3[i] ?? "",
        s.trend.hcho?.[i] ?? "",
      ]);
      rows.push([date, ...vals]);
    }
    autoTable(doc, { startY: 28, head: [headers], body: rows, styles: { fontSize: 7 }, headStyles: { fillColor: [34, 197, 94] } });
    savePdf(doc, "vaayu_14day_trend_all_stations.pdf");
  } else if (format === "json") {
    const payload = {
      title: "VAAYU DARSHIYA — 14-Day Trend (All Stations)",
      stations: allData.map((s) => ({
        name: s.name,
        site_id: s.id,
        data: s.trend.days.map((day, i) => ({
          date: day,
          no2_avg_ppb: s.trend.no2[i],
          o3_avg_ppb: s.trend.o3[i],
          hcho_avg_umol_m2: s.trend.hcho?.[i] ?? null,
        })),
      })),
    };
    triggerDownload(JSON.stringify(payload, null, 2), `vaayu_14day_trend_all_stations.json`, "application/json");
  } else {
    const headers = ["Date", ...allData.map((s) => `${s.name} NO2 (ppb)`), ...allData.map((s) => `${s.name} O3 (ppb)`), ...allData.map((s) => `${s.name} HCHO (umol/m2)`)];
    // Use the longest days array
    const maxDays = Math.max(...allData.map((s) => s.trend.days.length));
    const rows: string[] = [];
    for (let i = 0; i < maxDays; i++) {
      const firstTrend = allData.find((s) => s.trend.days.length > i);
      const date = firstTrend ? firstTrend.trend.days[i] : "";
      const no2Vals = allData.map((s) => (s.trend.no2[i] !== undefined ? String(s.trend.no2[i]) : ""));
      const o3Vals = allData.map((s) => (s.trend.o3[i] !== undefined ? String(s.trend.o3[i]) : ""));
      const hchoVals = allData.map((s) => (s.trend.hcho?.[i] != null ? String(s.trend.hcho[i]) : ""));
      rows.push([date, ...no2Vals, ...o3Vals, ...hchoVals].map(escapeCSV).join(","));
    }
    const csv = [headers.join(","), ...rows].join("\n");
    triggerDownload(csv, `vaayu_14day_trend_all_stations.csv`, "text/csv");
  }
}

export async function download24HourForecast(_forecast: CityForecast, siteName: string, timeIdx: number, format: "csv" | "json" | "pdf") {
  // Fetch data for all 7 stations
  const allData = await Promise.all(
    ALL_STATIONS.map(async (s) => ({
      ...s,
      forecast: await fetchForecastForSite(s.id),
    }))
  );

  if (format === "pdf") {
    const doc = pdfDoc("VAAYU DARSHIYA — 24-Hour Forecast", `Current time: ${TIMES[timeIdx]}`);
    const headers = ["Time", ...allData.flatMap((s) => [`${s.name} NO₂`, `${s.name} O₃`, `${s.name} HCHO`])];
    const rows = TIMES.map((t, i) => {
      const vals = allData.flatMap((s) => [
        s.forecast.no2[i],
        s.forecast.o3[i],
        s.forecast.hcho?.[i] ?? "—",
      ]);
      return [t, ...vals];
    });
    autoTable(doc, { startY: 28, head: [headers], body: rows, styles: { fontSize: 7 }, headStyles: { fillColor: [34, 197, 94] } });
    savePdf(doc, "vaayu_24hour_forecast_all_stations.pdf");
  } else if (format === "json") {
    const payload = {
      title: "VAAYU DARSHIYA — 24-Hour Forecast (All Stations)",
      current_time: TIMES[timeIdx],
      stations: allData.map((s) => ({
        name: s.name,
        site_id: s.id,
        data: TIMES.map((t, i) => ({
          time: t,
          no2_ppb: s.forecast.no2[i],
          o3_ppb: s.forecast.o3[i],
          hcho_umol_m2: s.forecast.hcho?.[i] ?? null,
          is_current: i === timeIdx,
        })),
      })),
    };
    triggerDownload(JSON.stringify(payload, null, 2), `vaayu_24hour_forecast_all_stations.json`, "application/json");
  } else {
    const headers = ["Time", "Is Current", ...allData.map((s) => `${s.name} NO2 (ppb)`), ...allData.map((s) => `${s.name} O3 (ppb)`), ...allData.map((s) => `${s.name} HCHO (umol/m2)`)];
    const rows = TIMES.map((t, i) => {
      const isCurrent = i === timeIdx ? "TRUE" : "";
      const no2Vals = allData.map((s) => String(s.forecast.no2[i]));
      const o3Vals = allData.map((s) => String(s.forecast.o3[i]));
      const hchoVals = allData.map((s) => (s.forecast.hcho?.[i] != null ? String(s.forecast.hcho[i]) : ""));
      return [t, isCurrent, ...no2Vals, ...o3Vals, ...hchoVals].map(escapeCSV).join(",");
    });
    const csv = [headers.join(","), ...rows].join("\n");
    triggerDownload(csv, `vaayu_24hour_forecast_all_stations.csv`, "text/csv");
  }
}

const HCHO_PDF_COLOR: [number, number, number] = [139, 92, 246]; // matches HCHO_COLOR #8B5CF6

/** 24-hour HCHO-only export across every station, mirroring download24HourForecast. */
export async function downloadHchoForecast(timeIdx: number, format: "csv" | "json" | "pdf") {
  const allData = await Promise.all(
    ALL_STATIONS.map(async (s) => ({
      ...s,
      forecast: await fetchForecastForSite(s.id),
    })),
  );

  if (format === "pdf") {
    const doc = pdfDoc(
      "VAAYU DARSHIYA — 24-Hour HCHO Forecast",
      `Current time: ${TIMES[timeIdx]}  |  HCHO (formaldehyde) in µmol/m²  |  Stations: ${allData.length}`,
    );
    autoTable(doc, {
      startY: 28,
      head: [["Time", ...allData.map((s) => s.name)]],
      body: TIMES.map((t, i) => [t, ...allData.map((s) => s.forecast.hcho?.[i] ?? "—")]),
      styles: { fontSize: 7 },
      headStyles: { fillColor: HCHO_PDF_COLOR },
    });
    savePdf(doc, "vaayu_24hour_hcho_all_stations.pdf");
  } else if (format === "json") {
    const payload = {
      title: "VAAYU DARSHIYA — 24-Hour HCHO Forecast (All Stations)",
      pollutant: "HCHO (formaldehyde)",
      unit: "µmol/m²",
      current_time: TIMES[timeIdx],
      stations: allData.map((s) => ({
        name: s.name,
        site_id: s.id,
        data: TIMES.map((t, i) => ({
          time: t,
          hcho_umol_m2: s.forecast.hcho?.[i] ?? null,
          is_current: i === timeIdx,
        })),
      })),
    };
    triggerDownload(
      JSON.stringify(payload, null, 2),
      "vaayu_24hour_hcho_all_stations.json",
      "application/json",
    );
  } else {
    const headers = ["Time", "Is Current", ...allData.map((s) => `${s.name} HCHO (umol/m2)`)];
    const rows = TIMES.map((t, i) =>
      [
        t,
        i === timeIdx ? "TRUE" : "",
        ...allData.map((s) => (s.forecast.hcho?.[i] != null ? String(s.forecast.hcho[i]) : "")),
      ]
        .map(escapeCSV)
        .join(","),
    );
    const csv = [headers.join(","), ...rows].join("\n");
    triggerDownload(csv, "vaayu_24hour_hcho_all_stations.csv", "text/csv");
  }
}

/** Single-site, single-pollutant 24-hour export (the 24-hour card's download button). */
export async function downloadPollutantForecast(
  forecast: CityForecast,
  siteName: string,
  timeIdx: number,
  pollutant: "NO2" | "O3" | "HCHO",
  format: "csv" | "json" | "pdf",
) {
  const meta = {
    NO2: { name: "NO₂ (Nitrogen Dioxide)", unit: "ppb", values: forecast.no2 },
    O3: { name: "O₃ (Ozone)", unit: "ppb", values: forecast.o3 },
    HCHO: { name: "HCHO (Formaldehyde)", unit: "µmol/m²", values: forecast.hcho ?? [] },
  }[pollutant];
  const file = `vaayu_24hour_${pollutant.toLowerCase()}_${siteSlug(siteName)}`;

  if (format === "pdf") {
    const doc = pdfDoc(
      `VAAYU DARSHIYA — 24-Hour ${meta.name} Forecast`,
      `Site: ${siteName}  |  Unit: ${meta.unit}  |  Current time: ${TIMES[timeIdx]}`,
    );
    autoTable(doc, {
      startY: 28,
      head: [["Time", `${pollutant} (${meta.unit})`]],
      body: TIMES.map((t, i) => [t, meta.values[i] ?? "—"]),
      styles: { fontSize: 8 },
      headStyles: { fillColor: HCHO_PDF_COLOR },
    });
    savePdf(doc, `${file}.pdf`);
  } else if (format === "json") {
    const payload = {
      title: `VAAYU DARSHIYA — 24-Hour ${meta.name} Forecast`,
      site: siteName,
      pollutant,
      unit: meta.unit,
      current_time: TIMES[timeIdx],
      data: TIMES.map((t, i) => ({
        time: t,
        value: meta.values[i] ?? null,
        is_current: i === timeIdx,
      })),
    };
    triggerDownload(JSON.stringify(payload, null, 2), `${file}.json`, "application/json");
  } else {
    const headers = ["Time", "Is Current", `${pollutant} (${meta.unit})`];
    const rows = TIMES.map((t, i) =>
      [t, i === timeIdx ? "TRUE" : "", meta.values[i] ?? ""].map(escapeCSV).join(",")
    );
    const csv = [headers.join(","), ...rows].join("\n");
    triggerDownload(csv, `${file}.csv`, "text/csv");
  }
}
