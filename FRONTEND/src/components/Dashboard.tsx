import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Hero, MapPanel, Metrics, Charts, AIInsights, PollutantGuide,
  Alerts, ModelMetrics, BestTime, HchoPanel, Footer, DataExport, CaseStudy, TIMES, FORECAST, formatTime,
} from "./dashboard/index";
import { Navbar } from "./dashboard/Navbar";
import { fetchForecast, fetchStations, fetchTime, fetchWeeklyTrend, fetchLiveStatus, CityForecast, StationData, WeeklyTrend } from "./dashboard/data";
import { useLiveStream } from "./dashboard/useLiveStream";

export interface StationInfo {
  name: string;
  city: string;
  value: number;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr + "T00:00:00");
  return d.toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
}

function formatShortDate(dateStr: string): string {
  // Date-only string: pin to local midnight so the day cannot shift by one
  // when the visitor's timezone is behind UTC.
  const d = new Date(dateStr + "T00:00:00");
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

/** "19:24" — the backend stamps every reading in IST, so render in IST
 *  rather than the visitor's timezone or the time would disagree with
 *  the "observed at" text shown on the station cards. */
function formatIst(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Kolkata",
  });
}

export default function Dashboard() {
  const [layer, setLayer] = useState<"ALL" | "NO2" | "O3" | "HCHO">("ALL");
  const [timeIdx, setTimeIdx] = useState(() => new Date().getHours());
  const [selectedStation, setSelectedStation] = useState<StationInfo | null>(null);
  const [selectedSite, setSelectedSite] = useState("Delhi");
  const [currentMinute, setCurrentMinute] = useState(() => new Date().getMinutes());
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { data: timeData, error: timeError } = useQuery({
    queryKey: ["time"],
    queryFn: fetchTime,
    staleTime: 60_000,
  });

  const { data: forecast, error: forecastError } = useQuery({
    queryKey: ["forecast", selectedSite],
    queryFn: () => fetchForecast(selectedSite),
    staleTime: 60_000,
  });

  const { data: apiStations = [], error: stationsError } = useQuery({
    queryKey: ["stations"],
    queryFn: fetchStations,
    staleTime: 60_000,
  });

  const { data: weeklyTrend, error: weeklyError } = useQuery({
    queryKey: ["weeklyTrend", selectedSite],
    queryFn: () => fetchWeeklyTrend(selectedSite),
    staleTime: 60_000,
  });

  const { data: liveStatus } = useQuery({
    queryKey: ["liveStatus"],
    queryFn: fetchLiveStatus,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });

  // One SSE connection for the whole dashboard: each validated refresh is
  // written straight into the query cache, so cards move without a reload.
  const live = useLiveStream();

  useEffect(() => {
    if (timeData) {
      setTimeIdx(timeData.current_hour);
      setCurrentMinute(timeData.current_minute);
    }
  }, [timeData]);

  // forecast.currentHour is cached at server startup and never updates,
  // so we intentionally skip it — the setInterval + timeData keep timeIdx fresh.

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setTimeIdx(now.getHours());
      setCurrentMinute(now.getMinutes());
    };
    tick();
    timerRef.current = setInterval(tick, 1000);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const handleStationSelect = (station: StationInfo | null) => {
    setSelectedStation(station);
    setSelectedSite(station ? station.name : "Delhi");
  };

  const matchedDate = (() => {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  })();
  // Real issue date of the forecast being served (from /api/time). The model
  // is offline — it is not re-run every hour — so this is intentionally shown
  // separately from today's live clock instead of being passed off as "now".
  const modelRunDate = timeData?.matched_date;
  const resolvedForecast: CityForecast = forecast ?? {
    aqi: Array(24).fill(0),
    no2: Array(24).fill(0),
    o3: Array(24).fill(0),
    label: "Delhi",
    healthAdvisory: "Loading...",
    bestHours: [],
  };
  const resolvedWeekly: WeeklyTrend = weeklyTrend ?? { days: [], no2: [], o3: [] };

  const aqi = resolvedForecast.no2[timeIdx] !== 0
    ? { aqi: resolvedForecast.aqi[timeIdx], no2: resolvedForecast.no2[timeIdx], o3: resolvedForecast.o3[timeIdx] }
    : FORECAST[TIMES[timeIdx]];

  // Live-layer state (spec §6). The dot must reflect the real stream, and an
  // expired 24 h window says so explicitly — a green dot next to data that
  // stopped arriving is exactly what "never present a stale number as live"
  // is meant to prevent.
  //
  // Only a backend that actually answered with `expired: true` closes the
  // banner: an unreachable server is not the same as a finished window, and
  // claiming the latter would be a statement we cannot back.
  const windowClosed = Boolean(liveStatus?.expired);
  const streamLabel =
    live.status === "live"
      ? "Live"
      : live.status === "polling"
        ? "Polling"
        : live.status === "connecting"
          ? "Connecting…"
          : "Offline";
  const streamDot =
    live.status === "live"
      ? "bg-emerald-500 animate-pulse"
      : live.status === "polling"
        ? "bg-amber-500"
        : live.status === "connecting"
          ? "bg-slate-400 animate-pulse"
          : "bg-slate-500";
  const closedMessage = windowClosed ? liveStatus?.message ?? "Live context ended" : null;

  // Data freshness, derived from the readings actually received. Station 3
  // reports "offline" permanently (it stopped in 2023), so it is excluded from
  // the count — otherwise every refresh would look stale.
  const liveCount = apiStations.filter((s) => s.status === "live").length;
  const staleCount = apiStations.filter((s) => s.status === "stale").length;
  const newestObserved = apiStations
    .map((s) => s.observedAt)
    .filter(Boolean)
    .sort()
    .pop();
  const freshness: "live" | "stale" | "none" = windowClosed
    ? "none"
    : liveCount > 0
      ? "live"
      : staleCount > 0
        ? "stale"
        : "none";
  const freshnessLabel = windowClosed
    ? "Live context ended"
    : freshness === "live"
      ? `Live · updated ${newestObserved ? formatIst(newestObserved) : ""}`
      : freshness === "stale"
        ? "Stale · showing last reading"
        : "No live readings yet";

  return (
    <main className="min-h-screen">
      <div className="grid-bg" />
      <Navbar />
      <Hero />

      {(forecastError || stationsError) && (
        <div className="mx-4 mt-4 md:mx-8">
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-sm text-red-400">
            Unable to fetch data from the server. Showing fallback data.
          </div>
        </div>
      )}

      {/* Date Banner */}
      <div className="mx-4 mt-4 md:mx-8">
        <div className="glass flex items-center justify-between rounded-xl px-4 py-2.5">
          <div className="flex items-center gap-3">
            <div
              className={`h-2 w-2 rounded-full ${windowClosed ? "bg-slate-500" : streamDot}`}
              title={closedMessage ?? `SSE ${streamLabel}`}
            />
            <span className="text-sm font-medium">{formatDate(matchedDate)}</span>
            {/* Spec §8: the header reports the age of the DATA, not the state
                of the socket. A healthy connection carrying a stale reading
                must not be allowed to say "Live". */}
            {!windowClosed && (
              <span
                className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                  freshness === "live"
                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-500"
                    : freshness === "stale"
                      ? "border-amber-500/30 bg-amber-500/10 text-amber-500"
                      : "border-border/60 bg-muted/50 text-muted-foreground"
                }`}
                title={`SSE ${streamLabel}`}
              >
                {freshnessLabel}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            {closedMessage && (
              <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-500">
                {closedMessage}
              </span>
            )}
            <span className="text-xs text-muted-foreground">
              Showing: <span className="font-mono font-semibold text-foreground">{formatTime(timeIdx, currentMinute)}</span>
            </span>
          </div>
        </div>
      </div>

      <Tabs defaultValue="overview" className="mx-4 mt-4 md:mx-8">
        <TabsList className="glass">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="casestudy">DELHI'S DRASTIC CASE</TabsTrigger>
          <TabsTrigger value="model">Model Details</TabsTrigger>
        </TabsList>
        <TabsContent value="overview">
          {selectedStation && (
            <div className="mb-2 flex items-center gap-2">
              <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                Viewing: {selectedStation.name}, {selectedStation.city}
              </span>
              <button onClick={() => handleStationSelect(null)} className="text-xs text-muted-foreground hover:text-foreground">
                Clear
              </button>
            </div>
          )}
          <AnimatePresence mode="wait">
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
              <MapPanel layer={layer} setLayer={setLayer} timeIdx={timeIdx} setTimeIdx={setTimeIdx} onStationSelect={handleStationSelect} onRefresh={() => {}} stations={apiStations} />
              <Metrics aqi={aqi} city="Delhi" timeIdx={timeIdx} currentMinute={currentMinute} forecast={resolvedForecast} />
            </motion.div>
          </AnimatePresence>
          <div className="mt-8 flex justify-end"><DataExport stations={apiStations} weeklyTrend={resolvedWeekly} forecast={resolvedForecast} timeIdx={timeIdx} selectedSite={selectedSite} /></div>
          <Charts city="Delhi" timeIdx={timeIdx} forecast={resolvedForecast} apiStations={apiStations} weeklyTrend={resolvedWeekly} />
          <HchoPanel forecast={resolvedForecast} timeIdx={timeIdx} currentMinute={currentMinute} />
          <Alerts forecast={resolvedForecast} />
          <BestTime forecast={resolvedForecast} />
        </TabsContent>
        <TabsContent value="insights">
          <AIInsights city="Delhi" forecast={resolvedForecast} />
          <PollutantGuide />
        </TabsContent>
        <TabsContent value="casestudy">
          <CaseStudy stations={apiStations} />
        </TabsContent>
        <TabsContent value="model">
          <ModelMetrics />
        </TabsContent>
      </Tabs>
      <Footer />
    </main>
  );
}
