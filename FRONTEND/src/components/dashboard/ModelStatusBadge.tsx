import { useEffect, useState } from "react";
import { Cpu, Database, Wifi, WifiOff } from "lucide-react";
import { fetchModels, ModelsData, ModelName, DataSourceLabel } from "./data";

/**
 * Status pill: "Model: XGBoost | Data: Live", colour-coded.
 *   Green  = live data + best (auto-selected) model
 *   Yellow = live data + fallback / override model
 *   Red    = static data fallback
 * Reads /api/models and refreshes on the same cadence as the dashboard.
 */
export function ModelStatusBadge() {
  const [data, setData] = useState<ModelsData | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => fetchModels().then((d) => alive && setData(d));
    load();
    const id = setInterval(load, 60000); // refresh each minute
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  if (!data || data.models.length === 0) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/50 px-2.5 py-0.5 text-[10px] font-medium text-muted-foreground">
        <Cpu className="h-3 w-3" /> Model: — | Data: —
      </span>
    );
  }

  const isFallbackModel =
    data.active_reason.includes("missing") ||
    data.active_reason.includes("features") ||
    data.override != null;

  const tone: Record<DataSourceLabel, string> = {
    live: isFallbackModel
      ? "border-sev-moderate/30 bg-sev-moderate/10 text-ink"
      : "border-sev-good/30 bg-sev-good/10 text-ink",
    static: "border-border/60 bg-muted/50 text-muted-foreground",
    "static (fallback)": "border-sev-poor/30 bg-sev-poor/10 text-accent",
  };

  const dataLabel: Record<DataSourceLabel, string> = {
    live: "Live",
    static: "Static",
    "static (fallback)": "Static (fallback)",
  };

  const DataIcon = data.data_source === "live" ? Wifi : WifiOff;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[10px] font-medium ${tone[data.data_source]}`}
      title={`${data.active_reason} · ${data.data_source_detail}`}
    >
      <Cpu className="h-3 w-3" />
      Model: {data.active_model}
      <span className="opacity-40">|</span>
      <DataIcon className="h-3 w-3" />
      Data: {dataLabel[data.data_source]}
    </span>
  );
}
