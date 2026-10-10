import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { Download, BarChart3, ChevronDown, TrendingUp, Clock, FlaskConical, FileSpreadsheet, FileJson, FileText, Loader2 } from "lucide-react";
import { downloadStationData, downloadWeeklyTrend, download24HourForecast, downloadHchoForecast } from "@/lib/export";
import { StationData, WeeklyTrend, CityForecast } from "./data";

type DataType = "stations" | "trend" | "forecast" | "hcho";
type FileFormat = "csv" | "json" | "pdf";

const DATA_OPTIONS: {
  value: DataType;
  label: string;
  description: string;
  icon: typeof BarChart3;
}[] = [
  {
    value: "stations",
    label: "Site-Wise Data",
    description: "All 7 monitoring stations",
    icon: BarChart3,
  },
  {
    value: "trend",
    label: "14-Day Trend",
    description: "Daily NO₂, O₃ & HCHO averages",
    icon: TrendingUp,
  },
  {
    value: "forecast",
    label: "24-Hour Forecast",
    description: "Hourly NO₂, O₃ & HCHO",
    icon: Clock,
  },
  {
    value: "hcho",
    label: "24-Hour HCHO",
    description: "Hourly HCHO (µmol/m²), all sites",
    icon: FlaskConical,
  },
];

interface DataExportProps {
  stations: StationData[];
  weeklyTrend: WeeklyTrend;
  forecast: CityForecast;
  timeIdx: number;
  selectedSite: string;
}

export function DataExport({ stations, weeklyTrend, forecast, timeIdx, selectedSite }: DataExportProps) {
  const [dataType, setDataType] = useState<DataType>("stations");
  const [format, setFormat] = useState<FileFormat>("csv");
  const [downloading, setDownloading] = useState(false);

  async function handleDownload() {
    setDownloading(true);
    try {
      switch (dataType) {
        case "stations":
          await downloadStationData(stations, timeIdx, format);
          break;
        case "trend":
          await downloadWeeklyTrend(weeklyTrend, selectedSite, format);
          break;
        case "forecast":
          await download24HourForecast(forecast, selectedSite, timeIdx, format);
          break;
        case "hcho":
          await downloadHchoForecast(timeIdx, format);
          break;
      }
    } catch (err) {
      console.error("Download failed:", err);
      alert("Download failed. The server may be waking up — please try again in 30 seconds.");
    } finally {
      setDownloading(false);
    }
  }

  const currentOption = DATA_OPTIONS.find((o) => o.value === dataType)!;

  return (
    <div className="glass rounded-xl px-4 py-3">
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs font-medium text-muted-foreground">Export Data</span>
        <div className="h-4 w-px bg-border" />

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex items-center gap-1.5 rounded-lg bg-muted/50 px-2.5 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted">
              <currentOption.icon className="h-3.5 w-3.5 text-primary" />
              {currentOption.label}
              <ChevronDown className="h-3 w-3 text-muted-foreground" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-56">
            <DropdownMenuLabel>Data Type</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuRadioGroup
              value={dataType}
              onValueChange={(v) => setDataType(v as DataType)}
            >
              {DATA_OPTIONS.map((opt) => (
                <DropdownMenuRadioItem key={opt.value} value={opt.value} className="flex flex-col items-start gap-0.5 py-2">
                  <div className="flex items-center gap-2">
                    <opt.icon className="h-3.5 w-3.5 text-primary" />
                    <span className="font-medium">{opt.label}</span>
                  </div>
                  <span className="text-[11px] text-muted-foreground ml-5.5">{opt.description}</span>
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>

        <div className="h-4 w-px bg-border" />

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex items-center gap-1.5 rounded-lg bg-muted/50 px-2.5 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted">
              {format === "csv" ? <FileSpreadsheet className="h-3.5 w-3.5 text-green-500" /> : format === "json" ? <FileJson className="h-3.5 w-3.5 text-blue-500" /> : <FileText className="h-3.5 w-3.5 text-red-500" />}
              {format.toUpperCase()}
              <ChevronDown className="h-3 w-3 text-muted-foreground" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-28">
            <DropdownMenuLabel>Format</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuRadioGroup value={format} onValueChange={(v) => setFormat(v as FileFormat)}>
              <DropdownMenuRadioItem value="csv" className="flex items-center gap-2">
                <FileSpreadsheet className="h-3.5 w-3.5 text-green-500" />
                CSV
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="json" className="flex items-center gap-2">
                <FileJson className="h-3.5 w-3.5 text-blue-500" />
                JSON
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="pdf" className="flex items-center gap-2">
                <FileText className="h-3.5 w-3.5 text-red-500" />
                PDF
              </DropdownMenuRadioItem>
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>

        <div className="h-4 w-px bg-border" />

        <Button
          size="sm"
          className="h-8 gap-1.5 text-xs font-semibold bg-primary text-primary-foreground hover:bg-primary/90"
          onClick={handleDownload}
          disabled={downloading}
        >
          {downloading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
          {downloading ? "Exporting..." : "Download"}
        </Button>
      </div>
    </div>
  );
}
