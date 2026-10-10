import { screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BestTime } from "./BestTime";
import { renderWithProvider } from "./renderWithProvider";
import type { WeatherSlot } from "./contextApi";
import type { CityForecast } from "./data";

const forecast: CityForecast = {
  aqi: new Array(24).fill(90),
  no2: new Array(24).fill(45),
  o3: new Array(24).fill(30),
  hcho: new Array(24).fill(18),
  label: "Delhi NCR",
  healthAdvisory: "Moderate pollution expected.",
  bestHours: [5, 6, 7],
};

const weather = {
  source: "IMD",
  url: "https://mausam.imd.gov.in/delhi/city_weather.php",
  fetched_at: "2026-10-10T13:40:00",
  text: "Mainly clear sky. Wind speed 8-12 kmph. Humidity 68%. Shallow fog in the morning.",
  wind_min_kmph: 8,
  wind_max_kmph: 12,
  humidity_pct: 68,
  temp_c: 34,
  fog: true,
  rain: false,
};

const notice = {
  id: "caqm-truck",
  kind: "traffic" as const,
  title: "Truck entry restricted in Delhi under GRAP",
  summary: "Heavy goods vehicles are barred from entering the city limits.",
  source: "CAQM",
  url: "https://caqm.nic.in/notices/truck-restriction",
  published_at: "2026-10-08",
  fetched_at: "2026-10-10T13:40:00",
};

const periods = ["Early Morning", "Morning Rush", "Afternoon Peak", "Evening", "Late Night"];

/** All five windows are Safe per the ML model; the backend downgrades only
 *  Early Morning (fog in the official forecast), with a visible reason. */
function makeSlots(): WeatherSlot[] {
  return periods.map((period) => ({
    period,
    hours: "",
    ml_status: "Safe",
    status: period === "Early Morning" ? "Moderate" : "Safe",
    adjusted: period === "Early Morning",
    reasons: period === "Early Morning" ? ["Fog forecast: pollutants linger in still air"] : [],
  }));
}

const okPayload = {
  status: "ok",
  stale: false,
  fetched_at: "2026-10-10T13:40:00",
  age_seconds: 30,
  ttl_seconds: 21600,
  weather,
  notices: [notice],
  slots: makeSlots(),
  sources: [{ name: "IMD", url: weather.url, fetched_at: weather.fetched_at }],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("BestTime", () => {
  it("adds the live forecast beside the ML windows and explains the downgrade", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => okPayload }));

    renderWithProvider(<BestTime forecast={forecast} />);

    // ML numbers untouched
    expect(screen.getAllByText("NO₂: 45").length).toBe(5);
    expect(screen.getAllByText("ML forecast").length).toBe(5);

    // live advisory values on every window + the strip with its source label
    expect((await screen.findAllByText("Live advisory · IMD")).length).toBe(5);
    expect(screen.getAllByText("Wind: 8–12 km/h").length).toBe(5);
    expect(screen.getAllByText("Humidity: 68%").length).toBe(5);
    expect(screen.getAllByText("Temp: 34°C").length).toBe(5);
    expect(screen.getByText(/Today's official forecast:/)).toBeInTheDocument();
    const stripLink = screen.getByRole("link", { name: /Source: IMD/ });
    expect(stripLink).toHaveAttribute("href", weather.url);

    // the downgrade is explicit: original verdict + reason, never silent
    const reason = screen.getByTestId("reason-Early Morning");
    expect(reason).toHaveTextContent("ML forecast: Safe → Moderate");
    expect(reason).toHaveTextContent("Fog forecast: pollutants linger in still air");
    expect(screen.queryByTestId("reason-Morning Rush")).not.toBeInTheDocument();
    expect(screen.getAllByText("Moderate")).toHaveLength(1);
    expect(screen.getAllByText("Safe")).toHaveLength(4);

    // notice with its own source + fetched-at label
    expect(screen.getByText("Truck entry restricted in Delhi under GRAP")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Source: CAQM/ })).toHaveAttribute("href", notice.url);

    expect(screen.getByTestId("weather-status")).toHaveTextContent(/retrieved/);
  });

  it("keeps the ML windows unchanged when the context fetch fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));

    renderWithProvider(<BestTime forecast={forecast} />);

    expect(
      await screen.findByText(/Live forecast unavailable — showing ML windows unchanged\./),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Safe")).toHaveLength(5);
    expect(screen.queryByText(/ML forecast:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Live advisory/)).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Source:/ })).not.toBeInTheDocument();
    // numbers still render — never a blank section
    expect(screen.getAllByText("NO₂: 45")).toHaveLength(5);
  });

  it("marks the last forecast as stale when the refresh fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          ...okPayload,
          status: "stale",
          stale: true,
          age_seconds: 1800,
        }),
      }),
    );

    renderWithProvider(<BestTime forecast={forecast} />);

    expect(await screen.findByText(/Forecast last updated 30m ago/)).toBeInTheDocument();
    // stale data is still shown, with its source
    expect(screen.getAllByText("Live advisory · IMD")).toHaveLength(5);
  });

  it("renders an unavailable payload with the ML verdicts and no live block", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          status: "unavailable",
          stale: false,
          fetched_at: null,
          age_seconds: null,
          ttl_seconds: 21600,
          weather: null,
          notices: [],
          slots: periods.map((period) => ({
            period,
            hours: "",
            ml_status: "Safe",
            status: "Safe",
            adjusted: false,
            reasons: [],
          })),
          sources: [],
          message: "Live context unavailable (timed out).",
        }),
      }),
    );

    renderWithProvider(<BestTime forecast={forecast} />);

    expect(
      await screen.findByText(/Live forecast unavailable — showing ML windows unchanged\./),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Safe")).toHaveLength(5);
    expect(screen.queryByText(/Live advisory/)).not.toBeInTheDocument();
    expect(screen.queryByTestId("weather-notices")).not.toBeInTheDocument();
    expect(screen.getAllByText("NO₂: 45")).toHaveLength(5);
  });
});
