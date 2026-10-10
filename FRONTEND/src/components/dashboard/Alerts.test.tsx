import { screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AlertTriangle, Info, ShieldAlert } from "lucide-react";
import { Alerts } from "./Alerts";
import { renderWithProvider } from "./renderWithProvider";
import { mergeAlerts, titleOverlaps } from "./alertMerge";
import type { Alert } from "./alertMerge";
import type { ContextAlert } from "./contextApi";
import type { CityForecast } from "./data";

const forecast: CityForecast = {
  aqi: new Array(24).fill(140),
  no2: new Array(24).fill(45),
  o3: new Array(24).fill(30),
  hcho: new Array(24).fill(18),
  label: "Delhi NCR",
  healthAdvisory: "Moderate pollution expected.",
  bestHours: [5, 6, 7],
};

const officialItem: ContextAlert = {
  id: "caqm-grap-3",
  title: "CAQM imposes GRAP Stage 3 restrictions in Delhi NCR",
  severity: "medium",
  summary: "Construction activities banned across Delhi NCR until further notice.",
  source: "CAQM",
  url: "https://caqm.nic.in/grap-stage-3-delhi",
  official: true,
  published_at: "2026-10-08",
  fetched_at: "2026-10-10T12:30:00",
};

const pressItem: ContextAlert = {
  ...officialItem,
  id: "ndtv-grap-3",
  title: "Delhi reels under poor air as restrictions kick in",
  summary: "Schools asked to shift to online classes as air quality worsens.",
  source: "NDTV",
  url: "https://www.ndtv.com/delhi-news/grap-stage-3",
  official: false,
};

const okPayload = {
  status: "ok",
  stale: false,
  fetched_at: "2026-10-10T12:30:00",
  age_seconds: 120,
  ttl_seconds: 1200,
  items: [officialItem, pressItem],
  sources: [{ name: "CAQM", url: officialItem.url, fetched_at: "2026-10-10T12:30:00" }],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Alerts", () => {
  it("shows official advisories above model alerts with source labels", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => okPayload }));

    renderWithProvider(<Alerts forecast={forecast} />);

    // live cards render
    expect(
      await screen.findByText("CAQM imposes GRAP Stage 3 restrictions in Delhi NCR"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Construction activities banned across Delhi NCR until further notice."),
    ).toBeInTheDocument();

    // Official badge + source label (name + fetched time + published date)
    expect(screen.getAllByText("Official").length).toBeGreaterThanOrEqual(1);
    const sourceLink = screen.getByRole("link", { name: /Source: CAQM/ });
    expect(sourceLink).toHaveAttribute("href", officialItem.url);
    expect(sourceLink).toHaveAttribute("target", "_blank");
    expect(sourceLink).toHaveTextContent("published 08 Oct");
    expect(screen.getByRole("link", { name: /Source: NDTV/ })).toBeInTheDocument();

    // model alerts still render, clearly labelled as model output
    expect(await screen.findByText("Morning Rush Hour Impact")).toBeInTheDocument();
    expect(screen.getAllByText("ML forecast").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Live advisory").length).toBeGreaterThanOrEqual(1);

    // live cards come before model cards in the document
    const origins = Array.from(document.querySelectorAll("[data-origin]")).map((el) =>
      el.getAttribute("data-origin"),
    );
    expect(origins[0]).toBe("live");
    expect(origins[origins.length - 1]).toBe("model");

    // retrieval status shown
    expect(screen.getByTestId("alerts-live-status")).toHaveTextContent(
      /Live advisories retrieved at/,
    );
  });

  it("keeps model alerts and shows a muted status when the context fetch fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));

    renderWithProvider(<Alerts forecast={forecast} />);

    expect(await screen.findByText("Morning Rush Hour Impact")).toBeInTheDocument();
    expect(
      await screen.findByText(/Live advisories unavailable — model alerts shown\./),
    ).toBeInTheDocument();
    expect(screen.queryByText("Official")).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Source:/ })).not.toBeInTheDocument();
    // never a blank card: model alert still rendered with its ML label
    expect(screen.getAllByText("ML forecast").length).toBeGreaterThanOrEqual(1);
  });

  it("marks the last cached advisories as stale when refresh fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          ...okPayload,
          status: "stale",
          stale: true,
          age_seconds: 1800,
          message: "Refresh failed (timed out); showing the last result.",
        }),
      }),
    );

    renderWithProvider(<Alerts forecast={forecast} />);

    expect(
      await screen.findByText("CAQM imposes GRAP Stage 3 restrictions in Delhi NCR"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("alerts-live-status")).toHaveTextContent(
      /Live advisories last updated 30m ago/,
    );
  });

  it("renders the unavailable payload without any live card", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          status: "unavailable",
          stale: false,
          fetched_at: null,
          age_seconds: null,
          ttl_seconds: 1200,
          items: [],
          sources: [],
          message: "Live context unavailable (TINYFISH_API_KEY is not set).",
        }),
      }),
    );

    renderWithProvider(<Alerts forecast={forecast} />);

    expect(await screen.findByText(/TINYFISH_API_KEY is not set/)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /Source:/ })).not.toBeInTheDocument();
    expect(screen.getByText("Morning Rush Hour Impact")).toBeInTheDocument();
  });
});

describe("mergeAlerts", () => {
  const model: Alert[] = [
    {
      id: "high-no2",
      severity: "high",
      title: "Elevated NO2 Levels Detected",
      message: "Model text.",
      icon: ShieldAlert,
    },
    {
      id: "safe",
      severity: "low",
      title: "Air Quality Within Safe Limits",
      message: "Model text.",
      icon: Info,
    },
  ];

  it("puts live advisories first and drops model duplicates", () => {
    const duplicate: ContextAlert = {
      ...officialItem,
      id: "dup",
      title: "Elevated NO2 Levels Detected over Delhi",
    };
    const merged = mergeAlerts([duplicate], model);

    expect(merged[0].origin).toBe("live");
    expect(merged.map((item) => item.id)).not.toContain("model-high-no2");
    expect(merged.map((item) => item.id)).toContain("model-safe");
    expect(merged).toHaveLength(2);
  });

  it("keeps every model alert when nothing overlaps", () => {
    const merged = mergeAlerts([officialItem], model);
    expect(merged).toHaveLength(3);
    expect(merged.filter((item) => item.origin === "model")).toHaveLength(2);
  });

  it("de-duplicates repeated live items by id", () => {
    const merged = mergeAlerts([officialItem, officialItem], []);
    expect(merged).toHaveLength(1);
  });

  it("titleOverlaps needs a real token overlap", () => {
    expect(
      titleOverlaps("GRAP Stage 3 restrictions in Delhi", "GRAP Stage 3 restrictions Delhi"),
    ).toBe(true);
    expect(titleOverlaps("GRAP Stage 3 restrictions", "Rain likely this week")).toBe(false);
    expect(titleOverlaps("", "anything")).toBe(false);
  });

  it("uses the severity icon for live items", () => {
    const merged = mergeAlerts([{ ...officialItem, severity: "high", official: true }], []);
    expect(merged[0].icon).toBe(ShieldAlert);
    expect(merged[0].severity).toBe("high");
  });

  it("keeps the model icon for model items", () => {
    const merged = mergeAlerts(
      [],
      [
        {
          id: "x",
          severity: "medium",
          title: "Afternoon Ozone Spike",
          message: "m",
          icon: AlertTriangle,
        },
      ],
    );
    expect(merged[0].icon).toBe(AlertTriangle);
    expect(merged[0].origin).toBe("model");
  });
});
