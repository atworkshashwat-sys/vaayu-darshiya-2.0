import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AIInsights } from "./AIInsights";
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

const researchPayload = {
  status: "ok",
  provider: "TinyFish Search",
  city: "Delhi",
  query: "Delhi air quality pollution news and official advisory",
  retrieved_at: "2026-10-10T12:00:00",
  cached: false,
  items: [
    {
      title: "CAQM issues advisory for Delhi NCR",
      url: "https://example.com/advisory",
      snippet: "Schools asked to avoid outdoor sports.",
      site_name: "example.com",
      publisher: "Example Times",
      date: "2 days ago",
    },
  ],
  excerpt: {
    title: "CAQM issues advisory for Delhi NCR",
    url: "https://example.com/advisory",
    site_name: "example.com",
    text: "The commission directed schools to avoid outdoor sports.",
  },
  message: null,
};

const ML_CARD_TITLES = [
  "Pollution Pattern",
  "AI Forecast — NO₂, O₃ & HCHO",
  "Health Risk Guidance",
  "Vulnerable Groups",
];

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("AIInsights", () => {
  it("keeps the ML cards and shows a retrieval status when live research fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));

    render(<AIInsights city="Delhi" forecast={forecast} />);

    for (const title of ML_CARD_TITLES) {
      expect(await screen.findByText(title)).toBeInTheDocument();
    }
    expect(screen.getAllByText("ML model").length).toBeGreaterThanOrEqual(4);

    expect(await screen.findByText(/Live research unavailable/)).toBeInTheDocument();
    expect(screen.queryByText("External source")).not.toBeInTheDocument();
  });

  it("renders retrieved items, source links and excerpt when research succeeds", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => researchPayload }),
    );

    render(<AIInsights city="Delhi" forecast={forecast} />);

    expect(await screen.findByText("CAQM issues advisory for Delhi NCR")).toBeInTheDocument();

    const link = screen.getByRole("link", { name: /CAQM issues advisory/ });
    expect(link).toHaveAttribute("href", "https://example.com/advisory");
    expect(link).toHaveAttribute("target", "_blank");

    expect(screen.getByText("External source")).toBeInTheDocument();
    expect(
      screen.getByText(/The commission directed schools to avoid outdoor sports\./),
    ).toBeInTheDocument();

    // ML cards are still present and still labelled as model output.
    expect(screen.getAllByText("ML model").length).toBeGreaterThanOrEqual(4);
  });

  it("renders the unavailable status when the endpoint reports it", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          status: "unavailable",
          provider: "TinyFish Search",
          city: "Delhi",
          query: "Delhi air quality pollution news and official advisory",
          retrieved_at: "2026-10-10T12:00:00",
          cached: false,
          items: [],
          excerpt: null,
          message: "Live research unavailable (TINYFISH_API_KEY is not set)",
        }),
      }),
    );

    render(<AIInsights city="Delhi" forecast={forecast} />);

    expect(await screen.findByText(/TINYFISH_API_KEY is not set/)).toBeInTheDocument();
    expect(screen.queryByText("External source")).not.toBeInTheDocument();
    for (const title of ML_CARD_TITLES) {
      expect(screen.getByText(title)).toBeInTheDocument();
    }
  });
});
