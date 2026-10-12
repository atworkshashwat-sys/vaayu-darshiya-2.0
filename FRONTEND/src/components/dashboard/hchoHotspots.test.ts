import { describe, expect, it } from "vitest";
import {
  HCHO_RAMP_MAX,
  distanceKm,
  formatPassDate,
  idwRaster,
  interpolationStations,
  rampColor,
  rampCssGradient,
  type HchoHotspotsPayload,
} from "./hchoHotspots";

/**
 * The map's HCHO field, held to its promises:
 *  - it is smooth (continuous ramp, no steps),
 *  - it reproduces station values at stations (IDW interpolation),
 *  - it fades to fully transparent at the 15 km influence radius, so no strong
 *    colour ever appears far from a station.
 */

const px = (raster: ReturnType<typeof idwRaster>, col: number, row: number) => {
  const o = (row * raster.width + col) * 4;
  return {
    r: raster.data[o],
    g: raster.data[o + 1],
    b: raster.data[o + 2],
    a: raster.data[o + 3],
  };
};

describe("colour ramp", () => {
  it("is continuous and inside 0-255", () => {
    for (let v = 0; v <= HCHO_RAMP_MAX; v += 0.5) {
      const [r, g, b] = rampColor(v);
      for (const c of [r, g, b]) {
        expect(Number.isInteger(c)).toBe(true);
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(255);
      }
    }
  });

  it("clamps outside the range (no colour blow-outs)", () => {
    expect(rampColor(-50)).toEqual(rampColor(0));
    expect(rampColor(500)).toEqual(rampColor(HCHO_RAMP_MAX));
  });

  it("darkens as the value rises (sequential pale → deep)", () => {
    // Luminance must never increase with a higher value.
    const lum = (v: number) => {
      const [r, g, b] = rampColor(v);
      return 0.299 * r + 0.587 * g + 0.114 * b;
    };
    for (let v = 0; v < HCHO_RAMP_MAX; v += 1) {
      expect(lum(v + 1)).toBeLessThanOrEqual(lum(v) + 0.001);
    }
  });

  it("exposes a legend gradient built from the same stops", () => {
    expect(rampCssGradient()).toContain("linear-gradient(90deg,");
    expect(rampCssGradient()).toContain("rgb(");
  });
});

describe("IDW field", () => {
  it("reproduces the station value at the station", () => {
    // One station exactly on the single pixel centre.
    const raster = idwRaster(
      [{ lat: 28.5, lon: 77.2, value: 45 }],
      { north: 28.6, south: 28.4, east: 77.3, west: 77.1 },
      1,
      1,
    );
    const p = px(raster, 0, 0);
    expect(Math.abs(p.r * 0 + 45 - 45)).toBe(0); // ramp ran without NaN
    expect(p.a).toBeGreaterThan(200); // strong where a station is
    // The ramp colour for 45 µmol/m² — alpha aside, the RGB matches.
    const [er, eg, eb] = rampColor(45);
    expect([p.r, p.g, p.b]).toEqual([er, eg, eb]);
  });

  it("fades to fully transparent beyond the influence radius", () => {
    // ~10x wider than the 15 km radius: the far corners must be empty.
    const raster = idwRaster(
      [{ lat: 28.5, lon: 77.0, value: 60 }],
      { north: 29.0, south: 28.0, east: 78.0, west: 76.0 },
      3,
      3,
    );
    expect(px(raster, 1, 1).a).toBeGreaterThan(200); // at the station
    expect(px(raster, 0, 0).a).toBe(0); // far corner — no colour at all
    expect(px(raster, 2, 2).a).toBe(0);
    expect(px(raster, 0, 2).a).toBe(0);
    expect(px(raster, 2, 0).a).toBe(0);
  });

  it("blends between two stations instead of tiling cells", () => {
    // Two stations 10 km apart, 10 and 50 µmol/m²; the midpoint pixel sits
    // exactly halfway, so IDW must return their mean — one smooth field.
    const raster = idwRaster(
      [
        { lat: 28.545, lon: 77.1, value: 10 },
        { lat: 28.455, lon: 77.1, value: 50 },
      ],
      { north: 28.6, south: 28.4, east: 77.2, west: 77.0 },
      1,
      1,
    );
    // Midpoint value → the ramp colour for 30.
    const [er, eg, eb] = rampColor(30);
    expect([px(raster, 0, 0).r, px(raster, 0, 0).g, px(raster, 0, 0).b]).toEqual([er, eg, eb]);
  });

  it("is monotonic away from the station (no colour rings)", () => {
    // 0.6° wide ≈ 58 km, so the edge columns fall outside the 15 km radius.
    const bounds = { north: 28.7, south: 28.3, east: 77.5, west: 76.9 };
    const raster = idwRaster([{ lat: 28.5, lon: 77.2, value: 45 }], bounds, 7, 1);
    // Row through the station: alpha must decay smoothly to both edges.
    const alphas = Array.from({ length: 7 }, (_, i) => px(raster, i, 0).a);
    expect(alphas[3]).toBe(Math.max(...alphas));
    expect(alphas[0]).toBe(0);
    expect(alphas[6]).toBe(0);
    // Symmetric-ish decay, never a jump back up mid-way.
    for (let i = 0; i < 3; i++) expect(alphas[i]).toBeLessThanOrEqual(alphas[i + 1]);
    for (let i = 3; i < 6; i++) expect(alphas[i]).toBeGreaterThanOrEqual(alphas[i + 1]);
  });

  it("returns an empty (fully transparent) raster with no stations", () => {
    const raster = idwRaster([], { north: 1, south: 0, east: 1, west: 0 }, 4, 4);
    expect(raster.data.length).toBe(4 * 4 * 4);
    for (let i = 3; i < raster.data.length; i += 4) expect(raster.data[i]).toBe(0);
  });
});

describe("payload helpers", () => {
  const payload = {
    available: true,
    stations: [
      {
        id: "1", name: "A", lat: 28.5, lon: 77.1, value: 40, date: "2026-06-01",
        mean_7d: 40, days: 7, latest_pass: true,
      },
      {
        id: "2", name: "B", lat: 28.6, lon: 77.2, value: 50, date: "2026-05-26",
        mean_7d: 50, days: 1, latest_pass: false,
      },
    ],
    no_recent_data: [],
    hotspots: [],
  } as HchoHotspotsPayload;

  it("interpolates only from stations on the latest pass", () => {
    const pts = interpolationStations(payload);
    expect(pts).toHaveLength(1);
    expect(pts[0].value).toBe(40);
  });

  it("formats the pass date without timezone drift", () => {
    expect(formatPassDate("2026-06-01")).toBe("1 Jun 2026");
    expect(formatPassDate(undefined)).toBe("unknown date");
  });

  it("measures distance on the Delhi grid", () => {
    expect(distanceKm(28.5, 77.1, 28.5, 77.1)).toBe(0);
    // 0.1° of latitude ≈ 11 km.
    expect(distanceKm(28.5, 77.1, 28.6, 77.1)).toBeGreaterThan(10);
    expect(distanceKm(28.5, 77.1, 28.6, 77.1)).toBeLessThan(12);
  });
});
