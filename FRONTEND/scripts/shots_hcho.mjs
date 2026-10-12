/**
 * HCHO hotspot layer — acceptance screenshots and checks.
 *
 * Run the backend (uvicorn api:app --port 8000) and the dev server first:
 *
 *     cd FRONTEND && node scripts/shots_hcho.mjs [all|a|b|c|d|zoom|toggle]
 *
 * Captures:
 *   (a) hcho-layer-hotspots   — HCHO layer with two ranked hotspot clusters
 *   (b) hcho-hotspot-panel    — a cluster panel open (desktop)
 *   (c) hcho-no-hotspot       — the real payload's "No clear hotspot" state
 *   (d) hcho-mobile           — mobile width, panel as a bottom sheet
 *   (e) hcho-zoom8/zoom12     — same zoom steps, for cell-edge inspection
 *
 * (a), (b) and (d) need a payload with hotspots. The real database pass
 * (2026-06-01, 4 stations, peak 13.0 µmol/m²) honestly has none — the P90
 * rule never clears the 31 µmol/m² Elevated floor — so those three shots
 * intercept /api/hcho-hotspots and serve a FIXTURE built from the same schema.
 * No production code is touched: the UI path, rule text and rendering are the
 * real ones, only the numbers are a fixture. (c) and (e) use the real payload.
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, "..", "screenshots");
const base = process.env.SHOT_BASE ?? "http://127.0.0.1:8081";
const mode = process.argv[2] ?? "all";

await mkdir(outDir, { recursive: true });

// ---------------------------------------------------------------- fixture ---

const pct = (xs, q) => {
  const s = [...xs].sort((a, b) => a - b);
  if (s.length === 1) return s[0];
  const pos = (s.length - 1) * (q / 100);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (pos - lo);
};

/** Closed [lon, lat] ring ~2 km around a point (same shape the API emits). */
const ring = (lon, lat, km = 2, n = 16) => {
  const dlat = km / 111.32;
  const dlon = km / (111.32 * Math.cos((lat * Math.PI) / 180));
  const out = [];
  for (let k = 0; k < n; k++) {
    const a = (2 * Math.PI * k) / n;
    out.push([
      +(lon + dlon * Math.cos(a)).toFixed(5),
      +(lat + dlat * Math.sin(a)).toFixed(5),
    ]);
  }
  out.push(out[0]);
  return out;
};

const BG = [
  { id: "fx-1", name: "Rohini Sector 5 (fixture)", lat: 28.7, lon: 77.1, value: 10 },
  { id: "fx-2", name: "Seemapuri (fixture)", lat: 28.63, lon: 77.3, value: 12 },
  { id: "fx-3", name: "Uttam Nagar (fixture)", lat: 28.5, lon: 77.05, value: 14 },
  { id: "fx-4", name: "Sadar Bazaar (fixture)", lat: 28.75, lon: 77.25, value: 16 },
  { id: "fx-5", name: "Mahipalpur (fixture)", lat: 28.38, lon: 77.15, value: 18 },
  { id: "fx-6", name: "Mayur Vihar (fixture)", lat: 28.55, lon: 77.5, value: 20 },
  { id: "fx-7", name: "Gurugram Sector 21 (fixture)", lat: 28.3, lon: 77.35, value: 22 },
];
const HOT = [
  { id: "fx-h", name: "Indirapuram (fixture)", lat: 28.62, lon: 77.4, value: 66 },
];

/** Real payload + fixture stations + two hotspots, schema-identical. */
function buildFixture(real) {
  const pass = (real.stations ?? []).filter((s) => s.latest_pass);
  const anand = (real.stations ?? []).find((s) => s.id === "6145551");
  // Raise one real station above the fixture threshold — same coordinates,
  // same dates, only the number is staged for the screenshot.
  const raised = { ...anand, value: 55 };
  const fixturePass = [
    ...pass.filter((s) => s.id !== "6145551").map((s) => ({ ...s })),
    ...BG.map((s) => ({ ...s, date: real.as_of, mean_7d: s.value, days: 7, latest_pass: true })),
    { ...raised, date: real.as_of, mean_7d: 40, days: 7, latest_pass: true },
    ...HOT.map((s) => ({ ...s, date: real.as_of, mean_7d: s.value, days: 7, latest_pass: true })),
  ];
  const values = fixturePass.map((s) => s.value);
  const p90 = pct(values, 90);
  const threshold = Math.max(p90, 31);
  const median = pct(values, 50);
  const hot = fixturePass.filter((s) => s.value >= threshold).sort((a, b) => b.value - a.value);

  const hotspots = hot.map((s, i) => ({
    rank: i + 1,
    peak: s.value,
    ratio_to_median: +(s.value / median).toFixed(1),
    persistence: i === 0 ? 5 : 4,
    days: 7,
    stations: [{ id: s.id, name: s.name, value: s.value }],
    polygon: ring(s.lon, s.lat),
    center: [+s.lon.toFixed(5), +s.lat.toFixed(5)],
  }));

  return {
    ...real,
    stations: [...(real.stations ?? []).filter((s) => !s.latest_pass), ...fixturePass],
    region_stats: {
      ...(real.region_stats ?? {}),
      as_of: real.as_of,
      n_stations: fixturePass.length,
      n_stations_window: fixturePass.length,
      median: +median.toFixed(1),
      p75: +pct(values, 75).toFixed(1),
      p90: +p90.toFixed(1),
      threshold: +threshold.toFixed(1),
      elevated_floor: 31,
    },
    hotspots,
    rule: `hotspot = value ≥ region P90 (${+p90.toFixed(1)}) and ≥ the Elevated floor (31 µmol/m²); adjacent hotspots within 8 km form one cluster`,
    fixture: true,
  };
}

// ---------------------------------------------------------------- helpers ---

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function open({ fixture = false, viewport = { width: 1440, height: 900 } } = {}) {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));

  if (fixture) {
    await page.route("**/api/hcho-hotspots", async (route) => {
      const res = await route.fetch();
      const body = await res.json();
      const shaped = typeof body === "string" ? JSON.parse(body) : body;
      await route.fulfill({ json: buildFixture(shaped) });
    });
  }

  // First dev-server load compiles on demand; give it room.
  await page.goto(base, { waitUntil: "domcontentloaded", timeout: 120000 });
  // The satellite chip only renders once the HCHO layer is selected, so wait
  // for the layer toggle itself (this also proves React has hydrated).
  await page
    .locator('button:has-text("HCHO")')
    .first()
    .waitFor({ state: "visible", timeout: 30000 });
  return { browser, page, errors };
}

async function selectHcho(page) {
  // The toggle can paint before effects (map init, layer handlers) attach —
  // give React a beat so the click lands on a live handler.
  await sleep(4000);
  const btn = page.locator('button:has-text("HCHO")').first();
  await btn.click();
  // Field raster + markers settle.
  await page.waitForFunction(
    () => /Satellite pass:/.test(document.body.innerText),
    { timeout: 15000 },
  ).catch(() => {});
  await sleep(2500);
}

async function state(page) {
  return page.evaluate(() => ({
    chip: (document.body.innerText.match(/Satellite pass:[^\n]*/) || [null])[0],
    // Exactly one status chip must exist (guards against double rendering).
    chipCount: [...document.querySelectorAll("div")].filter(
      (el) => el.className?.includes?.("pointer-events-none") && /Satellite pass:/.test(el.textContent ?? ""),
    ).length,
    noHotspot: /No clear hotspot on/.test(document.body.innerText),
    archived: /Archived/.test(document.body.innerText),
    legendNote: /Satellite column, not ground level\./.test(document.body.innerText),
    sliderNote: /Daily satellite pass/.test(document.body.innerText),
    interpNote: /Estimated between stations\./.test(document.body.innerText),
    // Rank labels are canvas symbols, not DOM — read them from the source.
    ranks: (() => {
      const src = window.__vdMap?.getSource("hcho-hotspots");
      const data = src?._data ?? src?.serialize?.()?.data;
      return (data?.features ?? []).map((f) => f.properties?.rankLabel).filter(Boolean);
    })(),
    markers: document.querySelectorAll(".maplibregl-marker").length,
  }));
}

async function tapHotspot(page, center) {
  // On mobile the map card can sit below the fold — scroll it into view so
  // the projected point is inside the viewport before clicking it.
  await page.locator(".maplibregl-canvas").first().scrollIntoViewIfNeeded();
  await sleep(400);
  const pt = await page.evaluate(([lon, lat]) => {
    const m = window.__vdMap;
    if (!m) return null;
    const c = m.getContainer().getBoundingClientRect();
    const p = m.project([lon, lat]);
    return { x: c.left + p.x, y: c.top + p.y };
  }, center);
  if (!pt) throw new Error("window.__vdMap missing (dev server?)");
  await page.mouse.click(pt.x, pt.y);
  await sleep(1200);
}

const shot = async (page, name) => {
  const file = path.join(outDir, `${name}.png`);
  await page.screenshot({ path: file, fullPage: true });
  console.log(`  → ${file}`);
};

// ------------------------------------------------------------------ modes ---

async function shotA() {
  const { browser, page, errors } = await open({ fixture: true });
  await selectHcho(page);
  const s = await state(page);
  await shot(page, "hcho-layer-hotspots");
  console.log("  chip:", s.chip, "| chips in DOM:", s.chipCount, "| ranks:", s.ranks.join(","), "| markers:", s.markers);
  await browser.close();
  return errors;
}

async function shotB() {
  const { browser, page, errors } = await open({ fixture: true });
  await selectHcho(page);
  // Rank 1 sits at the fixture's hottest station (Indirapuram).
  await tapHotspot(page, [77.4, 28.62]);
  const panel = await page.locator('div[role="dialog"]').first();
  const visible = await panel.isVisible().catch(() => false);
  await shot(page, "hcho-hotspot-panel");
  console.log("  panel visible:", visible, "|", (await panel.innerText().catch(() => "")).replace(/\n/g, " | "));
  await browser.close();
  return errors;
}

async function shotC() {
  const { browser, page, errors } = await open(); // real payload
  await selectHcho(page);
  const s = await state(page);
  await shot(page, "hcho-no-hotspot");
  console.log("  chip:", s.chip, "| no-hotspot text:", s.noHotspot, "| archived:", s.archived);
  console.log("  legend note:", s.legendNote, "| slider note:", s.sliderNote, "| interp:", s.interpNote);
  await browser.close();
  return errors;
}

async function shotD() {
  const { browser, page, errors } = await open({
    fixture: true,
    viewport: { width: 390, height: 844 },
  });
  await selectHcho(page);
  await tapHotspot(page, [77.4, 28.62]);
  const panel = page.locator('div[role="dialog"]').first();
  const box = await panel.boundingBox().catch(() => null);
  await shot(page, "hcho-mobile");
  console.log("  bottom sheet box:", box && JSON.stringify(box));
  // Tap-target audit: every interactive element in the sheet ≥ 44 px.
  const small = await panel.evaluate((el) =>
    [...el.querySelectorAll("button")]
      .map((b) => ({ label: b.getAttribute("aria-label") || b.textContent?.trim(), h: Math.round(b.getBoundingClientRect().height) }))
      .filter((b) => b.h < 44),
  );
  console.log("  buttons under 44px:", JSON.stringify(small));
  await browser.close();
  return errors;
}

async function zoomShots(fixture = false) {
  const { browser, page, errors } = await open({ fixture });
  await selectHcho(page);
  for (const target of [8, 12]) {
    // Fit puts us near zoom 10; step with the built-in controls.
    const cur = await page.evaluate(() => Math.round(window.__vdMap?.getZoom() ?? 10));
    let steps = target - cur;
    const sel = steps > 0 ? ".maplibregl-ctrl-zoom-in" : ".maplibregl-ctrl-zoom-out";
    for (let i = 0; i < Math.abs(steps); i++) {
      await page.locator(sel).first().click();
      await sleep(700);
    }
    await sleep(2500);
    const z = await page.evaluate(() => +window.__vdMap.getZoom().toFixed(2));
    await shot(page, `hcho-zoom${target}${fixture ? "-fixture" : ""}`);
    console.log(`  requested zoom ${target}, actual ${z}`);
  }
  await browser.close();
  return errors;
}

async function toggleCheck() {
  const { browser, page, errors } = await open();
  const vis = () =>
    page.evaluate(() => {
      const m = window.__vdMap;
      const v = (id) => (m.getLayer(id) ? m.getLayoutProperty(id, "visibility") ?? "visible" : "missing");
      return {
        heatmap: v("pollutant-heatmap"),
        circles: v("pollutant-circles"),
        idw: v("hcho-idw"),
        hotspots: v("hcho-hotspot-line"),
      };
    });
  const LABEL = { NO2: "NO₂", O3: "O₃", HCHO: "HCHO", ALL: "All" };
  const out = {};
  for (const layer of ["NO2", "O3", "HCHO", "ALL"]) {
    await page.locator(`button:has-text("${LABEL[layer]}")`).first().click();
    await sleep(1500);
    out[layer] = await vis();
  }
  console.log("  layer visibility by toggle:", JSON.stringify(out, null, 1));
  await browser.close();
  return errors;
}

// ------------------------------------------------------------------- main ---

const runners = {
  a: shotA, b: shotB, c: shotC, d: shotD,
  zoom: zoomShots, "zoom-fixture": () => zoomShots(true), toggle: toggleCheck,
};
const wanted = mode === "all" ? ["a", "b", "c", "d", "zoom", "zoom-fixture", "toggle"] : [mode];

let failures = 0;
for (const key of wanted) {
  const run = runners[key];
  if (!run) {
    console.error(`unknown mode: ${key}`);
    failures++;
    continue;
  }
  console.log(`[${key}] …`);
  try {
    const errors = await run();
    const real = (errors || []).filter((e) => !/favicon|net::ERR|Failed to load resource/.test(e));
    if (real.length) {
      console.log("  console errors:", real.slice(0, 5));
      failures++;
    } else {
      console.log("  no console errors");
    }
  } catch (e) {
    console.log("  FAILED:", e.message);
    failures++;
  }
}
process.exit(failures ? 1 : 0);
