/**
 * Colour-system before/after screenshots.
 *
 *   node scripts/shots.js <before|after> [baseUrl]
 *
 * Captures: hero, map (NO2 + HCHO layers), 24h chart card, alerts card,
 * model-metrics table, and mobile (hero + map) at 390x844.
 */
const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");

const TAG = process.argv[2] || "before";
const BASE = process.argv[3] || "http://localhost:8081/";
const OUT = path.join(__dirname, "..", "shots", TAG);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Screenshot the enclosing shadcn Card (rounded-xl border) of a heading. */
async function shotCard(page, headingText, file) {
  const heading = page.getByText(headingText, { exact: false }).first();
  await heading.scrollIntoViewIfNeeded();
  await sleep(600);
  const card = heading.locator(
    'xpath=ancestor::div[contains(@class,"rounded-xl")][1]'
  );
  await card.screenshot({ path: path.join(OUT, file) });
  console.log("shot", file);
}

async function clickLayer(page, label) {
  const btn = page.getByRole("button", { name: label, exact: true }).first();
  await btn.scrollIntoViewIfNeeded();
  await btn.click();
  await sleep(3500); // let the raster + markers settle
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();

  // ── Desktop ────────────────────────────────────────────────────────────
  const desk = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1,
  });
  const page = await desk.newPage();
  await page.goto(BASE, { waitUntil: "domcontentloaded", timeout: 90000 });
  await sleep(9000); // react-query fetches + map init

  await page.screenshot({ path: path.join(OUT, "01-hero.png") });
  console.log("shot 01-hero.png");

  const map = page.locator(".maplibregl-map").first();
  await map.scrollIntoViewIfNeeded();
  await clickLayer(page, "NO\u2082");
  await map.screenshot({ path: path.join(OUT, "02-map-no2.png") });
  console.log("shot 02-map-no2.png");

  await clickLayer(page, "HCHO");
  await map.screenshot({ path: path.join(OUT, "03-map-hcho.png") });
  console.log("shot 03-map-hcho.png");

  await shotCard(page, "24-Hour Forecast", "04-chart.png");

  // Alerts: the heading lives in the section header, not the card grid.
  const alertsHeading = page.getByText("Active Alerts", { exact: false }).first();
  await alertsHeading.scrollIntoViewIfNeeded();
  await sleep(600);
  await alertsHeading
    .locator('xpath=ancestor::section[1]')
    .screenshot({ path: path.join(OUT, "05-alerts.png") });
  console.log("shot 05-alerts.png");

  // Model Details tab → metrics table
  const modelTab = page.getByRole("tab", { name: "Model Details" });
  await modelTab.click();
  await sleep(5000);
  const table = page.locator("table").first();
  await table.scrollIntoViewIfNeeded();
  await sleep(500);
  await table.screenshot({ path: path.join(OUT, "06-table.png") });
  console.log("shot 06-table.png");
  await desk.close();

  // ── Mobile ─────────────────────────────────────────────────────────────
  const mob = await browser.newContext({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
  });
  const mp = await mob.newPage();
  await mp.goto(BASE, { waitUntil: "domcontentloaded", timeout: 90000 });
  await sleep(9000);
  await mp.screenshot({ path: path.join(OUT, "07-mobile-hero.png") });
  console.log("shot 07-mobile-hero.png");

  const mmap = mp.locator(".maplibregl-map").first();
  await mmap.scrollIntoViewIfNeeded();
  await sleep(1500);
  await mmap.screenshot({ path: path.join(OUT, "08-mobile-map.png") });
  console.log("shot 08-mobile-map.png");

  await browser.close();
  console.log("done ->", OUT);
}

main().catch((e) => {
  console.error("FAIL", e);
  process.exit(1);
});
