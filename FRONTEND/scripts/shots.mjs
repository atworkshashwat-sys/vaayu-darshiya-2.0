/**
 * Step 8 screenshots: the dashboard on a fresh load, with the live layer
 * attached and the SSE status indicator visible.
 *
 * Run the backend (uvicorn api:app --port 8000) and the dev server (npm run
 * dev) first, then:
 *
 *     cd FRONTEND && node scripts/shots.mjs
 *
 * The dev server proxies /api to the backend, so the page talks to the same
 * instance a visitor would.
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, "..", "screenshots");
const base = process.env.SHOT_BASE ?? "http://127.0.0.1:8080";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });

const errors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(msg.text());
});
page.on("pageerror", (err) => errors.push(String(err)));

await mkdir(outDir, { recursive: true });

console.log(`opening ${base}/ ...`);
const response = await page.goto(`${base}/`, { waitUntil: "domcontentloaded" });
console.log(`  HTTP ${response?.status()}`);

// Give the SSE hook and the TanStack queries time to land.
await page.waitForTimeout(6000);

await page.screenshot({ path: path.join(outDir, "01-dashboard-live.png"), fullPage: true });
console.log("  wrote 01-dashboard-live.png");

// The status pill the SSE hook writes — proof the browser is on the stream.
const bodyText = await page.evaluate(() => document.body.innerText);
const signals = {
  "SSE connected": /connected to live updates|live/i.test(bodyText),
  "case-study figures": /Live from official sources/i.test(bodyText),
  "alerts panel": /Active Alerts|Alerts/i.test(bodyText),
  "best-time panel": /Best Time|windows/i.test(bodyText),
};
for (const [label, hit] of Object.entries(signals)) {
  console.log(`  ${hit ? "✓" : "✗"} ${label}`);
}

// Scroll the case-study strip into view for a focused shot.
const caseStudy = page.locator("text=Live from official sources").first();
if (await caseStudy.count()) {
  await caseStudy.scrollIntoViewIfNeeded();
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(outDir, "02-case-study-live.png") });
  console.log("  wrote 02-case-study-live.png");
}

if (errors.length) {
  console.log(`\n${errors.length} console error(s):`);
  for (const e of errors.slice(0, 12)) console.log(`  - ${e.slice(0, 200)}`);
} else {
  console.log("\nno console errors");
}

await browser.close();
