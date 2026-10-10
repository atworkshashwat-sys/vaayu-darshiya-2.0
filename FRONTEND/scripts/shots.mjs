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

// Give the SSE hook and the TanStack queries time to land, including the
// first invalidation the stream sends after the scheduler's staggered run.
await page.waitForTimeout(15000);

await page.screenshot({ path: path.join(outDir, "01-dashboard-live.png"), fullPage: true });
console.log("  wrote 01-dashboard-live.png");

// The status pill the SSE hook writes — proof the browser is on the stream.
const bodyText = await page.evaluate(() => document.body.innerText);
const signals = {
  "SSE live pill": /Live updated/i.test(bodyText),
  "alerts fetched (not still checking)": /Live advisories retrieved at/i.test(bodyText),
  "best-time panel": /Best Time To Go Outside/i.test(bodyText),
};
for (const [label, hit] of Object.entries(signals)) {
  console.log(`  ${hit ? "✓" : "✗"} ${label}`);
}

// The case study lives on its own tab, so it never appears in the shot above.
const caseTab = page.locator("button,[role=tab]").filter({ hasText: /DRASTIC CASE/i }).first();
if (await caseTab.count()) {
  await caseTab.click();
  await page.waitForTimeout(6000);
  await page.screenshot({ path: path.join(outDir, "02-case-study-live.png"), fullPage: true });
  console.log("  wrote 02-case-study-live.png");
  const caseText = await page.evaluate(() => document.body.innerText);
  console.log(`  ${/Live from official sources/i.test(caseText) ? "✓" : "✗"} case-study figures`);
  console.log(`  ${/Latest Delhi air quality report/i.test(caseText) ? "✓" : "✗"} case-study report strip`);
}

if (errors.length) {
  console.log(`\n${errors.length} console error(s):`);
  for (const e of errors.slice(0, 12)) console.log(`  - ${e.slice(0, 200)}`);
} else {
  console.log("\nno console errors");
}

await browser.close();
