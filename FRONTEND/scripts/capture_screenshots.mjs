#!/usr/bin/env node
/**
 * Capture the four acceptance states (spec §9).
 *
 *   node scripts/capture_screenshots.mjs live      (a) fresh load with live data
 *   node scripts/capture_screenshots.mjs broadcast (b) a push updates the page
 *   node scripts/capture_screenshots.mjs stale     (c) a station gone quiet
 *   node scripts/capture_screenshots.mjs expired   (d) the live window ended
 *
 * "broadcast" is the interesting one: it proves the page changed WITHOUT a
 * reload. A marker is planted in the window object before the refresh; if the
 * page reloaded the marker would be gone, so its survival is the evidence.
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";

const BASE = process.env.BASE_URL ?? "http://127.0.0.1:8081";
const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
const ADMIN = process.env.LIVE_ADMIN_TOKEN ?? "";
const mode = process.argv[2] ?? "live";

const OUT = new URL("../screenshots/", import.meta.url).pathname.replace(
  /^\/([A-Za-z]:)/,
  "$1",
);
await mkdir(OUT, { recursive: true });

const shot = (page, name) => page.screenshot({
  path: `${OUT}${name}.png`,
  fullPage: true,
});

async function forceRefresh(job) {
  const res = await fetch(
    `${BACKEND}/api/admin/tinyfish/refresh?job=${job}`,
    { method: "POST", headers: { Authorization: `Bearer ${ADMIN}` } },
  );
  return res.status;
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

try {
  if (mode === "live" || mode === "stale" || mode === "expired") {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    // Wait for the header to report a data-freshness state rather than "…yet".
    await page.waitForFunction(
      () => !/No live readings yet/i.test(document.body.innerText),
      { timeout: 60000 },
    ).catch(() => {});
    await page.waitForTimeout(4000);
    await shot(page, `${mode}-state`);
    console.log(`captured ${mode}`);

  } else if (mode === "broadcast") {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(
      () => !/No live readings yet/i.test(document.body.innerText),
      { timeout: 60000 },
    ).catch(() => {});
    await page.waitForTimeout(4000);
    await shot(page, "broadcast-before");

    // Prove no reload: a property that survives only if the document is never
    // torn down between now and the moment the pushed event lands.
    await page.evaluate(() => { window.__noReloadMarker = true; });
    const before = await page.evaluate(() => document.body.innerText);

    const status = await forceRefresh("readings");
    console.log("admin refresh ->", status, ADMIN ? "" : "(no admin token)");

    // The stream pushes the new readings into the React Query cache, so the
    // rendered text changes while the marker stays put.
    await page.waitForFunction(
      (prev) => document.body.innerText !== prev,
      before,
      { timeout: 180000 },
    ).catch(() => console.log("warning: UI text did not change in time"));
    await page.waitForTimeout(3000);

    const marker = await page.evaluate(() => window.__noReloadMarker === true);
    await shot(page, "broadcast-after");
    console.log("no-reload marker survived:", marker);
    console.log("captured broadcast");
  }
} finally {
  await browser.close();
}
