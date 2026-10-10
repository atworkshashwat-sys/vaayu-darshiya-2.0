#!/usr/bin/env node
/** Print the header's freshness label verbatim, so it can be checked against
 *  the API's observedAt without squinting at a screenshot. */
import { chromium } from "playwright";

const BASE = process.env.BASE_URL ?? "http://127.0.0.1:8081";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
try {
  await page.goto(BASE, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => !/No live readings yet/i.test(document.body.innerText),
    { timeout: 60000 },
  ).catch(() => {});
  await page.waitForTimeout(4000);

  const header = await page.evaluate(() => {
    // The bar that carries the date + freshness badge.
    const el = [...document.querySelectorAll("span,div")]
      .find((n) => /Live · updated|Stale|Live context ended|No live readings/i
        .test(n.textContent ?? "") && (n.textContent ?? "").length < 80);
    return el ? el.textContent.trim() : null;
  });
  console.log("header label :", header ?? "<not found>");

  const alerts = await page.evaluate(() => {
    const tags = [...document.querySelectorAll("*")]
      .map((n) => n.childNodes.length === 1 && n.textContent === "OFFICIAL")
      .filter(Boolean).length;
    return tags;
  });
  console.log("OFFICIAL tags rendered:", alerts);
} finally {
  await browser.close();
}
