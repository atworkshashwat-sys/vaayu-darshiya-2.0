/** Dump the rendered dashboard text — used to check what the panels actually
 * say, rather than guessing at it from the component source.
 *
 *     cd FRONTEND && node scripts/dump-text.mjs [tab]
 * where [tab] is "overview" (default) or "case".
 */
import { chromium } from "playwright";

const base = process.env.SHOT_BASE ?? "http://127.0.0.1:8080";
const tab = process.argv[2] ?? "overview";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
await page.goto(`${base}/`, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(12000);

if (tab === "case") {
  const caseTab = page
    .locator("button,[role=tab]")
    .filter({ hasText: /DRASTIC CASE/i })
    .first();
  if (await caseTab.count()) {
    await caseTab.click();
    await page.waitForTimeout(5000);
  }
}

const text = await page.evaluate(() => document.body.innerText);
console.log(text.split("\n").filter(Boolean).join("\n"));
await browser.close();
