/** Diagnose computed text colours on the AI Insights + Case Study tabs. */
const { chromium } = require("playwright");
const path = require("node:path");

const BASE = process.argv[2] || "http://localhost:8082/";
const OUT = path.join(__dirname, "..", "shots", "check");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function probe(page, label, selector) {
  const colors = await page.$$eval(selector, (els) =>
    els.slice(0, 4).map((el) => ({
      cls: el.className.slice(0, 70),
      color: getComputedStyle(el).color,
      text: (el.textContent || "").slice(0, 50),
    }))
  );
  console.log(`\n== ${label} ==`);
  for (const c of colors) console.log(JSON.stringify(c));
}

async function main() {
  const fs = require("node:fs");
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await ctx.newPage();
  await page.goto(BASE, { waitUntil: "domcontentloaded", timeout: 90000 });
  await sleep(9000);
  await page.screenshot({ path: path.join(OUT, "hero-3d.png") });

  // AI Insights tab
  const aiTab = page.getByRole("tab", { name: "AI Insights" }).first();
  await aiTab.click();
  await sleep(3500);
  await probe(page, "AI Insights paragraphs", '[class*="leading-relaxed"]');
  await probe(page, "AI Insights muted (control)", '[class*="text-muted-foreground"]');
  const aiSection = page.getByText("AI Insights", { exact: false }).first();
  await aiSection.scrollIntoViewIfNeeded();
  await sleep(600);
  await page.screenshot({ path: path.join(OUT, "ai-insights.png"), fullPage: false });

  // Case study tab
  const csTab = page.getByRole("tab", { name: /DRASTIC|CASE|Delhi/i }).first();
  await csTab.click().catch(() => {});
  await sleep(3000);
  await probe(page, "Case study text-secondary", '[class*="text-secondary"]');
  await page.screenshot({ path: path.join(OUT, "case-study-top.png") });
  await page.mouse.wheel(0, 1400);
  await sleep(900);
  await page.screenshot({ path: path.join(OUT, "case-study-mid.png") });
  await page.mouse.wheel(0, 1400);
  await sleep(900);
  await page.screenshot({ path: path.join(OUT, "case-study-mid2.png") });

  await browser.close();
  console.log("\nshots ->", OUT);
}

main().catch((e) => { console.error("FAIL", e); process.exit(1); });
