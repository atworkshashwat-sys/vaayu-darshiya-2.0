/** One-off: confirm remaining `text-secondary` elements now render ink-muted. */
const { chromium } = require("playwright");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const b = await chromium.launch();
  const p = await (await b.newContext({ viewport: { width: 1600, height: 1000 } })).newPage();
  await p.goto("http://localhost:8082/", { waitUntil: "domcontentloaded", timeout: 90000 });
  await sleep(9000);
  const c = await p.$$eval('[class*="text-secondary"]', (els) =>
    els
      .filter((el) => String(el.className).split(/\s+/).includes("text-secondary"))
      .slice(0, 6)
      .map((el) => ({
        cls: String(el.className).slice(0, 60),
        color: getComputedStyle(el).color,
        text: (el.textContent || "").slice(0, 40),
      }))
  );
  console.log(JSON.stringify(c, null, 1));
  await b.close();
})();
