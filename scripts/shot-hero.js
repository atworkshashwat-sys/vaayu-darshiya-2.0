/** Screenshot the text-only hero (desktop + mobile). */
const { chromium } = require("playwright");
const path = require("node:path");
const OUT = path.join(__dirname, "..", "shots", "check");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await chromium.launch();

  const desk = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const dp = await desk.newPage();
  await dp.goto("http://localhost:8082/", { waitUntil: "domcontentloaded", timeout: 90000 });
  await sleep(9000);
  await dp.screenshot({ path: path.join(OUT, "hero-no3d.png") });
  await desk.close();

  const mob = await browser.newContext({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
  });
  const mp = await mob.newPage();
  await mp.goto("http://localhost:8082/", { waitUntil: "domcontentloaded", timeout: 90000 });
  await sleep(9000);
  await mp.screenshot({ path: path.join(OUT, "hero-no3d-mobile.png") });
  await mob.close();

  await browser.close();
  console.log("done");
})();
