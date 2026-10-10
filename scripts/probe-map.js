/**
 * Probe the running dashboard headlessly over CDP: counts map markers, reads
 * their pixel positions and derives the map's zoom / visible bounds.
 *
 *   node scripts/probe-map.js [url]
 */
const { spawn } = require("node:child_process");

const URL_ = process.argv[2] || "http://localhost:8081/";
const EDGE =
  process.env.EDGE ||
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const PORT = 9333;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getTarget() {
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json`);
      const list = await res.json();
      const page = list.find((t) => t.type === "page" && t.url.startsWith("http"));
      if (page) return page;
    } catch {}
    await sleep(1000);
  }
  throw new Error("no CDP target");
}

function cdp(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  });
  const open = new Promise((res, rej) => {
    ws.addEventListener("open", res);
    ws.addEventListener("error", rej);
  });
  return {
    async send(method, params = {}) {
      await open;
      const mid = ++id;
      ws.send(JSON.stringify({ id: mid, method, params }));
      return new Promise((res) => pending.set(mid, res));
    },
    close: () => ws.close(),
  };
}

const EXPR = `(() => {
  const markers = [...document.querySelectorAll(".maplibregl-marker")];
  const MAP_W = document.querySelector(".maplibregl-canvas-container")?.getBoundingClientRect();
  const labels = markers.map((m) => (m.textContent || "").trim());
  const pos = markers.map((m) => {
    const t = m.style.transform || "";
    const mm = t.match(/translate3?d?\\(([-\\d.]+)px,\\s*([-\\d.]+)px/);
    return mm ? { x: +mm[1], y: +mm[2] } : null;
  });
  // visible container rect
  const cont = document.querySelector(".maplibregl-canvas-container")?.parentElement;
  const rect = cont ? cont.getBoundingClientRect() : null;
  return JSON.stringify({
    markerCount: markers.length,
    labels,
    positions: pos,
    container: rect ? { w: Math.round(rect.width), h: Math.round(rect.height), top: Math.round(rect.top) } : null,
    basemapWatermark: document.body.innerText.includes("API KEY REQUIRED"),
    liveText: (document.body.innerText.match(/\\d+ live stations|Connecting to model/) || [null])[0],
    hasCanvas: !!document.querySelector(".maplibregl-canvas"),
    styleButtons: [...document.querySelectorAll("button")].map(b => b.textContent.trim()).filter(Boolean).slice(0, 12),
  });
})()`;

async function main() {
  const shotPath = process.argv[3];
  const child = spawn(
    EDGE,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-sandbox",
      "--user-data-dir=" + require("node:os").tmpdir() + "\\vd-edge-profile",
      `--remote-debugging-port=${PORT}`,
      "--window-size=1600,1900",
      URL_,
    ],
    { stdio: "ignore", windowsHide: true },
  );

  try {
    const target = await getTarget();
    const client = cdp(target.webSocketDebuggerUrl);

    // give the app time to boot, fetch the API and place markers
    let data = null;
    for (let i = 0; i < 40; i++) {
      await sleep(1500);
      const r = await client.send("Runtime.evaluate", { expression: EXPR, returnByValue: true });
      const out = r.result?.result?.value;
      if (!out) continue;
      data = JSON.parse(out);
      if (data.markerCount >= 7) break;
      console.log(`  ... ${data.markerCount} markers so far`);
      data = null;
    }
    if (!data) throw new Error("did not reach 7 markers");
    report(data);

    if (shotPath) {
      // real wall-clock wait so vector tiles finish painting (virtual-time
      // screenshots fire while the tile queue is still in flight)
      await sleep(9000);
      await client.send("Page.enable");

      const capture = async (label) => {
        const shot = await client.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
        const p = shotPath.replace(/\.png$/, `-${label}.png`);
        require("node:fs").writeFileSync(p, Buffer.from(shot.result.data, "base64"));
        const r = await client.send("Runtime.evaluate", { expression: EXPR, returnByValue: true });
        const d = JSON.parse(r.result.result.value);
        console.log(
          `  ${label}: markers=${d.markerCount} watermark=${d.basemapWatermark} ${p}`,
        );
      };

      await capture("streets");
      if (process.argv[4] === "cycle") {
        // the style button cycles streets -> satellite -> dark
        const CLICK = `(() => {
          const b = [...document.querySelectorAll("button")].find(x => /Voyager|Satellite|Dark|Streets/.test(x.textContent));
          if (!b) return "no button";
          b.click(); return b.textContent.trim();
        })()`;
        for (const label of ["satellite", "dark"]) {
          const c = await client.send("Runtime.evaluate", { expression: CLICK, returnByValue: true });
          console.log(`  clicked -> ${c.result.result.value}`);
          await sleep(7000);
          await capture(label);
        }
      }
    }
  } finally {
    try {
      child.kill();
    } catch {}
  }
}

function report(d) {
  console.log(JSON.stringify(d, null, 2));
  const W = d.container?.w ?? 0;
  const H = d.container?.h ?? 0;
  const outside = d.labels.map((l, i) => ({ l, p: d.positions[i] }))
    .filter((m) => m.p && (m.p.x < 0 || m.p.x > W || m.p.y < 0 || m.p.y > H));
  console.log(`\ncontainer ${W}x${H}`);
  console.log(`markers outside the viewport: ${outside.length}`);
  outside.forEach((m) => console.log(`   ${m.l}  at ${Math.round(m.p.x)},${Math.round(m.p.y)}`));
}

main().catch((e) => {
  console.error("FAIL", e);
  process.exit(1);
});
