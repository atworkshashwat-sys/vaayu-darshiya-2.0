/**
 * Click through the dashboard tabs headlessly over CDP and screenshot each one,
 * so a UI change can be verified without a human clicking around.
 *
 *   node scripts/probe-tabs.js "Model Details" out-prefix
 */
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");

const TAB = process.argv[2] || "Model Details";
const PREFIX = process.argv[3] || os.tmpdir() + "\\vd_tab";
const URL_ = process.argv[4] || "http://localhost:8081/";
const EDGE =
  process.env.EDGE ||
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const PORT = 9334;

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

async function main() {
  const child = spawn(
    EDGE,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-sandbox",
      "--user-data-dir=" + os.tmpdir() + "\\vd-edge-tabs",
      `--remote-debugging-port=${PORT}`,
      "--window-size=1600,1900",
      URL_,
    ],
    { stdio: "ignore", windowsHide: true },
  );

  try {
    const target = await getTarget();
    const client = cdp(target.webSocketDebuggerUrl);
    await client.send("Page.enable");

    // Support a "Parent > Child" chain, e.g. "Model Details > Validation".
    for (const step of TAB.split(">").map((s) => s.trim()).filter(Boolean)) {
      const CLICK = `(() => {
        const want = ${JSON.stringify(step)};
        let b;
        if (want.startsWith("@marker:")) {
          const name = want.slice(8);
          b = [...document.querySelectorAll(".maplibregl-marker")]
                .find(x => (x.textContent || "").includes(name));
        } else {
          b = [...document.querySelectorAll("button,[role=tab]")]
                .find(x => (x.textContent || "").trim() === want)
           || [...document.querySelectorAll("button,[role=tab]")]
                .find(x => (x.textContent || "").trim().includes(want));
        }
        if (!b) return "NOT FOUND";
        b.scrollIntoView({ block: "center" });
        if (want.startsWith("@marker:")) {
          // station popups are hover-driven (mouseenter/mouseleave)
          b.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
          b.dispatchEvent(new MouseEvent("mouseenter", { bubbles: false }));
        } else {
          // Radix Tabs activates on mousedown, so a plain .click() does nothing.
          b.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, button: 0 }));
          b.dispatchEvent(new MouseEvent("mouseup",   { bubbles: true, button: 0 }));
          b.click();
        }
        return "clicked:" + (b.textContent || "").trim();
      })()`;

      // React has to mount before there is anything to click.
      let result = "NOT FOUND";
      for (let i = 0; i < 30 && result === "NOT FOUND"; i++) {
        await sleep(1500);
        const r = await client.send("Runtime.evaluate", {
          expression: CLICK,
          returnByValue: true,
        });
        result = r.result?.result?.value || "NOT FOUND";
      }
      console.log(`click "${step}" ->`, result);
      if (result === "NOT FOUND") throw new Error(`tab "${step}" not found`);
      await sleep(1200);
    }

    await sleep(7000);

    // pull visible text so the render can be checked programmatically too
    const TXT = `(() => {
      const c = document.querySelector(".maplibregl-popup-content");
      return "__POPUPHTML_START\\n" + (c ? c.innerHTML : "(none)") + "\\n__POPUPHTML_END";
    })()`;
    const t = await client.send("Runtime.evaluate", { expression: TXT, returnByValue: true });
    const text = t.result?.result?.value || "";
    fs.writeFileSync(PREFIX + ".txt", text, "utf8");
    console.log("text ->", PREFIX + ".txt", `(${text.length} chars)`);

    const shot = await client.send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: true,
    });
    fs.writeFileSync(PREFIX + ".png", Buffer.from(shot.result.data, "base64"));
    console.log("shot ->", PREFIX + ".png");
  } finally {
    try {
      child.kill();
    } catch {}
  }
}

main().catch((e) => {
  console.error("FAIL", e);
  process.exit(1);
});
