/**
 * generate-static-api.js
 * 
 * Run this script WHILE the backend is running locally:
 *   1. Start backend: cd backend && uvicorn api:app --port 8000
 *   2. Run this: node scripts/generate-static-api.js
 * 
 * It saves all API responses as JSON files in FRONTEND/public/api/
 * so Vercel can serve them as static files — no backend needed.
 *
 * Override the backend location with API_BASE, e.g.:
 *   API_BASE=http://localhost:8011 node scripts/generate-static-api.js
 */

const fs = require("fs");
const path = require("path");

const API_BASE = process.env.API_BASE || "http://localhost:8000";
const OUT_DIR = path.join(__dirname, "..", "FRONTEND", "public", "api");

const ENDPOINTS = [
  { path: "/api/forecast?site=1", file: "forecast-1.json" },
  { path: "/api/forecast?site=2", file: "forecast-2.json" },
  { path: "/api/forecast?site=3", file: "forecast-3.json" },
  { path: "/api/forecast?site=4", file: "forecast-4.json" },
  { path: "/api/forecast?site=5", file: "forecast-5.json" },
  { path: "/api/forecast?site=6", file: "forecast-6.json" },
  { path: "/api/forecast?site=7", file: "forecast-7.json" },
  { path: "/api/stations", file: "stations.json" },
  { path: "/api/time", file: "time.json" },
  { path: "/api/metrics", file: "metrics.json" },
  { path: "/api/weekly-trend?site=1", file: "weekly-trend-1.json" },
  { path: "/api/weekly-trend?site=2", file: "weekly-trend-2.json" },
  { path: "/api/weekly-trend?site=3", file: "weekly-trend-3.json" },
  { path: "/api/weekly-trend?site=4", file: "weekly-trend-4.json" },
  { path: "/api/weekly-trend?site=5", file: "weekly-trend-5.json" },
  { path: "/api/weekly-trend?site=6", file: "weekly-trend-6.json" },
  { path: "/api/weekly-trend?site=7", file: "weekly-trend-7.json" },
];

async function generate() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  for (const ep of ENDPOINTS) {
    const url = `${API_BASE}${ep.path}`;
    process.stdout.write(`Fetching ${ep.path} ... `);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const outPath = path.join(OUT_DIR, ep.file);
      fs.writeFileSync(outPath, JSON.stringify(data, null, 2));
      console.log(`OK → ${ep.file}`);
    } catch (err) {
      console.log(`FAILED: ${err.message}`);
    }
  }

  console.log(`\nDone! Static API files saved to ${OUT_DIR}`);
}

generate();
