// make.fun — Forecast Polymarket's next NVIDIA CEO prices, 4 hours ahead.
//
// Model: empirical predictive distribution of standardised price changes.
//   sigma_t = MAD of 1-minute changes over the trailing 1440 minutes (updated
//             every 30 minutes of history; no lookahead past the checkpoint)
//   z(t,h)  = (p[t+h] - p[t]) / (sigma_t * sqrt(h))      for h in HG
//   The z sample of every market of the case is pooled; its empirical quantiles
//   at the five levels give the shape (fat tails, down-skew, discreteness), and
//   the forecast is q_tau(h) = clip(x + sigma_now * sqrt(h) * z_tau(h), 0, 1).
// No network, no randomness: same input file, same output bytes.

import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const QS = [0.05, 0.25, 0.5, 0.75, 0.95];
const HG = [5, 10, 15, 30, 60, 90, 120, 180, 240];
const HMAX = 240;
const STEP = 5;
const WINSOR = 12;
const WIN = 1440;
const EVERY = 30;

function quantileSorted(s: number[], q: number): number {
  const n = s.length;
  if (!n) return 0;
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(n - 1, lo + 1);
  const f = pos - lo;
  return s[lo] * (1 - f) + s[hi] * f;
}

function medianOf(a: number[]): number {
  const s = a.slice().sort((x, y) => x - y);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

// sigma at every index, recomputed every EVERY minutes from the trailing WIN diffs
function trailSigma(p: number[]): number[] {
  const n = p.length;
  const sig = new Array<number>(n).fill(0.003);
  let cur = 0.003;
  for (let i = 0; i < n; i++) {
    if (i % EVERY === 0) {
      const from = Math.max(1, i - WIN);
      const d: number[] = [];
      for (let t = from; t < i; t++) d.push(p[t] - p[t - 1]);
      if (d.length >= 20) {
        const med = medianOf(d);
        const mad = medianOf(d.map((x) => Math.abs(x - med))) / 0.6745;
        cur = Math.max(mad, 2e-4);
      }
    }
    sig[i] = cur;
  }
  return sig;
}

function buildCase(markets: any[]) {
  const series = markets.map((m) => m.prices as number[]);
  const sigs = series.map((p) => trailSigma(p));
  // pool standardised h-step changes across the case's markets
  const pool: Record<number, number[]> = {};
  for (const h of HG) pool[h] = [];
  for (let mi = 0; mi < series.length; mi++) {
    const p = series[mi];
    const s = sigs[mi];
    for (const h of HG) {
      if (p.length <= h + 30) continue;
      for (let t = 30; t + h < p.length; t += STEP) {
        let z = (p[t + h] - p[t]) / (s[t] * Math.sqrt(h));
        if (z > WINSOR) z = WINSOR;
        if (z < -WINSOR) z = -WINSOR;
        pool[h].push(z);
      }
    }
  }
  for (const h of HG) pool[h].sort((a, b) => a - b);
  const shape = HG.map((h) => QS.map((q) => quantileSorted(pool[h], q)));
  return { shape, sigs };
}

function interpolate(h: number, shape: number[][]): number[] {
  // linear in sqrt(h) between the HG nodes
  const xs = Math.sqrt(h);
  let i = 0;
  while (i < HG.length - 1 && Math.sqrt(HG[i + 1]) < xs) i++;
  const x0 = Math.sqrt(HG[i]);
  const j = Math.min(i + 1, HG.length - 1);
  const x1 = Math.sqrt(HG[j]);
  const y0 = shape[i];
  const y1 = shape[j];
  const f = x1 === x0 ? 0 : Math.min(1, Math.max(0, (xs - x0) / (x1 - x0)));
  return y0.map((v, k) => v * (1 - f) + y1[k] * f);
}

function main() {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  for (const f of readdirSync(inDir).filter((x) => x.endsWith(".in")).sort()) {
    const c = JSON.parse(readFileSync(join(inDir, f), "utf8"));
    const horizon = c.horizon_minutes || HMAX;
    const { shape, sigs } = buildCase(c.markets);
    const forecasts = c.markets.map((m: any, i: number) => {
      const p = m.prices as number[];
      const x = p[p.length - 1];
      const s = sigs[i][p.length - 1];
      const q: number[][] = [];
      for (let h = 1; h <= horizon; h++) {
        const z = interpolate(h, shape);
        const sc = s * Math.sqrt(h);
        const row = z.map((zz) => {
          let v = x + sc * zz;
          v = v < 0 ? 0 : v > 1 ? 1 : v;
          return Math.round(v * 1e6) / 1e6;
        });
        for (let k = 1; k < 5; k++) if (row[k] < row[k - 1]) row[k] = row[k - 1];
        q.push(row);
      }
      return { id: m.id, q };
    });
    writeFileSync(join(outDir, f.replace(/\.in$/, ".out")), JSON.stringify({ forecasts }));
  }
}

main();
