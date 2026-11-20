// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Forecast: each trophy's Exophase earn rate (%) at the case's target.
// Model (beats the persistence baseline by using the history, not just the last read):
//   1. MECHANISTIC (counts): rate = 100 * earned / players_tracked. earned_i(t) and
//      players(t) each get a shrunk log-linear trend; extrapolate both to the target and
//      divide. When players grow faster than a trophy's earners (new players who have not
//      earned it yet), the rate FALLS - the dilution effect the baseline misses.
//   2. LOGIT TREND: weighted linear regression of logit(rate) on time, slope shrunk by
//      its own standard error (James-Stein-style factor b^2/(b^2+se(b)^2)); evaluated at
//      the target time. Median = 50/50 blend of (1) and (2).
//   3. SPREAD: baseline random-walk sigma 0.5*sqrt(d) in quadrature with the estimation
//      uncertainty of the two trends at the horizon (grows with horizon, shrinks with
//      more/better history). Thin or flat history -> falls back to persistence.
//      The slope's contribution to the spread uses its POSTERIOR variance after the
//      same shrinkage (f*se(b)^2, f = b^2/(b^2+se(b)^2)): a slope shrunk to zero must
//      not inflate the band. Without this, a 3-hour flat history would widen the band
//      ~2x over the baseline for no reason (v1 bug, fixed in v2).
//      v3: the standard error is HAC (Newey-West, Bartlett kernel): a random walk's
//      residuals are autocorrelated, so the OLS se understates slope uncertainty and a
//      wandering series looks like a significant trend (spurious regression). Slopes
//      are also multiplied by a SPLIT-HALF CONSISTENCY factor: a real drift has the
//      same slope in both halves of the history, a random walk's half-slopes are
//      independent. In a no-drift random-walk stress test v2 lost -55% pinball to the
//      baseline; these two guards exist to close that failure mode. Player growth is
//      exempt from the consistency factor (a smooth monotone count, never a walk).
//      v4: the consistency gate damps the rate's implied slope (h - g), never the
//      earner slope h alone: damping h while g stayed full manufactured fake
//      dilution in wandering series (v3 bug).
// All slopes are shrunk, all extrapolations clamped, no randomness, no network.
// Output: exactly {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]} per case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Baseline random-walk spread, logit units per square-root day. */
const SIGMA_RW = 0.5;
/** Recency half-life for history weights, in days. */
const HALF_LIFE_DAYS = 1;
/** Weak prior variance (logit^2) keeping thin-history fits honest. */
const PRIOR_VAR = 0.04;
const PRIOR_DF = 2;
/** Clamps on extrapolated per-day log growth (players). */
const G_MIN = -0.2;
const G_MAX = 2.0;
/** Clamp on the rate's implied log-slope from counts, per day. */
const S_MECH_MIN = -1.5;
const S_MECH_MAX = 1.5;
/** Cap on the players multiplier over the horizon (launch-week sanity bound). */
const P_MULT_CAP = 100;
/** Cap on how far the median logit may move from the last read. */
const DELTA_LOGIT_CAP = 3.5;
/** Cap on total spread, logit units. */
const SIGMA_CAP = 3.0;

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  levels: number[];
  horizon_minutes: number;
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => Math.log(clampPct(pct) / (100 - clampPct(pct)));
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Weighted least squares of y on t. Returns slope/intercept plus standard errors.
 * seB is HAC (Newey-West, Bartlett kernel, lag floor(n^(1/3))): residuals of a
 * wandering series are autocorrelated, and the OLS se would call a random walk's
 * apparent slope significant. sePred at t0 = level uncertainty + posterior
 * (shrunk-slope) uncertainty at t0. */
function wls(
  t: number[],
  y: number[],
  w: number[],
): { a: number; b: number; seB: number; sePred: (t0: number) => number; n: number; tBar: number; yBar: number } {
  const n = t.length;
  const W = w.reduce((s, v) => s + v, 0);
  const tBar = t.reduce((s, v, i) => s + w[i] * v, 0) / W;
  const yBar = y.reduce((s, v, i) => s + w[i] * v, 0) / W;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += w[i] * (t[i] - tBar) * (t[i] - tBar);
    sxy += w[i] * (t[i] - tBar) * (y[i] - yBar);
  }
  const resid = (a: number, b: number): number[] => y.map((v, i) => v - (a + b * t[i]));
  if (sxx < 1e-12) {
    // All times identical: slope unidentified, level = weighted mean only.
    const e = resid(yBar, 0);
    let rss = 0;
    for (let i = 0; i < n; i++) rss += w[i] * e[i] * e[i];
    const s2 = (rss + PRIOR_DF * PRIOR_VAR) / (Math.max(W - 1, 0.5) + PRIOR_DF);
    return { a: yBar, b: 0, seB: Infinity, sePred: () => Math.sqrt(s2 / W), n, tBar, yBar };
  }
  const b = sxy / sxx;
  const a = yBar - b * tBar;
  const e = resid(a, b);
  let rss = 0;
  for (let i = 0; i < n; i++) rss += w[i] * e[i] * e[i];
  const df = Math.max(W - 2, 0.5);
  const s2 = (rss + PRIOR_DF * PRIOR_VAR) / (df + PRIOR_DF);
  // HAC variance of b: M / sxx^2, M = sum_l K(l) sum_i w_i w_{i+l} e_i e_{i+l} dt_i dt_{i+l}.
  const L = Math.max(1, Math.floor(Math.pow(n, 1 / 3)));
  let meat = 0;
  for (let l = 0; l <= L; l++) {
    const k = l === 0 ? 1 : 2 * (1 - l / (L + 1));
    for (let i = 0; i + l < n; i++) {
      meat +=
        k *
        w[i] *
        w[i + l] *
        e[i] *
        e[i + l] *
        (t[i] - tBar) *
        (t[i + l] - tBar);
    }
  }
  const seB = Math.sqrt(Math.max(meat, 0) / (sxx * sxx));
  // Posterior variance of the shrunk slope: f * seB^2, f = b^2/(b^2+seB^2).
  const f = seB === 0 ? 1 : (b * b) / (b * b + seB * seB);
  const sePred = (t0: number): number =>
    Math.sqrt(s2 / W + f * seB * seB * (t0 - tBar) * (t0 - tBar));
  return { a, b, seB, sePred, n, tBar, yBar };
}

/** Split-half consistency of a slope: 1 when both halves agree, 0 when they
 * disagree in sign (a wander, not a drift), ratio in between. */
function consistency(t: number[], y: number[], w: number[]): number {
  const n = t.length;
  if (n < 4) return 0;
  const mid = Math.floor(n / 2);
  const half = (idx: number[]): number => {
    const th = idx.map((i) => t[i]);
    const yh = idx.map((i) => y[i]);
    const wh = idx.map((i) => w[i]);
    return wls(th, yh, wh).b;
  };
  const b1 = half(Array.from({ length: mid }, (_, i) => i));
  const b2 = half(Array.from({ length: n - mid }, (_, i) => mid + i));
  if (!Number.isFinite(b1) || !Number.isFinite(b2) || b1 === 0 || b2 === 0) return 0;
  if (Math.sign(b1) !== Math.sign(b2)) return 0;
  return Math.min(Math.abs(b1), Math.abs(b2)) / Math.max(Math.abs(b1), Math.abs(b2));
}

/** Shrink a slope toward 0 by its own noise: factor b^2/(b^2+se^2). */
function shrink(b: number, seB: number): number {
  if (!Number.isFinite(seB)) return 0;
  if (seB === 0) return b;
  return (b * b) / (b * b + seB * seB) * b;
}

/** Forecast one trophy. Returns [medianLogit, sigma]. */
function forecastTrophy(
  times: number[],
  players: number[],
  rates: Array<number | null>,
  tLast: number,
  dDays: number,
  fallbackPct: number,
): [number, number] {
  const sigmaRw = SIGMA_RW * Math.sqrt(dDays);
  const usable: number[] = [];
  for (let j = 0; j < rates.length; j++) {
    if (rates[j] !== null && Number.isFinite(rates[j] as number)) usable.push(j);
  }
  if (usable.length === 0) return [logit(fallbackPct), sigmaRw];
  const lastIdx = usable[usable.length - 1];
  const rLast = clampPct(rates[lastIdx] as number);
  const lLast = logit(rLast);
  if (usable.length < 2) return [lLast, sigmaRw];

  const tU = usable.map((j) => times[j]);
  const wU = tU.map((t) => Math.pow(0.5, (tLast - t) / HALF_LIFE_DAYS));

  // (2) Logit trend.
  const yLogit = usable.map((j) => logit(rates[j] as number));
  const trend = wls(tU, yLogit, wU);
  // Consistency gate: a real drift has the same slope in both halves; a wander's
  // half-slopes disagree. One gate for the rate's movement, shared with (1).
  const cRate = consistency(tU, yLogit, wU);
  const bTrend = shrink(trend.b, trend.seB) * cRate;
  const tFuture = tLast + dDays;
  // Pivot the shrunk slope through the data's centre (tBar, yBar), not through the
  // unshrunk fit's intercept: shrinking the slope must not shift the level.
  let yTrend = trend.yBar + bTrend * (tFuture - trend.tBar);
  yTrend = clamp(yTrend, lLast - DELTA_LOGIT_CAP, lLast + DELTA_LOGIT_CAP);
  const seTrend = trend.sePred(tFuture);

  // (1) Mechanistic counts: earned = rate/100 * players.
  const pU = usable.map((j) => players[j]);
  const logP = pU.map((p) => Math.log(Math.max(p, 1)));
  const gp = wls(tU, logP, wU);
  const g = clamp(shrink(gp.b, gp.seB), G_MIN, G_MAX);
  const logE = usable.map((j, k) => Math.log(Math.max((rates[j] as number) / 100, 0.0001) * pU[k]));
  const ge = wls(tU, logE, wU);
  // The rate's implied slope from counts is h - g. Damp THAT by the consistency
  // gate, never h alone: damping h while g stays full manufactures fake dilution
  // (v3 bug: in a wandering series it dragged every rate down ~10x over 3 days).
  const sMech = clamp(shrink(ge.b, ge.seB) - g, S_MECH_MIN, S_MECH_MAX);
  const h = g + sMech * cRate;
  const pLast = Math.max(players[lastIdx], 1);
  const pFuture = pLast * Math.min(Math.exp(g * dDays), P_MULT_CAP);
  const eLast = Math.max((rLast / 100) * pLast, 0.0001);
  const eFuture = eLast * Math.exp(h * dDays);
  const rMech = clampPct((100 * eFuture) / pFuture);
  const lMech = clamp(logit(rMech), lLast - DELTA_LOGIT_CAP, lLast + DELTA_LOGIT_CAP);
  // Mechanistic uncertainty: log-space se of both trends, converted to logit units.
  const seMechLog = Math.sqrt(ge.sePred(tFuture) ** 2 + gp.sePred(tFuture) ** 2);
  const logitFactor = Math.min(3, 1 / (1 - rLast / 100));
  const seMech = seMechLog * logitFactor;

  const median = clamp(0.5 * lMech + 0.5 * yTrend, lLast - DELTA_LOGIT_CAP, lLast + DELTA_LOGIT_CAP);
  const seEst = 0.5 * seTrend + 0.5 * seMech;
  const sigma = Math.min(SIGMA_CAP, Math.sqrt(sigmaRw * sigmaRw + seEst * seEst));
  return [median, sigma];
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);

  const t0 = Date.parse(c.history[0].at);
  const times = c.history.map((h, j) => {
    const ms = Date.parse(h.at);
    if (Number.isFinite(ms) && Number.isFinite(t0)) return (ms - t0) / 86_400_000;
    return j * (1 / 48); // fallback: 30-minute slots
  });
  const tLast = times[times.length - 1];
  const players = c.history.map((h) =>
    Number.isFinite(h.players_tracked) && h.players_tracked > 0 ? h.players_tracked : 1,
  );
  const dDays = c.horizon_minutes / 1440;

  // Fallback rate for a trophy never observed: overall median of the last read.
  const lastRates = c.history[c.history.length - 1].rates.filter(
    (r): r is number => r !== null && Number.isFinite(r),
  );
  const sortedLast = [...lastRates].sort((a, b) => a - b);
  const fallbackPct = sortedLast.length > 0 ? sortedLast[Math.floor(sortedLast.length / 2)] : 0.78;

  const trophies = c.trophies.map((t, i) => {
    const rates = c.history.map((h) => h.rates[i] ?? null);
    const [median, sigma] = forecastTrophy(times, players, rates, tLast, dDays, fallbackPct);
    const q = Z.map((z) => round4(clamp(sigmoidPct(median + z * sigma), 0, 100)));
    // Enforce non-decreasing after rounding (ties allowed).
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: t.id, q };
  });
  const out = `${JSON.stringify({ trophies })}\n`;
  if (out.length > 65_536) throw new Error(`${file} output ${out.length} bytes exceeds 65,536`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), out);
}
