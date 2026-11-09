// The Squealer computation. The user grabs the curve and drags it; each grabbed
// point (a pin) becomes a hard constraint g(x_j | theta) = y_j, and theta* is the
// posterior mode subject to those constraints, found starting from the previous estimate.
import type { CurveModel } from "./models/types";
import { laplaceCovariance, maximize } from "./stats/optimize";
import { mvnDraws } from "./stats/random";

export interface FitSummary {
  theta: number[];
  logPrior: number;
  logLik: number;
  logPost: number;
  logLikPoints: number[];
}

export interface BaseFit {
  u: number[];
  summary: FitSummary;
  cov: number[][]; // Laplace covariance of u
}

const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);

export function summarize(model: CurveModel, theta: number[]): FitSummary {
  const logLikPoints = model.logLikPoints(theta);
  const logLik = sum(logLikPoints);
  const logPrior = model.logPrior(theta);
  return { theta, logPrior, logLik, logPost: logPrior + logLik, logLikPoints };
}

function logPostU(model: CurveModel, u: number[]): number {
  const theta = model.toTheta(u);
  return model.logPrior(theta) + sum(model.logLikPoints(theta));
}

export function fitBase(model: CurveModel): BaseFit {
  const f = (u: number[]) => logPostU(model, u);
  const u = maximize(f, model.fromTheta(model.thetaInit));
  const cov = laplaceCovariance(f, u);
  return { u, summary: summarize(model, model.toTheta(u)), cov };
}

// Approximate posterior sd of g(x) by the delta method on the Laplace approximation.
export function curveSd(model: CurveModel, base: BaseFit, x: number): number {
  const n = base.u.length;
  const h = 1e-5;
  const grad = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const up = base.u.slice();
    const dn = base.u.slice();
    up[i] += h;
    dn[i] -= h;
    grad[i] = (model.curve(x, model.toTheta(up)) - model.curve(x, model.toTheta(dn))) / (2 * h);
  }
  let v = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) v += grad[i] * base.cov[i][j] * grad[j];
  return Math.sqrt(Math.max(v, 0));
}

export interface Pin {
  id: number;
  x: number;
  y: number;
  // Which panel the pin is on, for examples with several curves (small multiples).
  panel?: number;
}

export interface ConstrainedFit {
  u: number[];
  // Lagrange multipliers by pin id, reused to warm-start the next drag step.
  lambda: Map<number, number>;
  // Largest |g(x_j | theta) - y_j| over pins, in units of the posterior sd of g(x_j).
  maxViolation: number;
}

// The posterior mode subject to the curve passing exactly through each pin:
//   maximize log p(theta | y)  subject to  g(x_j | theta) = y_j.
// This is the limit of the paper's pseudo-data method as sigma* -> 0.
// Solved by an augmented Lagrangian with Nelder-Mead inner steps. Constraints
// are scaled by the posterior sd of g(x_j), so a violation of 1e-3 is a
// thousandth of the curve's own uncertainty. If no theta passes through all
// pins, the result approaches the closest it can get.
export function fitConstrained(
  model: CurveModel,
  base: BaseFit,
  pins: Pin[],
  warm: { u: number[]; lambda: Map<number, number> },
): ConstrainedFit {
  if (pins.length === 0) return { u: base.u, lambda: new Map(), maxViolation: 0 };
  const scale = pins.map((p) => Math.max(curveSd(model, base, p.x), 1e-12));
  const viol = (theta: number[]) => pins.map((p, j) => (model.curve(p.x, theta) - p.y) / scale[j]);
  let lambda = pins.map((p) => warm.lambda.get(p.id) ?? 0);
  let u = warm.u;
  let mu = 10;
  let prev = Infinity;
  let c = viol(model.toTheta(u));
  for (let iter = 0; iter < 40; iter++) {
    const lam = lambda;
    const m = mu;
    const F = (v: number[]) => {
      const theta = model.toTheta(v);
      let val = model.logPrior(theta) + sum(model.logLikPoints(theta));
      const cv = viol(theta);
      for (let j = 0; j < cv.length; j++) val += lam[j] * cv[j] - 0.5 * m * cv[j] * cv[j];
      return val;
    };
    u = maximize(F, u, 0.02);
    c = viol(model.toTheta(u));
    const norm = Math.max(...c.map(Math.abs));
    if (norm < 1e-3) break;
    lambda = lambda.map((l, j) => l - mu * c[j]);
    if (norm > 0.25 * prev) mu = Math.min(mu * 10, 1e9);
    prev = norm;
  }
  const out = new Map<number, number>();
  pins.forEach((p, j) => out.set(p.id, lambda[j]));
  return { u, lambda: out, maxViolation: Math.max(...c.map(Math.abs)) };
}

export interface Draw {
  theta: number[];
  summary: FitSummary;
}

// Posterior simulations theta^s for the dashboard and the light curves. Drawn from
// the Laplace approximation (normal in u at u-hat with covariance base.cov), which
// stands in for MCMC draws. Seeded, so the display is the same on every load.
export function posteriorDraws(model: CurveModel, base: BaseFit, S: number, seed = 1): Draw[] {
  return mvnDraws(base.u, base.cov, S, seed).map((u) => {
    const theta = model.toTheta(u);
    return { theta, summary: summarize(model, theta) };
  });
}
