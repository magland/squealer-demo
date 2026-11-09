// The Squealer computation for models with many parameters, using analytic gradients.
// Same method as squealer.ts: theta-hat is the posterior mode, and dragging the curve
// solves  maximize log p(u | y)  subject to  g(x_j | u, panel_j) = y_j  for each pin,
// by an augmented Lagrangian, here with L-BFGS inner steps instead of Nelder-Mead.
import type { Pin } from "./squealer";
import { laplaceCovarianceFromGrad, lbfgsMaximize } from "./stats/optimize";

export interface GradModel {
  n: number;
  u0: number[];
  // log p(u | y) up to a constant; fills grad (same length as u) when given.
  logPost(u: number[], grad: Float64Array | null): number;
  parts(u: number[]): { logPrior: number; logLik: number; pointLL: number[] };
  // The curve in panel `panel` at x; fills its gradient in u when given.
  curve(x: number, panel: number, u: number[], grad: Float64Array | null): number;
}

export interface GradBase {
  u: number[];
  cov: number[][];
}

export function fitGradBase(model: GradModel): GradBase {
  const fg = (u: number[], g: Float64Array) => model.logPost(u, g);
  const u = lbfgsMaximize(fg, model.u0);
  return { u, cov: laplaceCovarianceFromGrad(fg, u) };
}

export function gradCurveSd(model: GradModel, base: GradBase, x: number, panel: number): number {
  const g = new Float64Array(model.n);
  model.curve(x, panel, base.u, g);
  // Only a few entries of g are nonzero; skip the rest.
  const nz: number[] = [];
  for (let i = 0; i < model.n; i++) if (g[i] !== 0) nz.push(i);
  let v = 0;
  for (const i of nz) for (const j of nz) v += g[i] * base.cov[i][j] * g[j];
  return Math.sqrt(Math.max(v, 0));
}

export interface GradConstrainedFit {
  u: number[];
  lambda: Map<number, number>;
  maxViolation: number;
}

export function fitGradConstrained(
  model: GradModel,
  base: GradBase,
  pins: Pin[],
  warm: { u: number[]; lambda: Map<number, number> },
): GradConstrainedFit {
  if (pins.length === 0) return { u: base.u, lambda: new Map(), maxViolation: 0 };
  const n = model.n;
  const scale = pins.map((p) => Math.max(gradCurveSd(model, base, p.x, p.panel ?? 0), 1e-9));
  let lambda = pins.map((p) => warm.lambda.get(p.id) ?? 0);
  let u = warm.u;
  let mu = 10;
  let prev = Infinity;
  const cg = new Float64Array(n);
  const viol = (v: number[]) => pins.map((p, j) => (model.curve(p.x, p.panel ?? 0, v, null) - p.y) / scale[j]);
  let c = viol(u);
  for (let iter = 0; iter < 40; iter++) {
    const lam = lambda;
    const m = mu;
    const F = (v: number[], g: Float64Array) => {
      let val = model.logPost(v, g);
      for (let j = 0; j < pins.length; j++) {
        cg.fill(0);
        const cv = (model.curve(pins[j].x, pins[j].panel ?? 0, v, cg) - pins[j].y) / scale[j];
        val += lam[j] * cv - 0.5 * m * cv * cv;
        const w = (lam[j] - m * cv) / scale[j];
        for (let i = 0; i < n; i++) if (cg[i] !== 0) g[i] += w * cg[i];
      }
      return val;
    };
    u = lbfgsMaximize(F, u, 500, 1e-6);
    c = viol(u);
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
