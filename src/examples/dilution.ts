// Serial dilution assay (paper Section 2.2, after Gelman, Chew and Shnaidman 2004).
//
// The paper does not state its model or publish its plate, so this is our reconstruction
// of the structure it describes. Each sample j has a curve in dilution d,
//   g_j(d) = beta1 + beta2 / (1 + (c_j d / beta3)^(-beta4)),
// the four-parameter logistic in concentration c_j d, shared across samples. The standard
// sample has known relative concentration c_0 = 1; the 23 unknown samples have
//   log c_j ~ normal(mu_c, tau_c)                (the paper's hierarchical "beta2" term)
// and every sample has its own error scale,
//   log y ~ normal(log g_j(d), sigma_j),  log sigma_j ~ normal(mu_s, 1).
// At low concentration g_j(d) - beta1 is proportional to c_j^beta4, so log c_j plays the
// role of the paper's per-sample log beta2. theta-hat is the joint posterior mode. tau_c
// has a gamma(2) prior and is estimated; the concentrations are well identified, so the
// mode stays at a sensible tau_c. The error scales are weakly identified (3 readings per
// sample), and with an estimated group scale for them the joint mode collapses (all
// sigma_j equal, tau_s -> 0; the problem noted in the paper's Section 1.4), so their
// group scale is fixed at 1.
// The data are simulated by scripts/gen_dilution.py; samples 3 and 23 do not follow
// the curve shape, as in the paper's Figure 7.
import { fitGradBase, fitGradConstrained, type GradConstrainedFit, type GradModel } from "../gradSquealer";
import { mvnDraws } from "../stats/random";
import type { Pin } from "../squealer";
import type { Example, Solution } from "./types";

const RAW: [number, number, number][] = [
  [0, 1, 9.844],
  [0, 0.5, 9.847],
  [0, 0.25, 9.804],
  [0, 0.125, 9.679],
  [0, 0.0625, 9.496],
  [0, 0.03125, 8.806],
  [0, 0.015625, 8.222],
  [0, 0.0078125, 7.321],
  [0, 0.00390625, 6.319],
  [0, 0.00195312, 5.352],
  [0, 0.000976562, 4.390],
  [0, 0.000488281, 3.352],
  [0, 0.000244141, 2.536],
  [0, 0.00012207, 1.991],
  [1, 0.0001, 1.445],
  [1, 0.01, 3.889],
  [1, 0.1, 7.457],
  [2, 0.0001, 1.352],
  [2, 0.01, 4.545],
  [2, 0.1, 7.672],
  [3, 0.0001, 4.300],
  [3, 0.01, 6.500],
  [3, 0.1, 6.500],
  [4, 0.0001, 1.378],
  [4, 0.01, 2.687],
  [4, 0.1, 5.837],
  [5, 0.0001, 1.182],
  [5, 0.01, 2.931],
  [5, 0.1, 6.566],
  [6, 0.0001, 1.437],
  [6, 0.01, 2.241],
  [6, 0.1, 4.949],
  [7, 0.0001, 1.456],
  [7, 0.01, 2.314],
  [7, 0.1, 5.596],
  [8, 0.0001, 1.418],
  [8, 0.01, 1.767],
  [8, 0.1, 4.670],
  [9, 0.0001, 1.595],
  [9, 0.01, 4.475],
  [9, 0.1, 7.706],
  [10, 0.0001, 1.636],
  [10, 0.01, 1.946],
  [10, 0.1, 4.538],
  [11, 0.0001, 1.398],
  [11, 0.01, 2.113],
  [11, 0.1, 4.780],
  [12, 0.0001, 1.602],
  [12, 0.01, 1.452],
  [12, 0.1, 3.148],
  [13, 0.0001, 1.375],
  [13, 0.01, 3.130],
  [13, 0.1, 6.069],
  [14, 0.0001, 1.577],
  [14, 0.01, 1.668],
  [14, 0.1, 3.627],
  [15, 0.0001, 1.413],
  [15, 0.01, 1.424],
  [15, 0.1, 1.515],
  [16, 0.0001, 1.245],
  [16, 0.01, 4.403],
  [16, 0.1, 7.602],
  [17, 0.0001, 1.496],
  [17, 0.01, 2.830],
  [17, 0.1, 5.611],
  [18, 0.0001, 1.449],
  [18, 0.01, 1.739],
  [18, 0.1, 2.922],
  [19, 0.0001, 1.383],
  [19, 0.01, 1.771],
  [19, 0.1, 3.760],
  [20, 0.0001, 1.266],
  [20, 0.01, 2.049],
  [20, 0.1, 4.634],
  [21, 0.0001, 1.128],
  [21, 0.01, 3.983],
  [21, 0.1, 7.341],
  [22, 0.0001, 1.662],
  [22, 0.01, 3.125],
  [22, 0.1, 6.353],
  [23, 0.0001, 3.600],
  [23, 0.01, 6.700],
  [23, 0.1, 7.600],
];

const NU = 23; // unknown samples
const LOG2PI = Math.log(2 * Math.PI);
const LN10 = Math.LN10;
// Parameter layout of u (unconstrained; scale parameters on the log scale).
const B1 = 0, B2 = 1, B3 = 2, B4 = 3, MUC = 4, LTC = 5, MUS = 6;
const LC = (j: number) => 6 + j; // log c_j, j = 1..23 -> 7..29
const LS = (j: number) => 30 + j; // log sigma_j, j = 0..23 -> 30..53
const N = 54;
const TAU_S = 1; // fixed group scale for log sigma_j (see header)

const OBS = RAW.map(([s, d, v]) => ({ s, logd: Math.log(d), v }));

// Weakly informative priors (normal on the log scale) for the shared parameters.
const PRIOR: [number, number, number][] = [
  [B1, Math.log(4), 2],
  [B2, Math.log(2e4), 3],
  [B3, Math.log(0.05), 3],
  [B4, 0, 1],
  [MUC, -4, 5],
  [MUS, Math.log(0.2), 2],
];

interface CurveEval {
  L: number; // log g
  dB1: number;
  dB2: number;
  dZ: number;
  z: number;
}

function curveAt(u: number[], s: number, logd: number): CurveEval {
  const lc = s === 0 ? 0 : u[LC(s)];
  const b1 = Math.exp(u[B1]);
  const b2 = Math.exp(u[B2]);
  const b4 = Math.exp(u[B4]);
  const z = b4 * (lc + logd - u[B3]);
  const sg = 1 / (1 + Math.exp(-z));
  const g = b1 + b2 * sg;
  return { L: Math.log(g), dB1: b1 / g, dB2: (b2 * sg) / g, dZ: (b2 * sg * (1 - sg)) / g, z };
}

// Add w * d(log g)/du into grad.
function addCurveGrad(grad: Float64Array, c: CurveEval, s: number, u: number[], w: number) {
  const b4 = Math.exp(u[B4]);
  grad[B1] += w * c.dB1;
  grad[B2] += w * c.dB2;
  grad[B4] += w * c.dZ * c.z;
  grad[B3] -= w * c.dZ * b4;
  if (s !== 0) grad[LC(s)] += w * c.dZ * b4;
}

function logPrior(u: number[], grad: Float64Array | null): number {
  let lp = 0;
  for (const [i, m, sd] of PRIOR) {
    const e = (u[i] - m) / sd;
    lp -= 0.5 * e * e;
    if (grad) grad[i] -= e / sd;
  }
  // gamma(2, scale 1) prior on tau_c, written on t = log tau_c (Jacobian included).
  lp += 2 * u[LTC] - Math.exp(u[LTC]);
  if (grad) grad[LTC] += 2 - Math.exp(u[LTC]);
  const tc = Math.exp(u[LTC]);
  for (let j = 1; j <= NU; j++) {
    const e = (u[LC(j)] - u[MUC]) / tc;
    lp += -u[LTC] - 0.5 * e * e;
    if (grad) {
      grad[LC(j)] -= e / tc;
      grad[MUC] += e / tc;
      grad[LTC] += -1 + e * e;
    }
  }
  for (let j = 0; j <= NU; j++) {
    const e = (u[LS(j)] - u[MUS]) / TAU_S;
    lp -= 0.5 * e * e;
    if (grad) {
      grad[LS(j)] -= e / TAU_S;
      grad[MUS] += e / TAU_S;
    }
  }
  return lp;
}

function pointLL(u: number[], grad: Float64Array | null): number[] {
  return OBS.map(({ s, logd, v }) => {
    const c = curveAt(u, s, logd);
    const ls = u[LS(s)];
    const sig2 = Math.exp(2 * ls);
    const r = v - c.L;
    if (grad) {
      addCurveGrad(grad, c, s, u, r / sig2);
      grad[LS(s)] += -1 + (r * r) / sig2;
    }
    return -ls - (r * r) / (2 * sig2) - 0.5 * LOG2PI;
  });
}

function initialU(): number[] {
  const u = new Array(N).fill(0);
  u[B1] = Math.log(4);
  u[B2] = Math.log(2e4);
  u[B3] = Math.log(0.05);
  u[B4] = 0;
  u[MUC] = -4;
  u[LTC] = 0;
  u[MUS] = Math.log(0.2);
  for (let j = 1; j <= NU; j++) u[LC(j)] = -4;
  for (let j = 0; j <= NU; j++) u[LS(j)] = Math.log(0.2);
  return u;
}

export const dilutionModel: GradModel = {
  n: N,
  u0: initialU(),
  logPost(u, grad) {
    if (grad) grad.fill(0);
    let lp = logPrior(u, grad);
    for (const v of pointLL(u, grad)) lp += v;
    return lp;
  },
  parts(u) {
    const pl = pointLL(u, null);
    const logLik = pl.reduce((a, b) => a + b, 0);
    return { logPrior: logPrior(u, null), logLik, pointLL: pl };
  },
  // x is log10 dilution; the curve is log g_j.
  curve(x, panel, u, grad) {
    const c = curveAt(u, panel, x * LN10);
    if (grad) {
      const tmp = new Float64Array(N);
      addCurveGrad(tmp, c, panel, u, 1);
      for (let i = 0; i < N; i++) if (tmp[i] !== 0) grad[i] = tmp[i];
    }
    return c.L;
  },
};

const toTheta = (u: number[]) => u;

const PARAM_NAMES = [
  "log beta1",
  "log beta2",
  "log beta3",
  "log beta4",
  "mu_c",
  "log tau_c",
  "mu_sigma",
  ...Array.from({ length: NU }, (_, k) => `log c[${k + 1}]`),
  ...Array.from({ length: NU + 1 }, (_, k) => `log sigma[${k === 0 ? "std" : k}]`),
];

const N_DRAWS = 400;
const N_CURVES = 30;

export function dilutionExample(): Example {
  const model = dilutionModel;
  const base = fitGradBase(model);
  const hatParts = model.parts(base.u);
  const hatPost = hatParts.logPrior + hatParts.logLik;
  const solution = (fit: GradConstrainedFit): Solution => {
    const u = fit.u;
    const p = model.parts(u);
    return {
      theta: toTheta(u),
      curve: (x, panel) => model.curve(x, panel ?? 0, u, null),
      totals: [p.logPrior, p.logLik, p.logPrior + p.logLik],
      pointLL: p.pointLL,
      cost: hatPost - (p.logPrior + p.logLik),
      costParts: [
        { name: "from prior", value: hatParts.logPrior - p.logPrior },
        { name: "from likelihood", value: hatParts.logLik - p.logLik },
      ],
      maxViolation: fit.maxViolation,
      warm: fit,
    };
  };
  const hat = solution({ u: base.u, lambda: new Map(), maxViolation: 0 });
  const draws = mvnDraws(base.u, base.cov, N_DRAWS, 3);

  // Panels: the standard first, then unknowns in decreasing order of mean measurement (as in Figure 7).
  const meanOf = (s: number) => {
    const v = OBS.filter((o) => o.s === s).map((o) => o.v);
    return v.reduce((a, b) => a + b, 0) / v.length;
  };
  const order = [0, ...Array.from({ length: NU }, (_, k) => k + 1).sort((a, b) => meanOf(b) - meanOf(a))];
  const panels = order.map((s) => ({
    title: s === 0 ? "Standard sample" : `Unknown sample ${s}`,
    sample: s,
    dataIndex: OBS.map((o, i) => (o.s === s ? i : -1)).filter((i) => i >= 0),
  }));
  const pos = new Map(panels.map((p, k) => [p.sample, k]));
  const dilOffset = (logd: number) => 0.12 * (logd / LN10 + 2.5);

  return {
    id: "dilution",
    title: "Serial dilution assay (hierarchical)",
    description:
      "One standard sample of known concentration and 23 unknown samples, each measured at several dilutions, fit with a shared four-parameter logistic curve and hierarchical priors on concentration and error scale. Model and data are our reconstruction of the paper's Figure 7 (simulated).",
    xLabel: "log10 dilution",
    yLabel: "log y",
    xDomain: [-4.4, 0.2],
    yDomain: [0, 11],
    xTicks: [-4, -3, -2, -1, 0],
    data: OBS.map((o) => ({ x: o.logd / LN10, y: o.v })),
    panels: panels.map((p) => ({ title: p.title, dataIndex: p.dataIndex, curve: p.sample })),
    pointX: OBS.map((o) => (pos.get(o.s) ?? 0) + dilOffset(o.logd)),
    pointXTicks: panels.map((p, k) => ({ x: k + 0.3, label: p.sample === 0 ? "std" : String(p.sample) })),
    pointXLabel: "Sample (panel order), dilutions left to right",
    paramNames: PARAM_NAMES,
    dashParams: [B1, B2, B3, B4, MUC, LTC, MUS],
    groupScatter: {
      xName: "log c (concentration)",
      yName: "log σ (error scale)",
      items: Array.from({ length: NU }, (_, k) => ({ label: String(k + 1), xi: LC(k + 1), yi: LS(k + 1) })),
    },
    totalNames: ["log prior", "log lik", "log post"],
    maxPins: 8,
    pinNote:
      "Each panel's curve depends on the shared curve parameters and that sample's concentration, so pins on one sample can move the others a little.",
    drawNote: `${N_DRAWS} draws from a normal approximation to p(θ | y) at θ̂`,
    clampX: (x) => Math.min(0.2, Math.max(-4.4, x)),
    hat,
    drawCurves: draws.slice(0, N_CURVES).map((u) => (x: number, panel?: number) => model.curve(x, panel ?? 0, u, null)),
    dashDraws: draws.map((u) => {
      const p = model.parts(u);
      return { theta: toTheta(u), totals: [p.logPrior, p.logLik, p.logPrior + p.logLik] };
    }),
    solve(pins: Pin[], warm: unknown) {
      if (pins.length === 0) return hat;
      return solution(fitGradConstrained(model, base, pins, warm as GradConstrainedFit));
    },
  };
}
