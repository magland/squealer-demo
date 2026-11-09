// Adapter from a parametric CurveModel, g(x | theta), to an Example.
import type { CurveModel } from "../models/types";
import {
  fitBase,
  fitConstrained,
  posteriorDraws,
  summarize,
  type ConstrainedFit,
  type Pin,
} from "../squealer";
import type { Example, Solution } from "./types";

const N_DRAWS = 400;
const N_CURVES = 40;

export function parametricExample(model: CurveModel): Example {
  const base = fitBase(model);
  const draws = posteriorDraws(model, base, N_DRAWS);
  const solution = (fit: ConstrainedFit): Solution => {
    const theta = model.toTheta(fit.u);
    const s = summarize(model, theta);
    return {
      theta,
      curve: (x) => model.curve(x, theta),
      totals: [s.logPrior, s.logLik, s.logPost],
      pointLL: s.logLikPoints,
      cost: base.summary.logPost - s.logPost,
      costParts: [],
      maxViolation: fit.maxViolation,
      warm: fit,
    };
  };
  const hat = solution({ u: base.u, lambda: new Map(), maxViolation: 0 });
  const n = model.paramNames.length;
  return {
    id: model.id,
    title: model.title,
    description: model.description,
    xLabel: model.xLabel,
    yLabel: model.yLabel,
    xDomain: model.xDomain,
    yDomain: model.yDomain,
    data: model.data,
    paramNames: model.paramNames,
    totalNames: ["log prior", "log lik", "log post"],
    // More pins than parameters would over-determine the curve.
    maxPins: n,
    pinNote: `With ${n} parameters, at most ${n} pins hold at once, and a further grab releases the oldest.`,
    drawNote: `${N_DRAWS} draws from a normal approximation to p(θ | y) at θ̂`,
    clampX: model.clampX,
    hat,
    drawCurves: draws.slice(0, N_CURVES).map((d) => (x: number) => model.curve(x, d.theta)),
    dashDraws: draws.map((d) => ({ theta: d.theta, totals: [d.summary.logPrior, d.summary.logLik, d.summary.logPost] })),
    solve(pins: Pin[], warm: unknown) {
      if (pins.length === 0) return hat;
      return solution(fitConstrained(model, base, pins, warm as ConstrainedFit));
    },
  };
}
