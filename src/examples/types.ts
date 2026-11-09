// What the UI needs from an example, whether the curve is a parametric g(x | theta)
// or the posterior mean of a Gaussian process.
import type { DataPoint } from "../models/types";
import type { Pin } from "../squealer";

// A curve; `panel` selects which curve for examples with several (small multiples).
export type Curve = (x: number, panel?: number) => number;

export interface Panel {
  title: string;
  // Indices into Example.data of the points shown in this panel.
  dataIndex: number[];
  // The `panel` argument passed to curves and stored on pins (default: the panel's position).
  curve?: number;
}

// Paired per-group parameters to show as a scatter with +-1 sd, as in the paper's Figure 9.
export interface GroupScatter {
  xName: string;
  yName: string;
  items: { label: string; xi: number; yi: number }[];
}

export interface Solution {
  theta: number[];
  curve: Curve;
  // Values for the dashboard's lower rows, aligned with Example.totalNames.
  totals: number[];
  // log p(y_i | x_i, ...) for each data point under this solution.
  pointLL: number[];
  // Drop from the best fit; drives the squeal. Zero at the best fit.
  cost: number;
  // Optional breakdown of cost, shown under the total.
  costParts: { name: string; value: number }[];
  // Largest miss at a pin, in posterior sd (0 when the curve passes through every pin).
  maxViolation: number;
  // Opaque state for warm-starting the next solve.
  warm: unknown;
}

export interface Example {
  id: string;
  title: string;
  description: string;
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
  formatX?: (x: number) => string;
  xTicks?: number[];
  data: DataPoint[];
  // Small multiples; absent means one panel with all the data.
  panels?: Panel[];
  // Positions and tick labels for the change-in-fit plot, if not data[i].x.
  pointX?: number[];
  pointXTicks?: { x: number; label: string }[];
  pointXLabel?: string;
  paramNames: string[];
  // Which parameters get dashboard rows (default: all).
  dashParams?: number[];
  groupScatter?: GroupScatter;
  totalNames: string[];
  maxPins: number;
  pinNote: string;
  drawNote: string;
  clampX(x: number): number;
  hat: Solution;
  drawCurves: Curve[];
  // Posterior draws for the dashboard: parameter values and the matching totals.
  dashDraws: { theta: number[]; totals: number[] }[];
  solve(pins: Pin[], warm: unknown): Solution;
}
