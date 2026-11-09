// A curve-fitting model in the sense of the paper: data (x_i, y_i), a parameter
// vector theta, and a curve g(x | theta) = E(y | x, theta) we can evaluate directly.

export interface DataPoint {
  x: number;
  y: number;
}

export interface CurveModel {
  id: string;
  title: string;
  description: string;
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
  paramNames: string[];
  data: DataPoint[];
  thetaInit: number[];
  // Optimization runs on an unconstrained vector u; theta = toTheta(u).
  toTheta(u: number[]): number[];
  fromTheta(theta: number[]): number[];
  curve(x: number, theta: number[]): number;
  // log p(y_i | x_i, theta) for each data point, up to constants.
  logLikPoints(theta: number[]): number[];
  logPrior(theta: number[]): number;
  // Pseudo-data x* must lie where the curve is defined.
  clampX(x: number): number;
}
