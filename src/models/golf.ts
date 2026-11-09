// Golf putting model (Broadie 2018; Gelman 2019 Stan case study), as in the
// authors' prototype (github.com/elc45/squealer-prototype):
//   g(x) = [2 Phi(asin((R - r)/x) / sigma_angle) - 1]
//        * [Phi(2 / ((x + 1) sigma_distance)) - Phi(-1 / ((x + 1) sigma_distance))]
//   y_i ~ binomial(n_i, g(x_i)),  sigma_angle, sigma_distance ~ half-normal(0, 1).
import { erf, erfc, normSf } from "../stats/special";
import type { CurveModel } from "./types";

// x (feet), n attempts, successes. 31 distance bins, ~1.25 million putts.
const RAW: [number, number, number][] = [
  [0.28, 45198, 45183], [0.97, 183020, 182899], [1.93, 169503, 168594],
  [2.92, 113094, 108953], [3.93, 73855, 64740], [4.94, 53659, 41106],
  [5.94, 42991, 28205], [6.95, 37050, 21334], [7.95, 33275, 16615],
  [8.95, 30836, 13503], [9.95, 28637, 11060], [10.95, 26239, 9032],
  [11.95, 24636, 7687], [12.95, 22876, 6432], [14.43, 41267, 9813],
  [16.43, 35712, 7196], [18.44, 31573, 5290], [20.44, 28280, 4086],
  [21.95, 13238, 1642], [24.39, 46570, 4767], [28.4, 38422, 2980],
  [32.39, 31641, 1996], [36.39, 25604, 1327], [40.37, 20366, 834],
  [44.38, 15977, 559], [48.37, 11770, 311], [52.36, 8708, 231],
  [57.25, 8878, 204], [63.23, 5492, 103], [69.18, 3087, 35],
  [75.19, 1742, 24],
];

const R = 0.17708333; // hole radius (ft)
const r = 0.07; // ball radius (ft)
const SQRT2 = Math.SQRT2;

// Success probability p and failure probability q = 1 - p, each computed
// without cancellation so that log(q) stays accurate when p is near 1.
function pq(x: number, sa: number, sd: number): [number, number] {
  const z = Math.asin((R - r) / x) / sa / SQRT2;
  const p1 = erf(z);
  const q1 = erfc(z);
  const s = sd * (x + 1);
  const q2 = normSf(2 / s) + normSf(1 / s);
  const p2 = 1 - q2;
  return [p1 * p2, q1 + p1 * q2];
}

export const golf: CurveModel = {
  id: "golf",
  title: "Golf putting",
  description:
    "Proportion of successful putts by distance from the hole, 31 bins and about 1.25 million attempts, with a two-parameter geometric model.",
  xLabel: "Distance from hole (feet)",
  yLabel: "Probability of success",
  xDomain: [0, 80],
  yDomain: [0, 1.02],
  paramNames: ["sigma_angle", "sigma_distance"],
  data: RAW.map(([x, n, y]) => ({ x, y: y / n })),
  thetaInit: [0.013, 0.2],
  toTheta: (u) => u.map(Math.exp),
  fromTheta: (theta) => theta.map(Math.log),
  curve(x, [sa, sd]) {
    return pq(x, sa, sd)[0];
  },
  logLikPoints([sa, sd]) {
    // Binomial log likelihood without the (theta-free) binomial coefficient.
    return RAW.map(([x, n, y]) => {
      const [p, q] = pq(x, sa, sd);
      if (!(p > 0) || !(q > 0)) return -Infinity;
      return y * Math.log(p) + (n - y) * Math.log(q);
    });
  },
  logPrior([sa, sd]) {
    return -0.5 * (sa * sa + sd * sd);
  },
  clampX: (x) => Math.min(80, Math.max(0.2, x)),
};
