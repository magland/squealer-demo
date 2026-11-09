// Nelder-Mead maximization, warm-started from the previous estimate so that
// re-fitting while the curve is dragged takes only a few iterations.

export type Objective = (u: number[]) => number;

export interface MaximizeOptions {
  step?: number; // initial simplex size
  tolF?: number; // relative tolerance on the objective spread
  tolX?: number; // tolerance on the simplex size
  restart?: boolean; // restart once from the result (guards against a collapsed simplex)
  maxIter?: number;
}

export function maximize(f: Objective, u0: number[], step = 0.05, opts: MaximizeOptions = {}): number[] {
  const o = { step, tolF: 1e-11, tolX: 1e-9, restart: true, maxIter: 4000, ...opts };
  let best = nelderMead(f, u0, o.step, o.maxIter, o.tolF, o.tolX);
  if (o.restart) best = nelderMead(f, best, o.step / 5, o.maxIter, o.tolF, o.tolX);
  return best;
}

function nelderMead(fMax: Objective, u0: number[], step: number, maxIter: number, tolF: number, tolX: number): number[] {
  const n = u0.length;
  const f = (u: number[]) => {
    const v = -fMax(u);
    return Number.isFinite(v) ? v : 1e300;
  };
  let simplex: number[][] = [u0.slice()];
  for (let i = 0; i < n; i++) {
    const v = u0.slice();
    v[i] += step;
    simplex.push(v);
  }
  let values = simplex.map(f);

  for (let iter = 0; iter < maxIter; iter++) {
    const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
    simplex = order.map((i) => simplex[i]);
    values = order.map((i) => values[i]);

    const spread = Math.abs(values[n] - values[0]);
    let size = 0;
    for (let i = 1; i <= n; i++)
      for (let j = 0; j < n; j++) size = Math.max(size, Math.abs(simplex[i][j] - simplex[0][j]));
    if (spread <= tolF * (Math.abs(values[0]) + 1e-12) && size < tolX) break;

    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j] += simplex[i][j] / n;
    const along = (t: number) => centroid.map((c, j) => c + t * (simplex[n][j] - c));

    const xr = along(-1);
    const fr = f(xr);
    if (fr < values[0]) {
      const xe = along(-2);
      const fe = f(xe);
      if (fe < fr) {
        simplex[n] = xe;
        values[n] = fe;
      } else {
        simplex[n] = xr;
        values[n] = fr;
      }
    } else if (fr < values[n - 1]) {
      simplex[n] = xr;
      values[n] = fr;
    } else {
      const outside = fr < values[n];
      const xc = along(outside ? -0.5 : 0.5);
      const fc = f(xc);
      if (fc < (outside ? fr : values[n])) {
        simplex[n] = xc;
        values[n] = fc;
      } else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j]));
          values[i] = f(simplex[i]);
        }
      }
    }
  }
  let bi = 0;
  for (let i = 1; i <= n; i++) if (values[i] < values[bi]) bi = i;
  return simplex[bi];
}

// Covariance from a Laplace approximation at the mode u: inverse of the
// negative Hessian of f, by central finite differences.
export function laplaceCovariance(f: Objective, u: number[], h = 1e-4): number[][] {
  const n = u.length;
  const H: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const at = (di: number, si: number, dj: number, sj: number) => {
    const v = u.slice();
    v[di] += si * h;
    v[dj] += sj * h;
    return f(v);
  };
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const d = (at(i, 1, j, 1) - at(i, 1, j, -1) - at(i, -1, j, 1) + at(i, -1, j, -1)) / (4 * h * h);
      H[i][j] = -d;
      H[j][i] = -d;
    }
  }
  return invert(H);
}

export function invert(A: number[][]): number[][] {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...row.map((_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c];
    for (let k = 0; k < 2 * n; k++) M[c][k] /= piv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const fct = M[r][c];
      for (let k = 0; k < 2 * n; k++) M[r][k] -= fct * M[c][k];
    }
  }
  return M.map((row) => row.slice(n));
}

// L-BFGS maximization of f, given a function returning f(u) and filling its gradient.
// Used for models with many parameters, where Nelder-Mead is too slow.
export type ValueGrad = (u: number[], grad: Float64Array) => number;

export function lbfgsMaximize(fg: ValueGrad, u0: number[], maxIter = 2000, gtol = 1e-7): number[] {
  const n = u0.length;
  const m = 12;
  let x = u0.slice();
  const g = new Float64Array(n);
  // Work with the minimization of -f.
  const eval_ = (v: number[], out: Float64Array) => {
    const f = fg(v, out);
    for (let i = 0; i < n; i++) out[i] = -out[i];
    return Number.isFinite(f) ? -f : Infinity;
  };
  let f = eval_(x, g);
  const S: Float64Array[] = [];
  const Y: Float64Array[] = [];
  const RHO: number[] = [];
  const gNew = new Float64Array(n);
  let stall = 0;
  for (let iter = 0; iter < maxIter; iter++) {
    let gmax = 0;
    for (let i = 0; i < n; i++) gmax = Math.max(gmax, Math.abs(g[i]));
    if (gmax < gtol) break;
    // Two-loop recursion for d = -H g.
    const q = Float64Array.from(g);
    const alpha = new Array(S.length);
    for (let k = S.length - 1; k >= 0; k--) {
      let a = 0;
      for (let i = 0; i < n; i++) a += S[k][i] * q[i];
      a *= RHO[k];
      alpha[k] = a;
      for (let i = 0; i < n; i++) q[i] -= a * Y[k][i];
    }
    if (S.length > 0) {
      const k = S.length - 1;
      let sy = 0;
      let yy = 0;
      for (let i = 0; i < n; i++) {
        sy += S[k][i] * Y[k][i];
        yy += Y[k][i] * Y[k][i];
      }
      const gamma = sy / yy;
      for (let i = 0; i < n; i++) q[i] *= gamma;
    } else {
      // First step: unit length along the steepest descent direction.
      let gn = 0;
      for (let i = 0; i < n; i++) gn += g[i] * g[i];
      const sc = 1 / Math.max(Math.sqrt(gn), 1e-300);
      for (let i = 0; i < n; i++) q[i] *= Math.min(sc, 1);
    }
    for (let k = 0; k < S.length; k++) {
      let b = 0;
      for (let i = 0; i < n; i++) b += Y[k][i] * q[i];
      b *= RHO[k];
      for (let i = 0; i < n; i++) q[i] += S[k][i] * (alpha[k] - b);
    }
    const d = q.map((v) => -v);
    let gd = 0;
    for (let i = 0; i < n; i++) gd += g[i] * d[i];
    if (!(gd < 0)) {
      // Not a descent direction: reset memory and use steepest descent.
      S.length = 0;
      Y.length = 0;
      RHO.length = 0;
      for (let i = 0; i < n; i++) d[i] = -g[i] * 1e-3;
      gd = 0;
      for (let i = 0; i < n; i++) gd += g[i] * d[i];
    }
    // Backtracking line search (Armijo).
    let t = 1;
    let xNew: number[] = x;
    let fNew = Infinity;
    for (let ls = 0; ls < 60; ls++) {
      xNew = x.map((v, i) => v + t * d[i]);
      fNew = eval_(xNew, gNew);
      if (fNew <= f + 1e-4 * t * gd) break;
      t *= 0.5;
    }
    if (!(fNew <= f)) break;
    const s = new Float64Array(n);
    const y = new Float64Array(n);
    let sy = 0;
    for (let i = 0; i < n; i++) {
      s[i] = xNew[i] - x[i];
      y[i] = gNew[i] - g[i];
      sy += s[i] * y[i];
    }
    if (sy > 1e-12) {
      S.push(s);
      Y.push(y);
      RHO.push(1 / sy);
      if (S.length > m) {
        S.shift();
        Y.shift();
        RHO.shift();
      }
    }
    const df = f - fNew;
    x = xNew;
    f = fNew;
    g.set(gNew);
    stall = df < 1e-13 * (1 + Math.abs(f)) ? stall + 1 : 0;
    if (stall >= 4) break;
  }
  return x;
}

// Laplace covariance from an analytic gradient: Hessian by central differences of the gradient.
export function laplaceCovarianceFromGrad(fg: ValueGrad, u: number[], h = 1e-5): number[][] {
  const n = u.length;
  const gp = new Float64Array(n);
  const gm = new Float64Array(n);
  const H: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let j = 0; j < n; j++) {
    const up = u.slice();
    const um = u.slice();
    up[j] += h;
    um[j] -= h;
    fg(up, gp);
    fg(um, gm);
    for (let i = 0; i < n; i++) H[i][j] = -(gp[i] - gm[i]) / (2 * h);
  }
  for (let i = 0; i < n; i++)
    for (let j = 0; j < i; j++) {
      const v = 0.5 * (H[i][j] + H[j][i]);
      H[i][j] = v;
      H[j][i] = v;
    }
  return invert(H);
}
