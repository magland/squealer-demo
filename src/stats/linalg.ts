// Small dense symmetric positive-definite linear algebra (row-major Float64Array).

export interface Chol {
  n: number;
  L: Float64Array;
}

// Returns null if the matrix is not numerically positive definite.
export function cholesky(A: Float64Array, n: number): Chol | null {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (!(s > 0)) return null;
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  return { n, L };
}

// Solve L z = b.
export function forward(c: Chol, b: ArrayLike<number>): Float64Array {
  const { n, L } = c;
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * z[k];
    z[i] = s / L[i * n + i];
  }
  return z;
}

// Solve A x = b with A = L L^T.
export function cholSolve(c: Chol, b: ArrayLike<number>): Float64Array {
  const { n, L } = c;
  const z = forward(c, b);
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

export function logDet(c: Chol): number {
  let s = 0;
  for (let i = 0; i < c.n; i++) s += Math.log(c.L[i * c.n + i]);
  return 2 * s;
}
