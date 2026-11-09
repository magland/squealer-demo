// Seeded draws from a multivariate normal, so displays are the same on every load.

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function normalSampler(seed: number): () => number {
  const rand = mulberry32(seed);
  return () => {
    const a = 1 - rand();
    const b = rand();
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
  };
}

// Lower Cholesky factor of a small covariance given as nested arrays.
export function choleskyRows(A: number[][]): number[][] {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 0)) : s / L[j][j];
    }
  }
  return L;
}

export function mvnDraws(mean: number[], cov: number[][], S: number, seed = 1): number[][] {
  const L = choleskyRows(cov);
  const normal = normalSampler(seed);
  const n = mean.length;
  const out: number[][] = [];
  for (let s = 0; s < S; s++) {
    const z = Array.from({ length: n }, normal);
    out.push(mean.map((m, i) => {
      let v = m;
      for (let j = 0; j <= i; j++) v += L[i][j] * z[j];
      return v;
    }));
  }
  return out;
}
