// Special functions with good relative accuracy in the tails, which the golf
// likelihood needs: with ~10^5 putts per bin, log(1 - p) matters when 1 - p ~ 1e-9.

const SQRT_PI = Math.sqrt(Math.PI);
const SQRT2 = Math.SQRT2;

// erf(x) by its Taylor series; used for |x| < 2.
function erfSeries(x: number): number {
  const x2 = x * x;
  let term = x;
  let sum = x;
  for (let n = 1; n < 200; n++) {
    term *= -x2 / n;
    const t = term / (2 * n + 1);
    sum += t;
    if (Math.abs(t) < 1e-17 * Math.abs(sum)) break;
  }
  return (2 / SQRT_PI) * sum;
}

// erfc(x) for x >= 2 by the continued fraction (modified Lentz).
//   erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
function erfcContinuedFraction(x: number): number {
  const tiny = 1e-300;
  let f = x;
  let C = x;
  let D = 0;
  for (let n = 1; n < 500; n++) {
    const a = n / 2;
    D = x + a * D;
    if (Math.abs(D) < tiny) D = tiny;
    C = x + a / C;
    if (Math.abs(C) < tiny) C = tiny;
    D = 1 / D;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return Math.exp(-x * x) / SQRT_PI / f;
}

export function erf(x: number): number {
  if (Math.abs(x) < 2) return erfSeries(x);
  return x > 0 ? 1 - erfcContinuedFraction(x) : erfcContinuedFraction(-x) - 1;
}

export function erfc(x: number): number {
  if (x >= 2) return erfcContinuedFraction(x);
  if (x <= -2) return 2 - erfcContinuedFraction(-x);
  return 1 - erfSeries(x);
}

// Standard normal CDF and its upper tail.
export function normCdf(z: number): number {
  return 0.5 * erfc(-z / SQRT2);
}

export function normSf(z: number): number {
  return 0.5 * erfc(z / SQRT2);
}
