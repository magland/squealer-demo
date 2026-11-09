export function niceTicks(lo: number, hi: number, n = 6): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw)!;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(+v.toPrecision(12));
  return out;
}

// Map a change in log likelihood to a color: red for worse fit, green for better,
// with intensity on a log scale (a change of 1000 is full intensity).
export function changeColor(d: number): string {
  const t = Math.min(1, Math.log10(1 + Math.abs(d)) / 3);
  if (t < 0.02) return "var(--ink)";
  const hue = d < 0 ? "var(--worse)" : "var(--better)";
  return `color-mix(in srgb, ${hue} ${Math.round(100 * t)}%, var(--ink))`;
}

export function fmt(v: number, digits = 4): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(3);
  return v.toFixed(digits);
}
