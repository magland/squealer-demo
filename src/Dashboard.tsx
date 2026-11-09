// The dashboard of Figure 1 (center): one row per parameter, then log prior, log
// likelihood and log posterior. Each row shows the posterior density of that
// quantity from the draws (light blue), the estimate at theta-hat (blue dot) and
// the perturbed value at theta* (red dot). Each row has its own scale, centered on
// theta-hat and measured in posterior sd, so a red dot far from the density means
// the dragged fit is far outside the posterior.
import { fmt } from "./plotUtil";

export interface DashRow {
  name: string;
  draws: number[];
  hat: number;
  star: number | null;
}

const W = 460;
const ROW_H = 34;
const LABEL_W = 104;
const STRIP_W = 220;
const PAD = 8;

function meanSd(v: number[]): [number, number] {
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  const s = Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / (v.length - 1));
  return [m, s];
}

function Row({ row, y }: { row: DashRow; y: number }) {
  const [, sd] = meanSd(row.draws);
  const s = sd > 0 ? sd : 1;
  const z = (v: number) => (v - row.hat) / s;
  const zs = row.draws.map(z);
  const zStar = row.star === null ? null : z(row.star);
  let lo = Math.min(...zs, zStar ?? 0);
  let hi = Math.max(...zs, zStar ?? 0);
  const pad = 0.06 * (hi - lo);
  lo -= pad;
  hi += pad;
  const px = (zz: number) => LABEL_W + PAD + ((zz - lo) / (hi - lo)) * STRIP_W;
  const base = y + ROW_H - 9;
  const height = ROW_H - 14;

  // Gaussian kernel density of the draws on a grid across the strip.
  const n = 120;
  const bw = 1.06 * Math.pow(zs.length, -0.2);
  const dens: number[] = [];
  for (let i = 0; i <= n; i++) {
    const g = lo + ((hi - lo) * i) / n;
    let d = 0;
    for (const v of zs) d += Math.exp(-0.5 * ((g - v) / bw) ** 2);
    dens.push(d);
  }
  const peak = Math.max(...dens) || 1;
  const area =
    `M${px(lo).toFixed(1)},${base}` +
    dens.map((d, i) => `L${px(lo + ((hi - lo) * i) / n).toFixed(1)},${(base - (d / peak) * height).toFixed(1)}`).join("") +
    `L${px(hi).toFixed(1)},${base}Z`;

  return (
    <g>
      <text x={LABEL_W - 4} y={base - 2} textAnchor="end" className="dash-label">{row.name}</text>
      <line x1={LABEL_W + PAD} x2={LABEL_W + PAD + STRIP_W} y1={base} y2={base} className="dash-base" />
      <path d={area} className="dash-density" />
      {zStar !== null && <line x1={px(0)} x2={px(zStar)} y1={base} y2={base} className="dash-shift" />}
      <circle cx={px(0)} cy={base} r={4} className="dash-hat" />
      {zStar !== null && <circle cx={px(zStar)} cy={base} r={4} className="dash-star" />}
      <text x={LABEL_W + 2 * PAD + STRIP_W} y={base - 2} className="dash-value">
        <tspan className="v-hat">{fmt(row.hat, 4)}</tspan>
        {zStar !== null && (
          <tspan className="v-star">
            {" "}→ {zStar >= 0 ? "+" : ""}
            {Math.abs(zStar) < 10 ? zStar.toFixed(1) : zStar.toFixed(0)} sd
          </tspan>
        )}
      </text>
      <title>{`${row.name}\nθ̂: ${fmt(row.hat, 6)}` + (row.star === null ? "" : `\nθ*: ${fmt(row.star, 6)}`) + `\nposterior sd: ${fmt(sd, 4)}`}</title>
    </g>
  );
}

export function Dashboard({ rows, nParams }: { rows: DashRow[]; nParams: number }) {
  const sepY = nParams * ROW_H + 4;
  const H = rows.length * ROW_H + 10;
  return (
    <svg className="dashboard" viewBox={`0 0 ${W} ${H}`}>
      {rows.map((r, i) => (
        <Row key={r.name} row={r} y={i * ROW_H + (i >= nParams ? 8 : 0)} />
      ))}
      <line x1={8} x2={W - 8} y1={sepY} y2={sepY} className="dash-sep" />
    </svg>
  );
}
