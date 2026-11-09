// Per-group estimates as in the paper's Figure 9: for each group (sample), the best-fit
// values of two of its parameters with +-1 posterior sd (blue), and where the dragged
// fit puts them (red), labeled by group.
import type { GroupScatter as Spec } from "./examples/types";
import { niceTicks } from "./plotUtil";

const W = 460;
const H = 330;
const M = { left: 52, right: 14, top: 12, bottom: 40 };

function sdOf(v: number[]): number {
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
}

export function GroupScatter({
  spec,
  hat,
  star,
  draws,
}: {
  spec: Spec;
  hat: number[];
  star: number[] | null;
  draws: number[][];
}) {
  const pts = spec.items.map((it) => ({
    label: it.label,
    x: hat[it.xi],
    y: hat[it.yi],
    sx: sdOf(draws.map((d) => d[it.xi])),
    sy: sdOf(draws.map((d) => d[it.yi])),
    xs: star ? star[it.xi] : null,
    ys: star ? star[it.yi] : null,
  }));
  const xsAll = pts.flatMap((p) => [p.x - p.sx, p.x + p.sx, p.xs ?? p.x]);
  const ysAll = pts.flatMap((p) => [p.y - p.sy, p.y + p.sy, p.ys ?? p.y]);
  const pad = (lo: number, hi: number): [number, number] => {
    const d = Math.max(hi - lo, 1e-6) * 0.06;
    return [lo - d, hi + d];
  };
  const [x0, x1] = pad(Math.min(...xsAll), Math.max(...xsAll));
  const [y0, y1] = pad(Math.min(...ysAll), Math.max(...ysAll));
  const px = (x: number) => M.left + ((x - x0) / (x1 - x0)) * (W - M.left - M.right);
  const py = (y: number) => H - M.bottom - ((y - y0) / (y1 - y0)) * (H - M.top - M.bottom);
  // Labels of groups that moved noticeably are shown at their new position.
  const moved = (p: (typeof pts)[number]) =>
    p.xs !== null && p.ys !== null && Math.hypot((p.xs - p.x) / p.sx, (p.ys - p.y) / p.sy) > 0.5;

  return (
    <svg className="groupscatter" viewBox={`0 0 ${W} ${H}`}>
      {niceTicks(x0, x1, 6).map((t) => (
        <g key={"x" + t}>
          <line x1={px(t)} x2={px(t)} y1={M.top} y2={H - M.bottom} className="grid" />
          <text x={px(t)} y={H - M.bottom + 15} textAnchor="middle" className="tick">{t}</text>
        </g>
      ))}
      {niceTicks(y0, y1, 5).map((t) => (
        <g key={"y" + t}>
          <line x1={M.left} x2={W - M.right} y1={py(t)} y2={py(t)} className="grid" />
          <text x={M.left - 6} y={py(t) + 4} textAnchor="end" className="tick">{t}</text>
        </g>
      ))}
      {pts.map((p) => (
        <g key={p.label}>
          <line x1={px(p.x - p.sx)} x2={px(p.x + p.sx)} y1={py(p.y)} y2={py(p.y)} className="gs-bar" />
          <line x1={px(p.x)} x2={px(p.x)} y1={py(p.y - p.sy)} y2={py(p.y + p.sy)} className="gs-bar" />
          {moved(p) && <line x1={px(p.x)} y1={py(p.y)} x2={px(p.xs!)} y2={py(p.ys!)} className="dash-shift" />}
          <circle cx={px(p.x)} cy={py(p.y)} r={2.5} className="dash-hat" />
          {moved(p) && <circle cx={px(p.xs!)} cy={py(p.ys!)} r={3.5} className="dash-star" />}
          <text
            x={(moved(p) ? px(p.xs!) : px(p.x)) + 4}
            y={(moved(p) ? py(p.ys!) : py(p.y)) - 4}
            className={moved(p) ? "gs-label moved" : "gs-label"}
          >
            {p.label}
          </text>
        </g>
      ))}
      <text x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle" className="axis-label">{spec.xName}</text>
      <text transform={`translate(14 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">
        {spec.yName}
      </text>
    </svg>
  );
}
