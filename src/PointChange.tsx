// Figure 1 (right): change in fit for each data point,
// log p(y_i | x_i, theta*) - log p(y_i | x_i, theta-hat), plotted against x_i.
import type { Example } from "./examples/types";
import { changeColor, niceTicks } from "./plotUtil";

const W = 460;
const H = 250;
const M = { left: 62, right: 22, top: 10, bottom: 40 };

export function PointChange({ model, change }: { model: Example; change: number[] | null }) {
  const d = change ?? model.data.map(() => 0);
  const X = model.pointX ?? model.data.map((p) => p.x);
  const [x0, x1] = model.pointX ? [Math.min(...X) - 0.5, Math.max(...X) + 0.5] : model.xDomain;
  let y0 = Math.min(0, ...d);
  let y1 = Math.max(0, ...d);
  if (y1 - y0 < 1) {
    y0 -= 0.5;
    y1 += 0.5;
  }
  const pad = 0.06 * (y1 - y0);
  y0 -= pad;
  y1 += pad;
  const sx = (x: number) => M.left + ((x - x0) / (x1 - x0)) * (W - M.left - M.right);
  const sy = (y: number) => H - M.bottom - ((y - y0) / (y1 - y0)) * (H - M.top - M.bottom);
  const fx = model.formatX ?? ((x: number) => String(x));
  const xt = model.pointXTicks ?? (model.xTicks ?? niceTicks(x0, x1)).map((x) => ({ x, label: fx(x) }));
  const yt = niceTicks(y0, y1, 5);

  return (
    <svg className="pointchange" viewBox={`0 0 ${W} ${H}`}>
      {yt.map((t) => (
        <g key={"y" + t}>
          <line x1={M.left} x2={W - M.right} y1={sy(t)} y2={sy(t)} className="grid" />
          <text x={M.left - 6} y={sy(t) + 4} textAnchor="end" className="tick">{t}</text>
        </g>
      ))}
      {xt.map((t) => (
        <text key={"x" + t.x} x={sx(t.x)} y={H - M.bottom + 15} textAnchor="middle" className={model.pointXTicks ? "tick tiny" : "tick"}>
          {t.label}
        </text>
      ))}
      <line x1={M.left} x2={W - M.right} y1={sy(0)} y2={sy(0)} className="zero" />
      {model.data.map((p, i) => (
        <g key={i}>
          <line x1={sx(X[i])} x2={sx(X[i])} y1={sy(0)} y2={sy(d[i])} className="stem" />
          <circle cx={sx(X[i])} cy={sy(d[i])} r={3.5} fill={changeColor(d[i])} className="datum">
            <title>{`x = ${fx(p.x)}\nchange in log lik: ${d[i].toFixed(2)}`}</title>
          </circle>
        </g>
      ))}
      <text x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle" className="axis-label">{model.pointXLabel ?? model.xLabel}</text>
      <text transform={`translate(14 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">
        Change in log likelihood
      </text>
    </svg>
  );
}
