import { useMemo, useRef, useState } from "react";
import type { Curve, Example } from "./examples/types";
import type { DataPoint } from "./models/types";
import { changeColor, niceTicks } from "./plotUtil";
import type { Pin } from "./squealer";

const LARGE = { W: 720, H: 440, M: { left: 58, right: 16, top: 14, bottom: 46 }, samples: 400 };
// Small multiples: one panel per sample, as in the paper's Figure 7.
const SMALL = { W: 260, H: 190, M: { left: 34, right: 8, top: 22, bottom: 24 }, samples: 160 };
const HIT_PX = 12;

interface Props {
  model: Example;
  // Points shown in this plot (default: all of model.data); pointChange is aligned with them.
  data?: DataPoint[];
  title?: string;
  compact?: boolean;
  curveHat: Curve;
  curveStar: Curve | null;
  // Posterior draws, shown as light curves.
  drawCurves: Curve[];
  pins: Pin[];
  activePin: number | null;
  // log p(y_i | theta*) - log p(y_i | theta-hat), one per data point
  pointChange: number[] | null;
  // Grab the current curve at x; returns the new pin's id.
  onGrab: (x: number, y: number) => number;
  onMove: (id: number, x: number, y: number) => void;
  onRemove: (id: number) => void;
  onDrag: (id: number | null) => void;
}

export function Plot(props: Props) {
  const { model, curveHat, curveStar, drawCurves, pins, activePin, pointChange } = props;
  const data = props.data ?? model.data;
  const { W, H, M, samples: N_SAMPLES } = props.compact ? SMALL : LARGE;
  const svgRef = useRef<SVGSVGElement>(null);
  // Offset from cursor to the grabbed point, in data units, so the curve does not jump on grab.
  const drag = useRef<{ id: number; dx: number; dy: number } | null>(null);
  const [hover, setHover] = useState<"curve" | "pin" | null>(null);

  const [x0, x1] = model.xDomain;
  const [y0, y1] = model.yDomain;
  const sx = (x: number) => M.left + ((x - x0) / (x1 - x0)) * (W - M.left - M.right);
  const sy = (y: number) => H - M.bottom - ((y - y0) / (y1 - y0)) * (H - M.top - M.bottom);
  const ix = (px: number) => x0 + ((px - M.left) / (W - M.left - M.right)) * (x1 - x0);
  const iy = (py: number) => y0 + ((H - M.bottom - py) / (H - M.top - M.bottom)) * (y1 - y0);

  const sample = (curve: Curve, nSamples = N_SAMPLES) => {
    const lo = model.clampX(x0);
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= nSamples; i++) {
      const x = lo + ((x1 - lo) * i) / nSamples;
      const y = curve(x);
      if (Number.isFinite(y)) pts.push({ x, y });
    }
    return pts;
  };
  const path = (pts: { x: number; y: number }[]) =>
    "M" + pts.map((p) => `${sx(p.x).toFixed(2)},${sy(p.y).toFixed(2)}`).join("L");

  // Draws do not change while dragging, so compute their paths once.
  const drawPaths = useMemo(() => drawCurves.map((c) => path(sample(c, 120))), [drawCurves, model]);
  const current = curveStar ?? curveHat;
  const hatPts = useMemo(() => sample(curveHat), [curveHat, model]);
  const starPts = curveStar ? sample(curveStar) : null;
  const currentPts = starPts ?? hatPts;

  const toSvg = (e: React.PointerEvent | React.MouseEvent) => {
    const svg = svgRef.current!;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM()!.inverse());
    return { px: pt.x, py: pt.y };
  };

  const hitPin = (px: number, py: number): number | null => {
    let best: number | null = null;
    let bd = HIT_PX;
    for (const p of pins) {
      const d = Math.hypot(sx(p.x) - px, sy(p.y) - py);
      if (d < bd) {
        bd = d;
        best = p.id;
      }
    }
    return best;
  };

  // Nearest point on the current curve within HIT_PX, refined between samples.
  const hitCurve = (px: number, py: number): { x: number; y: number } | null => {
    let bi = -1;
    let bd = HIT_PX;
    currentPts.forEach((p, i) => {
      const d = Math.hypot(sx(p.x) - px, sy(p.y) - py);
      if (d < bd) {
        bd = d;
        bi = i;
      }
    });
    if (bi < 0) return null;
    const lo = currentPts[Math.max(0, bi - 1)].x;
    const hi = currentPts[Math.min(currentPts.length - 1, bi + 1)].x;
    let best = currentPts[bi];
    for (let k = 0; k <= 40; k++) {
      const x = lo + ((hi - lo) * k) / 40;
      const y = current(x);
      if (Math.hypot(sx(x) - px, sy(y) - py) < Math.hypot(sx(best.x) - px, sy(best.y) - py)) best = { x, y };
    }
    return best;
  };

  const clampY = (y: number) => Math.min(y1, Math.max(y0, y));

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const { px, py } = toSvg(e);
    const cx = ix(px);
    const cy = iy(py);
    const pinId = hitPin(px, py);
    if (pinId !== null) {
      const p = pins.find((q) => q.id === pinId)!;
      drag.current = { id: pinId, dx: p.x - cx, dy: p.y - cy };
    } else {
      const c = hitCurve(px, py);
      if (!c) return;
      const id = props.onGrab(c.x, c.y);
      drag.current = { id, dx: c.x - cx, dy: c.y - cy };
    }
    svgRef.current!.setPointerCapture(e.pointerId);
    props.onDrag(drag.current.id);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const { px, py } = toSvg(e);
    const d = drag.current;
    if (d === null) {
      setHover(hitPin(px, py) !== null ? "pin" : hitCurve(px, py) ? "curve" : null);
      return;
    }
    props.onMove(d.id, model.clampX(ix(px) + d.dx), clampY(iy(py) + d.dy));
  };

  const endDrag = () => {
    if (drag.current === null) return;
    drag.current = null;
    props.onDrag(null);
  };

  const onContextMenu = (e: React.MouseEvent) => {
    const { px, py } = toSvg(e);
    const id = hitPin(px, py);
    if (id !== null) {
      e.preventDefault();
      props.onRemove(id);
    }
  };

  const xt = model.xTicks ?? niceTicks(x0, x1);
  const fx = model.formatX ?? ((x: number) => String(x));
  const yt = niceTicks(y0, y1, 5);
  const cursor = activePin !== null ? "grabbing" : hover ? "grab" : "default";

  return (
    <svg
      ref={svgRef}
      className={props.compact ? "plot compact" : "plot"}
      style={{ cursor }}
      viewBox={`0 0 ${W} ${H}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onPointerLeave={() => setHover(null)}
      onContextMenu={onContextMenu}
    >
      <rect x={M.left} y={M.top} width={W - M.left - M.right} height={H - M.top - M.bottom} className="plot-bg" />
      {xt.map((t) => (
        <g key={"x" + t}>
          <line x1={sx(t)} x2={sx(t)} y1={M.top} y2={H - M.bottom} className="grid" />
          <text x={sx(t)} y={H - M.bottom + (props.compact ? 13 : 16)} textAnchor="middle" className="tick">{fx(t)}</text>
        </g>
      ))}
      {yt.map((t) => (
        <g key={"y" + t}>
          <line x1={M.left} x2={W - M.right} y1={sy(t)} y2={sy(t)} className="grid" />
          <text x={M.left - (props.compact ? 4 : 8)} y={sy(t) + 4} textAnchor="end" className="tick">{t}</text>
        </g>
      ))}
      {props.compact ? (
        <text x={M.left} y={14} className="panel-label">{props.title}</text>
      ) : (
        <>
          <text x={(M.left + W - M.right) / 2} y={H - 8} textAnchor="middle" className="axis-label">{model.xLabel}</text>
          <text transform={`translate(16 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">
            {model.yLabel}
          </text>
        </>
      )}

      {drawPaths.map((d, i) => (
        <path key={"draw" + i} d={d} className="curve-draw" />
      ))}
      <path d={path(hatPts)} className={starPts ? "curve-hat faded" : "curve-hat"} />
      {starPts && <path d={path(starPts)} className="curve-star" />}
      <path d={path(currentPts)} className={hover === "curve" && activePin === null ? "curve-hover on" : "curve-hover"} />

      {data.map((d, i) => (
        <circle
          key={i}
          cx={sx(d.x)}
          cy={sy(d.y)}
          r={4}
          fill={pointChange ? changeColor(pointChange[i]) : "var(--ink)"}
          className="datum"
        >
          <title>
            {`x = ${fx(d.x)}, y = ${d.y.toFixed(4)}` +
              (pointChange ? `\nchange in log lik: ${pointChange[i].toFixed(2)}` : "")}
          </title>
        </circle>
      ))}

      {pins.map((p) => (
        <circle
          key={p.id}
          cx={sx(p.x)}
          cy={sy(p.y)}
          r={p.id === activePin ? 7 : 5.5}
          className="pin"
        />
      ))}
    </svg>
  );
}
