import { useEffect, useMemo, useRef, useState } from "react";
import { Dashboard, type DashRow } from "./Dashboard";
import { dilutionExample } from "./examples/dilution";
import { gpTemperatureExample } from "./examples/gpTemperature";
import { GroupScatter } from "./GroupScatter";
import { parametricExample } from "./examples/parametric";
import type { Curve, Example, Solution } from "./examples/types";
import { golf } from "./models/golf";
import { Plot } from "./Plot";
import { fmt } from "./plotUtil";
import { PointChange } from "./PointChange";
import { Squeal, squealLevel } from "./squeal";
import type { Pin } from "./squealer";

const EXAMPLES: { id: string; label: string; create: () => Example }[] = [
  { id: "golf", label: "Golf putting", create: () => parametricExample(golf) },
  { id: "dilution", label: "Dilution assay", create: dilutionExample },
  { id: "gp-temperature", label: "Body temperature (GP)", create: gpTemperatureExample },
];
const cache = new Map<string, Example>();
function getExample(id: string): Example {
  let ex = cache.get(id);
  if (!ex) {
    ex = EXAMPLES.find((e) => e.id === id)!.create();
    cache.set(id, ex);
  }
  return ex;
}

const squeal = new Squeal();

export default function App() {
  const [exampleId, setExampleId] = useState(() => {
    const h = window.location.hash.slice(1);
    return EXAMPLES.some((e) => e.id === h) ? h : EXAMPLES[0].id;
  });
  const example = useMemo(() => getExample(exampleId), [exampleId]);
  const choose = (id: string) => {
    window.location.hash = id;
    setExampleId(id);
  };

  return (
    <div className="app">
      <header>
        <h1>Squealer</h1>
        <p className="lede">
          Drag a fitted curve away from its best fit and hear what it costs. After Gelman, Jaffe, Carlson and
          Greengard, <a href="https://arxiv.org/abs/2606.29842">The Squealer: sensification of model exploration and model misfit</a> (2026).
        </p>
        <nav className="tabs">
          {EXAMPLES.map((e) => (
            <button key={e.id} className={e.id === exampleId ? "tab on" : "tab"} onClick={() => choose(e.id)}>
              {e.label}
            </button>
          ))}
        </nav>
      </header>
      <ExampleView key={example.id} example={example} />
    </div>
  );
}

function ExampleView({ example }: { example: Example }) {
  const [pins, setPins] = useState<Pin[]>([]);
  const [activePin, setActivePin] = useState<number | null>(null);
  const [muted, setMuted] = useState(squeal.muted);
  const nextId = useRef(1);
  const warm = useRef<unknown>(example.hat.warm);

  const star: Solution | null = useMemo(() => {
    if (pins.length === 0) {
      warm.current = example.hat.warm;
      return null;
    }
    const s = example.solve(pins, warm.current);
    warm.current = s.warm;
    return s;
  }, [pins, example]);

  const hat = example.hat;
  const cost = star ? star.cost : 0;
  const pointChange = star ? star.pointLL.map((v, i) => v - hat.pointLL[i]) : null;
  const dragging = activePin !== null;

  useEffect(() => {
    squeal.muted = muted;
    if (star) squeal.update(cost, dragging);
    else squeal.silence();
  }, [star, cost, dragging, muted]);
  useEffect(() => () => squeal.silence(), []);

  const onGrab = (x: number, y: number, panel?: number) => {
    squeal.start();
    const id = nextId.current++;
    setPins((ps) => [...ps, { id, x, y, panel }].slice(-example.maxPins));
    return id;
  };
  const onMove = (id: number, x: number, y: number) =>
    setPins((ps) => ps.map((p) => (p.id === id ? { ...p, x, y } : p)));
  const onRemove = (id: number) => setPins((ps) => ps.filter((p) => p.id !== id));
  const onDrag = (id: number | null) => {
    squeal.start();
    setActivePin(id);
  };

  const dashParams = example.dashParams ?? example.paramNames.map((_, k) => k);
  const nParams = dashParams.length;
  const dashRows: DashRow[] = [
    ...dashParams.map((k) => ({
      name: example.paramNames[k],
      draws: example.dashDraws.map((d) => d.theta[k]),
      hat: hat.theta[k],
      star: star ? star.theta[k] : null,
    })),
    ...example.totalNames.map((name, k) => ({
      name,
      draws: example.dashDraws.map((d) => d.totals[k]),
      hat: hat.totals[k],
      star: star ? star.totals[k] : null,
    })),
  ];
  const fx = example.formatX ?? ((x: number) => fmt(x, 2));

  // Small multiples: per-panel curves, kept stable so panels only recompute what changed.
  const panels = example.panels;
  const panelCurve = (k: number) => panels?.[k].curve ?? k;
  const hatCurves = useMemo(
    () => (panels ?? []).map((_, k) => ((x: number) => example.hat.curve(x, panelCurve(k))) as Curve),
    [example],
  );
  const drawCurvesByPanel = useMemo(
    () => (panels ?? []).map((_, k) => example.drawCurves.map((c) => ((x: number) => c(x, panelCurve(k))) as Curve)),
    [example],
  );
  const level = squealLevel(cost);

  return (
    <>
      <main>
        <section className="plot-panel">
          <div className="panel-title">
            <strong>{example.title}</strong>
            <span>{example.description}</span>
          </div>
          {panels ? (
            <div className="panel-grid">
              {panels.map((panel, k) => {
                const cIdx = panelCurve(k);
                const starCurve = star ? (x: number) => star.curve(x, cIdx) : null;
                return (
                  <Plot
                    key={k}
                    compact
                    title={panel.title}
                    model={example}
                    data={panel.dataIndex.map((i) => example.data[i])}
                    curveHat={hatCurves[k]}
                    curveStar={starCurve}
                    drawCurves={drawCurvesByPanel[k]}
                    pins={pins.filter((p) => p.panel === cIdx)}
                    activePin={activePin}
                    pointChange={pointChange ? panel.dataIndex.map((i) => pointChange[i]) : null}
                    onGrab={(x, y) => onGrab(x, y, cIdx)}
                    onMove={onMove}
                    onRemove={onRemove}
                    onDrag={onDrag}
                  />
                );
              })}
            </div>
          ) : (
            <Plot
              model={example}
              curveHat={hat.curve}
              curveStar={star ? star.curve : null}
              drawCurves={example.drawCurves}
              pins={pins}
              activePin={activePin}
              pointChange={pointChange}
              onGrab={onGrab}
              onMove={onMove}
              onRemove={onRemove}
              onDrag={onDrag}
            />
          )}
          <p className="hint">
            Grab the curve anywhere and drag it; it passes through the point you hold, and the parameters follow
            as the most probable values given that constraint. On release the point stays pinned; drag a pin to
            move it, right-click it to release it. {example.pinNote}
          </p>
          <div className="legend">
            <span><i className="sw sw-hat" /> best fit</span>
            <span><i className="sw sw-draw" /> posterior draws</span>
            <span><i className="sw sw-star" /> dragged fit</span>
            <span><i className="sw-pin" /> pin</span>
          </div>
        </section>

        <aside>
          <div className="card">
            <div className="cost-label">Drop in log posterior</div>
            <div className="cost-value">{fmt(cost, 2)}</div>
            <div className="meter"><div style={{ width: `${Math.min(100, (level / 4.5) * 100)}%` }} /></div>
            {star && star.costParts.length > 0 && (
              <ul className="cost-parts">
                {star.costParts.map((p) => (
                  <li key={p.name}><span>{p.name}</span><span>{fmt(p.value, 2)}</span></li>
                ))}
              </ul>
            )}
            {star && star.maxViolation > 0.05 && (
              <p className="warn">
                No parameter values put the curve through every pin; it misses by up to {star.maxViolation.toFixed(1)} posterior sd.
              </p>
            )}
            <div className="controls">
              <button onClick={() => setMuted((m) => !m)}>{muted ? "Unmute" : "Mute"}</button>
              <button onClick={() => setPins([])} disabled={pins.length === 0}>Reset</button>
            </div>
          </div>

          {pins.length > 0 && (
            <div className="card">
              <div className="cost-label">Pins</div>
              <ul className="pin-list">
                {pins.map((p) => (
                  <li key={p.id}>
                    <span>
                      {p.panel !== undefined && example.panels ? `${example.panels.find((q, k) => (q.curve ?? k) === p.panel)?.title}: ` : ""}
                      x = {fx(p.x)}, y = {fmt(p.y, 3)}
                    </span>
                    <button onClick={() => onRemove(p.id)} title="Release this pin">×</button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>
      </main>

      <section className="lower">
        <div className="card">
          <div className="panel-title">
            <strong>Dashboard</strong>
            <span>
              Posterior density of each quantity (light blue, {example.drawNote}), the best fit θ̂ (blue) and the
              dragged fit θ* (red). Each row is centered at θ̂; numbers on the right give θ̂ and the shift of θ*
              in posterior sd.
            </span>
          </div>
          <Dashboard rows={dashRows} nParams={nParams} />
        </div>
        {example.groupScatter && (
          <div className="card">
            <div className="panel-title">
              <strong>Per-sample estimates</strong>
              <span>
                As in the paper's Figure 9: best-fit values ±1 posterior sd (blue) for each sample; samples whose
                estimates the drag moves are shown at their new values (red).
              </span>
            </div>
            <GroupScatter
              spec={example.groupScatter}
              hat={hat.theta}
              star={star ? star.theta : null}
              draws={example.dashDraws.map((d) => d.theta)}
            />
          </div>
        )}
        <div className="card">
          <div className="panel-title">
            <strong>Change in fit by data point</strong>
            <span>log p(yᵢ | xᵢ, θ*) − log p(yᵢ | xᵢ, θ̂). Negative values are points the dragged curve fits worse.</span>
          </div>
          <PointChange model={example} change={pointChange} />
        </div>
      </section>
    </>
  );
}
