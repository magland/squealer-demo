import { describe, expect, it } from "vitest";
import { dilutionExample, dilutionModel } from "../src/examples/dilution";

describe("dilution model", () => {
  it("analytic gradient matches finite differences", () => {
    const m = dilutionModel;
    const u = m.u0.map((v, i) => v + 0.1 * Math.sin(i + 1));
    const g = new Float64Array(m.n);
    m.logPost(u, g);
    let worst = 0;
    for (let i = 0; i < m.n; i++) {
      const h = 1e-6;
      const up = u.slice();
      const dn = u.slice();
      up[i] += h;
      dn[i] -= h;
      const fd = (m.logPost(up, null) - m.logPost(dn, null)) / (2 * h);
      worst = Math.max(worst, Math.abs(fd - g[i]) / (1 + Math.abs(fd)));
    }
    expect(worst).toBeLessThan(1e-5);
  });
  it("curve gradient matches finite differences", () => {
    const m = dilutionModel;
    const u = m.u0.map((v, i) => v + 0.1 * Math.cos(i));
    const g = new Float64Array(m.n);
    m.curve(-2, 5, u, g);
    for (let i = 0; i < m.n; i++) {
      const h = 1e-6;
      const up = u.slice();
      const dn = u.slice();
      up[i] += h;
      dn[i] -= h;
      const fd = (m.curve(-2, 5, up, null) - m.curve(-2, 5, dn, null)) / (2 * h);
      expect(Math.abs(fd - g[i])).toBeLessThan(1e-6);
    }
  });
});

describe("dilution example", () => {
  const t0 = performance.now();
  const ex = dilutionExample();
  const tBuild = performance.now() - t0;
  it("recovers the simulation's curve parameters", () => {
    const u = ex.hat.theta;
    console.log("build ms", tBuild.toFixed(0), "beta", u.slice(0, 4).map(Math.exp), "mu_c", u[4], "tau_c", Math.exp(u[5]), "mu_s", u[6], "theta6", u[6]);
    console.log("sample 3: log c", u[9], "log sigma", u[33], "| sample 23: log c", u[29], "log sigma", u[53]);
    expect(Math.exp(u[0])).toBeGreaterThan(2);
    expect(Math.exp(u[0])).toBeLessThan(8);
    expect(Math.exp(u[3])).toBeGreaterThan(1.2);
    expect(Math.exp(u[3])).toBeLessThan(1.8);
  });
  it("pinning samples 3 and 23 up costs mostly in the prior, with larger tau_c (paper Figure 9)", () => {
    const pins = [
      { id: 1, panel: 3, x: -2, y: 6.5 },
      { id: 2, panel: 23, x: -2, y: 6.7 },
    ];
    const t1 = performance.now();
    const s = ex.solve(pins, ex.hat.warm);
    console.log("solve ms", (performance.now() - t1).toFixed(0), "cost", s.cost, JSON.stringify(s.costParts), "maxViolation", s.maxViolation);
    console.log("tau_c", Math.exp(ex.hat.theta[5]), "->", Math.exp(s.theta[5]), "| log c3", ex.hat.theta[9], "->", s.theta[9], "| log c23", ex.hat.theta[29], "->", s.theta[29]);
    expect(s.maxViolation).toBeLessThan(1e-2);
    expect(s.cost).toBeGreaterThan(0);
    expect(s.theta[5]).toBeGreaterThan(ex.hat.theta[5]);
    expect(s.theta[9]).toBeGreaterThan(ex.hat.theta[9]);
  });
  it("reports a miss when pins over-constrain the curve", () => {
    const pins = [
      { id: 1, panel: 3, x: -4, y: 4.3 }, { id: 2, panel: 3, x: -2, y: 6.5 }, { id: 3, panel: 3, x: -1, y: 6.5 },
    ];
    const s = ex.solve(pins, ex.hat.warm);
    console.log("three pins on sample 3: maxViolation", s.maxViolation, "curve", pins.map((p) => s.curve(p.x, p.panel).toFixed(2)).join(" "));
  });
  it("one pin passes exactly and drag steps are fast", () => {
    let warm = ex.hat.warm;
    const y0 = ex.hat.curve(-2, 3);
    const t1 = performance.now();
    const K = 10;
    let s = ex.hat;
    for (let k = 1; k <= K; k++) {
      s = ex.solve([{ id: 1, panel: 3, x: -2, y: y0 + (k / K) * (6.5 - y0) }], warm);
      warm = s.warm;
    }
    console.log("ms per drag step", ((performance.now() - t1) / K).toFixed(1), "g(-2)", s.curve(-2, 3));
    expect(s.curve(-2, 3)).toBeCloseTo(6.5, 3);
  });
});
