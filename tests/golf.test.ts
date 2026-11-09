import { describe, expect, it } from "vitest";
import { golf } from "../src/models/golf";
import { fitBase, fitConstrained, summarize } from "../src/squealer";
import { erfc } from "../src/stats/special";

describe("special functions", () => {
  it("erfc matches reference values", () => {
    expect(erfc(0.5)).toBeCloseTo(0.4795001221869535, 14);
    expect(erfc(2.5) / 4.06952017444959e-4).toBeCloseTo(1, 12);
    expect(erfc(8) / 1.122429717298293e-29).toBeCloseTo(1, 12);
    expect(erfc(-1)).toBeCloseTo(1.842700792949715, 14);
  });
});

describe("golf", () => {
  const base = fitBase(golf);
  const cold = { u: base.u, lambda: new Map<number, number>() };

  it("finds the posterior mode reported by the prototype", () => {
    const [sa, sd] = base.summary.theta;
    expect(sa).toBeCloseTo(0.0133, 4);
    expect(sd).toBeCloseTo(0.1372, 3);
  });

  it("grabbing the curve without moving it leaves theta unchanged", () => {
    const y = golf.curve(20, base.summary.theta);
    const fit = fitConstrained(golf, base, [{ id: 1, x: 20, y }], cold);
    const theta = golf.toTheta(fit.u);
    expect(theta[0] / base.summary.theta[0]).toBeCloseTo(1, 6);
    expect(theta[1] / base.summary.theta[1]).toBeCloseTo(1, 6);
  });

  it("passes exactly through a dragged point and lowers the log posterior", () => {
    const fit = fitConstrained(golf, base, [{ id: 1, x: 20, y: 0.145 }], cold);
    const theta = golf.toTheta(fit.u);
    const s = summarize(golf, theta);
    console.log("theta*", theta, "g(20)", golf.curve(20, theta), "drop", base.summary.logPost - s.logPost, "viol", fit.maxViolation);
    expect(golf.curve(20, theta)).toBeCloseTo(0.145, 5);
    expect(s.logPost).toBeLessThan(base.summary.logPost);
  });

  it("follows a drag in small warm-started steps to the same answer", () => {
    let warm = cold;
    const t0 = performance.now();
    const y0 = golf.curve(20, base.summary.theta);
    const N = 40;
    for (let k = 1; k <= N; k++) warm = fitConstrained(golf, base, [{ id: 1, x: 20, y: y0 + ((0.145 - y0) * k) / N }], warm);
    console.log("ms per drag step", ((performance.now() - t0) / N).toFixed(2));
    const direct = fitConstrained(golf, base, [{ id: 1, x: 20, y: 0.145 }], cold);
    const a = golf.toTheta(warm.u);
    const b = golf.toTheta(direct.u);
    expect(a[0] / b[0]).toBeCloseTo(1, 4);
    expect(a[1] / b[1]).toBeCloseTo(1, 4);
  });

  it("two pins determine the two-parameter curve", () => {
    const pins = [{ id: 1, x: 10, y: 0.43 }, { id: 2, x: 30, y: 0.08 }];
    const fit = fitConstrained(golf, base, pins, cold);
    const theta = golf.toTheta(fit.u);
    console.log("two pins theta*", theta, "viol", fit.maxViolation);
    expect(golf.curve(10, theta)).toBeCloseTo(0.43, 5);
    expect(golf.curve(30, theta)).toBeCloseTo(0.08, 5);
  });
});
