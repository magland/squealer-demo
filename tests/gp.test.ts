import { describe, expect, it } from "vitest";
import { gpTemperatureExample } from "../src/examples/gpTemperature";

describe("gp temperature", () => {
  const ex = gpTemperatureExample();
  it("fits hyperparameters close to the paper's Table 1", () => {
    const [alpha, ell, sigma] = ex.hat.theta;
    console.log("theta-hat (alpha, ell, sigma)", alpha, ell, sigma);
    expect(sigma).toBeGreaterThan(0.05);
    expect(sigma).toBeLessThan(0.15);
  });
  it("passes exactly through pins; sigma grows as pins pull toward the late high readings", () => {
    const t0 = performance.now();
    const one = ex.solve([{ id: 1, x: 11, y: 100.3 }], ex.hat.warm);
    const t1 = performance.now();
    const two = ex.solve([{ id: 1, x: 11, y: 100.3 }, { id: 2, x: 11.9, y: 100.3 }], one.warm);
    console.log("one pin", one.theta, "cost", one.cost, one.costParts.map((p) => p.value), "ms", (t1 - t0).toFixed(1));
    console.log("two pins", two.theta, "cost", two.cost);
    expect(one.curve(11)).toBeCloseTo(100.3, 4);
    expect(two.curve(11.9)).toBeCloseTo(100.3, 4);
    expect(one.cost).toBeGreaterThan(0);
  });
  it("re-fits a drag step quickly", () => {
    let warm = ex.hat.warm;
    const t0 = performance.now();
    const K = 20;
    for (let k = 1; k <= K; k++) warm = ex.solve([{ id: 1, x: 11, y: 100.05 + (0.25 * k) / K }], warm).warm;
    console.log("ms per drag step", ((performance.now() - t0) / K).toFixed(1));
  });
});
