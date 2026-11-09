# Simulated thermometer readings shaped like Figure 11a of the paper.
import numpy as np
rng = np.random.default_rng(20261109)
# readings every ~4.5 minutes from 6:00 to 12:00, a little irregular
t = np.sort(np.concatenate([np.arange(6.0, 12.0001, 0.075) + rng.normal(0, 0.008, 81)]))
t = np.clip(t, 6.0, 12.0)
f = 98.95 + 1.03 / (1 + np.exp(-(t - 7.45) / 0.26))
# noise small early (resting, steady readings), larger later
sd = 0.012 + 0.09 / (1 + np.exp(-(t - 8.3) / 0.3))
y = f + rng.normal(0, 1, t.size) * sd
# upward drift in the last 45 minutes: a handful of readings above 100.2 F
late = t > 11.25
y[late] += np.linspace(0.05, 0.36, late.sum()) * (rng.uniform(0, 1, late.sum()) > 0.4)
print("n", t.size, "late", late.sum(), "max", y.max().round(3), "n>100.2", (y > 100.2).sum())
print("const T = [" + ", ".join(f"{v:.3f}" for v in t) + "];")
print("const Y = [" + ", ".join(f"{v:.3f}" for v in y) + "];")
