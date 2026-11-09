# Simulated serial dilution assay plate shaped like Figure 7 of the paper (the paper's
# plate data are not published). One standard sample of known concentration (relative
# concentration 1) at 14 two-fold dilutions, and 23 unknown samples at dilutions
# 1e-4, 1e-2 and 1e-1. Measurements are log y with
#   y = b1 + b2 / (1 + (c d / b3)^(-b4)),  log y ~ normal(log of that, sigma_j).
# Samples 3 and 23 do not follow the curve shape (as if contaminated): high readings at
# the strongest dilution and a flat top, roughly matching their panels in Figure 7.
import numpy as np

rng = np.random.default_rng(20261110)
b1, b2, b3, b4 = 4.0, 19000.0, 0.04, 1.5

def g(c, d):
    return b1 + b2 / (1 + (c * d / b3) ** (-b4))

rows = []  # (sample, dilution, log y)
for k in range(14):
    d = 2.0 ** (-k)
    rows.append((0, d, np.log(g(1.0, d)) + rng.normal(0, 0.06)))

dils = [1e-4, 1e-2, 1e-1]
log_c = rng.normal(-3.9, 1.0, 23)
log_sigma = rng.normal(np.log(0.15), 0.35, 23)
for j in range(1, 24):
    if j == 3:
        vals = [4.3, 6.5, 6.5]
    elif j == 23:
        vals = [3.6, 6.7, 7.6]
    else:
        vals = [np.log(g(np.exp(log_c[j - 1]), d)) + rng.normal(0, np.exp(log_sigma[j - 1])) for d in dils]
    for d, v in zip(dils, vals):
        rows.append((j, d, v))

print("// sample (0 = standard), dilution, log y")
print("const RAW: [number, number, number][] = [")
for s, d, v in rows:
    print(f"  [{s}, {d:.6g}, {v:.3f}],")
print("];")
