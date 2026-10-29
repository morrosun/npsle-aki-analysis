"""v31 审计 2：eICU 修订暴露（14 例）后的跨库合并估计。"""
import os
import json
import numpy as np

BASE = os.environ.get("SLECI_BASE", ".")
OUT = []
def log(*a):
    OUT.append(" ".join(str(x) for x in a))

def sesd(orv, lo, hi):
    return (np.log(hi) - np.log(lo)) / (2 * 1.96)

def dl(items):
    """DerSimonian–Laird random effects on log scale. items=[(label, OR, LO, HI)]"""
    y = np.array([np.log(x[1]) for x in items])
    se = np.array([sesd(*x[1:]) for x in items])
    w = 1 / se ** 2
    fe = (w * y).sum() / w.sum()
    q = (w * (y - fe) ** 2).sum()
    k = len(items)
    dfree = k - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - dfree) / c) if c > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    re = (wr * y).sum() / wr.sum()
    re_se = 1 / np.sqrt(wr.sum())
    i2 = max(0.0, (q - dfree) / q) * 100 if q > 0 else 0.0
    for (lab, orv, lo, hi), wi, wri in zip(items, w, wr):
        log(f"   {lab}: OR={orv} ({lo}-{hi})  FE weight={100*wi/w.sum():.1f}%  RE weight={100*wri/wr.sum():.1f}%")
    log(f"   Q={q:.4f} df={dfree} tau2={tau2:.5f} I2={i2:.1f}")
    return np.exp(re), np.exp(re - 1.96 * re_se), np.exp(re + 1.96 * re_se), round(i2, 1)

log("== 1. AKI 主分析（eICU 16 例暴露, aOR 3.17）==")
base = [("MIMIC-IV", 2.94, 1.59, 5.44), ("eICU", 3.17, 1.01, 10.00)]
r0 = dl(base)
log("   pooled = %.2f (%.2f-%.2f) I2=%.1f" % r0)

log("\n== 2. 敏感性：eICU 两例 encephalitis-only 归为未暴露（14 例, aOR 2.50）==")
alt = [("MIMIC-IV", 2.94, 1.59, 5.44), ("eICU", 2.50, 0.76, 8.28)]
r1 = dl(alt)
log("   pooled = %.2f (%.2f-%.2f) I2=%.1f" % r1)

log("\n== 3. 交叉核对：若 MIMIC 也用 narrowed 定义，权重变化 ==")
log("   (未重算 MIMIC，仅提示权重对 eICU 修订不敏感)")

json.dump(dict(primary=dict(OR=round(float(r0[0]), 2), LO=round(float(r0[1]), 2),
                            HI=round(float(r0[2]), 2)),
               encephalitis_recoded=dict(OR=round(float(r1[0]), 2), LO=round(float(r1[1]), 2),
                                         HI=round(float(r1[2]), 2))),
          open(f"{BASE}/analysis/v31_pooled.json", "w"), indent=2)
open(f"{BASE}/analysis/v31_pooled_report.txt", "w", encoding="utf-8").write("\n".join(OUT))
print("\n".join(OUT))
