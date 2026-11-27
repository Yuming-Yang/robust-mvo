"""Print the README results table for the N ~ T comparison.

    uv run python scripts/n_close_to_t.py

N = 100 assets, 20 seeds per ratio, medians reported. Deterministic: every
random draw is seeded.
"""

from robust_mvo.experiments import run, summarize

REGIMES = {
    "strong market factor": {"market_vol": 0.01, "style_vol": 0.005},
    "weak factors": {"market_vol": 0.003, "style_vol": 0.003},
}
RATIOS = (0.25, 0.5, 0.9, 1.0, 1.5)


def fmt(x: float) -> str:
    if x != x or abs(x) == float("inf"):
        return "inf"
    if abs(x) >= 1e4 or (x != 0 and abs(x) < 1e-2):
        return f"{x:.1e}"
    return f"{x:.2f}"


for name, kw in REGIMES.items():
    print(f"\n### {name}\n")
    print("| N/T | method | condition no. | rel. Frobenius error | efficiency | risk ratio |")
    print("| --- | --- | --- | --- | --- | --- |")
    for s in summarize(run(ratios=RATIOS, n_seeds=20, **kw)):
        print(
            f"| {s['q']:g} | {s['method']} | {fmt(s['cond'])} | {fmt(s['frob_err'])} "
            f"| {fmt(s['efficiency'])} | {fmt(s['risk_ratio'])} |"
        )
