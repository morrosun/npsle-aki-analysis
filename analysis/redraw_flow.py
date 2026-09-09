# -*- coding: utf-8 -*-
"""Redraw the cohort flow figure with text wrapping so no label overflows its box."""
import os
import matplotlib.pyplot as plt

BASE = os.environ.get("SLECI_BASE", ".")
ANA = os.path.join(BASE, "analysis")
OUT = os.path.join(BASE, "output")

def draw_flow(save_paths):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    def box(x, y, w2, h2, text, fc="#eef3f8", fs=8.6):
        ax.add_patch(plt.Rectangle((x - w2 / 2, y - h2 / 2), w2, h2,
                                    fc=fc, ec="#2166ac", lw=1.2, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                zorder=3, linespacing=1.15)

    def arrow(x1, y1, x2, y2, color="#2166ac"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    # Top box
    box(5.0, 9.0, 8.5, 0.9,
        "Three critical-care databases screened for SLE with \u22651 ICU stay\n"
        "Total SLE ICU stays = 809")
    arrow(5.0, 8.55, 5.0, 7.85)

    # Database boxes
    box(2.0, 7.1, 2.6, 1.1,
        "MIMIC-IV\nn = 550\n(derivation)", fc="#dbe7f3")
    box(5.0, 7.1, 2.6, 1.1,
        "eICU-CRD\nn = 210\n(external validation)", fc="#e4efdc")
    box(8.0, 7.1, 2.6, 1.1,
        "NWICU\nn = 49\n(descriptive only)", fc="#f3e9dc")

    for xc in (2.0, 5.0, 8.0):
        arrow(xc, 6.55, xc, 5.75, color="#888")

    # Prevalence boxes
    box(2.0, 4.95, 2.6, 1.2,
        "Narrow NPSLE\n92 (16.7%)\nBroad 37.1%")
    box(5.0, 4.95, 2.6, 1.2,
        "Narrow NPSLE\n16 (7.6%)\nBroad 27.6%")
    box(8.0, 4.95, 2.6, 1.2,
        "Narrow NPSLE\n5 (10.2%)\nBroad 16.3%")

    for xc in (2.0, 5.0, 8.0):
        arrow(xc, 4.35, xc, 3.75, color="#888")

    # Bottom summary box — wrapped to stay inside
    bottom_text = (
        "Pooled narrow prevalence 11.4% (95% CI 6.2\u201320.1%)\n"
        "Pooled broad prevalence 28.4% (95% CI 19.6\u201339.2%)\n"
        "Primary analysis: MIMIC-IV (logistic regression + propensity-score matching);\n"
        "eICU external validation; cross-DB random-effects meta-analysis"
    )
    box(5.0, 2.75, 8.5, 1.45, bottom_text, fc="#f7f7f7")

    ax.set_title("Study cohort flow", fontweight="bold", fontsize=11)
    plt.tight_layout()

    for p in save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print("Saved:", p)
    plt.close()

if __name__ == "__main__":
    draw_flow([os.path.join(ANA, "flow.png"), os.path.join(OUT, "flow.png")])
