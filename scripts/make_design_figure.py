"""Draw the agent-design figure for Lab Report 2 (template item C1).

Usage:
    python scripts/make_design_figure.py            # -> reports/figures/agent_design.{pdf,png}

Layout follows the Lab 1 debrief's version B, drawn as one loop: the tools run up the left
(optimizer -> train -> evaluate -> target check), the target check hands over to the one LLM
agent in the centre, and the agent's decision goes down through the validator and back to the
optimizer. The human gate and the no-agent baseline are on the right; the literature agent is
below, outside the research loop. Each box says whether it is built or planned. The script fails
if any text runs outside its box.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures"

INK, MUTED = "#1f2328", "#57606a"
TOOL = {"fc": "#f3f4f6", "ec": "#1f2328", "ls": "-"}
LLM = {"fc": "#fae3d0", "ec": "#eb6834", "ls": "-"}
HUMAN = {"fc": "white", "ec": "#1f2328", "ls": (0, (4, 3))}
ACCENT = "#c4501a"   # arrows that carry the agent's decision back into the loop

FS_TITLE, FS_BODY, FS_LABEL = 8, 7, 6.5
PAD = 0.12           # minimum gap between text and box edge, in data units

boxes = []           # (extent, texts) for the overflow check
labels = []          # arrow labels, which must stay clear of every box


def box(ax, x, y, w, h, style, title, body, status=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=style["fc"], ec=style["ec"], ls=style["ls"], lw=1.2))
    texts = [ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=FS_TITLE,
                     fontweight="bold", color=INK),
             ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=FS_BODY,
                     color=INK, linespacing=1.35, multialignment="center")]
    if status:
        texts.append(ax.text(x + w - 0.15, y + 0.1, status, ha="right", va="bottom",
                             fontsize=FS_LABEL, color=MUTED, style="italic"))
    boxes.append(((x, y, w, h), texts))


def arrow(ax, start, end, label="", color=INK, label_xy=None, ha="center", ls="-", both=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<|-|>" if both else "-|>",
                                 mutation_scale=10, color=color, lw=1.1, ls=ls,
                                 shrinkA=2, shrinkB=2))
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        labels.append(ax.text(lx, ly, label, ha=ha, va="center", fontsize=FS_LABEL, color=color,
                              style="italic", linespacing=1.2))


def check_overflow(fig, ax):
    """Raise if any text comes closer than PAD to its box edge or overlaps another text in it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()
    problems = []
    for (x, y, w, h), texts in boxes:
        extents = []
        for t in texts:
            (x0, y0), (x1, y1) = to_data.transform(t.get_window_extent(renderer).get_points())
            extents.append((x0, y0, x1, y1))
            if x0 < x + PAD or x1 > x + w - PAD or y0 < y + 0.05 or y1 > y + h - 0.05:
                problems.append(f"text runs outside its box: {t.get_text()[:40]!r}")
        for i in range(len(extents)):
            for j in range(i + 1, len(extents)):
                a, b = extents[i], extents[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    problems.append(f"texts overlap: {texts[i].get_text()[:25]!r} / "
                                    f"{texts[j].get_text()[:25]!r}")
    for t in labels:
        (x0, y0), (x1, y1) = to_data.transform(t.get_window_extent(renderer).get_points())
        for (x, y, w, h), _ in boxes:
            if x0 < x + w + 0.05 and x - 0.05 < x1 and y0 < y + h + 0.05 and y - 0.05 < y1:
                problems.append(f"arrow label touches a box: {t.get_text()!r}")
    if problems:
        raise SystemExit("Figure layout problems:\n  " + "\n  ".join(problems))


def main():
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.set_xlim(-0.1, 12.75)
    ax.set_ylim(-2.35, 8.45)
    ax.axis("off")
    LX, LW = 0.0, 3.3      # left column: the deterministic loop, read bottom to top
    CX, CW = 4.55, 3.7     # centre: the agent, with the validator below it
    RX, RW = 9.55, 3.1     # right column: human gate and baseline
    H, GAP = 1.75, 0.4     # height of and gap between the left-column boxes
    rows = [0.1 + i * (H + GAP) for i in range(4)]   # optimizer, train, evaluate, target check

    # Left: the loop's tools, bottom to top.
    box(ax, LX, rows[0], LW, H, TOOL, "Tool 1 · Optimizer",
        "Optuna (TPE): 5 training\nruns per round", "built")
    box(ax, LX, rows[1], LW, H, TOOL, "Tool 2 · Train",
        "train(config) → run_id;\nseeded, locked split", "built")
    box(ax, LX, rows[2], LW, H, TOOL, "Tool 3 · Evaluate",
        "per-family errors and\nPD check → eval.json", "built")
    box(ax, LX, rows[3], LW, H, TOOL, "Tool 4 · Target check",
        "target met, budget used\nor no progress → stop", "built")
    mid = LX + LW / 2
    for (lo, label) in zip(rows[:3], ("config", "run_id", "eval.json")):
        arrow(ax, (mid, lo + H), (mid, lo + H + GAP), label, label_xy=(mid + 0.12, lo + H + GAP / 2),
              ha="left")

    # Centre: the one LLM decision, then the validator.
    agent_y = 3.5
    box(ax, CX, agent_y, CW, rows[3] + H - agent_y, LLM, "Diagnostician (LLM)",
        "claude-sonnet-5-5, claude -p;\nJSON schema, no own tools\n\n"
        "Reads eval.json, trials.json\nand its own decision log\n\n"
        "Decides one of:\n(a) retune → new ranges\n(b) change structure → one\n"
        "menu option, kept afterwards\n(c) escalate → human", "planned")
    box(ax, CX, 0.9, CW, 1.4, TOOL, "Tool 5 · Validator",
        "checks schema, menu option\nand budget (planned)")

    gap_l = (LX + LW + CX) / 2
    arrow(ax, (LX + LW, rows[3] + H / 2), (CX, rows[3] + H / 2), "continue",
          label_xy=(gap_l, rows[3] + H / 2 + 0.2))
    arrow(ax, (CX + CW / 2, agent_y), (CX + CW / 2, 2.3), "decision", color=ACCENT,
          label_xy=(CX + CW / 2 + 0.12, 2.9), ha="left")
    arrow(ax, (CX, 1.3), (LX + LW, 1.3), "apply", color=ACCENT, label_xy=(gap_l, 1.5))

    # Right: human gate and the no-agent baseline.
    box(ax, RX, 6.4, RW, 1.9, HUMAN, "Human gate",
        "approves structural\nchanges and the\ntest-set unlock;\nanswers escalations")
    box(ax, RX, 1.8, RW, 3.9, TOOL, "Baseline, no agent",
        "Optuna alone searches\nhyperparameters and\nthe structural menu,\n"
        "same tools, same\n20-run budget\n\n"
        "Both run on the 64\nlocked test families,\n3 repeats each", "built")
    gap_r = (CX + CW + RX) / 2
    arrow(ax, (CX + CW, 7.8), (RX, 7.8), "(b), (c)", label_xy=(gap_r, 8.02))
    arrow(ax, (RX, 6.95), (CX + CW, 6.95), "reply", label_xy=(gap_r, 6.73))
    arrow(ax, (CX + CW, 4.1), (RX, 4.1), "compared", label_xy=(gap_r, 4.32), ls=(0, (3, 2)),
          both=True)

    # Outside the research loop.
    ax.plot([0, 12.65], [-0.42, -0.42], color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.text(12.65, -0.34, "outside the research loop", ha="right", va="bottom", fontsize=FS_LABEL,
            color=MUTED, style="italic")
    # The literature (retrieval) agent from Labs 4/5, with its own human check.
    lit_w, check_x = 4.6, 5.55
    box(ax, LX, -2.3, lit_w, 1.65, {**LLM, "ls": (0, (4, 3))}, "Literature agent (support)",
        "retrieval over my own paper corpus;\nreturns claims with page citations", "built")
    box(ax, check_x, -2.3, CX + CW - check_x, 1.65, HUMAN, "Human check",
        "verifies each page\ncitation by hand")
    arrow(ax, (LX + lit_w, -1.475), (check_x, -1.475), "claims",
          label_xy=((LX + lit_w + check_x) / 2, -1.27))

    # Legend.
    for i, (style, label) in enumerate([(LLM, "LLM agent"), (TOOL, "deterministic tool / code"),
                                        (HUMAN, "human checkpoint")]):
        y = -1.15 - i * 0.33
        ax.add_patch(FancyBboxPatch((RX, y - 0.1), 0.4, 0.2, boxstyle="square,pad=0",
                                    fc=style["fc"], ec=style["ec"], ls=style["ls"], lw=1.0))
        ax.text(RX + 0.55, y, label, va="center", fontsize=FS_LABEL, color=INK)

    check_overflow(fig, ax)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "agent_design.pdf", bbox_inches="tight")
    fig.savefig(OUT / "agent_design.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT / 'agent_design.pdf'} and .png")


if __name__ == "__main__":
    main()
