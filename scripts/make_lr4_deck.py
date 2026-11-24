"""Build the Lab Report 4 demo deck from the Lab Report 3 deck (same course template and styles).

Usage (repo root; needs python-pptx and Pillow, e.g. the system Python):
    python scripts/make_lr4_deck.py          # -> reports/lab_report_4_demo.pptx

Inputs: reports/lab_report_3_demo.pptx (slides reused for their layout), reports/figures/
agent_design_lr4.png (scripts/make_design_figure.py --report 4) and reports/figures/lr4/*.png
(scripts/make_lr4_figures.py). Every number on a slide comes from LOG.md or a run folder; slides
that wait on the planted-failure study say REWORK and are filled in once it has run.
"""
from __future__ import annotations

import copy
import io
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "lab_report_3_demo.pptx"
OUT = ROOT / "reports" / "lab_report_4_demo.pptx"
FIG = ROOT / "reports" / "figures"
DESIGN = FIG / "agent_design_lr4.png"
GREEN, RED, AMBER, BLACK = "188038", "D93025", "B06000", "000000"

# Pixel boxes of each module in agent_design_lr4.png (1125 x 964), for the section crops.
CROPS = {"checks": (20, 25, 310, 190), "harness": (20, 790, 425, 945),
         "agent": (409, 25, 730, 450), "rules": (827, 483, 1105, 703)}


# ---------------------------------------------------------------- helpers

def set_paragraphs(frame, paras):
    """Replace a text frame's text, one entry per paragraph, keeping the first run's format of
    each existing paragraph (extra paragraphs copy the last one). An entry is a string or a list
    of (text, {bold, color}) runs."""
    ps = frame.paragraphs
    while len(ps) < len(paras):
        frame._txBody.append(copy.deepcopy(ps[-1]._p))
        ps = frame.paragraphs
    for p in ps[len(paras):]:
        frame._txBody.remove(p._p)
    for p, entry in zip(frame.paragraphs, paras):
        runs = entry if isinstance(entry, list) else [(entry, {})]
        while len(p.runs) < len(runs):
            p._p.append(copy.deepcopy(p.runs[-1]._r))
        for r in p.runs[len(runs):]:
            p._p.remove(r._r)
        for r, (text, fmt) in zip(p.runs, runs):
            r.text = text
            if "bold" in fmt:
                r.font.bold = fmt["bold"]
            if "color" in fmt:
                r.font.color.rgb = RGBColor.from_string(fmt["color"])


def shape(slide, startswith):
    for sh in slide.shapes:
        if sh.name.startswith(startswith):
            return sh
    raise KeyError(startswith)


def title(slide, text):
    set_paragraphs(slide.shapes.title.text_frame if slide.shapes.title else shape(slide, "Google Shape").text_frame, [text])


def text_box(slide, startswith_text, paras):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith(startswith_text):
            set_paragraphs(sh.text_frame, paras)
            return sh
    raise KeyError(startswith_text)


def pictures(slide):
    return [sh for sh in slide.shapes if sh.shape_type == 13]


def replace_picture(pic, png_bytes, fit=True):
    """Swap the image of a picture shape; keep it inside the old box, centred, at its own aspect."""
    part = pic.part.related_part(pic._element.blipFill.blip.rEmbed)
    part._blob = png_bytes
    if fit:
        w, h = Image.open(io.BytesIO(png_bytes)).size
        bx, by, bw, bh = pic.left, pic.top, pic.width, pic.height
        scale = min(bw / w, bh / h)
        pic.width, pic.height = int(w * scale), int(h * scale)
        pic.left, pic.top = bx + (bw - pic.width) // 2, by + (bh - pic.height) // 2
    pic._element.blipFill.getparent()  # keep crop untouched
    pic.crop_left = pic.crop_right = pic.crop_top = pic.crop_bottom = 0


def png(path):
    return Path(path).read_bytes()


def crop(name):
    buf = io.BytesIO()
    Image.open(DESIGN).crop(CROPS[name]).save(buf, "PNG")
    return buf.getvalue()


def remove(sh):
    sh._element.getparent().remove(sh._element)


def status(kind, *notes):
    mark = {"Working": ("✓ Working", GREEN), "Fixed": ("✓ Fixed", GREEN),
            "Limitation": ("✗ Limitation", RED), "Failing": ("✗ Failing", RED),
            "Pending": ("… Pending", AMBER)}[kind]
    return [[(mark[0], {"bold": True, "color": mark[1]})]] + [[(n, {"bold": False, "color": BLACK})] for n in notes]


def fill_table(slide, rows):
    """rows: (input, output, status paragraphs) for Example 1, Example 2, Failure case."""
    table = next(sh for sh in slide.shapes if sh.has_table).table
    for i, (inp, out, st) in enumerate(rows, start=1):
        set_paragraphs(table.cell(i, 1).text_frame, [inp])
        set_paragraphs(table.cell(i, 2).text_frame, [out])
        set_paragraphs(table.cell(i, 3).text_frame, st)


def what(slide, text, label="What this section does:"):
    text_box(slide, "What this section does", [[(label, {"bold": True})], text])


def source(slide, text):
    text_box(slide, "Source:", [f"Source: {text}"])


def caption(slide, text):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith(("What to check", "Orange =")):
            set_paragraphs(sh.text_frame, [text])
            return
    raise KeyError("caption")


def delete_slides(prs, keep):
    """Keep only the slides at the given 0-based indices, in that order."""
    ids = prs.slides._sldIdLst
    items = list(ids)
    for el in items:
        ids.remove(el)
    for i in keep:
        ids.append(items[i])
    for i, el in enumerate(items):          # unreachable parts (slide, its images) are not saved
        if i not in keep:
            prs.part.drop_rel(el.rId)


# ---------------------------------------------------------------- the deck

def build():
    prs = Presentation(SRC)
    s = list(prs.slides)

    # 1 Title
    text_box(s[0], "Lab Report 3", ["Lab Report 4 Demo: The Diagnostician Loop under Planted Failures and a Compute Budget"])
    text_box(s[0], "Jamie Carpenter", ["Jamie Carpenter · AM/ME/CE 295 · 29 November 2026 · github.com/jamie-carpenter/lattice-gnn"])

    # 2 Design diagram
    replace_picture(pictures(s[1])[0], png(DESIGN))
    caption(s[1], "Orange = LLM agent, grey = deterministic tool or code, dashed = human. Green 'finished' = built, "
                  "tested and run in a full study. New since Lab Report 3: CPU-hours per run, the CPU-hour budget "
                  "and the planted-failure harness. The Optuna baseline runs on the planted cases by 9 December.")

    # 3 Section 1: CPU-hour budget
    sl = s[2]
    title(sl, "Section 1: CPU-hour budget (code)")
    replace_picture(pictures(sl)[0], crop("checks"))
    what(sl, "Every run records its CPU time, summed over all threads, and its wall time. A study stops when its "
             "22 CPU-hours run out, and a decision is allowed only if what is left pays for a full round at the "
             "mean cost so far. The agent sees each run's CPU-hours and the remainder.")
    fill_table(sl, [
        ("One 3-epoch run at width 512 on the 2,000-lattice subset, 8 threads (smoke test).",
         "0.230 CPU-hours against 0.038 h on the clock, about 6×. So 22 CPU-hours match the 3.6 h that "
         "Lab Report 3's Optuna study trained.", status("Working")),
        ("Two-epoch runs with the rule table and a budget of 0.5 CPU-hours (smoke test).",
         "Width 512 cost 0.16 CPU-hours per run, width 64 0.003–0.007. 26 runs in 13 rounds; stopped with "
         "0.02 CPU-hours left, against about 0.04 for a round.", status("Working")),
        ("The same smoke test, before the loop itself checked whether a round was affordable.",
         "The leftover 0.02 CPU-hours still reached the decider; the validator rejected its retune, and code "
         "logged a forced escalation.",
         status("Fixed", "Expected: stop on budget, no decision.",
                "Fix: the loop now stops before asking (validate.round_shortfall; 1 test).")),
    ])
    source(sl, "lattice_gnn/training.py, optimize.py, validate.py; agents/diagnostician/loop.py; "
               "config/loop_lr4.yaml; LOG.md 2026-11-24")

    # 4 Section 1 outputs (waits on the study)
    sl = s[3]
    title(sl, "Section 1 outputs: what each width costs")
    remove(pictures(sl)[0])
    caption(sl, "REWORK: CPU-hours per run against width, from every run of the planted study "
                "(add a 'cost' figure to scripts/make_lr4_figures.py), and how much of each study width 512 used.")

    # 5 Section 2: planted-failure harness
    sl = s[4]
    title(sl, "Section 2: Planted-failure harness (code)")
    replace_picture(pictures(sl)[0], crop("harness"))
    what(sl, "Injects a fault into the Lab Report 4 setup, either a label fault on a copy of the data or a bad "
             "search range, and runs each case with the agent and with the rule table. Code scores every "
             "decision against an answer key fixed beforehand, which never enters the agent's input.")
    fill_table(sl, [
        ("units_low_lr: moduli of validation family tet_Z04.8 (20 lattices) ×10, lr 1e-5 to 3e-5; "
         "2-epoch smoke run.",
         "tet_Z04.8 is the second-worst validation family, at a 1,276% median modulus error.",
         status("Working")),
        ("Six cases (config/planted.yaml): collapse, units_low_lr, width_trap with a correct remedy; control, "
         "hidden_noise, two_causes without one.",
         "Example key: collapse passes only if the first decision retunes lr below 3×10⁻³; any structural "
         "change first is wrong. 8 tests, one checking that no prompt contains a key.",
         status("Working")),
        ("A frozen agent call while the Claude Code sign-in had expired (smoke test).",
         "'OAuth session expired' was read as an invalid answer, retried, and logged as the agent's forced "
         "escalation.",
         status("Fixed", "Expected: an error, never a decision.",
                "Fix: a failed call now stops the run (agent.decide; 1 test).")),
    ])
    source(sl, "config/planted.yaml; agents/diagnostician/planted.py; lattice_gnn/training.py "
               "(plant_label_faults); tests/test_planted.py; LOG.md 2026-11-24")

    # 6 Section 2 outputs: the probes
    sl = s[5]
    title(sl, "Section 2 outputs: probes behind the cases")
    replace_picture(pictures(sl)[0], png(FIG / "lr4" / "probes.png"))
    caption(sl, "What to check: lr 1e-1 and 3e-2 collapse to the mean (loss ≈ 1), stop early and give no non-PD "
                "predictions; non-PD appears only in the partly trained lr 1e-5 run (0.042). So the planned "
                "'divergence' case became collapse, and two_causes uses a low lr.")

    # 7 Section 3: the diagnostician on the planted cases (waits on the study)
    sl = s[7]
    title(sl, "Section 3: Diagnostician on planted cases (LLM agent)")
    replace_picture(pictures(sl)[0], crop("agent"))
    what(sl, "Unchanged since Lab Report 3 except that it now sees each run's CPU-hours and the CPU-hours left. "
             "claude-sonnet-5-5 via claude -p, no tools. Each planted case: one full loop, then 10 frozen "
             "calls on the loop's first-round report.")
    pend = status("Pending", "REWORK: fill from runs/planted-*-agent and *-repeats.")
    fill_table(sl, [("collapse: lr range 1e-2 to 1e-1, PD layer off.", "REWORK: first decision and verdict.", pend),
                    ("units_low_lr: tet_Z04.8 ×10 and lr 1e-5 to 3e-5.", "REWORK: did it raise lr and name tet_Z04.8?", pend),
                    ("width_trap: widths 256/512 at lr 5e-4 to 1.5e-3.", "REWORK: did it drop 512 or lower lr?", pend)])
    source(sl, "runs/planted-*-agent/decisions.jsonl, *-repeats/repeats.jsonl; "
               "python -m agents.diagnostician.planted score runs/planted-*")

    # 8 Section 3 outputs (waits on the study)
    sl = s[8]
    title(sl, "Section 3 outputs: agent vs rule table on planted cases")
    remove(pictures(sl)[0])
    caption(sl, "REWORK: the scored table for the six cases (agent loop, 10 frozen calls, rule-table loop), "
                "with CPU-hours used and best loss per loop.")

    # 9 Section 4: the rule table on the planted cases (waits on the study)
    sl = s[13]
    title(sl, "Section 4: Rule table on planted cases (code)")
    replace_picture(pictures(sl)[0], crop("rules"))
    what(sl, "Unchanged since Lab Report 3: non-PD predictions with the PD layer off → PD layer on; otherwise "
             "zoom every range on the best run. Each planted case is built so that this rule gives the wrong "
             "answer, or so that no answer is right.")
    pend = status("Pending", "REWORK: fill from runs/planted-*-rules.")
    fill_table(sl, [("collapse: non-PD 0 (collapsed runs predict the mean).", "Expected: rule 2 zooms around a high lr and stays above 3×10⁻³.", pend),
                    ("units_low_lr: the low-lr probe left non-PD 0.042.", "Expected: rule 1 turns the PD layer on first; it never escalates.", pend),
                    ("width_trap: best runs mix widths 256 and 512.", "Expected: the zoom keeps the top three runs' widths, so 512 may stay.", pend)])
    source(sl, "agents/diagnostician/rules.py; runs/planted-*-rules; LOG.md 2026-11-24")

    # 10 Current failure modes
    sl = s[10]
    title(sl, "Current failure modes")
    remove(pictures(sl)[0])
    text_box(sl, "Input / Output Examples", [[("Where the system breaks", {"bold": True})]])
    table = next(sh for sh in sl.shapes if sh.has_table).table
    for i, label in enumerate(("Failure 1", "Failure 2", "Failure 3"), start=1):
        set_paragraphs(table.cell(i, 0).text_frame, [label])
    fill_table(sl, [
        ("Training on the CPU: the agent and the rule table trained identical settings in Lab Report 3.",
         "Their losses differed by up to 0.024, more than the 0.02 that counts as an improvement.",
         status("Limitation", "Single verdicts near the threshold mean little; repeats are needed.")),
        ("A planted case whose premise was not checked: lr 1e-2 to 1e-1 meant to cause non-PD predictions.",
         "The model collapsed to the mean instead, with no non-PD predictions at all.",
         status("Fixed", "Every case now cites the probe behind its premise.")),
        ("An escalation that describes the corrupted family without naming it.",
         "Scored as a miss: the check is a substring search for 'tet_Z04.8'.",
         status("Limitation", "Read every escalation by hand as well.")),
    ])
    what(sl, "The 100-epoch cap still limits accuracy (best epochs 92–100), the agent's free-text evidence is not "
             "checked, and CPU-hours compare conditions on this laptop only. REWORK: add what the study reveals.",
         label="Also open:")
    source(sl, "LOG.md 2026-11-11, 11-15, 11-24; config/planted.yaml; runs/probe-lr")

    # 11 Next steps
    sl = s[15]
    title(sl, "Next steps to the final report (9 December)")
    remove(pictures(sl)[0])
    box = next(sh for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.startswith("What to check"))
    box.top, box.height = Emu(int(1.35 * 914400)), Emu(int(3.2 * 914400))
    set_paragraphs(box.text_frame, [
        "1.  Optuna baseline on every planted case, at the same 22 CPU-hours.",
        "2.  Repeated loops for the cases where the agent and the rule table differ.",
        "3.  One evaluation of each condition's best model on the 64 locked test families.",
        "4.  Menu options family_balanced_loss and physics_residual.",
        "5.  Final presentation and four-page paper."])
    for para in box.text_frame.paragraphs:
        para.space_after = Emu(int(0.12 * 914400))
        for r in para.runs:
            r.font.size = Emu(int(16 * 12700))

    delete_slides(prs, [0, 1, 2, 3, 4, 5, 7, 8, 13, 10, 15])
    prs.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    build()
