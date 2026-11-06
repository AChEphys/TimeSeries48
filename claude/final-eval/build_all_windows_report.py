"""
Build a PDF report of all-windows_performances.ipynb for the baseline's author.

Two steps, because they need different environments:

    .venv/bin/python claude/final-eval/build_all_windows_report.py parts PARTS.json
    <env with playwright + markdown-it-py>/python \
        claude/final-eval/build_all_windows_report.py pdf PARTS.json REPORT.pdf

"parts" runs the notebook's code cells (project environment) and saves the grand
table's HTML, the mix weights, the figure and the notebook's footnote and caption text.
"pdf" assembles the report (no code, paths or file names: the footnotes' few internal
references are removed, see CLEANUPS) and prints it with headless Chromium.
"""
# ruff: noqa: E501, RUF001, INP001, T201

import argparse
import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "all-windows_performances.ipynb"
FIGURE = ROOT / "runs/run-final/plots/all_windows_per_window.png"
FINAL_WINDOW = 7

# Internal references (paths, file names, notes, notebook sections) removed or reworded
# in the notebook's footnotes; everything else is printed as the notebook shows it.
CLEANUPS = [
    (r"\s*\(`claude/project-notes\.md` N-24\)", ""),
    (r"\s*The training scripts save these as\s+`predictions/validation_\*\.parquet`\.", ""),
    (r"Section 2 lists the weights of every window\.", "Table 2 lists the weights of every window."),
    (r"\s*\(`runs/run-final/post-processing\.json`\)", ""),
    (r"for this notebook,", "for this report,"),
    (r"\s*\(issue I-1\)", ""),
    (r";\s*section 1 shows the counts", ""),
    (r"\s*\(W1–W6: the original runs; W7: `runs/run-final`\)", ""),
    (r"`", ""),  # The one remaining code span is a formula: plain text.
]
CAPTION_CLEANUPS = [
    (r"\s*The figure is also\s+saved as `[^`]+`\.", ""),
    (r"`", ""),
]

LABELS = {"run-11": "Run-11", "run-13": "Run-13", "run-14": "Run-14", "run-15": "Run-15",
          "run-16": "Run-16", "run-17": "Run-17",
          "mix-11-17-fitted": "Run-11 &amp; 17 (weighted mix)",
          "lagmix-fitted": "Run-11 &amp; 13-16 (weighted mix)"}

INTRO = """
<p>This report compares the original baseline (<b>Run-2</b>: the 48 raw features of the
prediction date) with models that also see each stock's recent past: the 48 features one
and five trading dates earlier (T–1, T–5) and the stock's own realized labels of T–6 to
T–10. All models are LightGBM models, trained anew for each rolling window on its 230
history dates; a 10-date gap separates the history from the test dates.</p>
<p>The models are scored on <b>seven consecutive test windows</b>, 804 test dates from
2022-12-29 to 2026-04-28, on the same dates and stocks for every column. The first six
windows (W1–W6) were used while developing and choosing the setups, so their numbers flatter
the chosen ones. The last window (W7, 2025-12-19 → 2026-04-28) was held back until the
setups were fixed and evaluated once; Run-22 had been named the main candidate beforehand.</p>
<p>Table 1 describes each column (setup rows) and gives its metrics; every term is defined
in the footnotes below it. Figure 1 shows the main metrics window by window. Table 2 lists
the weights of the two weighted mixes.</p>
"""

SUMMARY = """
<h2>Summary</h2>
<ul>
<li><b>Ranking.</b> Adding the 48 features of T–1 and T–5 and the stock's labels of T–6 to
T–10 (Run-22) raises the mean daily Rank IC from 0.0592 (Run-2) to 0.0730 over the 804
test dates (+0.0138, t 6.4): better in 6 of 7 windows and on 65% of the dates. Against
Run-5, which is trained like Run-22 but on the 48 features of the prediction date only, the
added inputs account for +0.0095 (t 4.5).</li>
<li><b>Magnitudes.</b> R² rises from 0.00740 to 0.01009. Part of this comes from the
training procedure and the demeaning, as Run-5 shows (0.00896). Calibration varies with the
period for every column (slopes from about 0.6 to 1.4); averaged over the windows, Run-22's
distance from a slope of 1 is the same as Run-2's (0.154 vs 0.155).</li>
<li><b>The two kinds of history.</b> The past features (Run-11) and the past labels
(Run-17) each help on their own. Combined in one model (Run-22) they give the best ranking;
the average of the two separate models (Run-11 &amp; 17 mix) gives a similar R² and Pearson
IC.</li>
<li><b>Dependence on the period.</b> The gain over Run-2 differs a lot between windows:
largest in W2, W5 and W6 (about +0.02 Rank IC), close to zero in W3 and in the final
window W7, the only window that played no part in choosing the setups. More recent data
are needed to see whether the improvement holds across market periods.</li>
</ul>
"""

CSS = """
@page { size: A4 landscape; margin: 11mm 10mm 12mm 10mm; }
body { font-family: "DejaVu Sans", sans-serif; font-size: 9pt; color: #1a1a1a; line-height: 1.35; }
h1 { font-size: 16pt; margin: 0 0 2pt 0; }
h2 { font-size: 12pt; margin: 14pt 0 4pt 0; }
.date { color: #666; margin-bottom: 8pt; }
.caption { font-size: 8.5pt; margin: 4pt 0 6pt 0; }
.grand table { border-collapse: collapse; font-size: 6.9pt; line-height: 1.22; width: 100%; }
.grand th { font-weight: bold; vertical-align: bottom; padding: 2px 4px; border-bottom: 1px solid #888; }
.grand td { padding: 1.5px 4px; border-bottom: 0.5px solid #e2e2e2; vertical-align: top; }
.grand td:first-child { min-width: 150px; max-width: 210px; }
.grand thead { display: table-header-group; }
.grand tr { break-inside: avoid; }
.footnotes { font-size: 8pt; line-height: 1.3; columns: 2; column-gap: 9mm; }
.footnotes h2 { column-span: all; }
.footnotes p { break-after: avoid; }
.footnotes li { orphans: 3; widows: 3; }
.footnotes ul, .footnotes ol { padding-left: 14pt; margin: 2pt 0; }
.footnotes li { margin: 1.5pt 0; }
.footnotes code { font-family: inherit; }
figure { margin: 0; break-inside: avoid; }
figure img { display: block; margin: 0 auto; width: auto; max-width: 100%; max-height: 132mm; }
table.weights { border-collapse: collapse; font-size: 8pt; margin-top: 4pt; }
table.weights th, table.weights td { padding: 2px 8px; border-bottom: 0.5px solid #ccc; text-align: right; }
table.weights td.l, table.weights th.l { text-align: left; }
.summary li { margin: 3pt 0; }
.break { break-before: page; }
"""


def notebook_parts(out: Path) -> None:
    """Run the notebook's code cells; save the table HTML, weights and text cells."""
    import IPython.display as ipd  # noqa: PLC0415  # The notebook's own environment.

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    html: list[str] = []

    def capture(obj: object) -> None:
        if isinstance(obj, ipd.HTML):
            html.append(obj.data)

    ns: dict = {"display": capture}
    ipd.display = capture
    for c in nb["cells"]:
        if c["cell_type"] == "code" and c["id"] != "plot":
            exec(compile("".join(c["source"]), c["id"], "exec"), ns)  # noqa: S102
    text = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"}
    parts = {
        "table": html[-1], "footnotes": text["footnotes"], "table_md": text["table-md"],
        "plot_md": text["plot-md"], "weights": ns["weight_rows"],
        "figure": base64.b64encode(FIGURE.read_bytes()).decode(),
    }
    out.write_text(json.dumps(parts), encoding="utf-8")
    print(f"wrote {out}")


def clean(text: str, rules: list[tuple[str, str]]) -> str:
    """Apply the cleanup rules; fail if a path or code span survives."""
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text)
    leftover = re.search(r"runs/|\.parquet|\.json|\.md\b|\.py\b|claude/", text)
    if leftover:
        msg = f"internal reference left in the report text: {leftover.group()}"
        raise ValueError(msg)
    return text


def weights_table(rows: list[dict]) -> str:
    """Table 2: the weighted mixes' weights per window."""
    windows = sorted({r["window"] for r in rows})
    head = '<tr><th class="l">Mix</th><th class="l">Model</th>' + "".join(
        f"<th>W{w}{' (final)' if w == FINAL_WINDOW else ''}</th>" for w in windows) + "</tr>"
    body = []
    for mix in dict.fromkeys(r["mix"] for r in rows):
        for member in dict.fromkeys(r["member"] for r in rows if r["mix"] == mix):
            cells = {r["window"]: r["weight"] for r in rows if r["mix"] == mix and r["member"] == member}
            body.append(f'<tr><td class="l">{LABELS[mix]}</td><td class="l">{LABELS[member]}</td>'
                        + "".join(f"<td>{cells[w]:.3f}</td>" for w in windows) + "</tr>")
    return f'<table class="weights">{head}{"".join(body)}</table>'


def build_pdf(parts_file: Path, pdf: Path) -> None:
    """Assemble the report HTML and print it to PDF."""
    from markdown_it import MarkdownIt  # noqa: PLC0415  # Report environment only.
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    parts = json.loads(parts_file.read_text(encoding="utf-8"))
    md = MarkdownIt("commonmark", {"html": True})
    footnotes = md.render(clean(parts["footnotes"], CLEANUPS))
    table_caption = md.render(parts["table_md"].split("\n", 1)[1].strip())
    plot_caption = md.render(clean(parts["plot_md"].split("\n", 1)[1].strip(), CAPTION_CLEANUPS))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Seven test windows</title>
<style>{CSS}</style></head><body>
<h1>Past features and past labels on top of the 48-feature baseline: seven test windows</h1>
<div class="date">2026-11-05</div>
{INTRO}
<h2>Table 1. Setups and metrics, all seven windows</h2>
<div class="caption">{table_caption}</div>
<div class="grand">{parts["table"].replace("13-16", "13&#8209;16")}</div>
<div class="footnotes">{footnotes}</div>
<div class="break"></div>
<h2>Figure 1. Per window: ranking, magnitudes and calibration</h2>
<div class="caption">{plot_caption}</div>
<figure><img src="data:image/png;base64,{parts["figure"]}"></figure>
<h2>Table 2. Weights of the weighted mixes, per window</h2>
<div class="caption">Fitted on each window's history only (footnote **); a weight of 0 drops that model in that window.</div>
{weights_table(parts["weights"])}
<div class="summary">{SUMMARY}</div>
</body></html>"""
    page = pdf.with_suffix(".html")
    page.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        tab = browser.new_page()
        tab.goto(page.resolve().as_uri())
        tab.pdf(path=str(pdf), prefer_css_page_size=True, print_background=True,
                display_header_footer=True, header_template="<span></span>",
                footer_template='<div style="font-size:7pt;width:100%;text-align:center;color:#777">'
                                '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        browser.close()
    page.unlink()
    print(f"wrote {pdf}")


def main() -> int:
    """Run one of the two steps."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)
    a = sub.add_parser("parts")
    a.add_argument("out", type=Path)
    b = sub.add_parser("pdf")
    b.add_argument("parts", type=Path)
    b.add_argument("pdf", type=Path)
    args = parser.parse_args()
    if args.step == "parts":
        notebook_parts(args.out)
    else:
        build_pdf(args.parts, args.pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
