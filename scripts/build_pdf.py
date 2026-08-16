"""
Build a single self-contained PDF from the Markdown documents.

Combines the project brief, the theory report and the quick-start into one
typeset document, embeds every referenced figure as a data URI so the file
stands alone, and renders it with headless Chrome.

    python build_report.py                    # -> cylinder_report.pdf
    python build_report.py --html-only        # inspect the intermediate HTML
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import subprocess
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DOCS = ["REPORT.md"]

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.2pt; line-height: 1.5;
       color: #1a1a1a; max-width: 100%; }
h1 { font-size: 17pt; margin: 0 0 .4em; padding-bottom: .2em;
     border-bottom: 2px solid #444; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 14pt; margin: 1.5em 0 .4em; padding-bottom: .15em;
     border-bottom: 1px solid #bbb; page-break-after: avoid; }
h3 { font-size: 11.6pt; margin: 1.2em 0 .3em; page-break-after: avoid; }
h4 { font-size: 10.6pt; margin: 1em 0 .3em; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code { font-family: "SF Mono", "Menlo", monospace; font-size: 8.6pt;
       background: #f2f2f0; padding: .1em .32em; border-radius: 3px; }
pre { background: #f7f7f5; border: 1px solid #e0e0dc; border-left: 3px solid #888;
      border-radius: 3px; padding: .7em .9em; overflow-x: auto;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.42; }
blockquote { border-left: 3px solid #7a9; background: #f4f8f6; margin: 1em 0;
             padding: .5em 1em; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: .35em .55em; text-align: left;
         vertical-align: top; }
th { background: #efefec; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
img { max-width: 100%; display: block; margin: 1em auto; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.6em 0; }
a { color: #245; text-decoration: none; }
em { color: #333; }
.doc-sep { page-break-before: always; }
.title-block { text-align: center; margin: 3em 0 2em; }
.title-block .t { font-size: 24pt; font-weight: 600; }
.title-block .s { font-size: 13pt; color: #555; margin-top: .5em; }
.title-block .d { font-size: 9.5pt; color: #777; margin-top: 2em; }
.toc { background: #f7f7f5; border: 1px solid #e2e2de; border-radius: 4px;
       padding: .8em 1.4em; margin: 1.5em 0; font-size: 9.5pt; }
"""


def embed_images(html: str, root: Path) -> str:
    """Replace <img src="file.png"> with inline base64 so the PDF stands alone."""
    # python-markdown emits <img alt="..." src="..." />, so the src is not
    # necessarily the first attribute -- match it wherever it appears.
    def repl(m):
        before, src = m.group(1), m.group(2)
        if src.startswith(("data:", "http")):
            return m.group(0)
        path = root / src
        if not path.exists():
            print(f"  ! missing figure, dropping: {src}")
            return ""
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode()
        print(f"  + embedded {src} ({path.stat().st_size/1e3:.0f} kB)")
        return f'<img{before}src="data:{mime};base64,{b64}"'

    return re.sub(r'<img([^>]*?)src="([^"]+)"', repl, html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cylinder_report.pdf")
    ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).parent
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists",
                                       "attr_list"])

    parts = []
    for i, name in enumerate(DOCS):
        path = root / name
        if not path.exists():
            print(f"  ! skipping missing {name}")
            continue
        md.reset()
        body = md.convert(path.read_text())
        sep = ' class="doc-sep"' if i else ""
        parts.append(f"<section{sep}>{body}</section>")
        print(f"  + {name}")

    title = """
    <div class="title-block">
      <div class="t">A Parametric Neural ODE for the<br>Cylinder-Wake Hopf Bifurcation</div>
      <div class="s">Problem, system and proposed modelling technique</div>
      <div class="d">Project brief &middot; one month, undergraduate</div>
    </div>
    """

    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Cylinder-wake parametric NODE — project brief</title>"
            f"<style>{CSS}</style></head><body>{title}"
            f"{''.join(parts)}</body></html>")
    html = embed_images(html, root)

    html_path = root / "cylinder_report.html"
    html_path.write_text(html)
    print(f"  wrote {html_path.name} ({len(html)/1e6:.1f} MB)")
    if args.html_only:
        return

    out = root / args.out
    cmd = [CHROME, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", f"--print-to-pdf={out}", str(html_path)]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    if not out.exists():
        # older Chrome builds use a different flag name for the header/footer
        cmd[cmd.index("--no-pdf-header-footer")] = "--print-to-pdf-no-header"
        r = subprocess.run(cmd, capture_output=True, timeout=180)
    if not out.exists():
        raise SystemExit(f"Chrome failed to produce a PDF:\n{r.stderr.decode()[:500]}")
    print(f"  wrote {out.name} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
