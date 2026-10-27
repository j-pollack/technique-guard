"""Build the arXiv submission from PAPER.md: arxiv/paper.tex, arxiv/figures/*.pdf, arxiv/paper.pdf, arxiv/submission.zip.

    python3 arxiv.py

Needs pandoc, pdflatex (arXiv compiles with pdflatex, not xelatex) and Google Chrome (SVG -> PDF, headless).
"""

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "arxiv"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
AUTHOR = r"Jonathan Pollack\\Independent Researcher\\\texttt{contact@arxiv.jonathanpollack.net}"
DATE = "October 2026"
HEADER = r"""
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{00D7}{\ensuremath{\times}}
\DeclareUnicodeCharacter{03BA}{\ensuremath{\kappa}}
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\usepackage{listings}
\lstset{basicstyle=\small\ttfamily, breaklines=true, columns=fullflexible, frame=single,
  literate={…}{{\ldots}}1 {→}{{$\rightarrow$}}1 {≥}{{$\geq$}}1 {×}{{$\times$}}1 {–}{{--}}1 {—}{{---}}1 {§}{{\S}}1}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize}
"""


def figures():
    """Each SVG to a tight one-page PDF via headless Chrome (light theme, the browser default)."""
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for svg in sorted((ROOT / "figures").glob("*.svg")):
        text = svg.read_text()
        w, h = re.search(r'viewBox="0 0 (\d+) (\d+)"', text).groups()
        html = OUT / "figures" / f"{svg.stem}.html"
        html.write_text(f"<!doctype html><style>@page {{ size: {w}px {h}px; margin: 0 }} body {{ margin: 0 }}</style>{text}")
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT / 'figures' / (svg.stem + '.pdf')}", html.as_uri()],
                       check=True, capture_output=True)
        html.unlink()


def markdown() -> tuple[str, str, str]:
    """(title, abstract, body) from PAPER.md, with figure captions moved into the image syntax."""
    md = (ROOT / "PAPER.md").read_text()
    title = re.match(r"# (.+)\n", md).group(1)
    abstract = re.search(r"## Abstract\n\n(.+?)\n## ", md, re.S).group(1).strip()
    note = re.search(r"\n\*(Draft, .+?)\*\n", md).group(1)
    note = re.sub(r"^Draft, [\d-]+\. ", "", note)
    note = re.sub(r" References were checked.*$", "", note)
    body = md[md.index("## 1. Introduction"):]
    # ![alt](figures/x.svg)\n\n*Figure N. caption* -> ![caption](figures/x.pdf); LaTeX numbers the figures in the same order
    body = re.sub(r"!\[[^\]]*\]\(figures/(\w+)\.svg\)\n\n\*Figure \d+\. (.+?)\*\n",
                  lambda m: f"![{m.group(2)}](figures/{m.group(1)}.pdf)\n", body)
    assert ".svg" not in body, "a figure is missing its caption line"
    # pandoc needs a blank line before a list that follows a paragraph line (GitHub doesn't)
    body = re.sub(r"(?m)^((?!\s*(?:[-*]|\d+\.) )\S.*)\n(\s*(?:-|\d+\.) )", r"\1\n\n\2", body)
    return title, abstract, f"*{note}*\n\n" + body


def main():
    OUT.mkdir(exist_ok=True)
    figures()
    title, abstract, body = markdown()
    src = OUT / "paper.md"
    meta = "".join(f"{k}: |\n" + "".join(f"  {l}\n" for l in v.splitlines()) for k, v in (("title", title), ("abstract", abstract)))
    src.write_text(f"---\n{meta}---\n\n{body}")  # YAML so the abstract's markdown is parsed
    (OUT / "header.tex").write_text(HEADER)
    subprocess.run(["pandoc", str(src), "-s", "-o", str(OUT / "paper.tex"), "--syntax-highlighting=idiomatic",
                    "-V", f"author={AUTHOR}", "-V", f"date={DATE}",
                    "-V", "geometry:margin=1in", "-V", "fontsize=11pt", "-V", "colorlinks=true",
                    "-H", str(OUT / "header.tex")], check=True)
    for _ in range(2):  # second pass settles references
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=OUT, check=True,
                       capture_output=True)
    with zipfile.ZipFile(OUT / "submission.zip", "w") as z:
        z.write(OUT / "paper.tex", "paper.tex")
        for f in sorted((OUT / "figures").glob("*.pdf")):
            z.write(f, f"figures/{f.name}")
    src.unlink()
    print("wrote", OUT / "paper.pdf", "and", OUT / "submission.zip")


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    main()
