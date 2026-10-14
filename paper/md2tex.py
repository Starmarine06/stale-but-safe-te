"""Convert paper/paper.md (the subset of Markdown it uses) to paper/tnsm/main.tex in IEEEtran journal format.
  python paper/md2tex.py"""
import re
from pathlib import Path

root = Path(__file__).resolve().parent
md = (root / "paper.md").read_text(encoding="utf-8").replace("\r\n", "\n")

SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}


def esc_plain(t):
    t = "".join(SPECIAL.get(ch, ch) for ch in t)
    t = t.replace("\\", r"\textbackslash{}") if "\\" in t.replace(r"\&", "").replace(r"\%", "").replace(r"\$", "").replace(r"\#", "").replace(r"\_", "").replace(r"\{", "").replace(r"\}", "") else t
    t = t.replace("^", r"\^{}")
    t = re.sub(r"~(?=\d)", r"$\\sim$", t)
    t = t.replace("~", r"\textasciitilde{}")
    t = t.replace("<=", r"$\le$").replace(">=", r"$\ge$")
    t = re.sub(r"(?<![\w])\+-(?=\s?\d)", r"$\\pm$", t)
    t = t.replace(" -> ", r" $\rightarrow$ ")
    t = t.replace("<", r"\textless{}").replace(">", r"\textgreater{}")
    t = t.replace(" x ", r" $\times$ ")
    t = re.sub(r"(?<=\d)x(?=\s)", r"$\\times$", t)
    t = re.sub(r"(\d)-(\d)", r"\1--\2", t)
    t = t.replace("\u2013", "--").replace("\u2014", "---").replace("\u00b1", r"$\pm$")
    return t


def inline(t):
    parts = re.split(r"(`[^`]+`)", t)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 2:
            code = p[1:-1].replace("\\", "/")
            for k, v in SPECIAL.items():
                code = code.replace(k, v)
            code = code.replace("^", r"\^{}").replace("~", r"\textasciitilde{}")
            out.append(r"\texttt{" + code + "}")
        else:
            for tk in re.split(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)", p):
                if tk.startswith("**") and tk.endswith("**") and len(tk) > 4:
                    out.append(r"\textbf{" + esc_plain(tk[2:-2]) + "}")
                elif tk.startswith("*") and tk.endswith("*") and len(tk) > 2:
                    out.append(r"\textit{" + esc_plain(tk[1:-1]) + "}")
                else:
                    out.append(esc_plain(tk))
    s = "".join(out)
    return s


def table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\s*\|[\s:|-]+\|\s*$", l)]
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    wide = ncol >= 6
    colspec = "l" * ncol if ncol <= 4 else "l" + "c" * (ncol - 1)
    env = "table*" if wide else "table"
    body = []
    for i, r in enumerate(rows):
        body.append(" & ".join(inline(c) for c in r) + r" \\")
        if i == 0:
            body.append(r"\hline")
    tab = "\\begin{tabular}{" + colspec + "}\n\\hline\n" + "\n".join(body) + "\n\\hline\n\\end{tabular}"
    size = r"\scriptsize" if ncol >= 5 else r"\footnotesize"
    width = r"\textwidth" if wide else r"\columnwidth"
    return (f"\\begin{{{env}}}[!t]\n\\centering\n{size}\n\\renewcommand{{\\arraystretch}}{{1.05}}\n"
            f"\\adjustbox{{max width={width}}}{{%\n{tab}}}\n\\end{{{env}}}\n")


def items_tex(items, ordered):
    env = "enumerate" if ordered else "itemize"
    return f"\\begin{{{env}}}\n" + "\n".join(r"\item " + inline(x) for x in items) + f"\n\\end{{{env}}}\n"


blocks = re.split(r"\n\s*\n", md)
out, abstract, refs = [], [], []
i, title, sec, in_refs = 0, "", None, False
while i < len(blocks):
    b = blocks[i].strip("\n")
    i += 1
    if not b.strip():
        continue
    if b.startswith("# "):
        title = b[2:].strip()
        continue
    if b.startswith("*Draft v"):
        continue
    m = re.match(r"^(#{2,3}) (.*)$", b)
    if m:
        lvl, name = len(m.group(1)), m.group(2).strip()
        name = re.sub(r"^\d+[a-z]?(\.\d+[a-z]?)? ", "", name)
        sec, in_refs = name, name == "References"
        if name == "Abstract":
            out.append("%ABSTRACT%")
        elif in_refs:
            out.append("%REFS%")
        else:
            out.append(("\\section{" if lvl == 2 else "\\subsection{") + inline(name) + "}")
        continue
    if in_refs:
        for line in b.splitlines():
            mm = re.match(r"^(\d+)\. (.*)$", line)
            if mm:
                refs.append(mm.group(2))
        continue
    if sec == "Abstract":
        abstract.append(inline(b))
        continue
    if b.startswith("!["):
        fn = re.search(r"\((figures/[^)]+)\)", b).group(1)
        cap = ""
        if i < len(blocks) and blocks[i].lstrip().startswith("**Figure"):
            cap = re.sub(r"^\*\*Figure \d+\.\*\*\s*", "", blocks[i].strip())
            i += 1
        wide = "fig1" in fn or "fig3" in fn
        star = "*" if wide else ""
        w = r"0.8\textwidth" if wide else r"\columnwidth"
        out.append(f"\\begin{{figure{star}}}[!t]\n\\centering\n\\includegraphics[width={w}]{{{fn}}}\n\\caption{{{inline(cap)}}}\n\\end{{figure{star}}}\n")
        continue
    lines = b.splitlines()
    if all(l.lstrip().startswith("|") for l in lines):
        out.append(table(lines))
        continue
    if all(re.match(r"^\s*[*-] ", l) or l.startswith("  ") for l in lines):
        items = []
        for l in lines:
            mm = re.match(r"^\s*[*-] (.*)$", l)
            if mm:
                items.append(mm.group(1))
            else:
                items[-1] += " " + l.strip()
        out.append(items_tex(items, False))
        continue
    if all(re.match(r"^\d+\. ", l) or l.startswith("   ") for l in lines):
        items = []
        for l in lines:
            mm = re.match(r"^\d+\. (.*)$", l)
            if mm:
                items.append(mm.group(1))
            else:
                items[-1] += " " + l.strip()
        out.append(items_tex(items, True))
        continue
    if b.startswith(">"):
        out.append("\\begin{quote}\n" + inline(" ".join(l.lstrip("> ").strip() for l in lines)) + "\n\\end{quote}\n")
        continue
    para, rest = [], []
    for k, l in enumerate(lines):
        if re.match(r"^\s*[*-] ", l) and para:
            rest = lines[k:]
            break
        para.append(l)
    out.append(inline(" ".join(x.strip() for x in para)) + "\n")
    if rest:
        items = []
        for l in rest:
            mm = re.match(r"^\s*[*-] (.*)$", l)
            if mm:
                items.append(mm.group(1))
            else:
                items[-1] += " " + l.strip()
        out.append(items_tex(items, False))

abs_tex = ("\\begin{abstract}\n" + "\n\n".join(abstract) + "\n\\end{abstract}\n\n\\begin{IEEEkeywords}\n"
           "traffic engineering, software-defined networking, network telemetry, learned control, baseline tuning\n\\end{IEEEkeywords}\n")
bib = "\\begin{thebibliography}{99}\n" + "\n".join(f"\\bibitem{{r{n + 1}}} {inline(r)}" for n, r in enumerate(refs)) + "\n\\end{thebibliography}\n"
ack_f = root / "tnsm" / "ack.tex"
ack = ack_f.read_text(encoding="utf-8") if ack_f.exists() else ""
head = r"""\documentclass[journal]{IEEEtran}
\usepackage{graphicx}
\usepackage{adjustbox}
\usepackage{amsmath,amssymb}
\usepackage{url}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage{microtype}
\usepackage{hyperref}
\hypersetup{hidelinks}
\graphicspath{{../}}
\sloppy
\begin{document}
\title{""" + inline(title) + r"""}
\author{Dev~Nambiar,~\IEEEmembership{Vellore Institute of Technology, Vellore, India}%
\thanks{Manuscript not yet submitted. Corresponding e-mail: [to be added].}}
\markboth{IEEE Transactions on Network and Service Management, draft}{Nambiar: The Cost of Stale Telemetry in Traffic Engineering}
\maketitle
"""
body = "\n".join(out).replace("%ABSTRACT%", abs_tex).replace("%REFS%", ack + "\n" + bib)
(root / "tnsm" / "main.tex").write_text(head + body + "\n\\end{document}\n", encoding="utf-8")
print("wrote main.tex;", len(refs), "refs;", len(out), "blocks")
