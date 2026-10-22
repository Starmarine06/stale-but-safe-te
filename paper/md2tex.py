"""Convert paper/paper.md (the subset of Markdown it uses) to paper/tnsm/main.tex in IEEEtran journal format.
  python paper/md2tex.py
Conventions in paper.md: '**Table N.** caption' directly before a table, '**Figure N.** caption' after an image, headings '## 6 Title' and
'### 6.1 Title' (numbers become \\label{sec:6.1}), '## Appendix A: Title'. Text references 'Table N', 'Fig. N', 'Section 6.1', 'Appendix A'
become \\ref{}s, so the printed numbers always follow LaTeX's own numbering."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent
md = (root / "paper.md").read_text(encoding="utf-8").replace("\r\n", "\n")

SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
SEC = r"\d(?:\.\d[a-z]?)?"


def esc_plain(t):
    t = "".join(SPECIAL.get(ch, ch) for ch in t)
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


def crossref(t):
    def secs(m):
        toks = re.split(r"(, | and |--|-)", m.group(2))
        out = []
        for tk in toks:
            out.append(("\\ref{sec:%s}" % tk) if re.fullmatch(SEC, tk) else tk)
        return m.group(1).rstrip() + "~" + "".join(out)
    t = re.sub(r"(Sections? )((?:%s)(?:(?:, | and |--|-)(?:%s))*)" % (SEC, SEC), secs, t)
    t = re.sub(r"Tables? (A?\d+)", lambda m: "Table~\\ref{tab:%s}" % m.group(1), t)
    t = re.sub(r"Fig\. (\d+)", lambda m: "Fig.~\\ref{fig:%s}" % m.group(1), t)
    t = re.sub(r"Appendix ([A-Z])\b", lambda m: "Appendix~\\ref{app:%s}" % m.group(1), t)
    return t


def inline(t, refs=True):
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
    if refs:
        s = re.sub(r"\[(\d+(?:, ?\d+)*)\]", lambda mm: r"\cite{" + ",".join("r" + x.strip() for x in mm.group(1).split(",")) + "}", s)
    return crossref(s) if refs else s


def table(lines, caption="", label=""):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\s*\|[\s:|-]+\|\s*$", l)]
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    wide = ncol >= 7
    colspec = "l" * min(ncol, 3) + "c" * max(0, ncol - 3) if ncol > 3 else "l" * ncol
    if str(label) == "1" and ncol == 5:
        wide = True
        colspec = "".join(">{\\raggedright\\arraybackslash}p{%.3f\\textwidth}" % w for w in (0.10, 0.20, 0.24, 0.22, 0.14))
    env = "table*" if wide else "table"
    body = []
    for i, r in enumerate(rows):
        body.append(" & ".join(inline(c) for c in r) + r" \\")
        if i == 0:
            body.append(r"\hline")
    tab = "\\begin{tabular}{" + colspec + "}\n\\hline\n" + "\n".join(body) + "\n\\hline\n\\end{tabular}"
    size = r"\scriptsize" if ncol >= 5 else r"\footnotesize"
    width = r"\textwidth" if wide else r"\columnwidth"
    cap = f"\\caption{{{inline(caption)}}}\n\\label{{tab:{label}}}\n" if caption else ""
    return (f"\\begin{{{env}}}[!t]\n{cap}\\centering\n{size}\n\\renewcommand{{\\arraystretch}}{{1.05}}\n"
            f"\\adjustbox{{max width={width}}}{{%\n{tab}}}\n\\end{{{env}}}\n")


def items_tex(items, ordered):
    env = "enumerate" if ordered else "itemize"
    return f"\\begin{{{env}}}\n" + "\n".join(r"\item " + inline(x) for x in items) + f"\n\\end{{{env}}}\n"


blocks = re.split(r"\n\s*\n", md)
out, abstract, refs = [], [], []
i, title, sec, in_refs, in_appendix = 0, "", None, False, False
pending_cap = None
warn = []
while i < len(blocks):
    b = blocks[i].strip("\n")
    i += 1
    if not b.strip():
        continue
    if b.startswith("# "):
        title = b[2:].strip()
        continue
    m = re.match(r"^(#{2,3}) (.*)$", b)
    if m:
        lvl, name = len(m.group(1)), m.group(2).strip()
        mm = re.match(r"^(\d+[a-z]?(?:\.\d+[a-z]?)?) (.*)$", name)
        num, name_clean = (mm.group(1), mm.group(2)) if mm else (None, name)
        sec, in_refs = name_clean, name_clean == "References"
        if name_clean == "Abstract":
            out.append("%ABSTRACT%")
        elif in_refs:
            out.append("%REFS%")
        elif name_clean.startswith("Appendix"):
            letter = re.match(r"Appendix ([A-Z]): (.*)", name_clean)
            if not in_appendix:
                out.append("\\appendices")
                in_appendix = True
            out.append(f"\\section{{{inline(letter.group(2))}}}\n\\label{{app:{letter.group(1)}}}")
        else:
            lab = f"\n\\label{{sec:{num}}}" if num else ""
            out.append(("\\section{" if lvl == 2 else "\\subsection{") + inline(name_clean) + "}" + lab)
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
    mt = re.match(r"^\*\*Table (A?\d+)\.\*\*\s*(.*)$", b, flags=re.S)
    if mt:
        pending_cap = (mt.group(1), mt.group(2).replace("\n", " "))
        continue
    if b.startswith("!["):
        fn = re.search(r"\((figures/[^)]+)\)", b).group(1)
        cap, fid = "", "?"
        if i < len(blocks) and blocks[i].lstrip().startswith("**Figure"):
            mf = re.match(r"^\*\*Figure (\d+)\.\*\*\s*(.*)$", blocks[i].strip(), flags=re.S)
            fid, cap = mf.group(1), mf.group(2).replace("\n", " ")
            i += 1
        if (root / fn).with_suffix(".pdf").exists():
            fn = fn[:-4]
        wide = "fig1" in fn or "fig3" in fn
        star = "*" if wide else ""
        w = r"0.68\textwidth" if wide else r"0.9\columnwidth"
        out.append(f"\\begin{{figure{star}}}[!t]\n\\centering\n\\includegraphics[width={w}]{{{fn}}}\n\\caption{{{inline(cap)}}}\n\\label{{fig:{fid}}}\n\\end{{figure{star}}}\n")
        continue
    lines = b.splitlines()
    if all(l.lstrip().startswith("|") for l in lines):
        if pending_cap is None:
            warn.append("table without caption near: " + lines[0][:60])
            out.append(table(lines))
        else:
            out.append(table(lines, pending_cap[1], pending_cap[0]))
            pending_cap = None
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
bib = "\\begin{thebibliography}{99}\n" + "\n".join(f"\\bibitem{{r{n + 1}}} {inline(r, refs=False)}" for n, r in enumerate(refs)) + "\n\\end{thebibliography}\n"
ack_f = root / "tnsm" / "ack.tex"
ack = ack_f.read_text(encoding="utf-8") if ack_f.exists() else ""
head = r"""\documentclass[journal]{IEEEtran}
\usepackage{graphicx}
\usepackage{adjustbox}
\usepackage{array}
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
\title{""" + inline(title, refs=False) + r"""}
\author{Dev~Nambiar and Jaya~Subalakshmi~Ramamoorthi%
\thanks{D. Nambiar is with the Vellore Institute of Technology, Vellore, India (e-mail: devnambiar06@gmail.com; ORCID: 0009-0007-5516-3316).}%
\thanks{J. S. Ramamoorthi is with the School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India (e-mail: jayasubalakshmi.r@vit.ac.in; ORCID: 0000-0001-8350-2871).}}
\markboth{IEEE Transactions on Machine Learning in Communications and Networking}{Nambiar and Ramamoorthi: The Cost of Stale Telemetry in Traffic Engineering}
\maketitle
"""
body = "\n".join(out).replace("%ABSTRACT%", abs_tex).replace("%REFS%", ack + "\n" + bib)
(root / "tnsm" / "main.tex").write_text(head + body + "\n\\end{document}\n", encoding="utf-8")
print("wrote main.tex;", len(refs), "refs;", len(out), "blocks;", len(warn), "warnings")
for w in warn:
    print("WARN", w)
