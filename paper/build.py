"""Build paper/paper.html (self-contained) and paper/paper.pdf (via headless Edge/Chrome) from paper/paper.md.
  python paper/build.py"""
import base64, re, subprocess, sys
from pathlib import Path
import markdown

root = Path(__file__).resolve().parent
md = (root / "paper.md").read_text(encoding="utf-8")
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
def embed(m):
    f = root / m.group(1)
    return f'src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}"' if f.exists() else m.group(0)
body = re.sub(r'src="(figures/[^"]+)"', embed, body)
title = re.search(r"<h1>(.*?)</h1>", body).group(1)
css = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10.5pt; line-height: 1.35; color: #111; max-width: 178mm; margin: 0 auto; }
h1 { font-size: 18pt; text-align: center; margin: 0 0 6pt; line-height: 1.2; }
h2 { font-size: 12.5pt; margin: 16pt 0 4pt; border-bottom: 0.5pt solid #888; padding-bottom: 1pt; }
h3 { font-size: 11pt; margin: 12pt 0 3pt; }
p { margin: 4pt 0; text-align: justify; hyphens: auto; }
table { border-collapse: collapse; margin: 6pt auto; font-size: 8.3pt; line-height: 1.2; page-break-inside: avoid; }
th, td { border: 0.4pt solid #999; padding: 1.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; }
img { display: block; max-width: 100%; margin: 6pt auto; page-break-inside: avoid; }
blockquote { margin: 6pt 14pt; font-style: italic; }
em { font-style: italic; }
ul, ol { margin: 3pt 0 3pt 16pt; padding: 0; }
li { margin: 1.5pt 0; }
"""
html = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{re.sub('<[^>]+>', '', title)}</title><style>{css}</style></head><body>{body}</body></html>"
(root / "paper.html").write_text(html, encoding="utf-8")
print("wrote", root / "paper.html")
for exe in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
    if Path(exe).exists():
        pdf = root / "paper.pdf"
        subprocess.run([exe, "--headless=new", "--disable-gpu", f"--print-to-pdf={pdf}", "--no-pdf-header-footer", (root / "paper.html").as_uri()], check=True, timeout=180)
        print("wrote", pdf); break
else:
    print("no Edge/Chrome found; open paper.html and print to PDF", file=sys.stderr)
