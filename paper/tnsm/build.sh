#!/bin/bash
# Builds paper/tnsm/main.pdf. MiKTeX fails on a malformed PATH entry on this machine, so use a minimal PATH.
cd "$(dirname "$0")"
export PATH="/c/Users/flame/AppData/Local/Programs/MiKTeX/miktex/bin/x64:/usr/bin:/c/WINDOWS/system32:/c/WINDOWS"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 && pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log 2>&1
tail -n 15 build.log
