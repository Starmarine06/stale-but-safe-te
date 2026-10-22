"""TCP emulation: does the guard flap between ECMP fallback and the learned proposal?
Reads runs/emu_tcp{,_r1,_r2} guard0.05 controller logs and counts fallback transitions and flow-mods per interval."""
import glob
import json
import sys

import numpy as np

if len(sys.argv) > 1 and sys.argv[1] == "keys":
    d = json.load(open(glob.glob("runs/emu_tcp/n14/guard0.05*/ctrl_*.json")[0]))
    print(list(d.keys()))
    print(d.get("guard"))
    print(d["rows"][5])
    sys.exit()
