"""독립 실행: python evaluate.py --seed 0."""

import argparse, json
import numpy as np
from experiment import ROOT, Net, Grid, no_grad

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
s = json.loads((ROOT / "results" / "q1" / "summary.json").read_text())
c = s["configs"][s["best"]]
net = Net(25, 4, c["width"], c["depth"])
net.load_weights(str(ROOT / "results" / "q1" / s["best"] / f"seed{a.seed}.npz"))
with no_grad():
    q = net(np.eye(25, dtype=np.float32)).data
print(json.dumps(Grid().evaluate(q, c["gamma"]), indent=2))
print(q)
