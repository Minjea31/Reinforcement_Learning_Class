"""독립 실행: python evaluate.py --seed 0."""

import argparse, json
import numpy as np
from experiment import ROOT, Net, no_grad, evaluate_cart

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
s = json.loads((ROOT / "results" / "q2" / "summary.json").read_text())
c = s["configs"][s["best"]]
net = Net(4, 2, c["width"], c["depth"])
net.load_weights(str(ROOT / "results" / "q2" / s["best"] / f"seed{a.seed}.npz"))
scores = evaluate_cart(net, range(900000 + a.seed * 100, 900100 + a.seed * 100))
print(
    json.dumps(dict(mean=np.mean(scores), std=np.std(scores), scores=scores), indent=2)
)
