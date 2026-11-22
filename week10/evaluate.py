"""독립 실행: python evaluate.py q1|q2 --seed 0."""

import argparse, json
import numpy as np
from experiment import ROOT, Net, Grid, no_grad, evaluate_cart

p = argparse.ArgumentParser()
p.add_argument("problem", choices=["q1", "q2"])
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
s = json.loads((ROOT / "results" / a.problem / "summary.json").read_text())
c = s["configs"][s["best"]]
net = Net(
    25 if a.problem == "q1" else 4,
    4 if a.problem == "q1" else 2,
    c["width"],
    c["depth"],
)
net.load_weights(str(ROOT / "results" / a.problem / s["best"] / f"seed{a.seed}.npz"))
if a.problem == "q1":
    with no_grad():
        q = net(np.eye(25, dtype=np.float32)).data
    print(json.dumps(Grid().evaluate(q, c["gamma"]), indent=2))
    print(q)
else:
    scores = evaluate_cart(net, range(900000 + a.seed * 100, 900100 + a.seed * 100))
    print(
        json.dumps(
            dict(mean=np.mean(scores), std=np.std(scores), scores=scores), indent=2
        )
    )
