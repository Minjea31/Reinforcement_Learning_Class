"""Q1 독립 제출 코드. python experiment.py --workers 8."""

import os

for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import sys, json, time, hashlib, copy, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/deep-learning-from-scratch-3"))
from dezero import Model, no_grad
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F


class Net(Model):
    def __init__(self, n, a, width, depth):
        super().__init__()
        self.layers = []
        for i, (x, y) in enumerate(zip([n] + [width] * depth, [width] * depth + [a])):
            l = L.Linear(y, in_size=x, dtype=np.float32)
            setattr(self, "l" + str(i), l)
            self.layers.append(l)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)


class Grid:
    walls = {(2, 1), (2, 2)}
    goal = (0, 4)
    start = (4, 0)
    bad = {(0, 3), (3, 4)}
    states = [
        (y, x)
        for y in range(5)
        for x in range(5)
        if (y, x) not in {(2, 1), (2, 2), (0, 4)}
    ]

    def transition(self, s, a):
        dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
        t = (s[0] + dy, s[1] + dx)
        if not (0 <= t[0] < 5 and 0 <= t[1] < 5) or t in self.walls:
            t = s
        return (t, float(t == self.goal) - float(t in self.bad), t == self.goal)

    def optimal(self, gamma):
        q = np.zeros((25, 4))
        for _ in range(10000):
            old = q.copy()
            for s in self.states:
                for a in range(4):
                    t, r, d = self.transition(s, a)
                    q[s[0] * 5 + s[1], a] = (
                        r + gamma * (not d) * old[t[0] * 5 + t[1]].max()
                    )
            if np.max(abs(q - old)) < 1e-10:
                break
        return q

    def evaluate(self, q, gamma):
        ref = self.optimal(gamma)
        ids = [y * 5 + x for y, x in self.states]
        success = []
        length = []
        returns = []
        paths = []
        for start in self.states:
            s = start
            path = [s]
            total = 0
            for t in range(200):
                a = int(q[s[0] * 5 + s[1]].argmax())
                s, r, d = self.transition(s, a)
                total += gamma**t * r
                path.append(s)
                if d:
                    break
            success.append(int(d))
            length.append(t + 1)
            returns.append(total)
            paths.append(path)
        chosen = q[ids].argmax(1)
        return dict(
            success=float(np.mean(success)),
            steps=float(np.mean(length)),
            value=float(np.mean(returns)),
            rmse=float(np.sqrt(np.mean((q[ids] - ref[ids]) ** 2))),
            agreement=float(np.mean(ref[ids, chosen] >= ref[ids].max(1) - 1e-07)),
            start_path=paths[self.states.index(self.start)],
        )


BASE = {"q1": dict(gamma=0.9, lr=0.01, epsilon=0.1, episodes=1000, width=100, depth=1)}
SWEEPS = {
    "q1": dict(
        gamma=[0.8, 0.9, 0.95, 0.99],
        lr=[0.001, 0.01, 0.05, 0.1],
        epsilon=[0.01, 0.05, 0.1, 0.2],
        episodes=[500, 1000, 2000, 4000],
        depth=[1, 2, 3],
        width=[32, 64, 100, 128],
    )
}


def key(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def train(task):
    problem, c, seed = task
    out = ROOT / "results" / "q1" / key(c)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"seed{seed}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    start = time.time()
    grid = Grid()
    eye = np.eye(25, dtype=np.float32)
    net = Net(25, 4, c["width"], c["depth"])
    opt = optimizers.SGD(c["lr"]).setup(net)
    rewards = []
    losses = []
    lengths = []
    trace = []
    failed = False
    for ep in range(c["episodes"]):
        s = grid.start
        total = 0
        ls = []
        for step in range(200):
            x = eye[s[0] * 5 + s[1]][None]
            with no_grad():
                a = (
                    int(rng.integers(4))
                    if rng.random() < c["epsilon"]
                    else int(net(x).data.argmax())
                )
            ns, r, term = grid.transition(s, a)
            trunc = False
            nx = eye[ns[0] * 5 + ns[1]][None]
            with no_grad():
                y = np.array(
                    [r + c["gamma"] * (not term) * net(nx).data.max()], np.float32
                )
            pred = net(x)[:, a]
            loss = F.mean_squared_error(pred, y)
            lv = float(loss.data)
            if not np.isfinite(lv) or lv > 1000000000000.0:
                failed = True
                break
            net.cleargrads()
            loss.backward()
            opt.update()
            ls.append(lv)
            total += r
            s = ns
            if term or trunc:
                break
        rewards.append(total)
        losses.append(float(np.mean(ls)) if ls else 0)
        lengths.append(step + 1)
        if False and (ep + 1) % c["sync_interval"] == 0:
            target = copy.deepcopy(net)
        if True and (ep + 1) % 100 == 0:
            with no_grad():
                met = grid.evaluate(net(eye).data, c["gamma"])
            trace.append([ep + 1, met["success"], met["rmse"]])
        if failed:
            break
    net.save_weights(str(out / f"seed{seed}.npz"))
    with no_grad():
        q = net(eye).data
    np.save(out / f"q_seed{seed}.npy", q)
    metrics = grid.evaluate(q, c["gamma"])
    result = dict(
        config=c,
        seed=seed,
        failed=failed,
        seconds=time.time() - start,
        rewards=rewards,
        losses=losses,
        lengths=lengths,
        trace=trace,
        metrics=metrics,
    )
    dest.write_text(json.dumps(result, indent=2))
    return result


def rank(problem, rows):
    if any((x["failed"] for x in rows)):
        return (-1000000000000.0,)
    m = [x["metrics"] for x in rows]
    return (
        np.mean([x["success"] for x in m]),
        np.mean([x["agreement"] for x in m]),
        -np.mean([x["steps"] for x in m]),
        -np.mean([x["rmse"] for x in m]),
    )


def run(problem, workers):
    configs = {key(BASE["q1"]): BASE["q1"]}
    for name, vals in SWEEPS["q1"].items():
        for v in vals:
            c = {**BASE["q1"], name: v}
            configs[key(c)] = c
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(train, ("q1", c, s)): (k, s)
            for k, c in configs.items()
            for s in range(3)
        }
        for f in as_completed(jobs):
            k, s = jobs[f]
            r = f.result()
            rows.setdefault(k, []).append(r)
            print("q1", k, s, r["metrics"], flush=True)
    combo = BASE["q1"].copy()
    for name, vals in SWEEPS["q1"].items():
        best = max(vals, key=lambda v: rank("q1", rows[key({**BASE["q1"], name: v})]))
        combo[name] = best
    if key(combo) not in rows:
        rows[key(combo)] = [train(("q1", combo, s)) for s in range(3)]
        configs[key(combo)] = combo
    best = max(rows, key=lambda k: rank("q1", rows[k]))
    summary = dict(
        problem="q1",
        baseline=BASE["q1"],
        sweeps=SWEEPS["q1"],
        configs=configs,
        rows=rows,
        combo=key(combo),
        best=best,
    )
    (ROOT / "results" / "q1" / "summary.json").write_text(json.dumps(summary, indent=2))
    print("COMPLETE", "q1", best, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run("q1", args.workers)
