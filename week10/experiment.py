"""DeZero Q-Network/DQN. python experiment.py q1|q2 --workers 8."""

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
        return t, float(t == self.goal) - float(t in self.bad), t == self.goal

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
            agreement=float(np.mean(ref[ids, chosen] >= ref[ids].max(1) - 1e-7)),
            start_path=paths[self.states.index(self.start)],
        )


BASE = {
    "q1": dict(gamma=0.9, lr=0.01, epsilon=0.1, episodes=1000, width=100, depth=1),
    "q2": dict(
        gamma=0.98,
        lr=0.0005,
        epsilon=0.05,
        episodes=300,
        width=128,
        depth=2,
        buffer_size=100000,
        batch_size=32,
        sync_interval=20,
    ),
}
SWEEPS = {
    "q1": dict(
        gamma=[0.8, 0.9, 0.95, 0.99],
        lr=[0.001, 0.01, 0.05, 0.1],
        epsilon=[0.01, 0.05, 0.1, 0.2],
        episodes=[500, 1000, 2000, 4000],
        depth=[1, 2, 3],
        width=[32, 64, 100, 128],
    ),
    "q2": dict(
        gamma=[0.9, 0.95, 0.98, 0.99],
        lr=[0.0001, 0.0005, 0.001, 0.005],
        epsilon=[0.01, 0.05, 0.1, 0.2],
        buffer_size=[1000, 10000, 50000, 100000],
        batch_size=[16, 32, 64, 128],
        episodes=[100, 200, 300, 500],
        sync_interval=[1, 5, 20, 50],
        depth=[1, 2, 3],
        width=[32, 64, 128, 256],
    ),
}


def key(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def evaluate_cart(net, seeds):
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    scores = []
    with no_grad():
        for seed in seeds:
            s, _ = env.reset(seed=int(seed))
            total = 0
            for _ in range(500):
                a = int(net(np.asarray(s, np.float32)[None]).data.argmax())
                s, r, term, trunc, _ = env.step(a)
                total += r
                if term or trunc:
                    break
            scores.append(total)
    env.close()
    return scores


def train(task):
    problem, c, seed = task
    out = ROOT / "results" / problem / key(c)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"seed{seed}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    start = time.time()
    grid = Grid()
    eye = np.eye(25, dtype=np.float32)
    net = Net(
        25 if problem == "q1" else 4,
        4 if problem == "q1" else 2,
        c["width"],
        c["depth"],
    )
    opt = (
        optimizers.SGD(c["lr"]) if problem == "q1" else optimizers.Adam(c["lr"])
    ).setup(net)
    rewards = []
    losses = []
    lengths = []
    trace = []
    failed = False
    if problem == "q2":
        import gymnasium as gym

        env = gym.make("CartPole-v1")
        target = copy.deepcopy(net)
        cap = c["buffer_size"]
        batch = c["batch_size"]
        count = 0
        bs = np.empty((cap, 4), np.float32)
        bn = bs.copy()
        ba = np.empty(cap, np.int64)
        br = np.empty(cap, np.float32)
        bd = br.copy()
    for ep in range(c["episodes"]):
        s = grid.start if problem == "q1" else env.reset(seed=seed * 100000 + ep)[0]
        total = 0
        ls = []
        for step in range(200 if problem == "q1" else 500):
            x = (
                eye[s[0] * 5 + s[1]][None]
                if problem == "q1"
                else np.asarray(s, np.float32)[None]
            )
            with no_grad():
                a = (
                    int(rng.integers(4 if problem == "q1" else 2))
                    if rng.random() < c["epsilon"]
                    else int(net(x).data.argmax())
                )
            if problem == "q1":
                ns, r, term = grid.transition(s, a)
                trunc = False
                nx = eye[ns[0] * 5 + ns[1]][None]
                with no_grad():
                    y = np.array(
                        [r + c["gamma"] * (not term) * net(nx).data.max()], np.float32
                    )
                pred = net(x)[:, a]
                loss = F.mean_squared_error(pred, y)
            else:
                ns, r, term, trunc, _ = env.step(a)
                i = count % cap
                bs[i] = s
                bn[i] = ns
                ba[i] = a
                br[i] = r
                bd[i] = term
                count += 1
                if count < batch:
                    s = ns
                    total += r
                    if term or trunc:
                        break
                    continue
                ix = rng.choice(min(count, cap), batch, replace=False)
                with no_grad():
                    y = br[ix] + c["gamma"] * (1 - bd[ix]) * target(bn[ix]).data.max(1)
                pred = net(bs[ix])[np.arange(batch), ba[ix]]
                loss = F.mean_squared_error(pred, y)
            lv = float(loss.data)
            if not np.isfinite(lv) or lv > 1e12:
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
        if problem == "q2" and (ep + 1) % c["sync_interval"] == 0:
            target = copy.deepcopy(net)
        if problem == "q1" and (ep + 1) % 100 == 0:
            with no_grad():
                met = grid.evaluate(net(eye).data, c["gamma"])
            trace.append([ep + 1, met["success"], met["rmse"]])
        if failed:
            break
    net.save_weights(str(out / f"seed{seed}.npz"))
    if problem == "q1":
        with no_grad():
            q = net(eye).data
        np.save(out / f"q_seed{seed}.npy", q)
        metrics = grid.evaluate(q, c["gamma"])
    else:
        env.close()
        scores = evaluate_cart(net, range(500000 + seed * 100, 500020 + seed * 100))
        metrics = dict(
            score=float(np.mean(scores)), score_std=float(np.std(scores)), scores=scores
        )
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
    if any(x["failed"] for x in rows):
        return (-1e12,)
    m = [x["metrics"] for x in rows]
    return (
        (
            np.mean([x["success"] for x in m]),
            np.mean([x["agreement"] for x in m]),
            -np.mean([x["steps"] for x in m]),
            -np.mean([x["rmse"] for x in m]),
        )
        if problem == "q1"
        else (np.mean([x["score"] for x in m]), -np.std([x["score"] for x in m]))
    )


def run(problem, workers):
    configs = {key(BASE[problem]): BASE[problem]}
    for name, vals in SWEEPS[problem].items():
        for v in vals:
            c = {**BASE[problem], name: v}
            configs[key(c)] = c
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(train, (problem, c, s)): (k, s)
            for k, c in configs.items()
            for s in range(3)
        }
        for f in as_completed(jobs):
            k, s = jobs[f]
            r = f.result()
            rows.setdefault(k, []).append(r)
            print(problem, k, s, r["metrics"], flush=True)
    combo = BASE[problem].copy()
    for name, vals in SWEEPS[problem].items():
        best = max(
            vals, key=lambda v: rank(problem, rows[key({**BASE[problem], name: v})])
        )
        combo[name] = best
    if key(combo) not in rows:
        rows[key(combo)] = [train((problem, combo, s)) for s in range(3)]
        configs[key(combo)] = combo
    best = max(rows, key=lambda k: rank(problem, rows[k]))
    summary = dict(
        problem=problem,
        baseline=BASE[problem],
        sweeps=SWEEPS[problem],
        configs=configs,
        rows=rows,
        combo=key(combo),
        best=best,
    )
    if problem == "q2":
        tests = []
        for s in range(3):
            c = configs[best]
            np.random.seed(s)
            net = Net(4, 2, c["width"], c["depth"])
            net.load_weights(str(ROOT / "results" / problem / best / f"seed{s}.npz"))
            tests.append(evaluate_cart(net, range(900000 + s * 100, 900100 + s * 100)))
        summary["independent_tests"] = tests
    (ROOT / "results" / problem / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("COMPLETE", problem, best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("problem", choices=["q1", "q2"])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    run(a.problem, a.workers)
