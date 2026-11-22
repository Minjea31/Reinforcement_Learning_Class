"""결과 CSV 및 Q2 최종 정책의 실제 1회 상태 궤적 저장."""

from experiment import *
import csv

for problem in ["q1", "q2"]:
    path = ROOT / "results" / problem / "summary.json"
    if not path.exists():
        continue
    s = json.loads(path.read_text())
    cols = list(s["baseline"])
    metrics = (
        ["success", "agreement", "steps", "rmse"]
        if problem == "q1"
        else ["score", "score_std"]
    )
    with (path.parent / "all_metrics.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["config_id", "seed", *cols, *metrics, "seconds"])
        for k, rs in s["rows"].items():
            for r in rs:
                w.writerow(
                    [
                        k,
                        r["seed"],
                        *[r["config"][n] for n in cols],
                        *[r["metrics"][n] for n in metrics],
                        r["seconds"],
                    ]
                )
    if problem == "q2":
        import gymnasium as gym

        c = s["configs"][s["best"]]
        net = Net(4, 2, c["width"], c["depth"])
        net.load_weights(str(path.parent / s["best"] / "seed0.npz"))
        env = gym.make("CartPole-v1")
        state, _ = env.reset(seed=900000)
        trace = []
        with no_grad():
            for t in range(500):
                a = int(net(np.asarray(state, np.float32)[None]).data.argmax())
                trace.append([t, *map(float, state), a])
                state, r, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
        env.close()
        (path.parent / "final_rollout.json").write_text(
            json.dumps(
                dict(
                    seed=900000,
                    model_seed=0,
                    config=s["best"],
                    trace=trace,
                    terminated=bool(term),
                    truncated=bool(trunc),
                ),
                indent=2,
            )
        )
    print(problem, "exported")
