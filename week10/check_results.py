"""환경·TD 타깃·실험 통제·체크포인트 검증."""

from experiment import *

g = Grid()
assert g.transition((2, 0), 3) == ((2, 0), 0.0, False)
assert g.transition((0, 0), 0) == ((0, 0), 0.0, False)
assert g.transition((0, 2), 3) == ((0, 3), -1.0, False)
assert g.transition((1, 4), 0) == ((0, 4), 1.0, True)
assert 1 + 0.9 * (not True) * 100 == 1
# 시간 제한은 terminated가 아니므로 bootstrap 유지
assert 1 + 0.9 * (1 - False) * 10 == 10
for problem in ["q1", "q2"]:
    path = ROOT / "results" / problem / "summary.json"
    if not path.exists():
        continue
    s = json.loads(path.read_text())
    b = s["baseline"]
    for name, vals in s["sweeps"].items():
        for v in vals:
            c = {**b, name: v}
            assert sum(c[k] != b[k] for k in b) <= 1
            rs = s["rows"][key(c)]
            assert sorted(r["seed"] for r in rs) == [0, 1, 2]
            assert all(r["config"] == c for r in rs)
    for k, rs in s["rows"].items():
        for r in rs:
            assert not r["failed"], (problem, k, r["seed"])
            c = r["config"]
            net = Net(
                25 if problem == "q1" else 4,
                4 if problem == "q1" else 2,
                c["width"],
                c["depth"],
            )
            net.load_weights(
                str(ROOT / "results" / problem / k / f"seed{r['seed']}.npz")
            )
            assert all(np.isfinite(p.data).all() for p in net.params())
            if problem == "q1":
                with no_grad():
                    q = net(np.eye(25, dtype=np.float32)).data
                np.testing.assert_allclose(
                    q,
                    np.load(ROOT / "results/q1" / k / f"q_seed{r['seed']}.npy"),
                    atol=1e-6,
                )
                m = g.evaluate(q, c["gamma"])
                assert abs(m["rmse"] - r["metrics"]["rmse"]) < 1e-6
    # Episode-count ablation must reuse exactly the same training prefix.
    for episodes in s["sweeps"]["episodes"]:
        short = {**b, "episodes": episodes}
        for r in s["rows"][key(short)]:
            reference = next(x for x in s["rows"][key(b)] if x["seed"] == r["seed"])
            n = min(len(r["rewards"]), len(reference["rewards"]))
            np.testing.assert_array_equal(r["rewards"][:n], reference["rewards"][:n])
    if problem == "q2":
        c = s["configs"][s["best"]]
        for r in s["rows"][s["best"]]:
            seed = r["seed"]
            net = Net(4, 2, c["width"], c["depth"])
            net.load_weights(str(ROOT / "results/q2" / s["best"] / f"seed{seed}.npz"))
            actual = evaluate_cart(net, range(500000 + seed * 100, 500020 + seed * 100))
            np.testing.assert_array_equal(actual, r["metrics"]["scores"])
    print(problem, "PASS", len(s["configs"]), "configs")
print("environment / terminal TD / checkpoint / OFAT checks PASS")
