"""DQN warm-up regression and no-gradient target tests, using temporary artifacts."""

import tempfile
import warnings
from pathlib import Path
import experiment as e
import numpy as np

# Replay warm-up may cross an episode boundary before batch_size experiences.
# Gymnasium warns if step() is called again after termination without reset().
with tempfile.TemporaryDirectory() as folder:
    original = e.ROOT
    try:
        e.ROOT = Path(folder)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            c = {**e.BASE["q2"], "episodes": 3, "batch_size": 128}
            r = e.train(("q2", c, 0))
        assert len(r["rewards"]) == 3 and not r["failed"]
    finally:
        e.ROOT = original

net = e.Net(4, 2, 32, 1)
target = e.copy.deepcopy(net)
x = np.zeros((4, 4), np.float32)
with e.no_grad():
    y = np.ones(4, np.float32) + 0.98 * target(x).data.max(1)
loss = e.F.mean_squared_error(net(x)[:, 0], y)
net.cleargrads()
loss.backward()
assert all(p.grad is None for p in target.params())
assert any(p.grad is not None for p in net.params())
print("PASS: warm-up terminal/reset and detached target gradients")
