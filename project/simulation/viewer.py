"""An optional, latest-frame-only view of the actual training/evaluation state."""
import multiprocessing as mp
import queue
import time


def _display(frames, title):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import mujoco
    import mujoco.viewer
    from simulation.curriculum import make_env
    env = make_env()
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat[:] = [0, 0, .8]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.55, 110, -35
            while viewer.is_running():
                try:
                    item = frames.get(timeout=.05)
                except queue.Empty:
                    continue
                if item is None:
                    break
                state, metrics = item
                with viewer.lock():
                    env.set_state(state)
                viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, title, metrics)])
                viewer.sync()
    finally:
        env.close()


class TrainingViewer:
    def __init__(self, enabled, title='PPO: actual training worker 0',interval=.075):
        self.frames = mp.get_context('spawn').Queue(maxsize=1) if enabled else None
        self.process = None
        self.last = 0.
        self.interval = interval
        if enabled:
            self.process = mp.get_context('spawn').Process(target=_display,args=(self.frames,title),daemon=True)
            self.process.start()

    def ready(self):
        return self.process is not None and self.process.is_alive() and time.monotonic()-self.last >= self.interval

    def send(self, state, metrics):
        if self.ready():
            self.last = time.monotonic()
            try:
                self.frames.put_nowait((state, metrics))
            except queue.Full:
                try:
                    self.frames.get_nowait()
                    self.frames.put_nowait((state, metrics))
                except (queue.Empty, queue.Full):
                    pass

    def close(self):
        if self.process is not None:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put(None, timeout=.2)
            except queue.Full:
                pass
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            self.frames.cancel_join_thread()
            self.frames.close()
