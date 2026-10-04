"""Q2 그림의 5×5 환경. 좌표는 (행, 열), 0부터 시작한다."""

import numpy as np
from gridworld import GridWorld


class GridWorldQ2(GridWorld):
    def __init__(self):
        super().__init__()
        self.reward_map = np.array([
            [0, 0, 0, -1.0, 1.0],
            [0, 0, 0, 0, 0],
            [0, None, None, 0, 0],
            [0, 0, 0, 0, -1.0],
            [0, 0, 0, 0, 0],
        ], dtype=object)
        self.goal_state = (0, 4)
        self.wall_states = {(2, 1), (2, 2)}
        self.wall_state = (2, 1)  # 기존 Renderer 호출과의 호환
        self.start_state = (4, 0)
        self.agent_state = self.start_state

    def states(self):
        for state in super().states():
            if state not in self.wall_states:
                yield state

    def next_state(self, state, action):
        next_state = super().next_state(state, action)
        if next_state in self.wall_states:
            return state
        return next_state
