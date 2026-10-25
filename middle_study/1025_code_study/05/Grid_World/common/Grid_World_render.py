"""강의 스크린샷에 구현이 없는 시각화 보조 모듈."""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


class Renderer:
    def __init__(self, reward_map, goal_state, wall_state):
        self.reward_map = reward_map
        self.goal_state = goal_state
        self.wall_state = wall_state
        self.wall_states = {(y, x) for y in range(reward_map.shape[0])
                            for x in range(reward_map.shape[1])
                            if reward_map[y, x] is None}

    def render_v(self, V=None, policy=None, print_value=True, save_path=None):
        V = {} if V is None else V
        height, width = self.reward_map.shape
        values = [V.get((y, x), 0) for y in range(height)
                  for x in range(width) if (y, x) not in self.wall_states]
        scale = max(max(abs(v) for v in values), 1e-8)
        norm = Normalize(-scale, scale)
        cmap = plt.get_cmap('RdYlGn')
        fig, ax = plt.subplots(figsize=(7, 5))
        arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        for y in range(height):
            for x in range(width):
                state = (y, x)
                color = '#666666' if state in self.wall_states else cmap(norm(V.get(state, 0)))
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color,
                                       edgecolor='#888888'))
                if state in self.wall_states:
                    continue
                if print_value:
                    ax.text(x + .95, y + .12, f'{V.get(state, 0):.2f}',
                            ha='right', va='top')
                reward = self.reward_map[state]
                if reward:
                    label = f'R {reward:.1f}'
                    if state == self.goal_state:
                        label += ' (GOAL)'
                    ax.text(x + .5, y + .88, label, ha='center', va='bottom')
                if policy is not None and state != self.goal_state:
                    probs = policy[state]
                    best = max(probs.values())
                    label = ' '.join(arrows[a] for a, p in probs.items() if p == best)
                    ax.text(x + .5, y + .5, label, ha='center', va='center')
        ax.set(xlim=(0, width), ylim=(height, 0), aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def render_q(self, Q=None, print_value=True):
        Q = {} if Q is None else Q
        height, width = self.reward_map.shape
        V = {(y, x): max(Q.get(((y, x), a), 0) for a in range(4))
             for y in range(height) for x in range(width)}
        policy = {}
        for state in V:
            qs = [Q.get((state, a), 0) for a in range(4)]
            best = max(qs)
            count = qs.count(best)
            policy[state] = {a: 1 / count if q == best else 0
                             for a, q in enumerate(qs)}
        self.render_v(V, policy, print_value)
