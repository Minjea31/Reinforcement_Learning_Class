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
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Polygon

        Q = {} if Q is None else Q
        height, width = self.reward_map.shape
        values = [Q.get(((y, x), action), 0)
                  for y in range(height) for x in range(width)
                  if (y, x) not in self.wall_states and (y, x) != self.goal_state
                  for action in range(4)]
        scale = max(max((abs(value) for value in values), default=0), 1e-8)
        norm = Normalize(-scale, scale)
        cmap = LinearSegmentedColormap.from_list(
            'q_values', ['#ef7777', '#ffffff', '#438e48'])
        fig, ax = plt.subplots(figsize=(7, 5))

        # 행동 0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽
        triangles = {
            0: [(0, 0), (1, 0), (.5, .5)],
            1: [(0, 1), (1, 1), (.5, .5)],
            2: [(0, 0), (0, 1), (.5, .5)],
            3: [(1, 0), (1, 1), (.5, .5)],
        }
        text_positions = {0: (.5, .16), 1: (.5, .82),
                          2: (.20, .5), 3: (.80, .5)}

        for y in range(height):
            for x in range(width):
                state = (y, x)
                if state in self.wall_states:
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor='#666666',
                                           edgecolor='#888888'))
                    continue

                if state != self.goal_state:
                    for action, vertices in triangles.items():
                        value = Q.get((state, action), 0)
                        points = [(x + dx, y + dy) for dx, dy in vertices]
                        ax.add_patch(Polygon(points, facecolor=cmap(norm(value)),
                                             edgecolor='#dddddd', linewidth=.5))
                        if print_value:
                            dx, dy = text_positions[action]
                            ax.text(x + dx, y + dy, f'{value:.2f}',
                                    ha='center', va='center', fontsize=10)

                ax.add_patch(Rectangle((x, y), 1, 1,
                                       facecolor='white' if state == self.goal_state else 'none',
                                       edgecolor='#888888'))
                reward = self.reward_map[state]
                if reward:
                    label = f'R {reward:.1f}'
                    if state == self.goal_state:
                        label += ' (GOAL)'
                    ax.text(x + .05, y + .97, label,
                            ha='left', va='bottom', fontsize=9)

        ax.set(xlim=(0, width), ylim=(height, 0), aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        plt.show()
        plt.close(fig)
