"""5x5 GridWorld에서 epsilon 비교 4개, alpha 비교 4개의 Q값과 정책을 저장.

실행: python week5/06_quiz.py
예시: python week5/06_quiz.py --episodes 10000
각 에피소드는 스텝 제한 없이 목표에 도달할 때까지 진행한다.
epsilon 비교는 alpha=0.1, alpha 비교는 epsilon=0.1로 고정한다.
"""

import argparse
import csv
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')  # 창을 띄우지 않고 이미지 파일로 저장
import matplotlib.pyplot as plt
import numpy as np

from GridWorld import GridWorld, McAgent, greedy_probs


EPSILONS = (0.1, 0.2, 0.4, 0.6)
ALPHAS = (0.05, 0.1, 0.3, 0.5)
FIXED_ALPHA = 0.1
FIXED_EPSILON = 0.1
DEFAULT_OUTPUT = Path(__file__).resolve().parent / '06_quiz_results'


class QuizGridWorld(GridWorld):
    """사진의 환경. 좌표는 (행, 열), 보상 -1 칸은 종료 상태가 아니다."""

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
        self.wall_state = (2, 1)
        self.wall_states = {(2, 1), (2, 2)}
        self.start_state = (4, 0)
        self.agent_state = self.start_state

    def next_state(self, state, action):
        next_state = super().next_state(state, action)
        # 기존 함수가 처리하는 벽 한 칸에 더해 두 번째 벽도 처리
        return state if next_state in self.wall_states else next_state


def train(epsilon, alpha, episodes, seed):
    # 모든 조건에서 같은 난수 시드와 기존의 초기 무작위 정책으로 시작
    np.random.seed(seed)
    env = QuizGridWorld()
    agent = McAgent()
    agent.epsilon = epsilon
    agent.alpha = alpha
    completed = 0
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()
        agent.reset()
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add(state, action, reward)
            total_steps += 1
            if done:
                agent.exponential_moving_average_update()
                completed += 1
                break
            state = next_state

    stats = {
        'epsilon': epsilon,
        'alpha': alpha,
        'episodes': episodes,
        'completed': completed,
        'total_steps': total_steps,
        'seed': seed,
    }
    return env, agent, stats


def save_results(env, agent, stats, output_dir):
    stem = f"epsilon_{stats['epsilon']:g}_alpha_{stats['alpha']:g}"
    title = f"epsilon={stats['epsilon']:g}, alpha={stats['alpha']:g}"
    subtitle = f"episodes={stats['episodes']}, goal={stats['completed']}"
    paths = {}

    # render_q/render_v는 show() 다음에 figure를 닫으므로, show 시점에 저장.
    # 기존 렌더링 함수 자체는 수정하지 않는다.
    def save_current(kind):
        fig = plt.gcf()
        fig.set_size_inches(10, 11)
        for text in fig.axes[0].texts:
            text.set_fontsize(8 if text.get_text().startswith('R ') else 9)
        if kind == 'policy':
            ax = fig.axes[0]
            for cell in ax.patches:
                x, y = cell.get_xy()
                if (int(y), int(x)) not in env.wall_states:
                    cell.set_facecolor('white')
            for text in ax.texts:
                if not text.get_text().startswith('R '):
                    text.set_fontsize(22)
        fig.suptitle(f"{kind.upper()} | {title}\n{subtitle}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        path = output_dir / f'{stem}_{kind}.png'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        paths[kind] = path

    with patch.object(plt, 'show', lambda: save_current('q')):
        env.render_q(agent.Q)

    # 학습에 사용한 epsilon-greedy 정책이 아닌, 최종 Q에 대한 greedy 정책.
    policy = {state: greedy_probs(agent.Q, state)
              for state in env.states()
              if state not in env.wall_states and state != env.goal_state}
    with patch.object(plt, 'show', lambda: save_current('policy')):
        env.render_v(policy=policy, print_value=False)
    return paths


def save_comparison(results, conditions, kind, group, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 18))
    for ax, (epsilon, alpha) in zip(axes.flat, conditions):
        ax.imshow(plt.imread(results[(epsilon, alpha)][kind]))
        ax.axis('off')
    fixed = (f'alpha={FIXED_ALPHA:g}' if group == 'epsilon'
             else f'epsilon={FIXED_EPSILON:g}')
    fig.suptitle(f'{kind.upper()} | {group} comparison ({fixed} fixed)', fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output_dir / f'{kind}_comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('0보다 큰 정수를 입력하세요.')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=positive_int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    groups = {
        'epsilon': [(epsilon, FIXED_ALPHA) for epsilon in EPSILONS],
        'alpha': [(FIXED_EPSILON, alpha) for alpha in ALPHAS],
    }
    for group, conditions in groups.items():
        output_dir = args.output_dir / f'{group}_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for epsilon, alpha in conditions:
            print(f'[{group} comparison] Training: epsilon={epsilon:g}, alpha={alpha:g}', flush=True)
            env, agent, stats = train(
                epsilon, alpha, args.episodes, args.seed)
            results[(epsilon, alpha)] = save_results(env, agent, stats, output_dir)
            rows.append({'comparison': group, **stats})
            print(f"  goal={stats['completed']}/{args.episodes}", flush=True)

        for kind in ('q', 'policy'):
            save_comparison(results, conditions, kind, group, output_dir)
    with (args.output_dir / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved 16 individual PNGs, 4 comparison PNGs and summary.csv: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
