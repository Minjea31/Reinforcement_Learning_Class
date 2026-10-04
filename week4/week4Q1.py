from policy_eval import policy_eval
from gridworld import GridWorld
from collections import defaultdict
from pathlib import Path
from common.gridworld_render import Renderer


def main():
    gamma = 0.9  # 감쇠률
    output_dir = Path(__file__).resolve().parent / 'week4Q1_graphs'
    output_dir.mkdir(exist_ok=True)

    policy = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.7, 0.1, 0.1, 0.1],
    ]

    for i, prob in enumerate(policy, 1):
        env=GridWorld()
        pi = defaultdict(lambda: dict(enumerate(prob)))
        V = policy_eval(pi, defaultdict(lambda: 0), env, gamma)
        print(f'\n정책 {i}: {prob} (UP, DOWN, LEFT, RIGHT)')
        for y in range(env.height):
            print(' '.join('     WALL' if (y, x) == env.wall_state
                           else f'{V[y, x]:9.4f}' for x in range(env.width)))
        print(f'시작 상태 가치: {V[env.start_state]:.4f}')
        save_path = output_dir / f'policy_{i}.png'
        renderer = Renderer(env.reward_map, env.goal_state, env.wall_state)
        renderer.render_v(V, pi, save_path=save_path)
        print(f'그래프 저장: {save_path}')

if __name__ == '__main__':
    main()
