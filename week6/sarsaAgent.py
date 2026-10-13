from collections import defaultdict, deque
import numpy as np
import Grid_World_render as render_helper
from GridWorld import GridWorld

def greedy_probs(Q, state, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    action_probs = {action : 0.0 for action in range(action_size)}
    # 이 시점에서 action_probs는 {0:0.0, 1:0.0, 2:0.0, 3:0.0}이 됨
    action_probs[max_action] = 1
    return action_probs

def epsilon_greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon/action_size
    action_probs = {action : base_prob for action in range(action_size)}
    # 이 시점에서 action_probs는 {0 : epsilon/action_size, 1 : epsilon/action_size, 2 : epsilon/action_size, 3 : epsilon/action_size}이 됨
    action_probs[max_action] += (1- epsilon)
    return action_probs

class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda : 0)
        self.memory = deque(maxlen=2) # deque를 사용하여 최근 2개의 상태, 행동, 보상을 저장


    def reset(self):
        self.memory.clear()

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) <2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        # 다음 Q 함수
        next_q = 0 if done else self.Q[next_state, next_action]

        # TD법으로 self.Q 갱신
        target = reward + self.gamma * next_q
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        # 정책 개선
        self.pi[state] = epsilon_greedy_probs(self.Q, state, self.epsilon)



env = GridWorld()
agent = SarsaAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done) # 매번 호출

        if done:
            # 목표에 도달했을 때도 호출
            agent.update(next_state, None, None, None)
            break
        state = next_state

env.render_q(agent.Q) # Q 함수 시각화
env.render_policy(agent.Q)