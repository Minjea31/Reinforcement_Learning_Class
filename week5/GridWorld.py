import numpy as np
import Grid_World_render as render_helper
from collections import defaultdict

class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0:'UP',
            1:'DOWN',
            2:'LEFT',
            3:'RIGHT',
        }

        self.reward_map = np.array(
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )

        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h,w)


    def next_state(self, state, action):
        action_move_map = [(-1,0), (1,0), (0,-1), (0,1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def render_v(self, V=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(V, policy, print_value)

    def render_q(self, Q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(Q, print_value)

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.V = defaultdict(lambda : 0)
        self.cnts = defaultdict(lambda : 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]

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

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.Q = defaultdict(lambda : 0) # V가 아닌 Q를 사용
        # self.cnts = defaultdict(lambda : 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        # print(self.pi[(2,0)])
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.cnts[key] += 1
            self.Q[key] += (G - self.Q[key]) / self.cnts[key]

            # state 의 정책 탐욕화
            self.pi[state] = epsilon_greedy_probs(self.Q, state, self.epsilon)

    def exponential_moving_average_update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            
            self.Q[key] += (G - self.Q[key]) * self.alpha

            # state 의 정책 탐욕화
            self.pi[state] = epsilon_greedy_probs(self.Q, state, self.epsilon)


"""
env = GridWorld()
action = 0 # 더미 행동
next_state, reward, done = env.step(action)

print('next_state : ', next_state)
print('reward : ', reward)
print('done : ', done)
"""

"""
env = GridWorld()
agent = RandomAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state) # 행동선택
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.eval()
            break

        state = next_state
    print(agent.memory)
    env.render_v(agent.V)

env.render_v(agent.V)
"""

if __name__ == '__main__':
    env = GridWorld()
    agent = McAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state) # 행동선택
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.exponential_moving_average_update()
                break

            state = next_state
        # print(agent.memory)
        # print(agent.pi[(2,0)])

    env.render_q(agent.Q)
