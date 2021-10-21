from os import stat
import numpy as np


class Strategy():
    def __init__(self, graph, actions, eposilon, alpha, gamma):
        self.graph = graph # graph
        self.actions = actions # actions set
        self.done = False # finished or not
        self.state = np.array([3, 0]) # current state
        self.action = 0 # current action 
        self.eposilon = eposilon # greedy strategy parameter
        self.row, self.col = self.graph.shape # size of the graph
        self.rewards = 0 # the reward of current episode
        self.reward_list = np.zeros(0) # restore the reward of all the episodes
        self.q = np.zeros((4, 12, 4))  # q function
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor

    def init(self): # init every episode
        self.done = False # restart 
        self.state = np.array([3, 0]) # init state
        self.rewards = 0 # init reward

    def move(self, state, action): # every step
        nstate = state + self.actions[action] # change state to next state(nstate)
        # ensure nstate is still in the graph
        if nstate[0] < 0: 
            nstate[0] = 0
        if nstate[0] >= self.row:
            nstate[0] = self.row-1
        if nstate[1] < 0:
            nstate[1] = 0
        if nstate[1] >= self.col:
            nstate[1] = self.col-1
        # gain reward
        if self.graph[nstate[0]][nstate[1]] == 0: # normal grid
            reward = -1
        elif self.graph[nstate[0]][nstate[1]] == 1: # goal
            self.done = True # finish
            reward = -1
        elif self.graph[nstate[0]][nstate[1]] == -1: # cliff
            self.state = [3, 0] # go back to starting point
            reward = -100
        return nstate, reward

    def epsilon_greedy(self, state): # epsilon-greedy to choose next action(naction)
        if np.random.random() < self.eposilon: # random choose
            naction = np.random.randint(4)
        else:
            naction = np.argmax(self.q[state[0]][state[1]]) # greedy
        return naction

    def smooth(self):
        smooth_reward_list = np.array([])
        cnt = 0
        rewards = 0
        cur = 0
        reward_window = np.array([])
        for reward in self.reward_list:
            if cnt < 50:
                reward_window = np.append(reward_window, reward)
                cnt += 1
            else:
                reward_window[cur] = reward
                cur = (cur+1) % 50
            smooth_reward_list = np.append(
                smooth_reward_list, np.mean(reward_window))
        return smooth_reward_list


class Qlearning(Strategy):  # Q-learning
    def __init__(self, graph, actions, eposilon, alpha, gamma):
        super().__init__(graph, actions, eposilon, alpha, gamma)

    def forward(self): # Qlearning step
        state = self.state # current state
        action = self.epsilon_greedy(state) # action 
        nstate, reward = self.move(state, action) # next action and reward
        self.q[state[0]][state[1]][action] += self.alpha * \
            (reward + self.gamma *
             np.max(self.q[nstate[0]][nstate[1]])-self.q[state[0]][state[1]][action]) # update q function value
        self.state = nstate # update state
        self.rewards += reward # gain reward
        if self.done: # finish
            self.reward_list = np.append(self.reward_list, self.rewards)


class Sarsa(Strategy):  # Sarsa
    def __init__(self, graph, actions, eposilon, alpha, gamma):
        super().__init__(graph, actions, eposilon, alpha, gamma)

    def forward(self):
        state = self.state # current state
        action = self.action # current action
        nstate, reward = self.move(state, action) # next state and the reward
        naction = self.epsilon_greedy(nstate) # choose next action through epsilon_greedy
        self.q[state[0]][state[1]][action] += self.alpha * \
            (reward + self.gamma * self.q[nstate[0]][nstate[1]]
             [naction]-self.q[state[0]][state[1]][action])
        self.state = nstate
        self.action = naction
        self.rewards += reward
        if self.done:
            self.reward_list = np.append(self.reward_list, self.rewards)
