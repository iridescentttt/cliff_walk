from numpy.lib.arraypad import pad
from numpy.testing._private.utils import suppress_warnings
from model import *
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
graph = np.zeros((4, 12))  # graph
graph[3][1:11] = -1  # cliff
graph[3][11] = 1  # goal
# move down, right, up, left
actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

episodes = 500
eposilon = 0.1
alpha = 0.5
gamma = 0.9

# qlearning
qlearning = Qlearning(graph, actions, eposilon, alpha, gamma)
for episode in range(1, episodes+1):
    qlearning.init()
    qlearning.action = qlearning.epsilon_greedy(qlearning.state)
    while not qlearning.done:
        qlearning.forward()
    print(f"-----Qlearning episode {episode} reward: {qlearning.rewards}-----")
Qlearning_reward_list = qlearning.smooth()

# Sarsa
sarsa = Sarsa(graph, actions, eposilon, alpha, gamma)
for episode in range(1, episodes+1):
    sarsa.init()
    sarsa.action = sarsa.epsilon_greedy(sarsa.state)
    while not sarsa.done:
        sarsa.forward()
    print(f"-----Sarsa episode {episode} reward: {sarsa.rewards}-----")

Sarsa_reward_list = sarsa.smooth()


print(
    f"Final reward\nQlearning: {Qlearning_reward_list[-1]} Sarsa: {Sarsa_reward_list[-1]}")

plt.figure()
plt.plot(Qlearning_reward_list, label="Qlearning")
plt.plot(Sarsa_reward_list, label="Sarsa")
plt.ylim((-150, 0))
plt.yticks(np.arange(-150, 0, 10))
plt.grid()
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.title("Reward of Q-Learning/SARSA")
plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('./rewards.png', format="png")


directions = ["down", "right", "up", "left"]
for direction in range(4):
    to_plot = np.zeros([4, 12])
    for i in range(4):
        for j in range(12):
            to_plot[i][j] = qlearning.q[i][j][direction]
    to_plot = np.around(to_plot, 1)
    # q function
    plt.figure()
    fig, ax = plt.subplots()
    # 将元组分解为fig和ax两个变量
    im = ax.imshow(to_plot, cmap="coolwarm_r")
    for i in range(4):
        for j in range(12):
            text = ax.text(j, i, to_plot[i][j],
                           ha="center", va="center", color="black")
    fig.tight_layout()
    plt.colorbar(im, ax=ax, pad=0.01, fraction=0.1)
    plt.title("Q function value in Q-learning to move "+directions[direction])
    plt.savefig('./qlearning_'+directions[direction]+'.png', format="png")
    
    # 显示图片

symbol = ["↓", "→", "↑", "←"]
qlearning_move = np.zeros([4, 12], dtype=np.int)
to_plot = np.zeros([4, 12])
plt.figure()
for i in range(4):
    for j in range(12):
        qlearning_move[i][j] = int(np.argmax(qlearning.q[i][j]))
fig, ax = plt.subplots()
# 将元组分解为fig和ax两个变量
im = ax.imshow(to_plot, cmap="hot_r")
for i in range(4):
    for j in range(12):
        text = ax.text(j+0.5, i+0.5, symbol[qlearning_move[i][j]],
                       ha="center", va="center", color="black")
plt.xlim(0, 12)
plt.ylim(0, 4)
plt.xticks(np.arange(0, 12), color='w')
plt.yticks(np.arange(0, 4), color='w')
ax.invert_yaxis()
plt.grid()
fig.tight_layout()
plt.title("Q-learning movement")
plt.savefig('./qlearning_move.png', format="png")



directions = ["down", "right", "up", "left"]
for direction in range(4):
    to_plot = np.zeros([4, 12])
    for i in range(4):
        for j in range(12):
            to_plot[i][j] = sarsa.q[i][j][direction]
    to_plot = np.around(to_plot, 1)
    # q function
    plt.figure()
    fig, ax = plt.subplots()
    # 将元组分解为fig和ax两个变量
    im = ax.imshow(to_plot, cmap="coolwarm_r")
    for i in range(4):
        for j in range(12):
            text = ax.text(j, i, to_plot[i][j],
                           ha="center", va="center", color="black")
    fig.tight_layout()
    plt.colorbar(im, ax=ax, pad=0.01, fraction=0.1)
    plt.title("Q function value in Sarsa to move "+directions[direction])
    plt.savefig('./sarsa_'+directions[direction]+'.png', format="png")
    
    # 显示图片

symbol = ["↓", "→", "↑", "←"]
sarsa_move = np.zeros([4, 12], dtype=np.int)
to_plot = np.zeros([4, 12])
plt.figure()
for i in range(4):
    for j in range(12):
        sarsa_move[i][j] = int(np.argmax(sarsa.q[i][j]))
fig, ax = plt.subplots()
# 将元组分解为fig和ax两个变量
im = ax.imshow(to_plot, cmap="hot_r")
for i in range(4):
    for j in range(12):
        text = ax.text(j+0.5, i+0.5, symbol[sarsa_move[i][j]],
                       ha="center", va="center", color="black")
plt.xlim(0, 12)
plt.ylim(0, 4)
plt.xticks(np.arange(0, 12), color='w')
plt.yticks(np.arange(0, 4), color='w')
ax.invert_yaxis()
plt.grid()
fig.tight_layout()
plt.title("Sarsa movement")
plt.savefig('./sarsa_move.png', format="png")

