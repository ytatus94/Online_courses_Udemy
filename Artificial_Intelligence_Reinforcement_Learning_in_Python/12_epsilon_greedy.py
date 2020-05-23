import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75] # 3 台吃角子老虎機個別的中獎機率

class Bandit:
    def __init__(self, p):
        # p: the win rate
        self.p = p # 事實上我們並不知道真正的中獎機率是多少
        self.p_estimate = 0. # 我們自己算的中獎機率
        self.N = 0. # num samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p # 只要比 p 小就當成中獎

    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits]) # 中獎機率最高的那台吃角子老虎機的 index
    print('optimal j:', optimal_j)

    for i in range(NUM_TRIALS):

        # use epsilon-greedy to select the next bandit
        if np.random.random() < EPS:
            num_times_explored += 1
            j = np.random.randint(len(bandits)) # 比 EPS 小的時候採取探索的方式，從三台吃角子老虎機中隨便選一台
        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits]) # 選中獎機率最高的那台吃角子老虎機來玩

        if j == optimal_j: # 表示選到了中獎機率最高的那台吃角子老虎機
            num_optimal += 1

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull() # 玩選中的那一台吃角子老虎機

        # update rewards log
        rewards[i] = x # 每一次玩，有中獎或沒中獎都記錄下來

        # update the distribution
        bandits[j].update(x) # 計算每一台吃角子老虎機的中獎機率

    # print mean estimates for each bandit
    for b in bandits:
        print('mean estimate:', b.p_estimate)

    # print total reward
    print('total reward earned:', rewards.sum())
    print('overall win rate:', rewards.sum() / NUM_TRIALS)
    print('num_times_explored', num_times_explored)
    print('num_times_exploited', num_times_exploited)
    print('num times selected optimal bandit:', num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == '__main__':
    experiment()
