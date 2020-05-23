import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m):
        self.m = m
        # self.mean = 0
        self.m_estimate = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.m # 現在 rewards 改成 Gaussian distribution with mean=m, variance=1

    def update(self, x):
        self.N += 1
        # self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
        self.m_estimate = (1 - 1.0 / self.N) * self.m_estimate + 1.0 / self.N * x

def run_experiment(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)] # 只有三台吃角子老虎機

    # count number of suboptimal choices
    means = np.array([m1, m2, m3])
    true_best = np.argmax(means)
    count_suboptimal = 0

    data = np.empty(N)

    for i in range(N):
        # epslion greedy
        p = np.random.random()
        if p < eps:
            # j = np.random.choice(3)
            j = np.random.choice(len(bandits)) # 探索
        else:
            # j = np.argmax([b.mean for b in bandits])
            j = np.argmax([b.m_estimate for b in bandits]) # 使用最大的中獎機率的吃角子老虎機
        x = bandits[j].pull()
        bandits[j].update(x)

        if j != true_best: # 選到中獎機率較低的吃角子老虎機
            count_suboptimal += 1

        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        # print(b.mean)
        print(b.m_estimate)

    print('percent suboptimal for epsilon = %s:' % eps, float(count_suboptimal) / N)

    return cumulative_average

if __name__ == '__main__':
    # c_1 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)
    # c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
    # c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)
    m1, m2, m3 = 1.5, 2.5, 3.5
    c_1 = run_experiment(m1, m2, m3, 0.1, 100000)
    c_05 = run_experiment(m1, m2, m3, 0.05, 100000)
    c_01 = run_experiment(m1, m2, m3, 0.01, 100000)

    # log scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()
