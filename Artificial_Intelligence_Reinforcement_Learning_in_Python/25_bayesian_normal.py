import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(1)
NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]

class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        # parameters for mu - prior is N(0, 1)
        self.predicted_mean = 0 # mean of \bar{X}
        self.lambda_ = 1
        self.sum_x = 0 # for convenience
        self.tau = 1 # precision = 1/variance
        self.N = 0 # 玩了幾次吃角子老虎機

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean # 拉拉霸後真正的中獎的值 x 它是一個 N(mu, tau^{-1}) 的高斯分佈

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean # 我們計算出來拉拉霸會得到的結果，這是 posterior 所以要用 lambda_ 和 predicted_mean 來計算

    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.predicted_mean = self.tau * self.sum_x / self.lambda_
        self.N += 1

'''
X ~ N(mu, tau^{-1})
mu ~ N(m0, lambda0^{-1})
mu|X ~ N(m, lambda^{-1}) 這是 posterior

lambda = tau * N + lambda0
m = (1/lambda) * (tau * sum_{i=1}^{N} Xi + lambda0 * m0)
當一個一個來 update 時，N=1

Z ~ N(0, 1), X = sigma * Z + mu 則 X ~ N(mu, sigma^2)
'''

def plot(bandits, trial):
    x = np.linspace(-3, 6, 200)
    for b in bandits:
        y = norm.pdf(x, b.predicted_mean, np.sqrt(1. / b.lambda_))
        plt.plot(x, y, label=f'real mean: {b.true_mean:.4f}, num plays: {b.N}')
    plt.title(f'Bandit distributions after {trial} trials')
    plt.legend()
    plt.show()

def run_experiment():
    bandits = [Bandit(m) for m in BANDIT_MEANS]

    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.empty(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        # Thompson sampling
        j = np.argmax([b.sample() for b in bandits]) # 選擇當前我們算出中獎率最高的機台

        # plt the posteriors
        if i in sample_points:
            plot(bandits, i)

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull() # 真正的中獎結果

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

        # update rewards
        rewards[i] = x

    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    # plt moving average ctr
    plt.plot(cumulative_average)
    for m in BANDIT_MEANS:
        plt.plot(np.ones(NUM_TRIALS) * m)
    plt.show()

    return cumulative_average

if __name__ == '__main__':
    run_experiment()
