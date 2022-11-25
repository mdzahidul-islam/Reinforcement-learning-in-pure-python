import numpy as np
import gym
import matplotlib.pyplot as plt


class SarsaMidterm:

    def __init__(self, env, epsilon, alpha, eps_decay_rate=1, min_epsilon=.01, gamma=1, toy_game='CliffWalking-v0',
                 decay_epsilon=False, dyna_alpha=False):
        self.episode_steps = None
        self.alpha_power = .8
        self.nsa = {}
        self.ns = {}
        self.episode_reward = None
        self.opt_policy = None
        self.learnedQ = None
        self.init_epsilon = epsilon
        self.eps_decay_rate = eps_decay_rate
        self.min_epsilon = min_epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay_epsilon = decay_epsilon
        self.dyna_alpha = dyna_alpha
        self.toy_game = toy_game
        self.epsilon = {}

        if toy_game == 'Blackjack-v1':
            self.env = gym.make('Blackjack-v1', natural=False, sab=True)
            # , render_mode = "human"
        elif toy_game == 'FrozenLake-v1':
            # self.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
            self.env = env
        else:
            self.env = gym.make(toy_game)

    def init_q(self):
        Q = {}
        if self.toy_game == 'Blackjack-v1':
            # pass
            for i in range(1, self.env.observation_space[0].n + 1):
                for j in range(1, self.env.observation_space[1].n + 1):
                    for k in [True, False]:
                        s = (i, j, k)
                        Q[s] = np.zeros(self.env.action_space.n)
                        self.epsilon[s] = self.init_epsilon
                        self.ns[s] = 1
                        self.nsa[s] = np.ones(self.env.action_space.n)
        else:
            for i in range(self.env.observation_space.n):
                Q[i] = np.zeros(self.env.action_space.n)
                self.ns[i] = 1
                self.nsa[i] = np.ones(self.env.action_space.n)
                self.epsilon[i] = self.init_epsilon
        return Q

    def e_greedy(self, Q, S):
        if self.decay_epsilon and (self.decay_epsilon >= self.min_epsilon):
            self.epsilon[S] = self.epsilon[S] / np.sqrt(self.ns[S])
            # self.time_step
        argmaxq = np.random.choice(np.flatnonzero(Q[S] == np.max(Q[S])))
        act_prob = np.ones(self.env.action_space.n) * self.epsilon[S] / self.env.action_space.n
        act_prob[argmaxq] += 1 - self.epsilon[S]
        action = np.random.choice(self.env.action_space.n, 1, p=act_prob)[0]
        return action

    def sarsa_nstep(self, n_episodes, runs, n):
        Q = self.init_q()
        self.env.reset(seed=222)
        self.episode_reward = np.zeros((runs, n_episodes))
        self.episode_steps = np.zeros((runs, n_episodes))
        for run in range(runs):
            self.time_step = 1
            Q = self.init_q()
            self.learnedQ = Q
            for episode in range(n_episodes):
                S, info = self.env.reset()
                action = self.e_greedy(Q, S)
                T = np.inf
                actions = [action]
                states = [S]
                rewards = [0]
                t = 0
                while True:
                    if t < T:
                        observation, reward, terminated, truncated, info = self.env.step(action)
                        rewards.append(reward)

                        states.append(observation)
                        if terminated:
                            T = t + 1
                        else:
                            action = self.e_greedy(Q, observation)
                            actions.append(action)
                    tao = t - n + 1
                    if tao >= 0:
                        G = 0
                        for i in np.arange(tao + 1, np.minimum(tao + n + 1, T + 1)):
                            G += np.power(self.gamma, i - tao - 1) * rewards[int(i)]
                        if tao + n < T:
                            G += np.power(self.gamma, n) * Q[states[tao + n]][actions[tao + n]]
                        if self.dyna_alpha:
                            self.alpha = self.alpha / (self.nsa[states[tao]][actions[tao]]) ** self.alpha_power
                        Q[states[tao]][actions[tao]] += self.alpha * (G - Q[states[tao]][actions[tao]])
                        self.nsa[states[tao]][actions[tao]] += 1
                        self.ns[states[tao]] += 1
                        self.time_step += 1
                        # action is taken using e-greedy based on latest Q for a state S
                        # however, optimal policy is not calculated for every state that are not visited
                        # it decreases time complexity
                        # policy_learned = self.e_greedy(Q)
                    if tao == T - 1:
                        break
                    t += 1
                self.episode_reward[run, episode] = np.sum(rewards.copy())
                self.episode_steps[run, episode] = t

            self.learnedQ = Q
            self.optimal_policy()
        return Q, self.opt_policy

    def optimal_policy(self):
        self.opt_policy = {}
        for key, value in self.learnedQ.items():
            # self.opt_policy[key] = np.argmax(value)
            act_prob = np.ones(self.env.action_space.n) * self.epsilon[key] / self.env.action_space.n
            act_prob[np.argmax(value)] += 1 - self.epsilon[key]
            action = np.random.choice(self.env.action_space.n, 1, p=act_prob)[0]
            self.opt_policy[key] = action

    def reward_plot(self):
        plt.figure(1, (10, 6))
        plt.plot(range(len(self.episode_reward[0, :])), np.mean(self.episode_reward, 0))
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        # plt.ylim([-300,20])
        plt.legend()
        plt.grid()
        plt.show()

    def step_plot(self):
        plt.figure(2, (10, 6))
        plt.plot(range(len(self.episode_steps[0, :])), np.mean(self.episode_steps, 0))
        plt.xlabel('Episodes')
        plt.ylabel('steps during episode')
        # plt.ylim([-200,10])
        plt.legend()
        plt.grid()
        plt.show()
