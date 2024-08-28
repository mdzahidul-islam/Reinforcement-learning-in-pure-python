# Reinforcement Learning: Pure Python Implementations

This repository contains the materials and code implementations from my graduate-level Reinforcement Learning (RL) class. The focus is on implementing various RL algorithms purely in Python, using only basic libraries like NumPy, Matplotlib, and OpenAI Gym. The content in this repository closely follows the concepts and algorithms presented in Sutton and Barto's classic "Reinforcement Learning: An Introduction."

## Project Structure

The repository is organized into folders corresponding to different topics and RL algorithms. Each folder contains Python scripts implementing the algorithms, along with relevant documentation and example problems. Here's an overview of the main sections:

- **Dynamic Programming:** Implementations of dynamic programming methods such as policy iteration and value iteration.
  
- **Finite MDP and Bellman Equations:** Solutions and examples involving finite Markov Decision Processes (MDPs) and Bellman equations.

- **Temporal Difference:** Implementations of temporal difference learning methods like SARSA and Q-learning.

- **Eligibility Traces and TD(lambda):** Examples and code related to eligibility traces and the TD(lambda) algorithm.

- **Monte Carlo Methods:** Implementation of Monte Carlo learning methods, including Monte Carlo control with exploring starts.

- **Actor-Critic with Eligibility Traces:** Code for actor-critic algorithms that utilize eligibility traces for learning.

- **On-policy Control with Approximation:** Implementation of on-policy control methods using function approximation.

- **Multi-arm Bandit Problems:** Solutions to various multi-arm bandit problems using different strategies like epsilon-greedy and UCB.

- **n-Step SARSA:** Implementation of n-step SARSA, a temporal difference learning method that balances between SARSA(1) and SARSA(lambda).

## Requirements

To run the code in this repository, you’ll need Python 3.x and the following Python libraries:

- `numpy`
- `matplotlib`
- `gym`

You can install the required libraries using `pip`:

```bash
pip install numpy matplotlib gym
