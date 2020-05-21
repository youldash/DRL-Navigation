# Deep Reinforcement Learning: Navigation Project

[![Twitter Follow](https://img.shields.io/twitter/follow/youldash.svg?style=social?style=plastic)](https://twitter.com/youldash)

[banana]: misc/banana.gif "Trained Agent"

## License

By using this site, you agree to the **Terms of Use** that are defined in [LICENSE](https://github.com/youldash/DRL-Navigation/blob/master/LICENSE).

## About

The goal of this project is to train a so-called [intelligent agent](https://en.wikipedia.org/wiki/Intelligent_agent) to navigate a virtual world (or environment), and collect as many *yellow* bananas as possible while avoiding *blue* bananas.

<div align="center">
	<img src="misc/banana.gif" width="100%" />
</div>

This project was developed in partial fulfillment of the requirements for Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.

## Reinforcement Learning

According to **skymind.ai**, the term [Reinforcement Learning (RL)](https://skymind.ai/wiki/deep-reinforcement-learning) refers to:

> Goal-oriented algorithms, which learn how to attain a complex objective (goal) or how to maximize along a particular dimension over many steps; for example, they can maximize the points won in a game over many moves. RL algorithms can start from a blank slate, and under the right conditions, they achieve superhuman performance. Like a pet incentivized by scolding and treats, these algorithms are penalized when they make the wrong decisions and rewarded when they make the right ones – this is reinforcement.

### Deep Reinforcement Learning

[Deep Reinforcement Learning (DRL)](https://skymind.ai/wiki/deep-reinforcement-learning) combines [Artificial Neural Networks (ANNs)](https://en.wikipedia.org/wiki/Artificial_neural_network) with an RL architecture that enables software-defined agents to learn the best actions possible in virtual environment in order to attain their goals. That is, it unites function approximation and target optimization, mapping state-action pairs to expected rewards.

## Project Environment

The project development environment is based on [Unity](https://unity.com)'s [Machine Learning Agents Toolkit (ML-Agents)](https://github.com/Unity-Technologies/ml-agents). The [toolkit](https://unity3d.ai/) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

The project environment is similar to, **but not identical to** the Banana Collector environment on the Unity ML-Agents GitHub page.

### The Banana Collector

A reward of` +1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

## The State Space

The state space has 37-dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- `0` - move forward.
- `1` - move backward.
- `2` - turn left.
- `3` - turn right.

