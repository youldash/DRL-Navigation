# Deep Reinforcement Learning: Navigation (Report)

[![Twitter Follow](https://img.shields.io/twitter/follow/youldash.svg?style=social?style=plastic)](https://twitter.com/youldash)

## License

By using this site, you agree to the **Terms of Use** that are defined in [LICENSE](https://github.com/youldash/DRL-Navigation/blob/master/LICENSE).

## Algorithm Implementations

As mentioned in the [`README.md`](https://github.com/youldash/DRL-Navigation/blob/master/README.md) file of this repo, The project was developed in partial fulfillment of the requirements for Udacity's [Deep Reinforcement Learning (DRL) Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program. To solve the challenges presented therein, we explored (and implemented) a selected number of DRL algorithms. These are:

1. The [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) algorithm.
2. The [Double-DQN](https://arxiv.org/abs/1509.06461) algorithm.
3. The [Dueling Q-Network (DN)](https://arxiv.org/abs/1511.06581) algorithm.
4. The [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) memory buffer algorithm.

## The Deep Q-Network Algorithm

Initial attempts were made for developing `model` implementations of a *value-based* method called the **Deep Q-Network (DQN)** algorithm, using Experience Replays and Fixed Q-targets (see `model.py` for more details). This model served as the *benchmark* for further experiments to come (and compare against).

### Early Attempts

- Our first `model` training configuration was based on a DQN with **two Fully-connected (FC) layers (hosting 512 nodes in each layer)**. This `model` configuration solved the virtual world (or environment) in a number of episodes that far exceeded 700. This was set as a point of reference to beat in our next attempts. The Neural Network architecture was based on the following configuration:
```
Input nodes (37) -> FC Layer (512 nodes, ReLU activation) -> Fully Connected Layer (512 nodes, ReLU activation) -> Output nodes (4)
```

- Further attempts were made by amending the `model` architecture (*i.e.* by increasing the number of layers, as well as increasing the number of nodes in the `model`). These experiments yielded **poorer** results when compared to the **benchmark** configuration above, leading us to further adjust the `model` by having **two FC layers (having 128 nodes in the first, and 32 nodes in the second)**. This architecture solved the environment in less than 500 episodes.

