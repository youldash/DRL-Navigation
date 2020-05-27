# Imports.
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim # For optimizer support.

# Networks and buffers.
from model import QNetwork, DuelingQNetwork
from buffer import ReplayBuffer, PrioritizedReplayBuffer

# Utility imports.
import random


""" Hyperparameter setup.
"""
BUFFER_SIZE = int(1e5)  # Replay buffer size.
BATCH_SIZE = 64         # Minibatch size.
LEARNING_RATE = 4.8e-4  # Learning rate.
THRESHOLD = 4           # How often to update the network.
GAMMA = 99e-2           # Discount factor.
TAU = 1e-2              # For soft update of target parameters.


""" Move the training model to either GPU (Cuda),
    or CPU (depending on availability).
"""
# Set the working device on the NVIDIA Tesla K80 accelerator GPU (if available).
# Otherwise we use the CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent():
    """ Class implementation of a so-called "intelligent" agent.
        This agent interacts with and learns from the environment.
    """

    double_dqn = False
    """ True for the Double-DQN method.
    """

    dueling_network = False
    """ True for the Dueling Network (DN) method.
    """

    prioritized_replay = False
    """ True for the Prioritized Replay memory buffer.
    """


    def __init__(
        self, state_size, action_size, seed, lr_decay=9999e-4,
        double_dqn=False, dueling_network=False, prioritized_replay=False):
        """ Initialize an Agent instance.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            lr_decay (float): Multiplicative factor of learning rate decay
            double_dqn (bool): Toogle for using the Double-DQN method
            dueling_network (bool): Toogle for using the Dueling Network (DN) method
            prioritized_replay (bool): Toogle for using the Prioritized Replay method
        """

        # Set the parameters.
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.double_dqn = double_dqn
        self.dueling_network = dueling_network
        self.prioritized_replay = prioritized_replay

        # Q-Network hidden layers.
        hidden_layers = [128,32]
        
        # Use the Dueling Network (DN) method.
        if self.dueling_network:

            # DN requires a hidden state value.
            hidden_state_value = [64,32]
            
            self.qnetwork_local = DuelingQNetwork(
                state_size, action_size, seed, hidden_layers, hidden_state_value).to(device)
            self.qnetwork_target = DuelingQNetwork(
                state_size, action_size, seed, hidden_layers, hidden_state_value).to(device)
            self.qnetwork_target.eval()
            
        else: # Use the Deep Q-Network (DQN) method.

            self.qnetwork_local = QNetwork(
                state_size, action_size, seed, hidden_layers).to(device)
            self.qnetwork_target = QNetwork(
                state_size, action_size, seed, hidden_layers).to(device)
            self.qnetwork_target.eval()
        
        # Optimize using Adam.
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)

        # Use the Prioritized Replay memory buffer if enabled.
        if self.prioritized_replay:

            self.memory = PrioritizedReplayBuffer(
                action_size, BUFFER_SIZE, BATCH_SIZE, seed, device,
                alpha=0.6, beta=0.4, beta_scheduler=1.0)

        else: # Use the Replay memory buffer instead.
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
        
        # Initialize the time step (until the THRESHOLD is reached).
        self.t_step = 0
    

    def step(self, state, action, reward, next_state, done):
        """ Update the network on each step.

        Params
        ======
            state (array_like): Current state
        """

        # Save experience in replay memory.
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every time step till THRESHOLD.
        self.t_step = (self.t_step + 1) % THRESHOLD

        if self.t_step == 0: # Initial time step.

            # If enough samples are available in memory, get random subset and learn.
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    
    def act(self, state, eps=0.):
        """ Return the actions for a given state as per current policy.
        
        Params
        ======
            state (array_like): Current state
            eps (float): Epsilon (ε), for epsilon-greedy action selection
        """

        # Epsilon-greedy action selection.
        if random.random() > eps:

            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            self.qnetwork_local.eval()

            with torch.no_grad():
                action_values = self.qnetwork_local(state)

            # Train the network.
            self.qnetwork_local.train()

            # Return the action.
            return np.argmax(action_values.cpu().data.numpy())
        
        else: # Return a random action.
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """ Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done, w) tuples 
            gamma (float): Discount factor
        """

        # Set the parameters.
        states, actions, rewards, next_states, dones, w = experiences

        # Compute and minimize the loss.
        with torch.no_grad():

            if self.double_dqn: # Use of Double-DQN method.

                # Select the greedy actions using the QNetwork Local.
                # Calculate the pair action/reward for each of the next_states.
                next_action_rewards_local = self.qnetwork_local(next_states)

                # Select the action with the maximum reward for each of the next actions.
                greedy_actions_local = next_action_rewards_local.max(dim=1, keepdim=True)[1]

                ## Get the rewards for the greedy actions using the QNetwork Target.
                # Calculate the pair action/reward for each of the next_states.
                next_action_rewards_target = self.qnetwork_target(next_states)

                # Get the target reward for each of the greedy actions selected,
                # following the local network.
                target_rewards = next_action_rewards_target.gather(1, greedy_actions_local)
                
            else: # Use of the fixed Q-target method.

                # Calculate the pair action/reward for each of the next_states.
                next_action_rewards = self.qnetwork_target(next_states)

                # Select the maximum reward for each of the next actions.
                target_rewards = next_action_rewards.max(dim=1, keepdim=True)[0]
                
            # Calculate the discounted target rewards.
            target_rewards = rewards + (gamma * target_rewards * (1 - dones))
            
        # Calculate the pair action/rewards for each of the states.
        # Here, shape: [batch_size, action_size].
        expected_action_rewards = self.qnetwork_local(states)

        # Get the reward for each of the actions.
        # Here, shape: [batch_size, 1].
        expected_rewards = expected_action_rewards.gather(1, actions)

        # If the Prioritized Replay memory buffer if enabled.
        if self.prioritized_replay:
            target_rewards.sub_(expected_rewards)
            target_rewards.squeeze_()
            target_rewards.pow_(2)
            
            with torch.no_grad():
                td_error = target_rewards.detach()
                td_error.pow_(0.5)
                self.memory.update_priorities(td_error)
            
            target_rewards.mul_(w)
            loss = target_rewards.mean()

        else: # Calculate the loss.
            loss = F.mse_loss(expected_rewards, target_rewards)

        # Perform the back-propagation.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # Update the target network.
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters:
            θ_target = τ * θ_local + (1 - τ) * θ_target.

        Params
        ======
            local_model (PyTorch model): Weights will be copied from
            target_model (PyTorch model): Weights will be copied to
            tau (float): Interpolation parameter 
        """

        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.-tau)*target_param.data)
