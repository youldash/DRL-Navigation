# Imports.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility imports.
import math

# For storing network layers.
from collections import OrderedDict


# Actor (Policy) Model.
class QNetwork(nn.Module):
    """ Class implementation of the Deep Q-Network (DQN) algorithm.
    """
    
    def __init__(self, state_size, action_size, seed, hidden_layers=[512, 512]):
        """ Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): List containing hidden layer sizes (defaults: 512x512 if not overridden)
        """

        super(QNetwork, self).__init__()
        """ Initialize a QNetwork instance.
        """

        # Manual seeding.
        self.seed = torch.manual_seed(seed)
        
        # Create an OrderedDict to store the DQN layers.
        layers = OrderedDict()
        
        # Include both state_size, and action_size as layers in the network.
        hidden_layers = [state_size] + hidden_layers + [action_size]
        
        # Iterate over the parameters to create the DQN layers.
        for idx, (hl_in, hl_out) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            
            # Add a linear layer to the DQN.
            layers['fc'+str(idx)] = nn.Linear(hl_in, hl_out)
            
            # Add an activation function to the DQN.
            layers['activation'+str(idx)] = nn.ReLU()
        
        # Remove the last activation layer from the DQN.
        layers.popitem()
        
        # Lastly, create the DQN.
        self.network = nn.Sequential(layers)


    def forward(self, state):
        """ Build a network that maps state -> action values. """
        
        # Perform a feed-forward pass through the DQN.
        return self.network(state)


# Actor (Policy) Model.
class DuelingQNetwork(nn.Module):
    """ Class implementation of the Dueling Network (commonly abbreviated DN) algorithm.
    """

    def __init__(self, state_size, action_size, seed, hidden_advantage=[512,512], hidden_state_value=[512,512]):
        """ Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_advantage (list): List containing hidden advantage sizes (defaults: 512x512 if not overridden)
            hidden_state_value (list): List containing state value sizes (defaults: 512x512 if not overridden)
        """
        
        super(DuelingQNetwork, self).__init__()
        """ Initialize a DuelingQNetwork instance.
        """
        
        # Manual seeding.
        self.seed = torch.manual_seed(seed)
        
        # Include the state_size as the 1st parameter to create the DN layers.
        hidden_layers = [state_size] + hidden_advantage
        
        # Create an OrderedDict instance for storing the DN advantage layers (1/2).
        advantage_layers = OrderedDict()

        # Iterate over the parameters to create the advantage layers.
        for idx, (hl_in, hl_out) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            
            # Add a linear layer to the advantage layers.
            advantage_layers['adv_fc_'+str(idx)] = nn.Linear(hl_in, hl_out)
            
            # Add an activation function to the advantage layers.
            advantage_layers['adv_activation_'+str(idx)] = nn.ReLU()
        
        # Create the output layer for the advantage layers.
        advantage_layers['adv_output'] = nn.Linear(hidden_layers[-1], action_size)
        
        # Lastly, add (assign) the advantage layers to the DN.
        self.network_advantage = nn.Sequential(advantage_layers)
         
        # Create an OrderedDict instance for storing the DN value layers (2/2).
        value_layers = OrderedDict()
        hidden_layers = [state_size] + hidden_state_value

        # Iterate over the parameters to create the value layers.
        for idx, (hl_in, hl_out) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):

            # Add a linear layer to the value layers.
            value_layers['val_fc_'+str(idx)] = nn.Linear(hl_in, hl_out)

            # Add an activation function to the value layers.
            value_layers['val_activation_'+str(idx)] = nn.ReLU()
        
        # Create the output layer for the value layers.
        value_layers['val_output'] = nn.Linear(hidden_layers[-1], 1)
        
        # Lastly, add (assign) the value layers to the DN.
        self.network_value = nn.Sequential(value_layers)
        
        
    def forward(self, state):
        """ Build a network that maps state -> action values.
        """
        
        # Perform a feed-forward pass through the DN (using both advantage and value layers).
        advantage = self.network_advantage(state)
        value = self.network_value(state)

        # Return the aggregated modules.
        return advantage.sub_(advantage.mean()).add_(value)
                   