import torch
import torch.nn as nn
import snntorch as snn
from leakysv import LeakySV


class SNNPolicy(nn.Module):
    """
    Spiking Neural Network policy for Deep Q-Network (DQN) reinforcement learning.

    This network uses spiking neurons to process observations and output Q-values
    for each possible action. The SNN simulates neural dynamics over multiple
    timesteps, with spikes accumulated to produce final Q-value estimates.

    Architecture:
        - Input layer: Linear transformation of observations to 128 features
        - Hidden layer 1: 128 spiking neurons (LIF)
        - Hidden layer 2: 128 spiking neurons (LIF)
        - Output layer: Linear transformation to Q-values (one per action)

    The network processes the same input observation for `num_steps` timesteps,
    accumulating Q-values across time and averaging them to produce the final
    action-value estimates.
    """

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        num_steps=30,
        beta=0.9,
        spike_grad=None,
        neuron_type="leaky",
    ):
        """
        - num_steps: number of timesteps to simulate the SNN per environment step
        - beta: membrane decay for LIF
        - spike_grad: surrogate gradient function from snntorch.surrogate (optional)
        - neuron_type: "leakysv" or "leaky" - type of spiking neuron to use
        """
        super().__init__()
        self.num_steps = num_steps
        self.n_actions = n_actions
        self.neuron_type = neuron_type

        # feedforward linear layers
        self.fc1 = nn.Linear(n_observations, 128)

        # Create neurons based on type
        if neuron_type == "leaky":
            self.lif1 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
            self.lif2 = snn.Leaky(
                beta=beta, init_hidden=True, spike_grad=spike_grad, output=True
            )
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        self.fc2 = nn.Linear(128, 128)
        # decode membrane potential to Q-values per timestep
        self.fc_out = nn.Linear(128, n_actions)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the spiking neural network.

        Processes observations through the SNN for num_steps timesteps,
        accumulating Q-values across time to produce final estimates.

        Args:
            observations: [batch, n_observations] float tensor of environment observations

        Returns:
            q_values: [batch, n_actions] tensor of Q-values (averaged over time)
        """
        # Reset snnTorch hidden states for all LIF instances
        # This avoids leak between separate forward passes / episodes.
        if self.neuron_type == "leakysv":
            LeakySV.reset_hidden()  # reset for LeakySV neurons (includes refractory counter)
        else:
            snn.Leaky.reset_hidden()  # reset for Leaky neurons

        batch_size = observations.size(0)
        out_accum = torch.zeros(batch_size, self.n_actions, device=observations.device)

        # Simulate for num_steps timesteps with the SAME input each step (rate coding)
        for _t in range(self.num_steps):
            h1 = self.fc1(observations)  # current input -> hidden current
            spk1 = self.lif1(h1)  # spike output from layer1
            h2 = self.fc2(spk1)  # pass spikes into next layer
            spk2, mem2 = self.lif2(h2)  # output spikes and membrane of final LIF
            q_t = self.fc_out(mem2)  # decode membrane -> Q-values at this step
            # The Accumulation Strategy
            # A key aspect here is that out_accum accumulates the Q-values
            # across all timesteps. Since the network processes the same input
            # repeatedly, neurons that respond more strongly to this input will
            # fire more frequently, building up higher accumulated values.
            # After the loop completes, the final Q-values are obtained by
            # averaging: q_pred = out_accum / float(self.num_steps).
            out_accum += q_t

        q_values = out_accum / float(self.num_steps)
        return q_values
