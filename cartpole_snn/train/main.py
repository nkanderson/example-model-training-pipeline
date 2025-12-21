import argparse
import math
import gymnasium as gym
import torch
import torch.optim as optim
from snntorch import surrogate
from snn_policy import SNNPolicy
from dqn_agent import DQNAgent, ReplayMemory
import matplotlib.pyplot as plt
from itertools import count
import random

# -----------------------
# Hyperparameters
# -----------------------
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon, or minimum exploration probability
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay (longer exploration)
# - If set too low, the agent may not explore enough and get stuck in suboptimal policies
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
# TODO: Consider moving the hyperparameters to a config file.
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005  # soft update rate for target network
LR = 3e-4

# TODO: Get this from the env.action_sapce.n, and len(state) instead of hardcoding
# Create networks
n_actions = 2  # CartPole has 2 actions: left (0) and right (1)
n_observations = (
    4  # CartPole has 4 observations: [position, velocity, angle, angular_velocity]
)

# SNN params
NUM_STEPS = 30  # simulation timesteps per env step (tunable)
BETA = 0.9
# optional surrogate: snn.surrogate.fast_sigmoid() or similar
spike_grad = surrogate.fast_sigmoid(slope=25)


# TODO: Consider moving this to a method of the DQNAgent class
# Select an action using epsilon-greedy policy
def select_action(state, steps_done, policy_net, device):
    """
    state: tensor shape [1, n_observations] on device
    returns action tensor shape [1,1]
    """
    sample = random.random()
    # Expotential decay of epsilon
    # math.exp(-1.0 * steps_done / EPS_DECAY) is the decay factor, which starts
    # at 1.0 when steps_done is 0, and approaches 0 as steps_done -> infinity.
    # This value should decay from EPS_START to EPS_END following an exponential curve.
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)  # forwards SNN, returns [1, n_actions]
            action = q_values.max(1).indices.view(1, 1)
            return action
    else:
        return torch.tensor(
            [[random.randint(0, n_actions - 1)]], device=device, dtype=torch.long
        )


# TODO: Consider creating a plotting util module
def plot_durations(episode_durations, show_result=False):
    """
    Plot episode durations and running average.

    Args:
        episode_durations: List of episode durations (number of steps per episode)
        show_result: If True, displays final result. If False, shows live training progress.
    """
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("SNN Training Results")
    else:
        plt.clf()
        plt.title("In-progress SNN Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    # Plot 100-episode moving average
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # For live updates: brief pause to update the plot
    # For final results: this ensures the plot is rendered before plt.show()
    plt.pause(0.001)


# -----------------------
# Optionally load a model, then train
# -----------------------
if __name__ == "__main__":
    #
    # Section 0: Parse command line arguments
    #
    parser = argparse.ArgumentParser(
        description="Train or evaluate an SNN-based DQN agent on CartPole"
    )

    # Model configuration
    parser.add_argument(
        "--neuron-type",
        "-n",
        type=str,
        default="leaky",
        choices=["leaky", "leakysv"],
        help="Type of spiking neuron to use (default: leaky)",
    )

    # Training/loading options
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        metavar="FILE",
        help="Load pre-trained model from FILE",
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate the loaded model, do not train",
    )

    # Hardware and rendering
    parser.add_argument(
        "--no-hw-acceleration",
        dest="hw_acceleration",
        action="store_false",
        help="Disable hardware acceleration (CUDA/MPS)",
    )
    parser.set_defaults(hw_acceleration=True)

    parser.add_argument(
        "--human-render",
        action="store_true",
        help="Show environment rendering and live training plot",
    )

    args = parser.parse_args()

    # Apply parsed arguments
    neuron_type = args.neuron_type
    pretrained_file = args.load
    evaluate_only = args.evaluate_only
    hw_acceleration = args.hw_acceleration
    human_render = args.human_render

    #
    # Section 1: Initialize plotting, device, default params, and basic training variables
    # like steps_done and episode_durations
    #
    if human_render:
        plt.ion()  # Enable interactive plotting for live updates

    # Set device based on hw_acceleration flag and availability
    if hw_acceleration and torch.cuda.is_available():
        device = torch.device("cuda")
    elif hw_acceleration and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Using neuron type: {neuron_type}")

    episode_durations = []
    steps_done = 0

    # Training parameters
    best_avg_reward = 0
    best_model_filename = "best_snn_dqn_cartpole.pth"  # Fixed filename, will overwrite

    # Set number of episodes based on hardware acceleration
    if hw_acceleration and (
        torch.cuda.is_available() or torch.backends.mps.is_available()
    ):
        num_episodes = 600
    else:
        num_episodes = 200

    #
    # Section 2: Initialize policy and target nets, optimizer, replay memory,
    # and create the CartPole environment
    #
    memory = ReplayMemory(10000)
    env = gym.make("CartPole-v1", render_mode="human" if human_render else None)

    # Create network instances
    policy_net = SNNPolicy(
        n_observations,
        n_actions,
        num_steps=NUM_STEPS,
        beta=BETA,
        spike_grad=spike_grad,
        neuron_type=neuron_type,
    ).to(device)
    target_net = SNNPolicy(
        n_observations,
        n_actions,
        num_steps=NUM_STEPS,
        beta=BETA,
        spike_grad=spike_grad,
        neuron_type=neuron_type,
    ).to(device)

    # Create optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    #
    # Section 3: If needed, load networks and optimizer
    # If loading a pre-trained model, load it and optionally evaluate then exit.
    # Otherwise, configure target_net from policy_net.
    #
    start_episode = 0

    if pretrained_file:
        # Load pre-trained model using DQNAgent.load()
        print(f"Loading pre-trained model from {pretrained_file}")
        agent = DQNAgent.load(
            pretrained_file,
            policy_net,
            target_net,
            optimizer,
            memory,
            device,
        )

        # Extract components from agent
        policy_net = agent.policy_net
        target_net = agent.target_net
        optimizer = agent.optimizer
        start_episode = agent.episode
        prev_avg = agent.avg_reward

        print(f"Resuming from episode {start_episode} (prev avg: {prev_avg:.2f})")

        # If evaluate-only mode, run evaluation and exit
        if evaluate_only:
            print("Running evaluation...")
            episode_rewards, avg_reward = agent.evaluate(
                env, num_episodes=10, render=True
            )
            print("Evaluation complete. Exiting.")
            env.close()
            exit(0)
    else:
        # Initialize target network from policy network for fresh start
        target_net.load_state_dict(policy_net.state_dict())

        # Create fresh agent instance
        agent = DQNAgent(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            memory=memory,
            n_observations=n_observations,
            n_actions=n_actions,
            num_steps=NUM_STEPS,
            beta=BETA,
            neuron_type=neuron_type,
            device=device,
            episode=0,
            avg_reward=0.0,
        )

    #
    # Section 4: Main training loop
    #
    for i_episode in range(start_episode, start_episode + num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(
            0
        )  # [1, obs]

        for t in count():
            action = select_action(state, steps_done, policy_net, device)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # store transition (state tensors already on device)
            memory.push(state, action, next_state, reward)

            # move to next state
            state = next_state

            # optimization step (on the policy network)
            agent.optimize(batch_size=BATCH_SIZE, gamma=GAMMA)

            # Soft update target network: θ' <- τ θ + (1 − τ) θ'
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                if human_render:
                    plot_durations(
                        episode_durations
                    )  # Only update plot live if human_render is enabled

                # Check if this is the best model so far (save only when we beat the record)
                if len(episode_durations) >= 100:
                    recent_avg = sum(episode_durations[-100:]) / 100
                    if recent_avg > best_avg_reward:
                        best_avg_reward = recent_avg
                        # Update agent's episode and avg_reward before saving
                        agent.episode = i_episode
                        agent.avg_reward = recent_avg
                        agent.save(best_model_filename)
                        print(f"New best model saved! Avg reward: {recent_avg:.2f}")

                break

    #
    # Section 5: Save final model
    #
    final_avg = sum(episode_durations[-min(len(episode_durations), 100) :]) / min(
        len(episode_durations), 100
    )
    # Update agent's episode and avg_reward before saving
    agent.episode = start_episode + num_episodes - 1
    agent.avg_reward = final_avg
    final_model_file = agent.save("final_snn_dqn_cartpole.pth")

    print("Complete")
    print(f"Final model saved to: {final_model_file}")
    if best_avg_reward > 0:
        print(
            f"Best model saved to: {best_model_filename} (avg reward: {best_avg_reward:.2f})"
        )

    if human_render:
        plt.ioff()  # Turn off interactive mode
    plot_durations(episode_durations, show_result=True)  # Always show final result
    plt.show()
