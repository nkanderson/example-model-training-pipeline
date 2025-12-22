# SNN-DQN CartPole Training

Deep Q-Network (DQN) implementation using Spiking Neural Networks (SNNs) for the CartPole-v1 environment.

## Quick Start

```bash
# Train with default baseline configuration
python main.py

# Train with live visualization
python main.py --human-render

# Load and evaluate a trained model
python main.py --load models/dqn_config-baseline-best.pth --evaluate-only --human-render
```

## Configuration

Training hyperparameters and network settings are defined in YAML configuration files located in the `configs/` directory.

### Configuration File Structure

```yaml
training:
  batch_size: 128          # Transitions sampled from replay buffer per optimization 
  gamma: 0.99              # Discount factor for future rewards
  eps_start: 0.9           # Initial exploration rate (epsilon-greedy)
  eps_end: 0.01            # Minimum exploration rate
  eps_decay: 2500          # Epsilon decay rate (higher = slower decay)
  tau: 0.005               # Target network soft update rate
  lr: 0.0003               # Learning rate for AdamW optimizer (3e-4)
  num_episodes: 600        # Total training episodes

snn:
  num_steps: 30            # SNN simulation timesteps per environment step
  beta: 0.9                # LIF neuron membrane decay rate
  surrogate_gradient_slope: 25  # Slope for fast_sigmoid surrogate gradient
  neuron_type: leaky       # Neuron type: 'leaky' or 'leakysv'
```

### Available Configurations

- **`config-baseline.yaml`**: Original hyperparameters used for initial experiments

## Command Line Arguments

### Configuration
- `--config`, `-c`: Path to YAML configuration file (default: `configs/config-baseline.yaml`)

### Model Options
- `--neuron-type`, `-n`: Type of spiking neuron (`leaky` or `leakysv`) - overrides config if specified
- `--load`: Load pre-trained model from file
- `--evaluate-only`: Only evaluate the loaded model without training

### Hardware & Visualization
- `--no-hw-acceleration`: Disable hardware acceleration (CUDA/MPS)
- `--human-render`: Show environment rendering and live training plot

## Usage Examples

### Training

```bash
# Basic training with baseline config
python main.py

# Train with custom configuration
python main.py --config configs/config-stable.yaml

# Train with live visualization (slower but useful for debugging)
python main.py --human-render

# Train without hardware acceleration (CPU only)
python main.py --no-hw-acceleration

# Train with LeakySV neurons (includes refractory period)
python main.py --neuron-type leakysv
```

### Evaluation

```bash
# Evaluate best model with visualization
python main.py --load models/dqn_config-baseline-best.pth --evaluate-only --human-render

# Evaluate final model
python main.py --load models/dqn_config-baseline-final.pth --evaluate-only
```

### Resuming Training

```bash
# Continue training from a saved checkpoint
python main.py --load models/dqn_config-baseline-best.pth

# Resume with different neuron type
python main.py --load models/dqn_config-baseline-best.pth --neuron-type leakysv
```

## Output Files

Training produces two model checkpoints in the `models/` directory:

- **`models/dqn_<config-name>-best.pth`**: Model with highest average reward (over last 100 episodes)
- **`models/dqn_<config-name>-final.pth`**: Model from the final training episode

Each checkpoint contains:
- Policy network state
- Target network state
- Optimizer state
- Episode number
- Average reward
- Network configuration parameters

## Training Visualization

When using `--human-render`, the script displays:
- **Environment rendering**: Live CartPole simulation
- **Training plot**: Episode durations and 100-episode moving average

The plot updates during training and shows final results when complete.

## Tips for Hyperparameter Tuning

### Common Issues and Solutions

**Performance degrades after initial improvement:**
- Try reducing `tau` (e.g., 0.001) for more stable target network updates
- Increase `eps_decay` (e.g., 5000-10000) to maintain exploration longer
- Lower `lr` (e.g., 0.0001) for more stable learning

**Training is too slow:**
- Reduce `num_episodes` for faster experiments
- Ensure hardware acceleration is enabled (remove `--no-hw-acceleration`)
- Disable rendering (don't use `--human-render`)

**Agent not learning:**
- Check that `eps_decay` isn't too low (agent stops exploring too early)
- Verify `lr` isn't too small (learning too slow)
- Ensure `batch_size` provides sufficient samples

## Project Structure

```
cartpole_snn/train/
├── README.md                 # This file
├── main.py                   # Main training script
├── snn_policy.py            # SNN-based policy network
├── dqn_agent.py             # DQN agent implementation
├── leakysv.py               # LeakySV neuron with refractory period
├── configs/                 # Configuration files
│   └── config-baseline.yaml
└── models/                  # Saved model checkpoints, e.g.:
    ├── dqn_config-baseline-best.pth
    └── dqn_config-baseline-final.pth
```

## Requirements

- Python 3.8+
- PyTorch
- snnTorch
- gymnasium
- PyYAML
- matplotlib

Install dependencies:
```bash
pip install -r ../../requirements-macos.txt  # macOS
# or
pip install -r ../../requirements.txt        # Linux/other
```
