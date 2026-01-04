# CartPole SNN Hardware Specification

## Overview

This document specifies the hardware architecture for a Spiking Neural Network (SNN) trained with snnTorch to solve the CartPole reinforcement learning task. The network uses direct current injection (no spike encoding at input) to match the trained weights from snn_policy.py.

This initial plan reflects training with snnTorch's Leaky neuron model. Small adjustments (e.g. changes to reset mechanism) may be necessary for usage with an alternative model.

## Network Architecture

### Default Layer Configuration
- **Input Layer**: 4 continuous observations (cart position, cart velocity, pole angle, pole angular velocity)
- **Hidden Layer 1**: 64 LIF neurons
- **Hidden Layer 2**: 16 LIF neurons
- **Output Layer**: 2 Q-values (one per action: left/right)
- **Timesteps**: 30 timesteps per inference

**NOTE**: The hidden layer sizes may vary depending on the model. For the snnTorch Leaky neuron model, sizes of 64 and 16 for the hidden layers were necessary to generate reasonably good performance.

### Data Format
- **Fixed-Point Format**: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
  - Scale factor: 2^13 = 8192
  - Range: [-4.0, 3.9998779296875]
- **LIF Parameters**:
  - Threshold: 1.0 (8192 in fixed-point)
  - Beta: 0.9 (115 in Q1.7 format for 8-bit)
  - Membrane potential: 24-bit signed (extra headroom)
  - Reset mechanism: subtract (reset_delay=True)

## Module Specifications

### 1. `linear_layer` Module

**Purpose**: Implements a fully connected layer (equivalent to nn.Linear in PyTorch)

**Interface**:
```systemverilog
module linear_layer #(
    parameter NUM_INPUTS = 4,
    parameter NUM_OUTPUTS = 64,
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS = 13,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE = "bias.mem"
) (
    input wire clk,
    input wire reset,
    input wire start,                                    // Start computation
    input wire signed [DATA_WIDTH-1:0] inputs [0:NUM_INPUTS-1],
    output logic signed [DATA_WIDTH-1:0] output_current, // One current per cycle
    output logic [IDX_WIDTH-1:0] output_idx,            // Which neuron (0 to NUM_OUTPUTS-1)
    output logic output_valid,                          // Current output is valid
    output logic done                                   // All outputs computed
);
```

**Behavior**:
1. When `start` asserted, latch input values
2. Compute one output neuron per cycle:
   - `output_current = Σ(input[i] × weight[neuron_idx][i]) + bias[neuron_idx]`
3. Assert `output_valid` each cycle with corresponding `output_idx`
4. Assert `done` after NUM_OUTPUTS cycles
5. Outputs remain stable until next `start`

**Resource Usage**:
- NUM_INPUTS multipliers (e.g., 4 for input layer, 64 for hidden layer 2)
- 1 accumulator
- Weights storage: NUM_OUTPUTS × NUM_INPUTS × 16 bits
- Bias storage: NUM_OUTPUTS × 16 bits

**Timing**:
- Latency: NUM_OUTPUTS cycles
- Throughput: 1 output current per cycle (after first cycle)

### 2. `lif` Module

**Purpose**: Leaky Integrate-and-Fire neuron with reset-by-subtraction

**Interface**:
```systemverilog
module lif #(
    parameter THRESHOLD = 8192,    // 1.0 in QS2.13
    parameter BETA = 115           // 0.9 in Q1.7
) (
    input wire clk,
    input wire reset,
    input wire start,                        // Latch current and begin timesteps
    input wire signed [15:0] current,        // Input current (QS2.13)
    output logic spike_out,                  // Current timestep spike
    output logic signed [23:0] membrane_out, // Current membrane potential
    output logic [4:0] timestep,             // Current timestep (0-29)
    output logic done                        // All timesteps complete
);
```

**Behavior**:
1. When `start` asserted, latch `current` input
2. Run for 30 timesteps internally:
   - Each cycle: update membrane, check threshold, output spike
   - `spike_out` valid each cycle with corresponding `timestep`
3. Assert `done` after 30 timesteps
4. Hold state until next `start`

**State Update (per timestep)**:
```
decay_potential = (membrane_potential × BETA) >> 7
reset_subtract = (spike_prev) ? THRESHOLD : 0
membrane_potential = decay_potential + current - reset_subtract
spike = (membrane_potential >= THRESHOLD)
```

**Resource Usage**:
- 1 multiplier (for beta decay)
- 24-bit membrane potential register
- 1-bit previous spike register

**Timing**:
- Internal: 30 cycles per inference
- External: Triggered by `start`, signals completion with `done`

### 3. `spike_buffer` Module

**Purpose**: Collect and synchronize spikes from all neurons in a layer

**Interface**:
```systemverilog
module spike_buffer #(
    parameter NUM_NEURONS = 64,
    parameter NUM_TIMESTEPS = 30
) (
    input wire clk,
    input wire reset,
    input wire [NUM_NEURONS-1:0] neuron_done,   // Done signal from each neuron
    input wire [NUM_NEURONS-1:0] spike_in,      // Current spike from each neuron
    input wire [4:0] timestep,                  // Current timestep
    output logic [NUM_NEURONS-1:0] spikes_out,  // Spike vector for current timestep
    output logic timestep_ready                 // All neurons ready for this timestep
);
```

**Behavior**:
1. Wait for all neurons to reach same timestep
2. When `neuron_done[63]` asserts (last neuron starts timestep 0), begin synchronization
3. Each cycle, collect spikes from all neurons at current timestep
4. Assert `timestep_ready` when all neurons have completed current timestep
5. Output synchronized spike vector

### 4. `membrane_buffer` Module

**Purpose**: Collect and synchronize membrane potentials from all neurons in final hidden layer

**Interface**:
```systemverilog
module membrane_buffer #(
    parameter NUM_NEURONS = 16,
    parameter NUM_TIMESTEPS = 30,
    parameter MEMBRANE_WIDTH = 24
) (
    input wire clk,
    input wire reset,
    input wire [NUM_NEURONS-1:0] neuron_done,              // Done signal from each neuron
    input wire signed [MEMBRANE_WIDTH-1:0] membrane_in [0:NUM_NEURONS-1], // Membrane potentials
    input wire [4:0] timestep,                             // Current timestep
    output logic signed [MEMBRANE_WIDTH-1:0] membranes_out [0:NUM_NEURONS-1], // Membrane vector
    output logic timestep_ready                            // All neurons ready for this timestep
);
```

**Behavior**:
1. Collect membrane potentials (not spikes) from final hidden layer neurons
2. Synchronize across all neurons for each timestep
3. Provide full membrane vector to output linear layer (fc_out)

## Pipelined Execution Flow

### Critical Path Analysis

The network processes data through sequential dependencies where each timestep of one layer must complete before the next layer can begin processing that timestep. The critical path determines overall latency.

### Phase 1: Hidden Layer 1 - Load Currents (Cycles 0-63)

**Linear Layer 0 (fc1)**: Produces currents for 64 neurons
```
Cycle 0:  linear_layer[0] outputs current[0] → LIF[0] latches and starts timestep 0
Cycle 1:  linear_layer[0] outputs current[1] → LIF[1] latches and starts timestep 0
          LIF[0] progresses to timestep 1
...
Cycle 63: linear_layer[0] outputs current[63] → LIF[63] latches and starts timestep 0
          LIF[0..62] are at various timesteps (0 started first, so is at timestep 63)
```

**Note**: LIFs start their timestep sequence immediately upon receiving current. This creates a "wave" where earlier neurons are further ahead in timesteps.

### Phase 2: Hidden Layer 1 - Complete Timesteps (Cycles 30-93)

```
Cycle 30: LIF[0] completes timestep 29 (last timestep) and asserts done
          LIF[63] is at timestep 0 (hasn't completed any timestep yet)
...
Cycle 64: LIF[63] is at timestep 1
...
Cycle 93: LIF[63] completes timestep 29 (last timestep) and asserts done
```

**spike_buffer[0]**: Collects spikes from all 64 neurons
- At cycle 63: All neurons have completed timestep 0 → spike vector for timestep 0 is ready
- At cycle 64: All neurons have completed timestep 1 → spike vector for timestep 1 is ready
- ...
- At cycle 92: All neurons have completed timestep 29 → spike vector for timestep 29 is ready

### Phase 3: Hidden Layer 2 - Process All Timesteps (Cycles 63-542)

For **each** of the 30 timesteps, linear_layer[1] must process the 64-bit spike vector:

**Timestep 0** (Cycles 63-78):
```
Cycle 63: spike_buffer[0] provides 64 spikes from timestep 0
          linear_layer[1] starts, outputs current[0] → LIF2[0] starts timestep 0
Cycle 64: linear_layer[1] outputs current[1] → LIF2[1] starts timestep 0
...
Cycle 78: linear_layer[1] outputs current[15] → LIF2[15] starts timestep 0
```

**Timestep 1** (Cycles 79-94):
```
Cycle 79: spike_buffer[0] provides 64 spikes from timestep 1
          linear_layer[1] outputs current[0] → LIF2[0] progresses to timestep 1
...
Cycle 94: linear_layer[1] outputs current[15] → LIF2[15] progresses to timestep 1
```

**Timestep 29** (Cycles 527-542):
```
Cycle 527: spike_buffer[0] provides 64 spikes from timestep 29 (last timestep)
           linear_layer[1] outputs current[0] → LIF2[0] progresses to timestep 29
...
Cycle 542: linear_layer[1] outputs current[15] → LIF2[15] progresses to timestep 29
```

**Total cycles for Hidden Layer 2**: 30 timesteps × 16 cycles/timestep = 480 cycles (cycles 63-542)

**Completion**:
```
Cycle 556: LIF2[0] completes timestep 29 (started at cycle 78, runs for 30 cycles)
...
Cycle 571: LIF2[15] completes timestep 29 (started at cycle 542, runs for 30 cycles)
```

### Phase 4: Output Layer - Process All Timesteps (Cycles 556-615)

**membrane_buffer[0]**: Collects membrane potentials (not spikes) from 16 neurons in hidden layer 2

For **each** of the 30 timesteps, linear_layer[2] (fc_out) processes 16 membrane potentials:

**Timestep 0** (Cycles 556-557):
```
Cycle 556: membrane_buffer[0] provides 16 membrane values from timestep 0
           linear_layer[2] outputs q[0] (Q-value for action 0)
Cycle 557: linear_layer[2] outputs q[1] (Q-value for action 1)
           Q_accum[0] += q[0], Q_accum[1] += q[1]
```

**Pattern continues**: 30 timesteps × 2 cycles/timestep = 60 cycles

**Cycles 556-615**: Output layer accumulates Q-values across all timesteps

### Phase 5: Q-Value Normalization and Action Selection (Cycles 616-617)
```
Cycle 616: Divide accumulated Q-values by number of timesteps (30)
           Q_final[0] = Q_accum[0] / 30
           Q_final[1] = Q_accum[1] / 30
Cycle 617: Compare Q_final[0] vs Q_final[1]
           Output action = argmax(Q_final) → 0 (left) or 1 (right)
```

**Note**: Division by 30 matches snn_policy.py behavior: `q_values = out_accum / float(self.num_steps)`

## Total Latency

**Critical path**:
- **Hidden Layer 1 load**: 64 cycles (to get last neuron started)
- **Hidden Layer 1 complete**: +29 cycles (for last neuron to finish all timesteps) = 93 cycles total
- **Hidden Layer 2 process all timesteps**: 30 timesteps × 16 cycles = 480 cycles (cycles 63-542)
- **Hidden Layer 2 complete**: +29 cycles (for last neuron to finish) = cycle 571
- **Output layer process all timesteps**: 30 timesteps × 2 cycles = 60 cycles (cycles 556-615)
- **Q-value normalization and action selection**: 2 cycles

**Total**: **617 cycles per inference**

@ 100MHz = **6.17µs per inference**

**Note**: This is fast enough for real-time CartPole control (typical requirement < 20ms response time)

## Weight File Format

Weights exported from snn_policy.py must be formatted as:

### Linear Layer Weights
- **Format**: Row-major flattened array
- **File**: `fc1_weights.mem` (or similar)
- **Structure**:
  ```
  weight[0][0]  // neuron 0, input 0
  weight[0][1]  // neuron 0, input 1
  ...
  weight[0][N-1]  // neuron 0, last input
  weight[1][0]  // neuron 1, input 0
  ...
  ```
- **Encoding**: 16-bit hex, QS2.13 fixed-point

### Biases
- **Format**: 1D array
- **File**: `fc1_bias.mem` (or similar)
- **Structure**: One bias per neuron
- **Encoding**: 16-bit hex, QS2.13 fixed-point

## Resource Estimates (64-16 Network)

### Linear Layers
- **fc1**: 4 multipliers, 64×4×16 = 4096 bits weights, 64×16 = 1024 bits bias
- **fc2**: 64 multipliers, 16×64×16 = 16384 bits weights, 16×16 = 256 bits bias
- **fc_out**: 16 multipliers, 2×16×16 = 512 bits weights, 2×16 = 32 bits bias

### LIF Neurons
- **Hidden Layer 1**: 64 instances × (1 multiplier + 24-bit membrane) = 64 multipliers, 1536 bits state
- **Hidden Layer 2**: 16 instances × (1 multiplier + 24-bit membrane) = 16 multipliers, 384 bits state

### Total Resource Usage
- **Multipliers**: 4 + 64 + 16 + 64 + 16 = 164 (plus multipliers for fractional-order neurons if used)
- **Memory**: ~22KB for weights + ~2KB for state

## Buffer Placement

Two buffer modules are required for synchronization:

1. **spike_buffer[0]**: Between Hidden Layer 1 LIF neurons and linear_layer[1]
   - Collects 64 spikes per timestep
   - Provides synchronized spike vectors to linear_layer[1]

2. **membrane_buffer[0]**: Between Hidden Layer 2 LIF neurons and linear_layer[2] (fc_out)
   - Collects 16 membrane potentials (24-bit each) per timestep
   - Provides synchronized membrane vectors to output layer
   - **Critical**: fc_out operates on membrane potentials, not spikes (per snn_policy.py)

## Software Integration

For hardware-in-the-loop evaluation, an alternative SNNPolicy class will be needed:

### Hardware-Accelerated Policy Class

```python
class SNNPolicyHardware(nn.Module):
    """
    Hardware-accelerated SNN policy that offloads forward pass to FPGA.
    Maintains same interface as SNNPolicy for seamless integration with DQN training loop.
    """

    def __init__(self, n_observations, n_actions, fpga_interface):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.fpga = fpga_interface  # FPGA communication interface

    def forward(self, observations):
        """
        Forward pass using hardware accelerator.

        Args:
            observations: [batch, 4] tensor of CartPole observations

        Returns:
            q_values: [batch, 2] tensor of Q-values
        """
        batch_size = observations.size(0)
        q_values = torch.zeros(batch_size, self.n_actions)

        # Process each sample in batch (hardware processes one at a time)
        for i in range(batch_size):
            obs = observations[i].cpu().numpy()

            # Convert to fixed-point and send to FPGA
            obs_fixed = self.float_to_qs2_13(obs)
            self.fpga.write_inputs(obs_fixed)
            self.fpga.start_inference()

            # Wait for completion and read Q-values
            self.fpga.wait_done()
            q_fixed = self.fpga.read_qvalues()

            # Convert back to float
            q_values[i] = torch.from_numpy(self.qs2_13_to_float(q_fixed))

        return q_values

    def float_to_qs2_13(self, values):
        """Convert float array to QS2.13 fixed-point."""
        return np.round(values * 8192).astype(np.int16)

    def qs2_13_to_float(self, values):
        """Convert QS2.13 fixed-point to float array."""
        return values.astype(np.float32) / 8192.0
```

### Usage in Evaluation

```python
# Load trained weights (already quantized for hardware)
# Initialize FPGA with trained weights

# Create hardware policy
fpga_interface = FPGAInterface(device_path="/dev/fpga0")  # Platform-specific
policy_hw = SNNPolicyHardware(n_observations=4, n_actions=2, fpga_interface=fpga_interface)

# Use in existing evaluation loop (no changes needed)
env = gym.make("CartPole-v1")
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_hw(state_tensor)  # Hardware forward pass
        action = q_values.argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
```

### Performance Validation

Compare hardware vs. software inference:
- Accuracy: Q-values should match within quantization error (~0.0001)
- Latency: Hardware should be 6.17µs vs. software ~100µs (on CPU)
- Throughput: Measure frames per second in CartPole environment

### Testing Strategy

1. **Unit tests**: Verify fixed-point conversion accuracy
2. **Layer-by-layer tests**: Compare hardware vs. software outputs after each layer
3. **End-to-end tests**: Compare final Q-values and action selection
4. **Episode tests**: Run full CartPole episodes, compare episode length distributions

## Notes and Future Enhancements

1. **Fractional-Order Neurons**: If using fractional-order LIF neurons, each instance requires additional DSP resources for the Grünwald-Letnikov approximation

2. **Time-Multiplexing**: Current design can be upgraded to process N neurons in parallel per cycle by adding N parallel accumulators to linear_layer and processing N LIF instances simultaneously. This would reduce latency proportionally (e.g., processing 4 neurons in parallel reduces Hidden Layer 2 from 480 cycles to 120 cycles).

3. **Reduced Precision**: Consider 8-bit or mixed-precision to reduce resource usage if accuracy permits

4. **Dynamic Timesteps**: Add early stopping mechanism if Q-values converge before 30 timesteps

5. **Membrane Potential Precision**: The output layer uses 24-bit membrane potentials but could potentially use reduced precision (16-bit) if testing shows acceptable accuracy
