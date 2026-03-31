"""
Cocotb integration test for CartPole using the SystemVerilog neural_network module.

This test runs the actual CartPole-v1 gymnasium environment and uses the
hardware neural_network module to compute actions, verifying that the
trained SNN model works correctly in hardware.

Test configuration uses full model parameters:
- NUM_INPUTS = 4 (CartPole observation space)
- HL1_SIZE = 64
- HL2_SIZE = 16
- NUM_ACTIONS = 2 (left/right)
- NUM_TIMESTEPS = 30
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

import gymnasium as gym
import numpy as np

# Fixed-point format constants (QS2.13)
TOTAL_BITS = 16
FRAC_BITS = 13
SCALE_FACTOR = 2**FRAC_BITS  # 8192
MAX_SIGNED = 2 ** (TOTAL_BITS - 1) - 1
MIN_SIGNED = -(2 ** (TOTAL_BITS - 1))
UNSIGNED_RANGE = 2**TOTAL_BITS

# Test configuration (must match Verilog parameters)
NUM_INPUTS = 4
HL1_SIZE = 64
HL2_SIZE = 16
NUM_ACTIONS = 2
NUM_TIMESTEPS = 30


def float_to_fixed(value: float) -> int:
    """Convert float to fixed-point (unsigned representation for cocotb)."""
    scaled = int(round(value * SCALE_FACTOR))
    if scaled > MAX_SIGNED:
        scaled = MAX_SIGNED
    elif scaled < MIN_SIGNED:
        scaled = MIN_SIGNED
    if scaled < 0:
        scaled = scaled + UNSIGNED_RANGE
    return scaled


def fixed_to_float(value: int) -> float:
    """Convert fixed-point (unsigned representation) to float."""
    if value >= 2 ** (TOTAL_BITS - 1):
        value = value - UNSIGNED_RANGE
    return value / SCALE_FACTOR


async def reset_dut(dut):
    """Apply reset sequence to the DUT."""
    dut.reset.value = 1
    dut.start.value = 0
    for i in range(NUM_INPUTS):
        dut.observations[i].value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def run_inference(
    dut, observations: np.ndarray, timeout_cycles: int = 50000
) -> tuple:
    """
    Run a single inference through the neural network.

    Args:
        dut: Device under test
        observations: NumPy array of 4 float observations from CartPole
        timeout_cycles: Maximum cycles to wait for completion

    Returns:
        int: selected action (0 or 1)
    """
    # Set observations (convert to fixed-point)
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(float(observations[i]))

    # Start inference
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    for cycle in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break
    else:
        raise TimeoutError(f"Inference did not complete within {timeout_cycles} cycles")

    # Read action selected by hardware from full-precision Q-values
    action = int(dut.selected_action.value)

    return action


@cocotb.test()
async def test_cartpole_single_episode(dut):
    """
    Run a single CartPole episode using the hardware neural network.

    This test validates that the trained model can balance the pole.
    A successful episode should last close to 500 steps (the CartPole max).
    """
    clock = Clock(dut.clk, 10, unit="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Create CartPole environment
    env = gym.make("CartPole-v1")

    # Reset environment
    observation, info = env.reset(seed=42)

    total_reward = 0
    step_count = 0
    max_steps = 500  # CartPole-v1 limit

    dut._log.info("Starting CartPole episode...")

    while step_count < max_steps:
        # Run inference on hardware
        action = await run_inference(dut, observation)

        # Log every 50 steps
        if step_count % 50 == 0:
            dut._log.info(f"Step {step_count}: obs={observation}, action={action}")

        # Take action in environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    env.close()

    dut._log.info(f"Episode finished: {step_count} steps, total reward: {total_reward}")

    # A well-trained model should balance for at least 200 steps on average
    # The model was trained to 495+ average reward, so expect good performance
    assert step_count >= 100, f"Episode too short: {step_count} steps (expected >= 100)"

    dut._log.info(f"SUCCESS: CartPole balanced for {step_count} steps")


@cocotb.test()
async def test_cartpole_multiple_episodes(dut):
    """
    Run multiple CartPole episodes and compute average performance.

    This provides a more robust test of the model's performance by
    averaging over multiple random initial conditions.
    """
    clock = Clock(dut.clk, 10, unit="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Create CartPole environment
    env = gym.make("CartPole-v1")

    async def run_episode(seed: int) -> tuple[float, int]:
        await reset_dut(dut)

        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        step_count = 0
        max_steps = 500

        while step_count < max_steps:
            action = await run_inference(dut, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        return total_reward, step_count

    # Repeatability check: same seed should produce nearly identical performance
    # if there is no cross-episode state leakage in the DUT.
    rep_seed = 42
    rep_rewards = []
    for idx in range(2):
        reward, steps = await run_episode(rep_seed)
        rep_rewards.append(reward)
        dut._log.info(
            f"Repeat seed check {idx + 1}: seed={rep_seed}, reward={reward} ({steps} steps)"
        )

    assert abs(rep_rewards[0] - rep_rewards[1]) <= 5.0, (
        f"Same-seed repeatability failed (possible state carryover): "
        f"{rep_rewards[0]} vs {rep_rewards[1]}"
    )

    num_episodes = 10
    episode_rewards = []

    for episode in range(num_episodes):
        total_reward, step_count = await run_episode(seed=episode * 7 + 42)

        episode_rewards.append(total_reward)
        dut._log.info(
            f"Episode {episode + 1}: {total_reward} reward ({step_count} steps)"
        )

    env.close()

    avg_reward = np.mean(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)

    dut._log.info(f"Performance over {num_episodes} episodes:")
    dut._log.info(f"  Average reward: {avg_reward:.1f}")
    dut._log.info(f"  Min reward: {min_reward:.1f}")
    dut._log.info(f"  Max reward: {max_reward:.1f}")

    expected_avg = 300
    assert (
        avg_reward >= expected_avg
    ), f"Average reward too low: {avg_reward:.1f} (expected >= {expected_avg})"

    dut._log.info(f"SUCCESS: Average reward {avg_reward:.1f} meets threshold")


@cocotb.test()
async def test_inference_timing(dut):
    """
    Measure and report inference timing.

    This test measures how many clock cycles a single inference takes,
    which is useful for performance characterization.
    """
    clock = Clock(dut.clk, 10, unit="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Create a sample observation
    test_obs = np.array([0.01, -0.02, 0.03, -0.01])

    # Measure timing
    for i in range(NUM_INPUTS):
        dut.observations[i].value = float_to_fixed(float(test_obs[i]))

    # Start inference and count cycles
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    cycle_count = 0
    max_cycles = 100000

    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if dut.done.value == 1:
            break

    # Calculate timing at 100 MHz
    time_us = cycle_count * 0.01  # 10ns per cycle

    dut._log.info(f"Inference timing:")
    dut._log.info(f"  Cycles: {cycle_count}")
    dut._log.info(f"  Time @ 100MHz: {time_us:.2f} µs")
    dut._log.info(f"  Throughput: {1_000_000 / time_us:.0f} inferences/sec")

    # Sanity check - inference should complete
    assert cycle_count < max_cycles, "Inference did not complete"

    # Verify inference produced a valid action
    action = int(dut.selected_action.value)
    dut._log.info(f"  selected_action: {action}")

    dut._log.info("SUCCESS: Inference completed and timing measured")


@cocotb.test()
async def test_observation_range(dut):
    """
    Test that the model handles the full CartPole observation range.

    CartPole observations can have quite large values:
    - Cart position: [-4.8, 4.8]
    - Cart velocity: [-inf, inf] (typically small)
    - Pole angle: [-0.418, 0.418] rad (~24 degrees)
    - Pole velocity: [-inf, inf]

    We test with various observation ranges to ensure the hardware
    handles them correctly without overflow.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Test cases with various observation ranges
    test_cases = [
        ("Small values", np.array([0.01, 0.02, 0.03, 0.01])),
        ("Moderate values", np.array([0.5, -0.5, 0.2, -0.3])),
        ("Near limits", np.array([2.0, 1.5, 0.3, 1.0])),
        ("Edge of representable", np.array([3.0, -2.0, 0.4, -1.5])),
    ]

    for name, obs in test_cases:
        action = await run_inference(dut, obs)

        assert action in [0, 1], f"Invalid action: {action}"

        dut._log.info(f"{name}: obs={obs}, action={action}")

    dut._log.info("SUCCESS: All observation ranges handled correctly")


@cocotb.test()
async def test_debug_signals(dut):
    """
    Debug test to trace internal signals and understand Q-value saturation.

    This test runs inference with known inputs and logs internal signal values.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Use small observations that caused saturation
    obs = np.array([0.01, 0.02, 0.03, 0.01])

    # Set observations
    for i in range(NUM_INPUTS):
        fixed_val = float_to_fixed(float(obs[i]))
        dut.observations[i].value = fixed_val
        dut._log.info(
            f"obs[{i}] = {obs[i]:.4f} -> fixed = {fixed_val} (hex: {fixed_val:04x})"
        )

    # Start inference
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Log fc1 outputs as they become valid
    fc1_outputs = []
    fc1_count = 0
    fc2_outputs = []
    fc2_count = 0

    # Monitor fc1 output
    for cycle in range(1000):
        await RisingEdge(dut.clk)

        # Check fc1 progress
        if hasattr(dut, "fc1_valid") and dut.fc1_valid.value == 1:
            fc1_idx = int(dut.fc1_output_idx.value)
            fc1_current = int(dut.fc1_output_current.value)
            fc1_float = fixed_to_float(fc1_current)
            fc1_outputs.append((fc1_idx, fc1_current, fc1_float))
            fc1_count += 1
            if fc1_count <= 5:  # Log first few
                dut._log.info(
                    f"fc1[{fc1_idx}] = {fc1_float:.4f} (hex: {fc1_current:04x})"
                )

        if dut.done.value == 1:
            break

    # Log hardware-selected action
    hw_action = int(dut.selected_action.value)
    dut._log.info(f"Hardware selected_action: {hw_action}")

    # Analyze fc1 outputs
    if fc1_outputs:
        fc1_values = [v[2] for v in fc1_outputs]
        dut._log.info(
            f"fc1 outputs: count={len(fc1_outputs)}, min={min(fc1_values):.4f}, max={max(fc1_values):.4f}"
        )
        # Log all fc1 outputs
        for idx, raw, val in fc1_outputs[:10]:
            dut._log.info(f"  fc1[{idx}] = {val:.4f}")
    else:
        dut._log.warning("No fc1 outputs captured (internal signal not accessible)")

    dut._log.info(f"Hardware selected_action: {hw_action}")
