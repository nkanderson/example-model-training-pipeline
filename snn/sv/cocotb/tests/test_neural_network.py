"""
Cocotb tests for the SNN neural_network module.
Tests XOR functionality with spike counting over timesteps.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


@cocotb.test()
async def test_neural_network_reset(dut):
    """Test that reset properly initializes the neural network"""

    # Start clock (10ns period = 100MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.reset.value = 1
    dut.start.value = 0
    dut.inputs.value = 0

    await ClockCycles(dut.clk, 5)

    # Check reset state
    assert dut.done.value == 0, "done should be 0 after reset"
    assert dut.spike_count.value == 0, "spike_count should be 0 after reset"

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_xor_00(dut):
    """Test XOR with inputs [0,0] - expect low/zero spike count"""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset.value = 1
    dut.start.value = 0
    dut.inputs.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)

    # Set inputs [0,0] (both bits low)
    dut.inputs.value = 0b00
    await RisingEdge(dut.clk)

    # Start processing
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done signal (with timeout)
    timeout_cycles = 50
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done signal should be high after processing"

    spike_count = int(dut.spike_count.value)
    dut._log.info(f"XOR [0,0] spike count: {spike_count}")

    # XOR(0,0) = 0, expect low spike count (<=2)
    assert spike_count <= 2, f"Expected low spike count for XOR(0,0), got {spike_count}"

    dut._log.info("XOR [0,0] test passed")


@cocotb.test()
async def test_xor_01(dut):
    """Test XOR with inputs [0,1] - expect high spike count"""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset.value = 1
    dut.start.value = 0
    dut.inputs.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)

    # Set inputs [0,1] - inputs[0]=0, inputs[1]=1 -> bit pattern 0b10
    dut.inputs.value = 0b10
    await RisingEdge(dut.clk)

    # Start processing
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done signal
    timeout_cycles = 50
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done signal should be high after processing"

    spike_count = int(dut.spike_count.value)
    dut._log.info(f"XOR [0,1] spike count: {spike_count}")

    # XOR(0,1) = 1, expect high spike count (>=3)
    assert (
        spike_count >= 3
    ), f"Expected high spike count for XOR(0,1), got {spike_count}"

    dut._log.info("XOR [0,1] test passed")


@cocotb.test()
async def test_xor_10(dut):
    """Test XOR with inputs [1,0] - expect high spike count"""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset.value = 1
    dut.start.value = 0
    dut.inputs.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)

    # Set inputs [1,0] - inputs[0]=1, inputs[1]=0 -> bit pattern 0b01
    dut.inputs.value = 0b01
    await RisingEdge(dut.clk)

    # Start processing
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done signal
    timeout_cycles = 50
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done signal should be high after processing"

    spike_count = int(dut.spike_count.value)
    dut._log.info(f"XOR [1,0] spike count: {spike_count}")

    # XOR(1,0) = 1, expect high spike count (>=3)
    assert (
        spike_count >= 3
    ), f"Expected high spike count for XOR(1,0), got {spike_count}"

    dut._log.info("XOR [1,0] test passed")


@cocotb.test()
async def test_xor_11(dut):
    """Test XOR with inputs [1,1] - expect low/zero spike count"""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset.value = 1
    dut.start.value = 0
    dut.inputs.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)

    # Set inputs [1,1] (both bits high)
    dut.inputs.value = 0b11
    await RisingEdge(dut.clk)

    # Start processing
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done signal
    timeout_cycles = 50
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break

    assert dut.done.value == 1, "done signal should be high after processing"

    spike_count = int(dut.spike_count.value)
    dut._log.info(f"XOR [1,1] spike count: {spike_count}")

    # XOR(1,1) = 0, expect low spike count (<=2)
    assert spike_count <= 2, f"Expected low spike count for XOR(1,1), got {spike_count}"

    dut._log.info("XOR [1,1] test passed")


@cocotb.test()
async def test_all_xor_cases(dut):
    """Test all XOR cases in sequence and verify the pattern"""

    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Test cases: (inputs, expected_behavior, label)
    test_cases = [
        (0b00, "low", "[0,0]"),
        (0b10, "high", "[0,1]"),
        (0b01, "high", "[1,0]"),
        (0b11, "low", "[1,1]"),
    ]

    results = []

    for inputs, expected, label in test_cases:
        # Reset
        dut.reset.value = 1
        dut.start.value = 0
        dut.inputs.value = 0
        await ClockCycles(dut.clk, 5)
        dut.reset.value = 0
        await ClockCycles(dut.clk, 2)

        # Set inputs
        dut.inputs.value = inputs
        await RisingEdge(dut.clk)

        # Start processing
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        # Wait for done
        timeout_cycles = 50
        for _ in range(timeout_cycles):
            await RisingEdge(dut.clk)
            if dut.done.value == 1:
                break

        spike_count = int(dut.spike_count.value)
        results.append((label, spike_count, expected))
        dut._log.info(f"XOR {label}: {spike_count} spikes (expected {expected})")

    # Verify pattern
    dut._log.info("\n=== XOR Test Results ===")
    for label, count, expected in results:
        status = (
            "PASS"
            if (expected == "low" and count <= 2) or (expected == "high" and count >= 3)
            else "FAIL"
        )
        dut._log.info(f"{label}: {count} spikes ({expected}) - {status}")

    # Check that high cases have more spikes than low cases
    low_avg = (results[0][1] + results[3][1]) / 2  # [0,0] and [1,1]
    high_avg = (results[1][1] + results[2][1]) / 2  # [0,1] and [1,0]

    dut._log.info(f"Average spikes - Low: {low_avg:.1f}, High: {high_avg:.1f}")
    assert high_avg > low_avg, f"High cases should spike more than low cases"

    dut._log.info("All XOR cases test passed")
