# Cocotb Testing Environment for CartPole SNN

Docker-based cocotb testing environment for the LIF neuron and linear layer modules using Icarus Verilog.

## Features

- **Multi-stage Dockerfile** with layer caching to avoid reinstalling tools on minor changes
- **oss-cad-suite** bundle (includes Icarus Verilog, cocotb, and other EDA tools)
- **No virtual environment** - uses oss-cad-suite's bundled cocotb to avoid version conflicts
- **Volume mounting** for easy access to source files and test results
- **Live reload** - changes to source and test files are immediately visible in container
- **FST waveform tracing** - enabled by default for all tests

## Directory Structure

```
cocotb/
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Docker Compose configuration
├── README.md            # This file
├── tests/               # Test directory (mounted as read-write)
│   ├── test_lif.py          # Tests for LIF neuron module
│   ├── test_linear_layer.py # Tests for linear layer module
│   ├── Makefile             # Makefile for running tests
│   └── weights/             # Test weight files
│       ├── test_weights.mem # Identity matrix weights
│       └── test_bias.mem    # Zero bias values
└── results/             # Test results and waveforms (generated)
    └── sim_build/       # Simulation artifacts and waveforms
        ├── lif.fst          # LIF neuron waveform
        └── linear_layer.fst # Linear layer waveform
```

## Prerequisites

- Docker
- Docker Compose (optional but recommended)

## Quick Start

### 1. Build the Docker Image

```bash
cd cocotb
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
```

This builds a multi-stage image with:
- Ubuntu 22.04 base
- oss-cad-suite (includes Verilator, Icarus, and cocotb bundled together)
- Non-root user matching host user's UID/GID (defaults to 1000)

**Layer caching:** The oss-cad-suite installation is cached unless the version is updated.

**File ownership:** The container runs as a non-root user. By default, it uses UID/GID 1000. To match the host user:

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
```

### 2. Start the Container

```bash
docker compose up -d
```

### 3. Run Tests

```bash
# Enter the container
docker compose exec cocotb bash

# Run tests for LIF module, for example
make test_lif
```

Or run tests directly:

```bash
docker compose exec cocotb bash -c "make test_lif"
```

### 4. View Results

Test results and waveforms are written to `results/`:
- `results.xml` - JUnit-style test results
- `sim_build/lif.fst` - LIF neuron waveform
- `sim_build/linear_layer.fst` - Linear layer waveform

### 5. Stop the Container

```bash
docker compose down
```

## Test Descriptions

### LIF Neuron Tests (`test_lif.py`)

| Test | Description |
|------|-------------|
| `test_lif_reset` | Verifies reset initializes neuron state |
| `test_lif_no_spike_below_threshold` | Small inputs don't cause spikes |
| `test_lif_spike_above_threshold` | Large inputs cause immediate spike |
| `test_lif_membrane_accumulation` | Membrane builds up over time and spikes |
| `test_lif_consecutive_spiking` | High sustained input causes consecutive spikes |
| `test_lif_reset_by_subtraction` | Verifies reset-by-subtraction prevents immediate re-spike |
| `test_lif_multiple_starts` | Module can be started multiple times |
| `test_lif_negative_input` | Negative inputs don't cause spikes |
| `test_lif_beta_decay` | Membrane decays when input removed |

### Linear Layer Tests (`test_linear_layer.py`)

| Test | Description |
|------|-------------|
| `test_linear_layer_reset` | Verifies reset initializes layer state |
| `test_linear_layer_identity_weights` | Identity weights pass inputs through unchanged |
| `test_linear_layer_multiple_runs` | Module can be started multiple times |
| `test_linear_layer_timing` | Verifies NUM_OUTPUTS valid outputs in correct timing |

## Fixed-Point Format

The LIF neuron uses **QS2.13** fixed-point format:
- 16-bit signed
- 2 integer bits, 13 fractional bits
- Scale factor: 8192 (2^13)
- Threshold 1.0 = 8192
- Beta 0.9 ≈ 115 in Q1.7 format

## Viewing Waveforms

After running tests, view waveforms with GTKWave:

```bash
# On host (if GTKWave installed)
gtkwave results/sim_build/lif.fst
gtkwave results/sim_build/linear_layer.fst
```

## Troubleshooting

### Permission Issues

If there are permission errors, rebuild with the host system user ID:

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build --no-cache
```

### Simulation Errors

Check the Icarus Verilog output in `results/sim_build/` for compilation errors.

### Test Failures

Examine the cocotb log output and `results/results.xml` for details.
