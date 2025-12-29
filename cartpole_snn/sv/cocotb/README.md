# Cocotb Testing Environment for CartPole SNN

Docker-based cocotb testing environment for the LIF neuron and neural network modules using Verilator.

## Features

- **Multi-stage Dockerfile** with layer caching to avoid reinstalling tools on minor changes
- **oss-cad-suite** bundle (includes Verilator, cocotb, and other EDA tools)
- **No virtual environment** - uses oss-cad-suite's bundled cocotb to avoid version conflicts
- **Volume mounting** for easy access to source files and test results
- **Live reload** - changes to source and test files are immediately visible in container

## Directory Structure

```
cocotb/
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Docker Compose configuration
├── README.md            # This file
├── tests/               # Test directory (mounted as read-write)
│   ├── test_lif.py      # Tests for LIF neuron module
│   └── Makefile         # Makefile for running tests
└── results/             # Test results and waveforms (generated)
```

Note: `results/` is created when tests run and contains FST waveforms, XML test results, and Verilator build artifacts.

## Prerequisites

- Docker
- Docker Compose (optional but recommended)

## Quick Start

### 1. Build the Docker Image

```bash
cd cocotb
docker compose build
```

This builds a multi-stage image with:
- Ubuntu 22.04 base
- oss-cad-suite (includes Verilator and cocotb bundled together)
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

# Run tests
make
```

Or run tests directly:

```bash
docker compose exec cocotb bash -c "make"
```

### 4. View Results

Test results are written to `results/`:
- `results.xml` - JUnit-style test results
- `sim_build/` - Verilator build artifacts
- `dump.fst` - Waveform file (viewable with GTKWave)

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
| `test_lif_reset_after_spike` | Verifies reset-by-subtraction behavior |
| `test_lif_enable_hold_state` | State holds when enable is low |
| `test_lif_negative_input` | Negative inputs don't cause spikes |
| `test_lif_beta_decay` | Membrane decays when input removed |

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
gtkwave results/sim_build/dump.fst
```

## Troubleshooting

### Permission Issues

If there are permission errors, rebuild with the host system user ID:

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build --no-cache
```

### Simulation Errors

Check the Verilator output in `results/sim_build/` for compilation errors.

### Test Failures

Examine the cocotb log output and `results/results.xml` for details.
