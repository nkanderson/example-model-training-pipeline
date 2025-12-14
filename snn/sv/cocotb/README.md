# Cocotb Testing Environment

Docker-based cocotb testing environment for SystemVerilog modules using Verilator.

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
├── tests/              # Test directory (mounted as read-write)
│   ├── test_neural_network.py  # Tests for neural network module
│   └── Makefile        # Makefile for running tests
└── results/            # Test results and waveforms (generated)
```

Note: `results/` is created when tests run and contains VCD/FST waveforms, XML test results, and Verilator build artifacts.

## Prerequisites

- Docker
- Docker Compose (optional but recommended)

## Quick Start

### 1. Build the Docker Image

```bash
cd cocotb
docker-compose build
```

This builds a multi-stage image with:
- Ubuntu 22.04 base
- oss-cad-suite (includes Verilator and cocotb bundled together)
- Non-root user matching host user's UID/GID (defaults to 1000)

**Layer caching:** The oss-cad-suite installation is cached unless you change the version.

**File ownership:** The container runs as a non-root user. By default, it uses UID/GID 1000. To match the host user:

```bash
# Use current user's UID/GID (recommended)
USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose build

# Or export them once in the shell session
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
docker-compose build

# To use a specific UID/GID (e.g., for a different user)
USER_ID=1001 GROUP_ID=1001 docker-compose build
```

Files created in the `results/` directory will be owned by the user with this UID/GID on the host machine. 

### 2. Run Tests

#### Option A: Using Docker Compose (Recommended)

```bash
# Start the container
docker-compose run --rm cocotb

# Or start the container and get a shell in one command
docker-compose run --rm cocotb /bin/bash

# SSH into the container if already running
docker-compose exec cocotb /bin/bash

# Inside the container, run tests
make

# Or run with parallel execution
make -j$(nproc)
```

#### Option B: Using Docker Directly

```bash
# Build the image
docker build -t cocotb:latest .

# Run tests
docker run --rm \
  -v $(pwd)/..:/workspace/sv:ro \
  -v $(pwd)/tests:/workspace/tests:rw \
  -v $(pwd)/results:/workspace/results:rw \
  -w /workspace/tests \
  cocotb:latest \
  make
```

### 3. View Results

Test results are written to `results/`:
- `results.xml` - JUnit-style test results
- `dump.fst` or `dump.vcd` - Waveform files (if enabled)
- `sim_build/` - Verilator build artifacts

View waveforms with GTKWave:
```bash
gtkwave results/dump.fst
```

## Writing Tests

Tests are written in Python using cocotb. Example structure:

```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

@cocotb.test()
async def test_my_module(dut):
    """Test description"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Apply reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    
    # Test logic
    dut.input_signal.value = 1
    await RisingEdge(dut.clk)
    assert dut.output_signal.value == 1
```

### Test Configuration

Edit `tests/Makefile` to configure:
- `VERILOG_SOURCES` - Path to SystemVerilog files
- `TOPLEVEL` - Top-level module name
- `MODULE` - Python test module name

## Example Tests

The included `test_lif.py` demonstrates:
- **test_lif_reset** - Verifying reset behavior
- **test_lif_threshold** - Testing spike generation at threshold
- **test_lif_enable** - Checking enable signal gating
- **test_lif_decay** - Validating membrane potential decay

Run specific tests:
```bash
# Inside the container
make MODULE=test_lif TESTCASE=test_lif_threshold
```

## Volume Mounts

The Docker setup uses three volume mounts:

1. **`..` → `/workspace/sv:ro`** (read-only)
   - Mounts the parent `snn/sv/` directory containing all SystemVerilog source files
   - **Read-only ensures source files can't be accidentally modified by the container**
   - Docker detects changes: when `.sv` files on the host are edited, the container sees the updates immediately
   
2. **`./tests` → `/workspace/tests:rw`** (read-write)
   - Mounts the `tests/` directory containing Python test files and Makefiles
   - **Read-write allows the container to create `__pycache__/` and other Python artifacts**
   - Docker detects changes: when test `.py` files on the host are edited, the container sees them immediately
   - The Makefile runs from this directory (`working_dir: /workspace/tests`)
   
3. **`./results` → `/workspace/results:rw`** (read-write)
   - Mounts the `results/` directory for test outputs
   - **Read-write required so container can write test results, waveforms (VCD/FST), and build artifacts**
   - **Files created here by the container will be owned by root on the host** (see Troubleshooting section)
   - Results persist after the container exits so waveforms may be viewed with GTKWave on the host

## Customization

### Using a Different Simulator

The setup uses Verilator by default. To use a different simulator:

```bash
# In tests/Makefile or as environment variable
SIM=icarus make

# Or with docker-compose
docker-compose run -e SIM=icarus cocotb make
```

### Updating oss-cad-suite Version

Edit the `Dockerfile` and change the `OSS_CAD_SUITE_VERSION` build arg:

```dockerfile
ARG OSS_CAD_SUITE_VERSION=2024-11-22
```

Then rebuild:
```bash
docker-compose build --no-cache
```

### Adding Python Dependencies

Add packages to `requirements.txt` and rebuild:

```bash
echo "numpy>=1.24.0" >> requirements.txt
docker-compose build
```

## Troubleshooting

### Tests fail to find modules

Ensure `VERILOG_SOURCES` in `tests/Makefile` points to the correct path relative to the Makefile location.

### Permission issues with results directory

**The container is configured to run as a non-root user by default.** The user's UID/GID is set during the build process:

- **Default**: Uses UID/GID 1000 (common default for first user on Linux systems)
- **Recommended**: Build with current UID/GID: `USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose build`

Files created in the `results/` directory will be owned by the user with the specified UID/GID on the host machine.

**If you encounter permission issues:**

1. Check the ownership of files in `results/`:
   ```bash
   ls -la results/
   ```

2. If files are owned by a different user, rebuild the image with correct UID/GID:
   ```bash
   USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose build
   ```

3. For files created by a previous container, fix ownership:
   ```bash
   sudo chown -R $USER:$USER results/
   ```

### Verilator compilation errors

Check the Verilator output in the console. Common issues:
- Syntax errors in SystemVerilog
- Missing `timescale directive
- Unsupported SystemVerilog constructs

## Interactive Development

For interactive debugging:

```bash
# Start container with bash
docker-compose run --rm cocotb

# Inside container, you have access to:
# - verilator --version
# - python3
# - make
# - All source files mounted under /workspace/
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Cocotb Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: cd snn/sv/cocotb && docker-compose build
      
      - name: Run tests
        run: cd snn/sv/cocotb && docker-compose run --rm cocotb make
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: snn/sv/cocotb/results/
```

## Additional Resources

- [cocotb Documentation](https://docs.cocotb.org/)
- [Verilator Manual](https://verilator.org/guide/latest/)
- [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build)
