# Makefile for running cocotb tests on SNN modules
#
# Usage:
#   make                    # Run LIF tests (default)
#   make test_lif           # Run LIF tests
#   make test_linear_layer  # Run linear_layer tests
#   make clean              # Clean build artifacts

# Default test module
TEST ?= lif

# Configuration based on TEST variable
ifeq ($(TEST),lif)
    VERILOG_SOURCES = $(PWD)/../sv/lif.sv
    TOPLEVEL = lif
    MODULE = test_lif
else ifeq ($(TEST),linear_layer)
    VERILOG_SOURCES = $(PWD)/../sv/linear_layer.sv
    TOPLEVEL = linear_layer
    MODULE = test_linear_layer
    # Parameters are set differently per simulator (see below)
else
    $(error Unknown TEST value: $(TEST). Use 'lif' or 'linear_layer')
endif

# Simulator - use Icarus for all tests (event-driven, correct timing)
SIM := icarus

# Simulator-specific flags
ifeq ($(SIM),icarus)
    # Enable waveform tracing (FST format by default)
    WAVES = 1
    # Icarus uses -P for parameter overrides
    ifeq ($(TEST),linear_layer)
        COMPILE_ARGS += -Plinear_layer.NUM_INPUTS=4 -Plinear_layer.NUM_OUTPUTS=4
        COMPILE_ARGS += -Plinear_layer.WEIGHTS_FILE=\"$(PWD)/weights/test_weights.mem\"
        COMPILE_ARGS += -Plinear_layer.BIAS_FILE=\"$(PWD)/weights/test_bias.mem\"
    endif
endif

# Output directory for simulation results
RESULTS_DIR = ../results
SIM_BUILD = $(RESULTS_DIR)/sim_build

# Export variables for cocotb
export COCOTB_RESULTS_FILE = $(RESULTS_DIR)/results.xml

# Include cocotb's make rules
include $(shell cocotb-config --makefiles)/Makefile.sim

# Convenience targets
test_lif:
	$(MAKE) TEST=lif

test_linear_layer:
	$(MAKE) TEST=linear_layer

# Clean up generated files
clean::
	rm -rf $(RESULTS_DIR)/*
	rm -rf __pycache__
	rm -rf .pytest_cache

.PHONY: clean test_lif test_linear_layer
