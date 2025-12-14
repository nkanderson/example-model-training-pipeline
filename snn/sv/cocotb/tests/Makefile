# Makefile for running cocotb tests on neural network module

# Paths to source files (relative to this Makefile in tests/)
# Include both neural_network.sv and the lif.sv module it instantiates
# In Docker, ../sv is mounted from the parent directory
VERILOG_SOURCES = $(PWD)/../sv/neural_network.sv $(PWD)/../sv/lif.sv

# Top-level module to test
TOPLEVEL = neural_network

# Python test module
MODULE = test_neural_network

# Simulator
SIM ?= verilator

# Verilator-specific flags
ifeq ($(SIM),verilator)
    COMPILE_ARGS += --timing
    COMPILE_ARGS += --trace
    COMPILE_ARGS += --trace-structs
    EXTRA_ARGS += --trace-fst
endif

# Output directory for simulation results
RESULTS_DIR = ../results
SIM_BUILD = $(RESULTS_DIR)/sim_build

# Export variables for cocotb
export COCOTB_RESULTS_FILE = $(RESULTS_DIR)/results.xml

# Include cocotb's make rules
include $(shell cocotb-config --makefiles)/Makefile.sim

# Clean up generated files
clean::
	rm -rf $(RESULTS_DIR)/*
	rm -rf __pycache__
	rm -rf .pytest_cache

.PHONY: clean
