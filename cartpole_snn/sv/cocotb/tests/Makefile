# Makefile for running cocotb tests on LIF neuron module

# Paths to source files (relative to this Makefile in tests/)
# In Docker, ../sv is mounted from the parent directory
VERILOG_SOURCES = $(PWD)/../sv/lif.sv

# Top-level module to test
TOPLEVEL = lif

# Python test module
MODULE = test_lif

# Simulator
SIM ?= verilator

# Verilator-specific flags
ifeq ($(SIM),verilator)
    COMPILE_ARGS += --timing
    EXTRA_ARGS += --trace --trace-fst --trace-structs
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
