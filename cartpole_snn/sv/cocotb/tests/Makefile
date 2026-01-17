# Makefile for running cocotb tests on SNN modules
#
# Usage:
#   make                      # Run LIF tests (default)
#   make test_lif             # Run LIF tests
#   make test_linear_layer    # Run linear_layer tests
#   make test_spike_buffer    # Run spike_buffer tests
#   make test_membrane_buffer # Run membrane_buffer tests
#   make clean                # Clean build artifacts

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
else ifeq ($(TEST),spike_buffer)
    VERILOG_SOURCES = $(PWD)/../sv/spike_buffer.sv
    TOPLEVEL = spike_buffer
    MODULE = test_spike_buffer
else ifeq ($(TEST),membrane_buffer)
    VERILOG_SOURCES = $(PWD)/../sv/membrane_buffer.sv
    TOPLEVEL = membrane_buffer
    MODULE = test_membrane_buffer
else ifeq ($(TEST),neuron_membrane_buffer)
    VERILOG_SOURCES = $(PWD)/../sv/neuron_membrane_buffer.sv
    TOPLEVEL = neuron_membrane_buffer
    MODULE = test_neuron_membrane_buffer
else ifeq ($(TEST),q_accumulator)
    VERILOG_SOURCES = $(PWD)/../sv/q_accumulator.sv
    TOPLEVEL = q_accumulator
    MODULE = test_q_accumulator
else
    $(error Unknown TEST value: $(TEST). Use 'lif', 'linear_layer', 'spike_buffer', 'membrane_buffer', 'neuron_membrane_buffer', or 'q_accumulator')
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
    ifeq ($(TEST),spike_buffer)
        COMPILE_ARGS += -Pspike_buffer.NUM_NEURONS=8 -Pspike_buffer.NUM_TIMESTEPS=4
    endif
    ifeq ($(TEST),membrane_buffer)
        COMPILE_ARGS += -Pmembrane_buffer.NUM_NEURONS=4 -Pmembrane_buffer.NUM_TIMESTEPS=3
    endif
    ifeq ($(TEST),neuron_membrane_buffer)
        COMPILE_ARGS += -Pneuron_membrane_buffer.NUM_TIMESTEPS=8
    endif
    ifeq ($(TEST),q_accumulator)
        COMPILE_ARGS += -Pq_accumulator.NUM_NEURONS=4 -Pq_accumulator.NUM_TIMESTEPS=4
        COMPILE_ARGS += -Pq_accumulator.NUM_ACTIONS=2 -Pq_accumulator.BATCH_SIZE=2
        COMPILE_ARGS += -Pq_accumulator.WEIGHTS_FILE=\"$(PWD)/weights/q_accum_weights.mem\"
        COMPILE_ARGS += -Pq_accumulator.BIAS_FILE=\"$(PWD)/weights/q_accum_bias.mem\"
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

test_spike_buffer:
	$(MAKE) TEST=spike_buffer

test_membrane_buffer:
	$(MAKE) TEST=membrane_buffer

test_neuron_membrane_buffer:
	$(MAKE) TEST=neuron_membrane_buffer

test_q_accumulator:
	$(MAKE) TEST=q_accumulator

# Clean up generated files
clean::
	rm -rf $(RESULTS_DIR)/*
	rm -rf __pycache__
	rm -rf .pytest_cache

.PHONY: clean test_lif test_linear_layer test_spike_buffer test_membrane_buffer test_neuron_membrane_buffer test_q_accumulator
