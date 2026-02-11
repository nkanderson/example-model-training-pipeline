# Makefile for running cocotb tests on SNN modules
#
# Usage:
#   make                             # Run LIF tests (default)
#   make test_lif                    # Run LIF (single-step) tests
#   make test_lif_timestep           # Run LIF with internal timestep loop tests
#   make test_linear_layer           # Run linear_layer tests
#   make test_spike_buffer           # Run spike_buffer tests
#   make test_spike_buffer_staggered # Run spike_buffer_staggered tests (legacy)
#   make test_membrane_buffer        # Run membrane_buffer tests
#   make test_neuron_membrane_buffer # Run neuron_membrane_buffer tests
#   make clean                       # Clean build artifacts

# Default test module
TEST ?= lif

# Configuration based on TEST variable
ifeq ($(TEST),lif)
    VERILOG_SOURCES = $(PWD)/../sv/lif.sv
    TOPLEVEL = lif
    MODULE = test_lif
else ifeq ($(TEST),lif_timestep)
    VERILOG_SOURCES = $(PWD)/../sv/lif_timestep.sv
    TOPLEVEL = lif_timestep
    MODULE = test_lif_timestep
else ifeq ($(TEST),linear_layer)
    VERILOG_SOURCES = $(PWD)/../sv/linear_layer.sv
    TOPLEVEL = linear_layer
    MODULE = test_linear_layer
else ifeq ($(TEST),spike_buffer)
    VERILOG_SOURCES = $(PWD)/../sv/spike_buffer.sv
    TOPLEVEL = spike_buffer
    MODULE = test_spike_buffer
else ifeq ($(TEST),spike_buffer_staggered)
    VERILOG_SOURCES = $(PWD)/../sv/spike_buffer_staggered.sv
    TOPLEVEL = spike_buffer_staggered
    MODULE = test_spike_buffer_staggered
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
else ifeq ($(TEST),neural_network)
    VERILOG_SOURCES = $(PWD)/../sv/neural_network.sv \
                      $(PWD)/../sv/linear_layer.sv \
                      $(PWD)/../sv/lif.sv \
                      $(PWD)/../sv/spike_buffer.sv \
                      $(PWD)/../sv/neuron_membrane_buffer.sv \
                      $(PWD)/../sv/q_accumulator.sv
    TOPLEVEL = neural_network
    MODULE = test_neural_network
else
    $(error Unknown TEST value: $(TEST). Use 'lif', 'lif_timestep', 'linear_layer', 'spike_buffer', 'spike_buffer_staggered', 'membrane_buffer', 'neuron_membrane_buffer', 'q_accumulator', or 'neural_network')
endif

# Simulator - use Icarus for all tests (event-driven, correct timing)
SIM := icarus

# Simulator-specific flags
# Icarus and Verilator differ in their naming of parameter overrides
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
    ifeq ($(TEST),spike_buffer_staggered)
        COMPILE_ARGS += -Pspike_buffer_staggered.NUM_NEURONS=8 -Pspike_buffer_staggered.NUM_TIMESTEPS=4
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
    ifeq ($(TEST),neural_network)
        COMPILE_ARGS += -Pneural_network.NUM_INPUTS=4 -Pneural_network.HL1_SIZE=8
        COMPILE_ARGS += -Pneural_network.HL2_SIZE=4 -Pneural_network.NUM_ACTIONS=2
        COMPILE_ARGS += -Pneural_network.NUM_TIMESTEPS=4 -Pneural_network.Q_BATCH_SIZE=2
        COMPILE_ARGS += -Pneural_network.FC1_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc1_weights.mem\"
        COMPILE_ARGS += -Pneural_network.FC1_BIAS_FILE=\"$(PWD)/weights/nn_fc1_bias.mem\"
        COMPILE_ARGS += -Pneural_network.FC2_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc2_weights.mem\"
        COMPILE_ARGS += -Pneural_network.FC2_BIAS_FILE=\"$(PWD)/weights/nn_fc2_bias.mem\"
        COMPILE_ARGS += -Pneural_network.FC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc_out_weights.mem\"
        COMPILE_ARGS += -Pneural_network.FC_OUT_BIAS_FILE=\"$(PWD)/weights/nn_fc_out_bias.mem\"
    endif
endif

# Verilator should be used when needing to trace unpacked arrays
ifeq ($(SIM),verilator)
    # Enable waveform tracing (VCD format)
    WAVES = 1
    EXTRA_ARGS += --trace --trace-fst
    # Verilator uses -G for parameter overrides
    ifeq ($(TEST),linear_layer)
        COMPILE_ARGS += -GNUM_INPUTS=4 -GNUM_OUTPUTS=4
        COMPILE_ARGS += -GWEIGHTS_FILE=\"$(PWD)/weights/test_weights.mem\"
        COMPILE_ARGS += -GBIAS_FILE=\"$(PWD)/weights/test_bias.mem\"
    endif
    ifeq ($(TEST),spike_buffer)
        COMPILE_ARGS += -GNUM_NEURONS=8 -GNUM_TIMESTEPS=4
    endif
    ifeq ($(TEST),spike_buffer_staggered)
        COMPILE_ARGS += -GNUM_NEURONS=8 -GNUM_TIMESTEPS=4
    endif
    ifeq ($(TEST),membrane_buffer)
        COMPILE_ARGS += -GNUM_NEURONS=4 -GNUM_TIMESTEPS=3
    endif
    ifeq ($(TEST),neuron_membrane_buffer)
        COMPILE_ARGS += -GNUM_TIMESTEPS=8
    endif
    ifeq ($(TEST),q_accumulator)
        COMPILE_ARGS += -GNUM_NEURONS=4 -GNUM_TIMESTEPS=4
        COMPILE_ARGS += -GNUM_ACTIONS=2 -GBATCH_SIZE=2
        COMPILE_ARGS += -GWEIGHTS_FILE=\"$(PWD)/weights/q_accum_weights.mem\"
        COMPILE_ARGS += -GBIAS_FILE=\"$(PWD)/weights/q_accum_bias.mem\"
    endif
    ifeq ($(TEST),neural_network)
        COMPILE_ARGS += -GNUM_INPUTS=4 -GHL1_SIZE=8
        COMPILE_ARGS += -GHL2_SIZE=4 -GNUM_ACTIONS=2
        COMPILE_ARGS += -GNUM_TIMESTEPS=4 -GQ_BATCH_SIZE=2
        COMPILE_ARGS += -GFC1_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc1_weights.mem\"
        COMPILE_ARGS += -GFC1_BIAS_FILE=\"$(PWD)/weights/nn_fc1_bias.mem\"
        COMPILE_ARGS += -GFC2_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc2_weights.mem\"
        COMPILE_ARGS += -GFC2_BIAS_FILE=\"$(PWD)/weights/nn_fc2_bias.mem\"
        COMPILE_ARGS += -GFC_OUT_WEIGHTS_FILE=\"$(PWD)/weights/nn_fc_out_weights.mem\"
        COMPILE_ARGS += -GFC_OUT_BIAS_FILE=\"$(PWD)/weights/nn_fc_out_bias.mem\"
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

test_lif_timestep:
	$(MAKE) TEST=lif_timestep

test_linear_layer:
	$(MAKE) TEST=linear_layer

test_spike_buffer:
	$(MAKE) TEST=spike_buffer

test_spike_buffer_staggered:
	$(MAKE) TEST=spike_buffer_staggered

test_membrane_buffer:
	$(MAKE) TEST=membrane_buffer

test_neuron_membrane_buffer:
	$(MAKE) TEST=neuron_membrane_buffer

test_q_accumulator:
	$(MAKE) TEST=q_accumulator

test_neural_network:
	$(MAKE) TEST=neural_network

# Clean up generated files
clean::
	rm -rf $(RESULTS_DIR)/*
	rm -rf __pycache__
	rm -rf .pytest_cache

.PHONY: clean test_lif test_lif_timestep test_linear_layer test_spike_buffer test_spike_buffer_staggered test_membrane_buffer test_neuron_membrane_buffer test_q_accumulator test_neural_network
