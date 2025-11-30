# Makefile for SNN Neural Network Verilator Simulation

# Verilator configuration
VERILATOR = verilator
VERILATOR_FLAGS = --binary --timing --trace -Wall -Wno-fatal
TOP_MODULE = neural_network_tb

# Source files
SOURCES = lif.sv neural_network.sv neural_network_tb.sv

# Build directory
BUILD_DIR = obj_dir

# Default target
all: compile run

# Compile with Verilator
compile:
	@echo "Compiling with Verilator..."
	$(VERILATOR) $(VERILATOR_FLAGS) --top-module $(TOP_MODULE) $(SOURCES)

# Run simulation
run:
	@echo "Running simulation..."
	@./$(BUILD_DIR)/V$(TOP_MODULE)

# View waveforms in GTKWave
wave:
	@echo "Opening waveform in GTKWave..."
	gtkwave dump.vcd &

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f dump.vcd

.PHONY: all compile run wave clean
