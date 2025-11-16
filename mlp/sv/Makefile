# Makefile for Neural Network Verilator Simulation

# Source files
TB = nn_tb.sv
SOURCES = neural_network.sv hidden_neuron.sv output_neuron.sv
SIM_MAIN = sim_main.cpp

# Build directory and binary
BUILD_DIR = obj_dir
BINARY = $(BUILD_DIR)/Vnn_tb

# VCD waveform file
VCD_FILE = nn_tb.vcd

# Verilator flags
VFLAGS = --cc --exe --build -j 0 --trace --timing

# Default target: compile and run
.PHONY: all
all: run

# Compile the simulation
.PHONY: compile
compile: $(BINARY)

$(BINARY): $(TB) $(SOURCES) $(SIM_MAIN)
	verilator $(VFLAGS) $(SIM_MAIN) $(TB) $(SOURCES)

# Run the simulation
.PHONY: run
run: $(BINARY)
	./$(BINARY)

# Open waveform in GTKWave
.PHONY: wave
wave: $(VCD_FILE)
	gtkwave $(VCD_FILE)

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(VCD_FILE)

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all     - Compile and run simulation (default)"
	@echo "  compile - Compile the simulation with Verilator"
	@echo "  run     - Run the compiled simulation binary"
	@echo "  wave    - Open waveform in GTKWave"
	@echo "  clean   - Remove build artifacts and waveforms"
	@echo "  help    - Show this help message"
