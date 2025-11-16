# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
DEBUG_FLAGS = -g -O0

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
LIB_DIR = lib
BIN_DIR = bin

# Library name
LIB_NAME = mlp
STATIC_LIB = $(LIB_DIR)/lib$(LIB_NAME).a
SHARED_LIB = $(LIB_DIR)/lib$(LIB_NAME).so

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

# XOR training executable
XOR_SRC = train_xor.cpp
XOR_BIN = $(BIN_DIR)/train_xor

# Create necessary directories
.PHONY: directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(BIN_DIR)

# Build static library
.PHONY: static
static: $(STATIC_LIB)

$(STATIC_LIB): $(OBJECTS)
	ar rcs $@ $^
	@echo "Static library created: $@"

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Build train_xor executable
.PHONY: train_xor
train_xor: directories static
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(XOR_SRC) -L$(LIB_DIR) -l$(LIB_NAME) -o $(XOR_BIN)
	@echo "Train XOR executable created: $(XOR_BIN)"

# Debug build
.PHONY: debug
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean all

# Clean build files
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR)
	@echo "Cleaned build files"
