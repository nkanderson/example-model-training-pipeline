#include "mlp.h"
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace mlp {

std::vector<float> MLP::generate_random_weights(size_t size) {
  std::random_device rd;  // True random seed
  std::mt19937 gen(rd()); // Fast PRNG (Mersenne Twister)
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // Range: [-1.0, 1.0]

  std::vector<float> weights(size);
  for (size_t i = 0; i < size; ++i) {
    weights[i] = dist(gen);
  }
  return weights;
}

float MLP::relu(float x) { return x > 0.0f ? x : 0.0f; }

float MLP::relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }

// NOTE: Sigmoid and its derivative are not currently used in training or
// inference. Sigmoid requires expensive exponential calculations (std::exp)
// which are difficult to implement efficiently in hardware. ReLU is preferred
// for hardware implementations due to its simplicity (just a comparator and
// multiplexer).
float MLP::sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float MLP::sigmoid_derivative(float sigmoid_output) {
  // Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  return sigmoid_output * (1.0f - sigmoid_output);
}

MLP::MLP(unsigned int input_size, unsigned int hidden_layer_size,
         const std::vector<std::vector<float>> &hidden_weights,
         const std::vector<float> &output_weights)
    : input_size_(input_size), hidden_layer_size_(hidden_layer_size),
      hidden_weights_(hidden_weights), output_weights_(output_weights) {

  // Initialize or validate hidden_weights (Input→Hidden)
  // Expected: hidden_layer_size vectors, each with input_size + 1 (for bias)
  // elements
  const size_t expected_weights_per_hidden_neuron = input_size_ + 1;

  if (hidden_weights_.empty()) {
    // Initialize with random weights for each hidden neuron
    hidden_weights_.resize(hidden_layer_size_);
    for (size_t i = 0; i < hidden_layer_size_; ++i) {
      hidden_weights_[i] =
          generate_random_weights(expected_weights_per_hidden_neuron);
    }
  } else {
    // Validate structure
    if (hidden_weights_.size() != hidden_layer_size_) {
      throw std::invalid_argument("hidden_weights size mismatch: expected " +
                                  std::to_string(hidden_layer_size_) +
                                  " neurons but got " +
                                  std::to_string(hidden_weights_.size()));
    }
    // Validate each neuron's weights
    for (size_t i = 0; i < hidden_weights_.size(); ++i) {
      if (hidden_weights_[i].size() != expected_weights_per_hidden_neuron) {
        throw std::invalid_argument(
            "hidden_weights[" + std::to_string(i) +
            "] size mismatch: expected " +
            std::to_string(expected_weights_per_hidden_neuron) + " but got " +
            std::to_string(hidden_weights_[i].size()));
      }
    }
  }

  // Initialize or validate output_weights (Hidden→Output)
  // Expected size: hidden_layer_size + 1 (for bias)
  const size_t expected_output_weights_size = hidden_layer_size_ + 1;

  if (output_weights_.empty()) {
    // Initialize with random weights
    output_weights_ = generate_random_weights(expected_output_weights_size);
  } else {
    // Validate size
    if (output_weights_.size() != expected_output_weights_size) {
      throw std::invalid_argument("output_weights size mismatch: expected " +
                                  std::to_string(expected_output_weights_size) +
                                  " but got " +
                                  std::to_string(output_weights_.size()));
    }
  }
}

MLP::~MLP() {
  // Cleanup if needed
}

float MLP::forward(const std::vector<float> &inputs) const {
  // Validate input size
  if (inputs.size() != input_size_) {
    throw std::invalid_argument("Input size mismatch: expected " +
                                std::to_string(input_size_) + " but got " +
                                std::to_string(inputs.size()));
  }

  // Forward propagation through hidden layer
  std::vector<float> hidden_outputs(hidden_layer_size_);
  for (size_t i = 0; i < hidden_layer_size_; ++i) {
    // Compute weighted sum (MAC operation) for hidden neuron i
    float sum = 0.0f;

    // Add weighted inputs
    for (size_t j = 0; j < input_size_; ++j) {
      sum += inputs[j] * hidden_weights_[i][j];
    }

    // Add bias (last element in weights vector)
    sum += hidden_weights_[i][input_size_];

    // Apply activation function
    hidden_outputs[i] = relu(sum);
  }

  // Forward propagation through output layer
  float output_sum = 0.0f;

  // Add weighted hidden outputs
  for (size_t i = 0; i < hidden_layer_size_; ++i) {
    output_sum += hidden_outputs[i] * output_weights_[i];
  }

  // Add bias (last element in output_weights)
  output_sum += output_weights_[hidden_layer_size_];

  // No activation function on output (linear output for hardware efficiency)
  // Output can be any value; threshold at 0.5 for binary classification
  return output_sum;
}

void MLP::train(const std::vector<std::vector<float>> &training_inputs,
                const std::vector<float> &training_targets, unsigned int epochs,
                float learning_rate) {
  // Validate training data
  if (training_inputs.empty() || training_targets.empty()) {
    throw std::invalid_argument("Training data cannot be empty");
  }
  if (training_inputs.size() != training_targets.size()) {
    throw std::invalid_argument(
        "Number of training inputs must match number of targets");
  }

  // Training loop
  for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
    // Iterate through each training sample
    for (size_t sample = 0; sample < training_inputs.size(); ++sample) {
      const std::vector<float> &inputs = training_inputs[sample];
      float target = training_targets[sample];

      // Validate input size
      if (inputs.size() != input_size_) {
        throw std::invalid_argument("Training input size mismatch at sample " +
                                    std::to_string(sample));
      }

      // === Forward Pass ===
      // Compute hidden layer outputs, same as in forward method
      std::vector<float> hidden_outputs(hidden_layer_size_);
      for (size_t i = 0; i < hidden_layer_size_; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < input_size_; ++j) {
          sum += inputs[j] * hidden_weights_[i][j];
        }
        sum += hidden_weights_[i][input_size_]; // Add bias
        hidden_outputs[i] = relu(sum);
      }

      // Compute output
      float output_sum = 0.0f;
      for (size_t i = 0; i < hidden_layer_size_; ++i) {
        output_sum += hidden_outputs[i] * output_weights_[i];
      }
      output_sum += output_weights_[hidden_layer_size_]; // Add bias
      float output = output_sum; // Linear output (no activation)

      // === Backward Pass ===
      // See Nielsen, "Neural Networks and Deep Learning" (2019), Chapter 2
      // http://neuralnetworksanddeeplearning.com/chap2.html
      //
      // Compute output layer error
      // Assumes a quadratic cost function: C = 1/2 * (target - output)^2
      // dC/doutput = output - target
      // See equations BP1 and 30 in Nielsen (2019)
      float output_cost = output - target;
      // Applying relu_derivative to the output layer would kill the gradient,
      // making training impossible.
      // float output_delta = output_cost * relu_derivative(output_sum);
      // Linear output has derivative of 1, so output_delta is just the cost
      float output_delta = output_cost;

      // Compute hidden layer errors
      // See equation BP2 in Nielsen (2019)
      std::vector<float> hidden_deltas(hidden_layer_size_);
      for (size_t i = 0; i < hidden_layer_size_; ++i) {
        float error = output_delta * output_weights_[i];
        hidden_deltas[i] = error * relu_derivative(hidden_outputs[i]);
      }

      // === Update Weights ===
      // See Nielsen, "Neural Networks and Deep Learning" (2019), Chapters 1 & 2
      // http://neuralnetworksanddeeplearning.com/chap1.html
      // http://neuralnetworksanddeeplearning.com/chap2.html
      // In Chapter 2, Equation BP4 shows how to calcaulate change in cost with
      // respect to an individual weight.
      // Applying this equation to Equations 16 & 17 in Chapter 1 gives the
      // weight and bias update equations below, where each weight or bias is
      // moved in the direction which will reduce the overall cost.
      //
      // Update output weights
      for (size_t i = 0; i < hidden_layer_size_; ++i) {
        output_weights_[i] -= learning_rate * output_delta * hidden_outputs[i];
      }
      // See equation BP3 in Nielsen (2019)
      output_weights_[hidden_layer_size_] -=
          learning_rate * output_delta; // Update bias

      // Update hidden weights
      for (size_t i = 0; i < hidden_layer_size_; ++i) {
        for (size_t j = 0; j < input_size_; ++j) {
          hidden_weights_[i][j] -= learning_rate * hidden_deltas[i] * inputs[j];
        }
        hidden_weights_[i][input_size_] -=
            learning_rate * hidden_deltas[i]; // Update bias
      }
    }
  }
}

void MLP::save_weights() const {
  // Generate filename in weights directory
  std::string filename = "weights/mlp_" + std::to_string(input_size_) + "_" +
                         std::to_string(hidden_layer_size_) + ".txt";

  // Open file for writing
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  // Write hidden layer weights and biases
  // Each hidden neuron gets its weights and bias on separate lines
  for (size_t i = 0; i < hidden_layer_size_; ++i) {
    // Write weights for this hidden neuron
    for (size_t j = 0; j < input_size_; ++j) {
      file << hidden_weights_[i][j];
      if (j < input_size_ - 1) {
        file << " ";
      }
    }
    // Write bias on the same line
    file << " " << hidden_weights_[i][input_size_] << "\n";
  }

  // Write output layer weights and bias on one line
  for (size_t i = 0; i < hidden_layer_size_; ++i) {
    file << output_weights_[i];
    if (i < hidden_layer_size_ - 1) {
      file << " ";
    }
  }
  // Write output bias
  file << " " << output_weights_[hidden_layer_size_] << "\n";

  file.close();
}

std::ostream &operator<<(std::ostream &os, const MLP &mlp) {
  os << "MLP(\n";
  os << "  input_size: " << mlp.input_size_ << "\n";
  os << "  hidden_layer_size: " << mlp.hidden_layer_size_ << "\n";

  // Print hidden weights (Input→Hidden)
  os << "  hidden_weights (Input→Hidden): [\n";
  for (size_t i = 0; i < mlp.hidden_weights_.size(); ++i) {
    os << "    neuron " << i << ": [";
    for (size_t j = 0; j < mlp.hidden_weights_[i].size(); ++j) {
      os << mlp.hidden_weights_[i][j];
      if (j < mlp.hidden_weights_[i].size() - 1) {
        os << ", ";
      }
    }
    os << "]";
    if (i < mlp.hidden_weights_.size() - 1) {
      os << ",";
    }
    os << "\n";
  }
  os << "  ]\n";

  // Print output weights (Hidden→Output)
  os << "  output_weights (Hidden→Output): [";
  for (size_t i = 0; i < mlp.output_weights_.size(); ++i) {
    os << mlp.output_weights_[i];
    if (i < mlp.output_weights_.size() - 1) {
      os << ", ";
    }
  }
  os << "]\n";
  os << ")";

  return os;
}

} // namespace mlp
