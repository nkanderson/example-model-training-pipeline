#include "mlp.h"
#include <iostream>
#include <vector>

int main() {

  // Training a network to learn XOR function
  std::cout << "Training a network to learn XOR" << std::endl;

  // Create network with random weights
  mlp::MLP xor_network(2, 2); // 2 inputs, 2 hidden neurons

  // XOR training data
  std::vector<std::vector<float>> xor_inputs = {
      {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
  std::vector<float> xor_targets = {0.0f, 1.0f, 1.0f, 0.0f};

  std::cout << "  Before training:" << std::endl;
  for (size_t i = 0; i < xor_inputs.size(); ++i) {
    float output = xor_network.forward(xor_inputs[i]);
    std::cout << "    [" << xor_inputs[i][0] << ", " << xor_inputs[i][1]
              << "] -> " << output << " (target: " << xor_targets[i] << ")"
              << std::endl;
  }

  // Train the network
  std::cout << "  Training for 5000 epochs..." << std::endl;
  xor_network.train(xor_inputs, xor_targets, 5000, 0.5f);

  std::cout << "  After training:" << std::endl;
  for (size_t i = 0; i < xor_inputs.size(); ++i) {
    float output = xor_network.forward(xor_inputs[i]);
    std::cout << "    [" << xor_inputs[i][0] << ", " << xor_inputs[i][1]
              << "] -> " << output << " (target: " << xor_targets[i] << ")"
              << std::endl;
  }

  // Save the trained weights
  std::cout << "  Saving weights to file..." << std::endl;
  xor_network.save_weights();
  std::cout << "  Weights saved to weights/mlp_2_2.txt" << std::endl;
  std::cout << std::endl;

  return 0;
}
