#ifndef MLP_H
#define MLP_H

#include <ostream>
#include <vector>

namespace mlp {

/**
 * @brief Multi-Layer Perceptron class
 *
 * A basic implementation of a multi-layer perceptron neural network
 * for boolean prediction.
 */
class MLP {
public:
  /**
   * @brief Construct a new MLP object
   *
   * @param input_size Number of input neurons
   * @param hidden_layer_size Number of neurons in the hidden layer (default: 2)
   * @param hidden_weights Weights and biases for the hidden layer
   * (Input→Hidden). Each inner vector contains weights and bias for one hidden
   * neuron (input_size + 1 (for bias) elements each).
   * @param output_weights Weights and bias for the output layer
   * (Hidden→Output). Single vector with hidden_layer_size + 1 elements.
   */
  explicit MLP(unsigned int input_size, unsigned int hidden_layer_size = 2,
               const std::vector<std::vector<float>> &hidden_weights =
                   std::vector<std::vector<float>>(),
               const std::vector<float> &output_weights = std::vector<float>());

  /**
   * @brief Destroy the MLP object
   */
  ~MLP();

  /**
   * @brief Forward propagation through the network
   *
   * @param inputs Input vector (must have size equal to input_size)
   * @return float Output prediction (sigmoid activated)
   */
  float forward(const std::vector<float> &inputs) const;

  /**
   * @brief Train the network using backpropagation
   *
   * @param training_inputs Vector of training input samples
   * @param training_targets Vector of target outputs (one per sample)
   * @param epochs Number of training iterations
   * @param learning_rate Learning rate for weight updates (default: 0.1)
   */
  void train(const std::vector<std::vector<float>> &training_inputs,
             const std::vector<float> &training_targets, unsigned int epochs,
             float learning_rate = 0.1f);

  /**
   * @brief Save weights and biases to a file
   *
   * Saves to a file named mlp_<input_size>_<hidden_size>.txt
   * Format: Each line contains weights followed by bias for one neuron/layer
   */
  void save_weights() const;

  /**
   * @brief Stream insertion operator for printing MLP
   */
  friend std::ostream &operator<<(std::ostream &os, const MLP &mlp);

private:
  /**
   * @brief Generate random weights
   *
   * @param size Number of weights to generate
   * @return std::vector<float> Vector of random weights in range [-1.0, 1.0]
   */
  static std::vector<float> generate_random_weights(size_t size);

  /**
   * @brief Sigmoid activation function
   *
   * @param x Input value
   * @return float Sigmoid output in range (0, 1)
   */
  static float sigmoid(float x);

  /**
   * @brief Derivative of sigmoid function
   *
   * @param sigmoid_output Output of sigmoid function
   * @return float Derivative value
   */
  static float sigmoid_derivative(float sigmoid_output);

  unsigned int input_size_;
  unsigned int hidden_layer_size_;
  std::vector<std::vector<float>> hidden_weights_; // Input→Hidden
  std::vector<float> output_weights_;              // Hidden→Output
};

} // namespace mlp

#endif // MLP_H
