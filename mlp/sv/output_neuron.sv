//////////////////////////////////////////////////////////////
// output_neuron.sv - Basic neuron for an MLP
//
//
// Description:
// ------------
// This module defines basic neuron functionalities for a multi-layer perceptron (MLP).
// It includes weighted sum computation and has a linear output (no activation function).
// Trained on XOR dataset for binary classification.
//
//////////////////////////////////////////////////////////////

module output_neuron #(parameter NUM_INPUTS = 2, parameter DATA_WIDTH = 16) (
    input logic signed [DATA_WIDTH-1:0] inputs [NUM_INPUTS],
    input logic signed [DATA_WIDTH-1:0] weights [NUM_INPUTS],
    input logic signed [DATA_WIDTH-1:0] bias,
    output logic signed [DATA_WIDTH-1:0] neuron_out
);

    // Internal signal for weighted sum
    // Need wider bit width for intermediate multiplication results
    logic signed [2*DATA_WIDTH-1:0] weighted_sum_extended;
    logic signed [2*DATA_WIDTH-1:0] weighted_sum_shifted;
    logic signed [2*DATA_WIDTH-1:0] bias_extended;
    logic signed [DATA_WIDTH-1:0] weighted_sum;

    // Compute weighted sum with proper fixed-point arithmetic
    // When multiplying Q3.12 * Q3.12, result is Q6.24, so we need to shift right by 12
    always_comb begin
        bias_extended = {{DATA_WIDTH{bias[DATA_WIDTH-1]}}, bias};  // Sign-extend bias
        weighted_sum_extended = bias_extended <<< 12;  // Shift to Q6.24
        for (int i = 0; i < NUM_INPUTS; i++) begin
            weighted_sum_extended += inputs[i] * weights[i];  // Q3.12 * Q3.12 = Q6.24
        end
        // Shift right by 12 to convert Q6.24 back to Q3.12
        weighted_sum_shifted = weighted_sum_extended >>> 12;
        // Take lower 16 bits
        weighted_sum = weighted_sum_shifted[DATA_WIDTH-1:0];
    end

    // Linear activation (no activation function)
    assign neuron_out = weighted_sum;
endmodule
