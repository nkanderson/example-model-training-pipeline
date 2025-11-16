//////////////////////////////////////////////////////////////
// hidden_neuron.sv - Hidden neuron for an MLP
//
//
// Description:
// ------------
// This module defines basic neuron functionalities for a multi-layer perceptron (MLP).
// It includes weighted sum computation and a ReLU activation function.
// Trained on XOR dataset for binary classification.
//
//////////////////////////////////////////////////////////////

module hidden_neuron #(parameter NUM_INPUTS = 2, parameter DATA_WIDTH = 16) (
    input logic inputs [NUM_INPUTS],
    input logic signed [DATA_WIDTH-1:0] weights [NUM_INPUTS],
    input logic signed [DATA_WIDTH-1:0] bias,
    output logic signed [DATA_WIDTH-1:0] neuron_out
);

    // Internal signal for weighted sum
    logic signed [DATA_WIDTH-1:0] weighted_sum;

    // Compute weighted sum
    always_comb begin
        weighted_sum = bias;
        for (int i = 0; i < NUM_INPUTS; i++) begin
            if (inputs[i])
                weighted_sum += weights[i];
        end
    end

    // ReLU activation function
    always_comb begin
        if (weighted_sum > 0)
            neuron_out = weighted_sum;
        else
            neuron_out = '0;
    end
endmodule
